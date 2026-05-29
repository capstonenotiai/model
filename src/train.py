import unsloth  # must be first — patches transformers/trl before they load

import json
import os
import re
import time

from datasets import Dataset, DatasetDict
from huggingface_hub import snapshot_download
from unsloth import FastLanguageModel
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from config import BASE_MODEL_NAME, OUTPUT_DIR, RANDOM_SEED, SYSTEM_PROMPT, TRAIN_DATA_PATH, VALID_DATA_PATH


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SEQ_LENGTH = 2048
DEBUG_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "debug")


def log(message):
    print(f"[train.py] {time.strftime('%Y-%m-%d %H:%M:%S')} | {message}", flush=True)


def find_latest_checkpoint(output_dir):
    if not os.path.isdir(output_dir):
        return None
    checkpoints = []
    for name in os.listdir(output_dir):
        match = re.fullmatch(r"checkpoint-(\d+)", name)
        if match:
            checkpoint_path = os.path.join(output_dir, name)
            if os.path.isdir(checkpoint_path) and os.path.exists(os.path.join(checkpoint_path, "trainer_state.json")):
                checkpoints.append((int(match.group(1)), checkpoint_path))
    return max(checkpoints, key=lambda item: item[0])[1] if checkpoints else None


def replace_system_prompt(messages):
    return [
        {"role": "system", "content": SYSTEM_PROMPT} if msg.get("role") == "system" else msg
        for msg in messages
    ]


def load_jsonl_dataset(train_path, valid_path):
    def read_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    return DatasetDict(
        {
            "train": Dataset.from_list(read_jsonl(train_path)),
            "validation": Dataset.from_list(read_jsonl(valid_path)),
        }
    )


def summarize_text_lengths(dataset, tokenizer):
    for split_name in ["train", "validation"]:
        token_lengths = []
        char_lengths = []
        for text in dataset[split_name]["text"]:
            char_lengths.append(len(text))
            token_lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        sorted_tokens = sorted(token_lengths)
        p95_index = max(0, int(len(sorted_tokens) * 0.95) - 1)
        over_limit = sum(length > MAX_SEQ_LENGTH for length in token_lengths)
        log(
            f"{split_name} text stats | samples={len(token_lengths)}, "
            f"chars_avg={sum(char_lengths) / len(char_lengths):.1f}, chars_max={max(char_lengths)}, "
            f"tokens_avg={sum(token_lengths) / len(token_lengths):.1f}, "
            f"tokens_p95={sorted_tokens[p95_index]}, tokens_max={max(token_lengths)}, "
            f"tokens_over_{MAX_SEQ_LENGTH}={over_limit}"
        )


class ProgressCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        log("trainer.train() entered")

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 0 or state.global_step % args.logging_steps == 0:
            log(f"training step begin | global_step={state.global_step}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        log(f"trainer log | step={state.global_step} | {logs}")

    def on_save(self, args, state, control, **kwargs):
        log(f"checkpoint save triggered | step={state.global_step}")


def main():
    debug_mode = os.environ.get("TRAIN_DEBUG") == "1"
    output_dir = DEBUG_OUTPUT_DIR if debug_mode else OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    log(f"debug_mode={debug_mode}")
    log(f"output_dir={output_dir}")
    log(f"train_data={TRAIN_DATA_PATH}")
    log(f"valid_data={VALID_DATA_PATH}")

    log("resolving local model path from HF cache")
    local_model_path = snapshot_download(BASE_MODEL_NAME, local_files_only=True)
    log(f"local_model_path={local_model_path}")

    log("loading model and tokenizer with Unsloth (4bit QLoRA)")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=local_model_path,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    log(f"model loaded | dtype={model.dtype}")

    log("applying LoRA with Unsloth")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
    )
    log("LoRA adapters applied")

    log("loading json datasets")
    dataset = load_jsonl_dataset(TRAIN_DATA_PATH, VALID_DATA_PATH)
    log(f"dataset loaded | train={len(dataset['train'])}, validation={len(dataset['validation'])}")

    if debug_mode:
        dataset["train"] = dataset["train"].select(range(min(8, len(dataset["train"]))))
        dataset["validation"] = dataset["validation"].select(range(min(4, len(dataset["validation"]))))
        log(f"debug subset selected | train={len(dataset['train'])}, validation={len(dataset['validation'])}")

    def format_example(example):
        messages = replace_system_prompt(example["messages"])
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    log("dataset.map formatting started")
    dataset = dataset.map(format_example)
    log("dataset.map formatting finished")
    summarize_text_lengths(dataset, tokenizer)

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=1 if debug_mode else 5,
        max_steps=5 if debug_mode else -1,
        logging_steps=1 if debug_mode else 5,
        eval_steps=5 if debug_mode else 100,
        save_steps=500 if debug_mode else 200,
        save_strategy="no" if debug_mode else "steps",
        save_total_limit=2,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        eval_strategy="steps" if debug_mode else "no",
        report_to="none",
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_first_step=True,
        torch_empty_cache_steps=50,
    )

    log("initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        callbacks=[ProgressCallback()],
    )
    log("SFTTrainer initialized")

    if os.environ.get("TRAIN_STOP_BEFORE_TRAIN") == "1":
        log("TRAIN_STOP_BEFORE_TRAIN=1, stopping before trainer.train()")
        return

    resume_checkpoint = None if debug_mode else find_latest_checkpoint(output_dir)
    if resume_checkpoint:
        log(f"resuming from checkpoint | path={resume_checkpoint}")
    else:
        log("no checkpoint resume requested; starting from scratch")

    log("calling trainer.train()")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    log("trainer.train() finished")

    if not debug_mode:
        trainer.model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

    log("training complete")
    log(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
