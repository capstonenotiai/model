import argparse
import json
import re

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import BASE_MODEL_NAME, MAX_NEW_TOKENS, OUTPUT_DIR, SYSTEM_PROMPT
from postprocess import postprocess


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        local_files_only=True,
    )
    model = PeftModel.from_pretrained(base_model, OUTPUT_DIR)
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.eval()
    return model, tokenizer


def build_user_content(args):
    if args.auto_user:
        return args.auto_user
    return (
        f"[공지 제목]\n{args.title}\n\n"
        f"[목록 날짜]\n{args.list_date}\n\n"
        f"[날짜 후보]\n{args.date_candidates}\n\n"
        f"[시간 후보]\n{args.time_candidates}\n\n"
        f"[본문 핵심]\n{args.body}"
    )


def _extract_json(text):
    text = text.strip().replace("```json", "```").replace("```JSON", "```")
    for block in re.findall(r"```(.*?)```", text, re.DOTALL):
        block = block.strip()
        if "{" in block and "}" in block:
            try:
                return json.loads(block)
            except Exception:
                pass
    stack, start = [], None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append(ch)
        elif ch == "}" and stack:
            stack.pop()
            if not stack and start is not None:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    start = None
    return None


def generate_answer(model, tokenizer, user_content):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    input_token_count = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_token_count:]
    raw = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    pred_obj = _extract_json(raw)
    if pred_obj is not None:
        pred_obj = postprocess(pred_obj, user_content)
        return json.dumps(pred_obj, ensure_ascii=False)
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto-user", default="", help="preformatted auto user message")
    parser.add_argument("--title", default="", help="notice title")
    parser.add_argument("--list-date", default="", help="list date")
    parser.add_argument("--date-candidates", default="", help="date candidates")
    parser.add_argument("--time-candidates", default="", help="time candidates")
    parser.add_argument("--body", default="", help="core notice body")
    args = parser.parse_args()

    model, tokenizer = load_model()
    print(generate_answer(model, tokenizer, build_user_content(args)))


if __name__ == "__main__":
    main()
