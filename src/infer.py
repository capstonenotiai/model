"""
infer.py — 단일 공지 추론 (Unsloth 4-bit QLoRA)

단건 추론만 담당합니다. Cascade(GPT fallback) 파이프라인은 scripts/cascade_infer.py를 사용하세요.

사용법:
    cd src
    python infer.py --auto-user "[제목]\n공지제목\n\n[본문]\n공지내용..."
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from unsloth import FastLanguageModel

from config import BASE_MODEL_NAME, MAX_NEW_TOKENS, MAX_SEQ_LENGTH, OUTPUT_DIR, SYSTEM_PROMPT
from postprocess import postprocess


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def _extract_json(text: str):
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


def generate(model, tokenizer, user_content: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True).strip()

    pred = _extract_json(raw)
    if pred is not None:
        pred = postprocess(pred, user_content)
        return json.dumps(pred, ensure_ascii=False, indent=2)
    return raw


def main():
    parser = argparse.ArgumentParser(
        description="단일 공지 추론. Cascade 파이프라인은 scripts/cascade_infer.py 사용."
    )
    parser.add_argument(
        "--auto-user", default="",
        help='공지 전문. 형식: "[제목]\\n제목\\n\\n[본문]\\n본문내용"'
    )
    parser.add_argument("--title", default="", help="공지 제목 (--auto-user 미사용 시)")
    parser.add_argument("--body",  default="", help="공지 본문 (--auto-user 미사용 시)")
    args = parser.parse_args()

    if args.auto_user:
        user_content = args.auto_user
    elif args.title or args.body:
        user_content = f"[제목]\n{args.title}\n\n[본문]\n{args.body}"
    else:
        parser.print_help()
        sys.exit(1)

    model, tokenizer = load_model()
    print(generate(model, tokenizer, user_content))


if __name__ == "__main__":
    main()
