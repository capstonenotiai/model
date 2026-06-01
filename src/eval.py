import argparse
import json
import os
import re
from collections import Counter

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import (
    BASE_MODEL_NAME,
    EVAL_RESULTS_PATH,
    EVAL_SUMMARY_PATH,
    MAX_NEW_TOKENS,
    OUTPUT_DIR,
    SYSTEM_PROMPT,
    VALID_DATA_PATH,
    normalize_location,
    normalize_text,
)
from postprocess import postprocess

# ── CLI 인자 파싱 ──────────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument("--model-dir",  default=None, help="어댑터 로드 경로 (기본: config.OUTPUT_DIR)")
_parser.add_argument("--output-dir", default=None, help="결과 저장 경로 (기본: config.OUTPUT_DIR)")
_parser.add_argument("--valid-data", default=None, help="평가용 JSONL 경로 (기본: config.VALID_DATA_PATH)")
_args, _ = _parser.parse_known_args()

if _args.model_dir:
    OUTPUT_DIR = os.path.abspath(_args.model_dir)
if _args.valid_data:
    VALID_DATA_PATH = os.path.abspath(_args.valid_data)
if _args.output_dir:
    _out = os.path.abspath(_args.output_dir)
    EVAL_RESULTS_PATH = os.path.join(_out, "eval_results.json")
    EVAL_SUMMARY_PATH = os.path.join(_out, "eval_summary.json")
elif _args.model_dir:
    EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "eval_results.json")
    EVAL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "eval_summary.json")


FIELDS = ["title", "start_date", "end_date", "location", "detail"]
DETAIL_KEYWORDS = [
    "접수",
    "신청",
    "기간",
    "대상",
    "참가자격",
    "시상",
    "장소",
    "방법",
    "제출",
    "마감",
    "시간",
    "발표",
]


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def normalize_json_dict(data):
    if not isinstance(data, dict):
        data = {}
    return {field: normalize_text(data.get(field, "")) for field in FIELDS}


def normalize_title_for_metric(value):
    text = normalize_text(value).lower()
    text = re.sub(r"\bnew\b|신규", "", text, flags=re.IGNORECASE)
    text = re.sub(r"d\s*[-+]?\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}|\d{1,2}[.\-/]\d{1,2}|[^)]*마감[^)]*)\)", "", text)
    text = re.sub(r"\[(?:\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}|\d{1,2}[.\-/]\d{1,2}|[^]]*마감[^]]*)\]", "", text)
    text = text.replace("…", "").replace("...", "")
    text = re.sub(r"[^\w가-힣]+", "", text)
    return text.strip()


def extract_json_candidates(text):
    text = text.strip().replace("```json", "```").replace("```JSON", "```")
    candidates = []
    for block in re.findall(r"```(.*?)```", text, re.DOTALL):
        block = block.strip()
        if "{" in block and "}" in block:
            candidates.append(block)

    stack = []
    start_idx = None
    for idx, char in enumerate(text):
        if char == "{":
            if not stack:
                start_idx = idx
            stack.append(char)
        elif char == "}":
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    candidates.append(text[start_idx : idx + 1])
                    start_idx = None

    unique = []
    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def safe_json_load(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def find_best_json(text):
    for candidate in extract_json_candidates(text):
        parsed = safe_json_load(candidate)
        if isinstance(parsed, dict):
            return candidate, parsed
    return "", None


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


def generate_answer(model, tokenizer, messages):
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
    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def tokenize_detail(text):
    return re.findall(r"[0-9a-zA-Z가-힣]+", normalize_text(text).lower())


def detail_token_metrics(gt_detail, pred_detail):
    gt_tokens = tokenize_detail(gt_detail)
    pred_tokens = tokenize_detail(pred_detail)
    gt_counter = Counter(gt_tokens)
    pred_counter = Counter(pred_tokens)
    overlap = sum((gt_counter & pred_counter).values())
    recall = overlap / len(gt_tokens) if gt_tokens else 1.0 if not pred_tokens else 0.0
    precision = overlap / len(pred_tokens) if pred_tokens else 1.0 if not gt_tokens else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1


def detail_keyword_recall(gt_detail, pred_detail):
    gt_text = normalize_text(gt_detail)
    pred_text = normalize_text(pred_detail)
    gt_keywords = [keyword for keyword in DETAIL_KEYWORDS if keyword in gt_text]
    if not gt_keywords:
        return 1.0
    matched = [keyword for keyword in gt_keywords if keyword in pred_text]
    return len(matched) / len(gt_keywords)


def is_truncated_unclosed_json(text):
    return text.count("{") > text.count("}") and "{" in text


def rate(count, total):
    return count / total if total else 0.0


def avg(values):
    return sum(values) / len(values) if values else 0.0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = load_jsonl(VALID_DATA_PATH)

    print(f"base_model: {BASE_MODEL_NAME}")
    print(f"adapter_path: {OUTPUT_DIR}")
    print(f"valid_file: {VALID_DATA_PATH}")
    print(f"valid_samples: {len(samples)}")

    model, tokenizer = load_model()

    strict_field_correct = {field: 0 for field in FIELDS}
    valid_json_count = 0
    strict_exact_match_count = 0
    exact_wo_detail_count = 0
    exact_wo_detail_norm_loc_count = 0
    normalized_location_correct_count = 0
    date_pair_correct_count = 0
    calendar_ready_raw_count = 0
    calendar_ready_norm_loc_count = 0
    title_normalized_correct_count = 0
    invalid_json_truncated_count = 0
    json_valid_with_date_pair_count = 0
    date_pair_but_title_failed_count = 0
    detail_f1_values = []
    detail_keyword_recall_values = []
    pred_detail_lengths = []
    results = []

    for idx, sample in enumerate(tqdm(samples, desc="evaluating")):
        messages = sample["messages"]
        user_msg = messages[1]
        gt_msg = messages[2]["content"]
        infer_messages = [{"role": "system", "content": SYSTEM_PROMPT}, user_msg]

        user_content = user_msg["content"]
        pred_text = generate_answer(model, tokenizer, infer_messages)
        pred_json_text, pred_obj = find_best_json(pred_text)
        gt_obj = safe_json_load(gt_msg)
        valid_json = pred_obj is not None
        valid_json_count += int(valid_json)
        invalid_json_truncated_count += int((not valid_json) and is_truncated_unclosed_json(pred_text))

        gt = normalize_json_dict(gt_obj)
        pred_raw_normalized = normalize_json_dict(pred_obj)
        pred = postprocess(pred_raw_normalized, user_content)
        pred_detail_length = len(pred["detail"])
        pred_detail_lengths.append(pred_detail_length)

        strict_field_match = {field: gt[field] == pred[field] for field in FIELDS}
        for field, matched in strict_field_match.items():
            strict_field_correct[field] += int(matched)

        title_norm_match = normalize_title_for_metric(gt["title"]) == normalize_title_for_metric(pred["title"])
        title_normalized_correct_count += int(title_norm_match)

        location_norm_match = normalize_location(gt["location"]) == normalize_location(pred["location"])
        normalized_location_correct_count += int(location_norm_match)
        date_pair_match = strict_field_match["start_date"] and strict_field_match["end_date"]
        date_pair_correct_count += int(date_pair_match)
        json_valid_with_date_pair_count += int(valid_json and date_pair_match)
        date_pair_but_title_failed_count += int(valid_json and date_pair_match and not strict_field_match["title"])

        strict_exact = all(strict_field_match.values())
        exact_wo_detail = all(strict_field_match[field] for field in FIELDS if field != "detail")
        exact_wo_detail_norm_loc = (
            strict_field_match["title"]
            and strict_field_match["start_date"]
            and strict_field_match["end_date"]
            and location_norm_match
        )
        calendar_ready_raw = valid_json and exact_wo_detail
        calendar_ready_norm_loc = valid_json and exact_wo_detail_norm_loc

        strict_exact_match_count += int(strict_exact)
        exact_wo_detail_count += int(exact_wo_detail)
        exact_wo_detail_norm_loc_count += int(exact_wo_detail_norm_loc)
        calendar_ready_raw_count += int(calendar_ready_raw)
        calendar_ready_norm_loc_count += int(calendar_ready_norm_loc)

        _, _, detail_f1 = detail_token_metrics(gt["detail"], pred["detail"])
        detail_kw_recall = detail_keyword_recall(gt["detail"], pred["detail"])
        detail_f1_values.append(detail_f1)
        detail_keyword_recall_values.append(detail_kw_recall)

        results.append(
            {
                "index": idx,
                "pred_raw": pred_text,
                "pred_json_text": pred_json_text,
                "gt": gt,
                "pred": pred,
                "valid_json": valid_json,
                "invalid_json_truncated": (not valid_json) and is_truncated_unclosed_json(pred_text),
                "strict_field_match": strict_field_match,
                "title_normalized_match": title_norm_match,
                "location_normalized_match": location_norm_match,
                "date_pair_match": date_pair_match,
                "calendar_ready_raw": calendar_ready_raw,
                "calendar_ready_norm_loc": calendar_ready_norm_loc,
                "exact_match_without_detail": exact_wo_detail,
                "exact_match_without_detail_location_normalized": exact_wo_detail_norm_loc,
                "strict_exact_match": strict_exact,
                "detail_token_overlap_f1": detail_f1,
                "detail_keyword_recall": detail_kw_recall,
                "pred_detail_length": pred_detail_length,
                "pred_detail_over_120": pred_detail_length > 120,
            }
        )

    total = len(samples)
    summary = {
        "total": total,
        "valid_json_count": valid_json_count,
        "valid_json_rate": rate(valid_json_count, total),
        "title_accuracy": rate(strict_field_correct["title"], total),
        "title_normalized_accuracy": rate(title_normalized_correct_count, total),
        "start_date_accuracy": rate(strict_field_correct["start_date"], total),
        "end_date_accuracy": rate(strict_field_correct["end_date"], total),
        "date_pair_accuracy": rate(date_pair_correct_count, total),
        "location_accuracy": rate(strict_field_correct["location"], total),
        "location_normalized_accuracy": rate(normalized_location_correct_count, total),
        "calendar_ready_accuracy_raw": rate(calendar_ready_raw_count, total),
        "calendar_ready_accuracy_norm_loc": rate(calendar_ready_norm_loc_count, total),
        "exact_match_without_detail": rate(exact_wo_detail_count, total),
        "exact_without_detail_location_normalized": rate(exact_wo_detail_norm_loc_count, total),
        "detail_exact_accuracy": rate(strict_field_correct["detail"], total),
        "detail_token_overlap_f1": avg(detail_f1_values),
        "detail_keyword_recall": avg(detail_keyword_recall_values),
        "strict_exact_match_accuracy": rate(strict_exact_match_count, total),
        "invalid_json_truncated_count": invalid_json_truncated_count,
        "avg_pred_detail_length": avg(pred_detail_lengths),
        "pred_detail_over_120_count": sum(1 for length in pred_detail_lengths if length > 120),
        "json_valid_with_date_pair_count": json_valid_with_date_pair_count,
        "date_pair_but_title_failed_count": date_pair_but_title_failed_count,
    }

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(EVAL_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n===== eval summary =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nsaved:")
    print(f"- {EVAL_RESULTS_PATH}")
    print(f"- {EVAL_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
