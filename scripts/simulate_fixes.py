"""코드 수정 효과를 기존 eval_results.json에 재적용해서 추정하는 시뮬레이션."""
import json, re, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from config import normalize_location, normalize_text

RESULTS = ROOT / "outputs" / "v1" / "eval_results.json"
VALID   = ROOT / "data" / "valid.jsonl"

with open(RESULTS, encoding="utf-8") as f:
    results = json.load(f)
samples = [json.loads(l) for l in open(VALID, encoding="utf-8") if l.strip()]

FIELDS = ["title", "start_date", "end_date", "location"]

def normalize_title_for_metric(value):
    text = normalize_text(value).lower()
    text = re.sub(r"\bnew\b|신규", "", text, flags=re.IGNORECASE)
    text = re.sub(r"d\s*[-+]?\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\((?:\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}|\d{1,2}[.\-/]\d{1,2}|[^)]*마감[^)]*)\)", "", text)
    text = re.sub(r"\[(?:\d{4}[.\-/]\d{1,2}[.\-/]\d{1,2}|\d{1,2}[.\-/]\d{1,2}|[^]]*마감[^]]*)\]", "", text)
    text = text.replace("…", "").replace("...", "")
    text = re.sub(r"[^\w가-힣]+", "", text)
    return text.strip()

total = len(results)
cal_ready = 0
title_ok = 0
date_ok = 0
loc_ok = 0

for r in results:
    idx = r["index"]
    gt = r["gt"]
    pred = r["pred"]

    gt_msg = json.loads(samples[idx]["messages"][2]["content"])
    gt_title    = normalize_text(gt_msg.get("title",""))
    gt_start    = normalize_text(gt_msg.get("start_date",""))
    gt_end      = normalize_text(gt_msg.get("end_date",""))
    gt_location = normalize_location(normalize_text(gt_msg.get("location","")))

    pred_title    = normalize_text(pred.get("title",""))
    pred_start    = normalize_text(pred.get("start_date",""))
    pred_end      = normalize_text(pred.get("end_date",""))
    pred_location = normalize_location(normalize_text(pred.get("location","")))

    t_ok = gt_title == pred_title
    t_norm_ok = normalize_title_for_metric(gt_title) == normalize_title_for_metric(pred_title)
    d_ok = (gt_start == pred_start) and (gt_end == pred_end)
    l_ok = gt_location == pred_location

    title_ok += int(t_norm_ok)
    date_ok  += int(d_ok)
    loc_ok   += int(l_ok)
    if t_ok and d_ok and l_ok:
        cal_ready += 1

print("=== 시뮬레이션: GT 라벨 수정 효과 (재학습 없음) ===")
print(f"  total            : {total}")
print(f"  title(정규화)    : {title_ok}/{total} = {title_ok/total:.1%}")
print(f"  date_pair        : {date_ok}/{total} = {date_ok/total:.1%}")
print(f"  location(정규화) : {loc_ok}/{total} = {loc_ok/total:.1%}")
print(f"  calendar_ready   : {cal_ready}/{total} = {cal_ready/total:.1%}")
print()
print("※ 이 수치는 기존 예측을 재사용한 추정치. 재학습 시 실제 성능은 더 개선됨.")
