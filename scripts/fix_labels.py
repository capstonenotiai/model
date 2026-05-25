"""
GT 라벨 정제 (백업 후 in-place 수정):
1. location='온라인'이면서 실제 온라인 행사 근거 없는 경우 → ''
2. title 불가시 특수문자 제거
3. GT detail 120자 초과 → postprocess_detail로 잘라냄
"""
import json, re, shutil, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from postprocess import _has_online_event, _is_apply_method_location, postprocess_detail

INVISIBLE_RE = re.compile(r"[​‌‍­﻿⁠᠎​‌‍­﻿⁠᠎]")

TARGET_FILES = [
    ROOT / "data" / "train.jsonl",
    ROOT / "data" / "valid.jsonl",
]


def clean_title(title: str) -> str:
    return INVISIBLE_RE.sub("", title).strip()


def fix_location(gt: dict, user_content: str) -> tuple[dict, str]:
    """Returns (gt, change_key) where change_key is '' if no change."""
    loc = (gt.get("location") or "").strip()
    # 접수 방법이 location에 들어온 경우 → ''
    if _is_apply_method_location(loc):
        gt = dict(gt)
        gt["location"] = ""
        return gt, "location_apply"
    # '온라인'인데 실제 온라인 행사 근거 없음 → ''
    if loc == "온라인" and not _has_online_event(user_content):
        gt = dict(gt)
        gt["location"] = ""
        return gt, "location"
    return gt, ""


def fix_title(gt: dict) -> tuple[dict, bool]:
    title = gt.get("title", "")
    cleaned = clean_title(title)
    if cleaned != title:
        gt = dict(gt)
        gt["title"] = cleaned
        return gt, True
    return gt, False


def fix_detail(gt: dict) -> tuple[dict, bool]:
    detail = gt.get("detail", "")
    trimmed = postprocess_detail(detail)
    if trimmed != detail:
        gt = dict(gt)
        gt["detail"] = trimmed
        return gt, True
    return gt, False


def fix_messages(sample: dict) -> tuple[dict, dict]:
    msgs = sample["messages"]
    user_content = msgs[1]["content"]
    try:
        gt = json.loads(msgs[2]["content"])
    except Exception:
        return sample, {}

    changes = {}
    gt, loc_key = fix_location(gt, user_content)
    if loc_key:
        changes[loc_key] = True

    gt, c = fix_title(gt)
    if c:
        changes["title"] = True

    gt, c = fix_detail(gt)
    if c:
        changes["detail"] = True

    if not changes:
        return sample, {}

    new_msgs = list(msgs)
    new_msgs[2] = dict(msgs[2])
    new_msgs[2]["content"] = json.dumps(gt, ensure_ascii=False)
    new_sample = dict(sample)
    new_sample["messages"] = new_msgs
    return new_sample, changes


def process_file(path: Path):
    backup = path.with_suffix(".jsonl.bak")
    shutil.copy2(path, backup)
    print(f"\n처리: {path.name} (백업: {backup.name})")

    with open(path, encoding="utf-8") as f:
        lines = [l for l in f if l.strip()]

    counts = {"location": 0, "title": 0, "detail": 0, "location_apply": 0}
    out_lines = []
    for line in lines:
        sample = json.loads(line)
        fixed, changes = fix_messages(sample)
        for key in changes:
            counts[key] += 1
        out_lines.append(json.dumps(fixed, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"  location(온라인) 수정: {counts['location']}건")
    print(f"  location(접수방법) 수정: {counts['location_apply']}건")
    print(f"  title 수정:    {counts['title']}건")
    print(f"  detail 수정:   {counts['detail']}건")
    print(f"  총 {len(out_lines)}건 → 저장 완료")


for fpath in TARGET_FILES:
    process_file(fpath)

print("\n모든 파일 정제 완료.")
