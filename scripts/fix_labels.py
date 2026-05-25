"""
GT 라벨 정제 (백업 후 in-place 수정):
1. location='온라인'이면서 실제 온라인 행사 근거 없는 경우 → ''
2. location에 접수 방법이 적힌 경우 → ''
3. '경 기' 같은 지역명 내부 공백 → '경기'로 합침
4. title 불가시 특수문자 제거
5. GT detail 120자 초과 → postprocess_detail로 잘라냄
6. valid.jsonl #51 연도 오류 수정: 2023 → 2026
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


def compact_region(loc: str) -> str:
    """'경 기' → '경기': 한 글자 음절이 공백으로 분리된 경우 합침."""
    parts = loc.split(" ")
    if len(parts) > 1 and all(len(p) == 1 and "가" <= p[0] <= "힣" for p in parts):
        return "".join(parts)
    return loc


def clean_title(title: str) -> str:
    return INVISIBLE_RE.sub("", title).strip()


def fix_location(gt: dict, user_content: str) -> tuple[dict, str]:
    loc = (gt.get("location") or "").strip()
    if _is_apply_method_location(loc):
        gt = dict(gt)
        gt["location"] = ""
        return gt, "location_apply"
    if loc == "온라인" and not _has_online_event(user_content):
        gt = dict(gt)
        gt["location"] = ""
        return gt, "location_online"
    compacted = compact_region(loc)
    if compacted != loc:
        gt = dict(gt)
        gt["location"] = compacted
        return gt, "location_space"
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


def fix_year(gt: dict, file_stem: str, idx: int) -> tuple[dict, bool]:
    """valid.jsonl #51 연도 오류 수정: 2023 → 2026"""
    if "valid" in file_stem and idx == 51:
        if gt.get("start_date", "").startswith("2023"):
            gt = dict(gt)
            gt["start_date"] = gt["start_date"].replace("2023", "2026")
            gt["end_date"] = gt["end_date"].replace("2023", "2026")
            return gt, True
    return gt, False


def fix_messages(sample: dict, file_stem: str = "", idx: int = -1) -> tuple[dict, dict]:
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
    gt, c = fix_year(gt, file_stem, idx)
    if c:
        changes["year_fix"] = True

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

    counts = {"location_online": 0, "location_apply": 0, "location_space": 0, "title": 0, "detail": 0, "year_fix": 0}
    out_lines = []
    for idx, line in enumerate(lines):
        sample = json.loads(line)
        fixed, changes = fix_messages(sample, path.stem, idx)
        for key in changes:
            counts[key] += 1
        out_lines.append(json.dumps(fixed, ensure_ascii=False))

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines) + "\n")

    print(f"  location(온라인) 수정:   {counts['location_online']}건")
    print(f"  location(접수방법) 수정: {counts['location_apply']}건")
    print(f"  location(공백) 수정:     {counts['location_space']}건")
    print(f"  title 수정:              {counts['title']}건")
    print(f"  detail 수정:             {counts['detail']}건")
    print(f"  연도 수정:               {counts['year_fix']}건")
    print(f"  총 {len(out_lines)}건 → 저장 완료")


for fpath in TARGET_FILES:
    process_file(fpath)

print("\n모든 파일 정제 완료.")
