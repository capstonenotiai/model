"""
test_baseline.py — 회귀 방지 테스트

현재 기준선 [QLoRA + Cascade, 개발셋(valid.jsonl) 75건]:
  calendar_ready: >= 61/75 = 81.3%   (strict, 4필드 exact — 회귀 가드)
  location      : >= 67/75 = 89.3%   (strict)
  auto precision: >= 83.3%           (72건 auto, 12건 false auto)
  test.jsonl    : 미사용 (최종 평가용 봉인)

실사용 핵심 지표(realuse_ready / date_safe / auto_dangerous_end_error_rate)는
cascade_summary.json에 저장된다. 이 테스트는 strict 기준선의 회귀만 감시한다.

코드 수정 후 이 테스트가 깨지면 즉시 확인 필요.
"""

import json, re, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT         = Path(__file__).resolve().parents[1]
DEV_RESULTS  = ROOT / "outputs" / "cascade_tier1" / "cascade_results.json"
DEV_JSONL    = ROOT / "data"    / "valid.jsonl"

# ── 기준선 (valid.jsonl 75건) ─────────────────────────────────────────────────
TARGETS = {
    "calendar_ready_min": 61,    # 61/75 = 81.3%
    "location_min":       67,    # 67/75 = 89.3%
    "auto_precision_min": 0.83,  # 83.3% (60/72), 부동소수점 허용
    "false_auto_max":     12,    # 12건 이하
}

FIELDS = ["title", "start_date", "end_date", "location"]

def norm(v):
    if not v: return ""
    return re.sub(r"\s+", " ", str(v).strip().replace("\n", " ")).strip()

def run():
    if not DEV_RESULTS.exists():
        print(f"❌  {DEV_RESULTS} 없음 — 먼저 cascade_infer.py를 실행하세요.")
        print(f"    python scripts/cascade_infer.py")
        sys.exit(1)

    with open(DEV_RESULTS, encoding='utf-8') as f:
        results = json.load(f)
    with open(DEV_JSONL, encoding='utf-8') as f:
        samples = [json.loads(l) for l in f if l.strip()]

    assert len(results) == len(samples), f"길이 불일치: {len(results)} vs {len(samples)}"
    n = len(samples)

    # ── 지표 계산 ─────────────────────────────────────────────────────────────
    cr_count  = 0
    loc_count = 0
    auto_ok   = 0
    auto_fail = 0
    review    = 0

    for r, s in zip(results, samples):
        fp = r.get("final_prediction", {})
        gt = json.loads(s["messages"][2]["content"])

        ok_all = all(norm(fp.get(f,"")) == norm(gt.get(f,"")) for f in FIELDS)
        if ok_all: cr_count += 1

        pv_loc = norm(fp.get("location",""))
        gv_loc = norm(gt.get("location",""))
        if pv_loc == gv_loc: loc_count += 1

        # auto gate
        status = r.get("auto_register_status", "unknown")
        if status == "auto":
            if ok_all: auto_ok += 1
            else: auto_fail += 1
        else:
            review += 1

    total_auto = auto_ok + auto_fail
    precision  = auto_ok / total_auto if total_auto else 0.0

    # ── 출력 ──────────────────────────────────────────────────────────────────
    print("=" * 65)
    print("회귀 방지 테스트 — QLoRA + Cascade, 개발셋(valid.jsonl) 기준")
    print("=" * 65)
    print(f"{'테스트':<28}  {'실제':>10}  {'기준':>10}  결과")
    print("-" * 65)

    total = 0
    passed = 0
    failed = []

    checks = [
        ("calendar_ready", cr_count,  TARGETS["calendar_ready_min"], ">=", f"{cr_count}/{n}={cr_count/n:.1%}"),
        ("location",       loc_count, TARGETS["location_min"],       ">=", f"{loc_count}/{n}={loc_count/n:.1%}"),
        ("auto_precision", precision, TARGETS["auto_precision_min"], ">=", f"{precision:.1%}"),
        ("false_auto",     auto_fail, TARGETS["false_auto_max"],     "<=", f"{auto_fail}건"),
    ]

    for name, actual, expected, op, display in checks:
        total += 1
        if op == ">=": ok = actual >= expected
        else:          ok = actual <= expected
        tag = "✅ PASS" if ok else "❌ FAIL"
        if ok: passed += 1
        else:  failed.append((name, actual, expected, op))
        exp_str = f">={expected}" if op == ">=" else f"<={expected}"
        print(f"  {name:<26}  {display:>12}  {exp_str:>10}  {tag}")

    print(f"\n  auto gate 분포: auto {total_auto}건 / needs_review {review}건")

    print(f"\n{'='*65}")
    if passed == total:
        print(f"✅  전체 통과 ({passed}/{total})  —  기준선 유지")
    else:
        print(f"❌  실패 {total-passed}건:")
        for name, actual, expected, op in failed:
            print(f"    {name}: 실제={actual!r}  기준={expected!r}  ({op})")
        sys.exit(1)


if __name__ == "__main__":
    run()
