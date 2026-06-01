"""
test_baseline.py — 회귀 방지 테스트

현재 기준선 (v10+cascade):
  구 valid(147건) 기준:
    safe fix 후 baseline_cr : 116/147 = 78.9%
    cascade 후  final_cr    : 123/147 = 83.7%
    regressed               : 0건
    tier2_status            : deferred
    auto_precision          : >= 85.4%  (111/130)
    false_auto              : <= 19건
    needs_review            : <= 17건

  공정 기준 (new_dev_fixed 75건):
    calendar_ready          : 59/75 = 78.7%
    location                : 65/75 = 86.7%
    auto_precision          : 80.6%  (72건 auto, 14건 false auto)

cascade_summary.json과 cascade_results.json을 읽어서 기준선이 유지되는지 확인.
코드 수정 후 이 테스트가 깨지면 즉시 확인 필요.
"""

import json, sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT         = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "outputs" / "cascade_tier1" / "cascade_summary.json"
RESULTS_PATH = ROOT / "outputs" / "cascade_tier1" / "cascade_results.json"
VALID_JSONL  = ROOT / "data" / "valid.jsonl"

# ── 기준선 ────────────────────────────────────────────────────────────────────
TARGETS = {
    "baseline_cr":      116,    # safe fix 후, cascade 전 (v10, 구 valid 기준)
    "final_cr":         123,    # cascade 후 (v10, 구 valid 기준)
    "regressed":        0,      # 회귀 0건
    "tier2_status":     "deferred",
    # auto gate v2 — 111/130 = 85.4%
    "auto_ok_min":      111,    # auto 중 정답 건수 >= 111건
    "false_auto":       19,     # auto 중 오답 건수 <= 19건
    "needs_review":     17,     # needs_review 건수 <= 17건
}

# ── 테스트 실행 ────────────────────────────────────────────────────────────────

def run():
    total = 0
    passed = 0
    failed = []

    # 1. summary 파일 로드
    if not SUMMARY_PATH.exists():
        print(f"❌  {SUMMARY_PATH} 없음 — cascade_infer.py를 먼저 실행하세요.")
        sys.exit(1)

    with open(SUMMARY_PATH, encoding='utf-8') as f:
        summary = json.load(f)

    # 2. cascade_results 로드 (auto_register 통계용)
    with open(RESULTS_PATH, encoding='utf-8') as f:
        results = json.load(f)

    # 3. 최신 GT로 final_cr 재계산 (valid_new.jsonl 기준)
    valid_data = []
    with open(VALID_JSONL, encoding='utf-8') as f:
        for line in f:
            if line.strip(): valid_data.append(json.loads(line))

    import re as _re
    def norm(v):
        if not v: return ""
        t = str(v).strip().replace("\n", " ")
        return _re.sub(r"\s+", " ", t).strip()
    def cr(p, g):
        return all(norm(p.get(f,"")) == norm(g.get(f,""))
                   for f in ["title","start_date","end_date","location"])

    real_final = 0
    real_regressed = 0
    for e, sample in zip(results, valid_data):
        gt_raw = json.loads(sample["messages"][2]["content"])
        gt     = {f: norm(gt_raw.get(f,"")) for f in ["title","start_date","end_date","location","detail"]}
        fp     = e.get("final_prediction", e.get("final_pred", {}))
        sfp    = e.get("safe_fixed_prediction", e.get("fixed_pred", {}))
        final_ok   = cr(fp, gt)
        baseline_ok = cr(sfp, gt)
        real_final += int(final_ok)
        if baseline_ok and not final_ok:
            real_regressed += 1

    n = len(results)

    # ── 테스트 케이스 ─────────────────────────────────────────────────────────
    checks = [
        # (이름, 실제값, 기대값, 방향)
        ("baseline_cr",  summary.get("baseline_cr", -1),  TARGETS["baseline_cr"],  ">="),
        ("final_cr",     real_final,                       TARGETS["final_cr"],     ">="),
        ("regressed",    real_regressed,                   TARGETS["regressed"],    "=="),
        ("tier2_status", summary.get("tier2_status",""),   TARGETS["tier2_status"], "=="),
    ]

    print("=" * 60)
    print("Cascade Tier 1 회귀 방지 테스트")
    print("=" * 60)
    print(f"{'테스트':<22}  {'실제':>8}  {'기준':>8}  {'결과':>6}")
    print("-" * 60)

    for name, actual, expected, op in checks:
        total += 1
        if op == ">=":
            ok = actual >= expected
            cmp = f"{actual} >= {expected}"
        elif op == "==":
            ok = actual == expected
            cmp = f"{actual!r} == {expected!r}"
        else:
            ok = False
            cmp = "?"

        tag = "✅ PASS" if ok else "❌ FAIL"
        if ok: passed += 1
        else:  failed.append((name, actual, expected, op))
        print(f"  {name:<20}  {str(actual):>8}  {str(expected):>8}  {tag}")

    # ── auto gate v2 체크 ─────────────────────────────────────────────────────
    # cascade_results.json의 auto_register_status + valid GT로 precision 계산
    import re as _re2
    auto_count = rev_count = auto_ok_cnt = false_auto_cnt = 0
    for e, sample in zip(results, valid_data):
        status = e.get("auto_register_status", "unknown")
        gt_raw = json.loads(sample["messages"][2]["content"])
        gt2    = {f: norm(gt_raw.get(f,"")) for f in ["title","start_date","end_date","location","detail"]}
        fp2    = e.get("final_prediction", {})
        actual = cr(fp2, gt2)
        if status == "auto":
            auto_count += 1
            if actual: auto_ok_cnt += 1
            else:      false_auto_cnt += 1
        else:
            rev_count += 1

    auto_prec = auto_ok_cnt / auto_count if auto_count else 0.0

    ag_checks = [
        ("auto_ok(>=114)",  auto_ok_cnt,     TARGETS["auto_ok_min"],  ">="),
        ("false_auto(<=18)", false_auto_cnt,  TARGETS["false_auto"],   "<="),
        ("needs_review(<=15)", rev_count,     TARGETS["needs_review"], "<="),
    ]
    for name, actual_v, expected_v, op in ag_checks:
        total += 1
        if op == ">=":
            ok = actual_v >= expected_v
        elif op == "<=":
            ok = actual_v <= expected_v
        else:
            ok = False
        tag = "✅ PASS" if ok else "❌ FAIL"
        if ok: passed += 1
        else:  failed.append((name, actual_v, expected_v, op))
        print(f"  {name:<20}  {str(round(actual_v,4) if isinstance(actual_v,float) else actual_v):>8}  {str(expected_v):>8}  {tag}")

    print(f"\n  auto_register 분포 (auto gate v2):")
    print(f"    auto         : {auto_count}/{n}  precision={auto_prec*100:.1f}%  false={false_auto_cnt}건")
    print(f"    needs_review : {rev_count}/{n}")

    # ── 결과 ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if passed == total:
        print(f"✅  전체 통과 ({passed}/{total})")
        print(f"    기준선 유지: {real_final}/{n} = {real_final/n*100:.1f}%  (회귀 0건)")
    else:
        print(f"❌  실패 {total-passed}건:")
        for name, actual, expected, op in failed:
            print(f"    {name}: 실제={actual!r}  기준={expected!r}  ({op})")
        sys.exit(1)


if __name__ == "__main__":
    run()
