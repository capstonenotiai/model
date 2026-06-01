"""
cascade_infer.py — Tier 1 날짜 cascade 구현

흐름:
  1. eval_results.json (v6 예측) + valid_new.jsonl 로드
  2. 새 postprocess 재적용 (사업장소 복원 포함)
  3. apply_safe_fixes() 적용 (R1, R2)
  4. DATE_TRIGGERS 기준 get_cascade_triggers() 실행
  5. trigger 없으면 fixed_pred 그대로 사용
  6. trigger 있으면 GPT-4o-mini fallback
  7. GPT 결과에 apply_safe_fixes() 적용
  8. 채택 조건 검증 → 통과 시 dates만 GPT로 교체, 실패 시 fixed_pred 유지

출력:
  outputs/cascade_tier1/cascade_results.json
  outputs/cascade_tier1/cascade_summary.json
"""

import json, os, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parents[1]

# .env 로드
env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path, encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

# ── 설정 ──────────────────────────────────────────────────────────────────────
EVAL_RESULTS  = ROOT / "outputs" / "v1" / "eval_results.json"
VALID_JSONL   = ROOT / "data" / "valid.jsonl"
OUT_DIR       = ROOT / "outputs" / "cascade_tier1"

GPT_MODEL     = "gpt-4o-mini"
GPT_TEMP      = 0.0
GPT_MAX_TOK   = 512
GPT_RETRIES   = 3

# True: API 호출 없이 cascade 대상 목록만 출력 (비용 절감 테스트용)
DRY_RUN = False

# Tier 1: 날짜만 GPT로 교체 (title/location/detail은 v6 유지)
MERGE_DATES_ONLY = True

# ── 의존성 ─────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(ROOT / "src"))
from postprocess import postprocess, _START_EXPLICIT_RE
from config import normalize_text

sys.path.insert(0, str(ROOT / "scripts"))
from cascade_trigger_sim import (
    apply_safe_fixes, get_cascade_triggers,
    T_NO_DATES, T_END_ANNOUNCE, T_YEAR_MISMATCH,
    T_END_NOT_IN_BODY, T_START_NOT_IN_BODY,
    _extract_dates, _extract_month_ev, extract_full_dates,
    FIELDS
)

DATE_TRIGGERS = {T_NO_DATES, T_END_ANNOUNCE, T_YEAR_MISMATCH,
                 T_END_NOT_IN_BODY, T_START_NOT_IN_BODY}

# ── location auto gate — 강한 장소 라벨만 (false positive 최소화) ──────────────
_STRONG_LOC_LABEL_RE = re.compile(
    r"(?:사업|행사|대회|교육|공연)\s*장소\s*[:：]?\s*"
    r"|오디션\s*장소\s*[:：]?\s*"
    r"|면접\s*장소\s*[:：]?\s*"
)
_LOC_GATE_INVALID_RE = re.compile(
    r"접수처|문의처|이메일|홈페이지|온라인\s*접수|http|@|\.[a-z]"
    r"|합격자|추후|개별|미정|별도|예정|발표|전국|\d{4}|\d{1,2}[./월]\s*\d",
    re.IGNORECASE
)

# ── OpenAI 클라이언트 ──────────────────────────────────────────────────────────
if not DRY_RUN:
    API_KEY = os.environ.get("OPENAI_API_KEY", "")
    if not API_KEY:
        print("OPENAI_API_KEY 가 설정되지 않았습니다.")
        print("  방법 1: .env 파일에  OPENAI_API_KEY=sk-proj-...  추가")
        print("  방법 2: $env:OPENAI_API_KEY = 'sk-proj-...'  실행")
        sys.exit(1)
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY)

# ── GPT 프롬프트 ───────────────────────────────────────────────────────────────
CASCADE_SYSTEM = (
    "당신은 한국어 공모전·공지문에서 캘린더 등록용 일정 정보를 추출하는 전문가입니다. "
    "반드시 JSON 객체 하나만 출력하세요. 코드블록, 설명문, 주석은 출력하지 마세요."
)

CASCADE_USER_TMPL = """{body}

위 공지문에서 일정 정보를 추출하세요.

날짜 규칙 (최우선 적용):
1. 접수/신청/모집/공모 기간이 있으면 그 시작일→start_date, 마감일→end_date로 사용하세요.
2. 마감일만 있고 시작일이 본문에 없으면 start_date는 반드시 ""로 쓰세요. start가 없을 때 end 값을 start에 복사하지 마세요.
3. 발표일·결과발표일·당첨자발표일·시상일은 start_date/end_date로 쓰지 마세요.
4. 행사일(대회일·공연일)과 접수기간이 둘 다 있으면 공지 성격에 따라 선택하세요.
   - 모집/공모 공지 → 접수기간을 start/end로
   - 행사/이벤트 공지 → 행사일을 start/end로
5. 본문에 없는 날짜를 절대 만들지 마세요.
6. 날짜는 반드시 YYYY-MM-DD 형식으로 출력하세요.

JSON 키: title, start_date, end_date, location, detail
JSON만 출력:"""


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _norm(v): return normalize_text(v) if v else ""

def _cr(pred, gt):
    return all(pred[f] == gt[f] for f in ["title", "start_date", "end_date", "location"])

def _safe_json(text):
    try:    return json.loads(text)
    except: return None

def _parse_gpt_text(text: str):
    """GPT 응답 텍스트에서 JSON 파싱."""
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text).strip()
    try:    return json.loads(text)
    except: pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:    return json.loads(m.group())
        except: pass
    return None


# ── GPT 호출 ──────────────────────────────────────────────────────────────────

def call_gpt(body: str) -> tuple:
    """
    GPT-4o-mini 호출. 반환: (parsed_dict_or_None, raw_text)
    """
    user_msg = CASCADE_USER_TMPL.format(body=body[:3000])
    for attempt in range(GPT_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=GPT_MODEL,
                messages=[
                    {"role": "system", "content": CASCADE_SYSTEM},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=GPT_TEMP,
                max_tokens=GPT_MAX_TOK,
            )
            raw = resp.choices[0].message.content.strip()
            parsed = _parse_gpt_text(raw)
            return parsed, raw
        except Exception as e:
            print(f"    [GPT 오류 attempt {attempt+1}] {e}")
            if attempt < GPT_RETRIES - 1:
                time.sleep(2)
    return None, ""


# ── GPT 결과 채택 검증 ─────────────────────────────────────────────────────────

def _year_changed_without_full_evidence(orig: str, gpt_d: str, full_dates: set) -> bool:
    """
    연도가 바뀌었는데 full date 근거가 없으면 True.
    - orig == gpt_d → 변화 없음 → False
    - orig[:4] == gpt_d[:4] → 연도 동일 → False
    - gpt_d in full_dates → 본문에 연도 명시 근거 있음 → False
    - 그 외 → 연도 변경인데 full 근거 없음 → True
    """
    if not orig or not gpt_d or orig == gpt_d:
        return False
    if orig[:4] == gpt_d[:4]:
        return False          # 연도 동일
    return gpt_d not in full_dates  # full_dates에 없으면 근거 없음


def is_adoptable(gpt_norm: dict, gpt_fixed: dict, body: str, orig_fixed: dict) -> tuple:
    """
    GPT 결과가 날짜 기준에서 신뢰할 수 있는지 검증.
    orig_fixed: safe_fix 후의 원래 pred (GPT와 비교 기준)
    반환: (채택여부, 거부이유)
    """
    gs = gpt_fixed["start_date"]
    ge = gpt_fixed["end_date"]
    os_ = orig_fixed["start_date"]   # 원래 start (GPT 비교용)
    oe_ = orig_fixed["end_date"]     # 원래 end

    body_dates   = _extract_dates(body)
    full_dates   = extract_full_dates(body)   # 연도 명시된 날짜만
    all_d        = set(body_dates)
    ann          = {d for d, rs in body_dates.items() if "announce" in rs}
    dead         = {d for d, rs in body_dates.items() if "deadline" in rs}
    rec          = {d for d, rs in body_dates.items() if "receipt"  in rs}
    deadline_any = dead | rec

    # 1. end_date 비어있는데 본문에 마감 날짜가 있으면 reject
    if not ge and deadline_any:
        return False, f"end='' 인데 본문에 마감일 {list(deadline_any)[:2]} 있음"

    # 2. start <= end (둘 다 있으면)
    if gs and ge and gs > ge:
        return False, f"start({gs}) > end({ge}): 날짜 역전"

    # 3. end가 발표/시상일 전용이면 reject
    if ge and ge in ann and ge not in dead and ge not in rec:
        return False, f"end({ge})가 발표/시상일 전용"

    # 4. start가 본문에 없는 날짜 (MD도 없음) → reject
    if gs and all_d and gs not in all_d:
        gs_md    = gs[5:]
        body_mds = {d[5:] for d in all_d}
        month_ev = _extract_month_ev(body)
        body_ms  = {d[5:7] for d in all_d} | {ym[5:7] for ym in month_ev}
        if gs_md not in body_mds and gs[5:7] not in body_ms:
            return False, f"start({gs})가 본문에 없는 hallucination"

    # 5. 연도 변경 시 full date 근거 필요 — [17] 같은 연도 오판 방지
    #    partial 날짜(default_year 적용)만으로 연도를 바꾸면 reject
    if _year_changed_without_full_evidence(os_, gs, full_dates):
        return False, f"start 연도 변경({os_}→{gs}): 본문에 full date 근거 없음"

    if _year_changed_without_full_evidence(oe_, ge, full_dates):
        return False, f"end 연도 변경({oe_}→{ge}): 본문에 full date 근거 없음"

    return True, ""


# ── 자동 등록 조건 판단 ────────────────────────────────────────────────────────

def get_auto_register_status(entry: dict, body: str = "") -> tuple:
    """
    자동 캘린더 등록 가능 여부 판단.
    반환: ("auto" | "needs_review", reason)
    """
    fp = entry["final_prediction"]
    gs = fp.get("start_date", "")
    ge = fp.get("end_date",   "")
    triggers     = entry.get("cascade_triggers", [])
    cascade_used = entry.get("cascade_used", False)
    accepted     = entry.get("accepted")

    # ── 기존 체크 ────────────────────────────────────────────────────────────
    if not ge:
        return "needs_review", "end_date 없음"
    if gs and ge and gs > ge:
        return "needs_review", f"start({gs}) > end({ge}): 날짜 역전"
    if cascade_used and accepted is False:
        reason = entry.get("reject_reason", "")
        return "needs_review", f"GPT 거부: {reason}"

    # cascade 채택 케이스: precision 40% (검증셋 기준) → 보수적 검토 필요
    if cascade_used and accepted is True:
        return "needs_review", "cascade 채택: 날짜 교정 결과 검토 필요"

    if "end=발표일_의심" in triggers and not cascade_used:
        return "needs_review", "end=발표일_의심 trigger 미해결"
    if triggers and not cascade_used:
        return "needs_review", f"날짜 trigger 미처리: {triggers}"

    # ── 날짜 body 근거 검증 ──────────────────────────────────────────────────
    if body:
        body_date_map = _extract_dates(body)
        all_d         = set(body_date_map)
        full_dates    = extract_full_dates(body)
        month_ev      = _extract_month_ev(body)

        # 1. cascade accepted인데 end_date가 본문에 없는 hallucination
        #    start check는 is_adoptable()에 있으나 end check는 없었음 — 여기서 보완
        if ge and cascade_used and accepted and all_d:
            ge_md    = ge[5:]
            body_mds = {d[5:] for d in all_d}
            body_ms  = {d[5:7] for d in all_d} | {ym[5:7] for ym in month_ev}
            if ge not in all_d and ge_md not in body_mds and ge[5:7] not in body_ms:
                return "needs_review", f"cascade end({ge})가 본문에 없는 hallucination"

        # 2. pred 날짜의 MM-DD가 본문 full date(마감/접수 역할)의 MM-DD와 같은데 연도 불일치
        #    → partial date default-year 오적용 방지 (연도만 다른 오류 검출)
        for pd in [gs, ge]:
            if not pd or len(pd) < 10:
                continue
            pred_md = pd[5:]   # "MM-DD"
            pred_yr = pd[:4]   # "YYYY"
            for fd, roles in body_date_map.items():
                if fd not in full_dates:
                    continue   # 연도가 명시된 날짜만 비교
                if fd[5:] != pred_md or fd[:4] == pred_yr:
                    continue   # MM-DD 불일치 또는 연도 일치
                if "deadline" in roles or "receipt" in roles:
                    return "needs_review", f"날짜 {pd}: 본문 마감일 {fd}와 연도 불일치"

    # ── location 신뢰도 검증 ─────────────────────────────────────────────────
    loc = fp.get("location", "")
    if not loc and body:
        m = _STRONG_LOC_LABEL_RE.search(body)
        if m:
            after     = body[m.end(): m.end() + 60].split('\n')[0]
            candidate = re.split(r'\s*[-–]\s*|[(,，/]|※', after)[0].strip()
            if (candidate
                    and not _LOC_GATE_INVALID_RE.search(candidate)
                    and re.search(r'[가-힣]{2,}', candidate)
                    and 2 <= len(candidate) <= 20):
                return "needs_review", f"location='' 인데 장소 후보 '{candidate}' 있음"

    return "auto", ""


# ── 단건 처리 공통 로직 ────────────────────────────────────────────────────────

def process_one(body: str, pred_json_text: str, gt: dict = None,
                verbose: bool = True) -> dict:
    """
    공지 1건 처리.
    batch 모드와 CLI 단건 모드 모두에서 사용.

    body          : 공지문 본문 (완전한 텍스트)
    pred_json_text: v6 모델 raw 출력 JSON 문자열
    gt            : 정답 (평가 시에만, 없으면 None)
    """
    # v6 예측 재처리
    raw_json = _safe_json(pred_json_text)
    if raw_json:
        v6_pred = postprocess({f: _norm(raw_json.get(f, "")) for f in FIELDS}, body)
    else:
        v6_pred = {f: "" for f in FIELDS}

    # safe fix
    fixed, fix_actions = apply_safe_fixes(v6_pred, body)

    # cascade trigger 탐지
    all_triggers  = get_cascade_triggers(v6_pred, fixed, body)
    date_triggers = [t for t in all_triggers if t in DATE_TRIGGERS]

    baseline_cr = _cr(fixed, gt) if gt else None

    entry = {
        # ── 예측 레이어 ─────────────────────────────────────────────────────
        "prediction":            v6_pred,
        "safe_fixed_prediction": fixed,
        "safe_fix_actions":      fix_actions,
        # ── cascade 정보 ────────────────────────────────────────────────────
        "cascade_triggers": date_triggers,
        "cascade_used":     False,
        "cascade_prediction": None,
        "gpt_raw":          None,
        "accepted":         None,
        "reject_reason":    None,
        # ── 최종 출력 ────────────────────────────────────────────────────────
        "final_prediction": fixed,
        # ── 자동 등록 판단 ──────────────────────────────────────────────────
        "auto_register_status": None,
        "auto_register_reason": None,
        # ── 평가 (gt 있을 때만 유효) ────────────────────────────────────────
        "gt":          gt,
        "baseline_cr": baseline_cr,
        "final_cr":    baseline_cr,
        "status":      "auto",
    }

    if not date_triggers:
        entry["auto_register_status"], entry["auto_register_reason"] = \
            get_auto_register_status(entry, body)
        return entry

    # cascade 대상
    if DRY_RUN:
        if verbose:
            print(f"  [DRY_RUN] cascade_triggers={date_triggers}")
            print(f"    safe_fixed: start={fixed['start_date']!r}  end={fixed['end_date']!r}")
        entry["status"] = "dry_run"
        entry["auto_register_status"], entry["auto_register_reason"] = \
            get_auto_register_status(entry, body)
        return entry

    if verbose:
        print(f"  🔄 GPT 호출  triggers={date_triggers}")

    gpt_parsed, gpt_raw = call_gpt(body)
    entry["cascade_used"] = True
    entry["gpt_raw"]      = gpt_raw

    if gpt_parsed is None:
        if verbose: print(f"    ❌ JSON 파싱 실패")
        entry["accepted"]      = False
        entry["reject_reason"] = "JSON 파싱 실패"
        entry["status"]        = "cascade_rejected"
    else:
        gpt_norm  = {f: _norm(gpt_parsed.get(f, "")) for f in FIELDS}
        gpt_fixed, _ = apply_safe_fixes(gpt_norm, body)
        entry["cascade_prediction"] = gpt_fixed

        adoptable, reason = is_adoptable(gpt_norm, gpt_fixed, body, fixed)
        if not adoptable:
            if verbose: print(f"    ❌ 채택 거부: {reason}")
            entry["accepted"]      = False
            entry["reject_reason"] = reason
            entry["status"]        = "cascade_rejected"
        else:
            final = {**fixed,
                     "start_date": gpt_fixed["start_date"],
                     "end_date":   gpt_fixed["end_date"]} if MERGE_DATES_ONLY else gpt_fixed
            entry["final_prediction"] = final
            entry["accepted"]         = True
            entry["reject_reason"]    = None
            entry["status"]           = "cascade_adopted"

            if gt:
                final_cr = _cr(final, gt)
                entry["final_cr"] = final_cr
                if verbose:
                    if final_cr and not baseline_cr:
                        print(f"    ✅ 채택 → calendar_ready: ❌→✅")
                    elif not final_cr and baseline_cr:
                        print(f"    ⚠️  채택했지만 calendar_ready: ✅→❌")
                    else:
                        print(f"    ✅ 채택 (calendar_ready: {'✅' if final_cr else '❌'})")
            else:
                if verbose: print(f"    ✅ 채택")

    entry["auto_register_status"], entry["auto_register_reason"] = \
        get_auto_register_status(entry, body)
    return entry


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main_batch():
    """배치 모드: eval_results.json 전체 처리 (평가 포함)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_RESULTS, encoding='utf-8') as f: results = json.load(f)
    valid_data = []
    with open(VALID_JSONL, encoding='utf-8') as f:
        for line in f:
            if line.strip(): valid_data.append(json.loads(line))
    assert len(results) == len(valid_data)
    total = len(results)

    print("=" * 65)
    print(f"Cascade Tier 1 — 날짜 trigger fallback  [배치 모드]")
    print(f"  DRY_RUN={DRY_RUN}  MERGE_DATES_ONLY={MERGE_DATES_ONLY}")
    print("=" * 65)

    cascade_results = []
    stats = Counter()
    trigger_dist = Counter()

    for i, (res, sample) in enumerate(zip(results, valid_data)):
        body   = sample["messages"][1]["content"]
        gt_raw = json.loads(sample["messages"][2]["content"])
        gt     = {f: _norm(gt_raw.get(f, "")) for f in FIELDS}
        pred_json_text = res.get("pred_json_text", "")

        # pred_json_text 없으면 old pred dict 사용 (fallback)
        if not pred_json_text and "pred" in res:
            pred_json_text = json.dumps(res["pred"], ensure_ascii=False)

        entry = process_one(body, pred_json_text, gt=gt, verbose=True)
        entry["index"] = i

        for t in entry["cascade_triggers"]: trigger_dist[t] += 1
        stats["baseline_ok"] += int(entry.get("baseline_cr") or 0)
        if not entry["cascade_triggers"]:
            stats["no_trigger"] += 1
        else:
            stats["triggered"] += 1
            if entry["cascade_used"]:
                stats["gpt_called"] += 1
                if entry["accepted"] is True:
                    stats["gpt_adopted"] += 1
                    if entry.get("final_cr") and not entry.get("baseline_cr"):
                        stats["improved"] += 1
                    elif not entry.get("final_cr") and entry.get("baseline_cr"):
                        stats["regressed"] += 1
                elif entry["accepted"] is False:
                    stats["gpt_rejected"] += 1

        cascade_results.append(entry)

    stats["final_ok"] = sum(int(e.get("final_cr") or 0) for e in cascade_results)

    # ── 결과 저장 ──────────────────────────────────────────────────────────
    results_path = OUT_DIR / "cascade_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(cascade_results, f, ensure_ascii=False, indent=2)

    summary = {
        # ── 기본 정보 ──────────────────────────────────────────────────────
        "model":           "final_llama8b_v6",
        "cascade_tier":    "Tier1_date_only",
        "gpt_model":       GPT_MODEL,
        "merge_dates_only": MERGE_DATES_ONLY,
        "dry_run":         DRY_RUN,
        # ── 성능 지표 ──────────────────────────────────────────────────────
        "total":           total,
        "baseline_cr":     int(stats["baseline_ok"]),
        "baseline_rate":   round(stats["baseline_ok"] / total, 4),
        "final_cr":        int(stats["final_ok"]),
        "final_rate":      round(stats["final_ok"] / total, 4),
        "improvement":     int(stats["final_ok"]) - int(stats["baseline_ok"]),
        # ── cascade 통계 ──────────────────────────────────────────────────
        "cascade_triggered": int(stats["triggered"]),
        "trigger_rate":      round(stats["triggered"] / total, 4),
        "gpt_called":        int(stats.get("gpt_called", 0)),
        "gpt_adopted":       int(stats.get("gpt_adopted", 0)),
        "gpt_rejected":      int(stats.get("gpt_rejected", 0)),
        "improved":          int(stats.get("improved", 0)),
        "regressed":         int(stats.get("regressed", 0)),
        "trigger_dist":      dict(trigger_dist),
        # ── Tier 2 보류 사유 (참고용) ──────────────────────────────────────
        "tier2_status": "deferred",
        "tier2_reason": (
            "title precision 32% (FP 17건): GT 라벨 정책 불일치 — "
            "notice의 [기관명] 대괄호를 GT 라벨러가 제거했고 모델도 따름. "
            "GPT 호출 시 대괄호를 복원해 GT와 충돌 → 회귀 위험. "
            "location precision 0% (FP 4건): GT='' 케이스에서 장소 라벨 감지 → 회귀 위험. "
            "cascade가 아닌 데이터 정비(GT 정책 통일 + 재학습) 대상으로 분류."
        ),
    }

    summary_path = OUT_DIR / "cascade_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # ── 출력 요약 ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"결과 요약")
    print(f"{'='*65}")
    print(f"  모델/cascade         : {summary['model']} + {summary['cascade_tier']}")
    print(f"  baseline             : {summary['baseline_cr']}/{total}  ({summary['baseline_rate']*100:.1f}%)")
    print(f"  cascade_triggered    : {summary['cascade_triggered']}건  ({summary['trigger_rate']*100:.1f}%)")
    if not DRY_RUN:
        print(f"  gpt_called           : {summary['gpt_called']}건")
        print(f"  gpt_adopted          : {summary['gpt_adopted']}건")
        print(f"  gpt_rejected         : {summary['gpt_rejected']}건")
        print(f"  improved (❌→✅)     : {summary['improved']}건")
        print(f"  regressed (✅→❌)    : {summary['regressed']}건")
        print(f"  final_cr             : {summary['final_cr']}/{total}  ({summary['final_rate']*100:.1f}%)")
        if summary['improvement'] > 0:
            print(f"  → 향상: +{summary['improvement']}건  ({summary['improvement']/total*100:.1f}%p)")
    print(f"\n  저장:")
    print(f"    {results_path}")
    print(f"    {summary_path}")

    print(f"\n  Trigger 분포:")
    for t, n in sorted(trigger_dist.items()):
        print(f"    {t:<22} : {n}건")


def main_single(input_path: str, output_path: str):
    """
    단건 모드: 공지 1건을 처리하고 결과를 JSON으로 저장.

    입력 JSON 형식:
      {
        "body": "공지문 본문 텍스트",
        "pred_json_text": "{\"title\": \"...\", ...}"   // v6 모델 raw 출력
      }

    출력 JSON:
      prediction / safe_fixed_prediction / cascade_* / final_prediction /
      auto_register_status / auto_register_reason / ...
    """
    with open(input_path, encoding='utf-8') as f:
        inp = json.load(f)

    body           = inp.get("body", "")
    pred_json_text = inp.get("pred_json_text", "")

    print("=" * 65)
    print(f"Cascade Tier 1 — 단건 모드")
    print(f"  input : {input_path}")
    print(f"  DRY_RUN={DRY_RUN}  MERGE_DATES_ONLY={MERGE_DATES_ONLY}")
    print("=" * 65)

    entry = process_one(body, pred_json_text, gt=None, verbose=True)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(entry, f, ensure_ascii=False, indent=2)

    fp = entry["final_prediction"]
    status = entry["auto_register_status"]
    reason = entry["auto_register_reason"]
    print(f"\n결과:")
    print(f"  title      : {fp.get('title','')!r}")
    print(f"  start_date : {fp.get('start_date','')!r}")
    print(f"  end_date   : {fp.get('end_date','')!r}")
    print(f"  location   : {fp.get('location','')!r}")
    print(f"  cascade    : {'사용' if entry['cascade_used'] else '미사용'}"
          f"  accepted={entry['accepted']}")
    print(f"  자동등록   : {status}"
          + (f"  ({reason})" if reason else ""))
    print(f"\n  저장: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cascade Tier 1 추론 파이프라인 (날짜 trigger only)"
    )
    parser.add_argument("--input",  "-i", type=str, default=None,
                        help="단건 모드: 입력 JSON 파일 경로 (body + pred_json_text)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="단건 모드: 출력 JSON 파일 경로")
    parser.add_argument("--dry-run", action="store_true",
                        help="GPT 호출 없이 trigger 목록만 출력")
    parser.add_argument("--eval-results", type=str, default=None,
                        help="배치 모드: eval_results.json 경로 오버라이드")
    parser.add_argument("--valid-data", type=str, default=None,
                        help="배치 모드: valid JSONL 경로 오버라이드")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="배치 모드: 결과 저장 디렉토리 오버라이드")
    args = parser.parse_args()

    if args.dry_run:
        DRY_RUN = True
    if args.eval_results:
        EVAL_RESULTS = Path(args.eval_results)
    if args.valid_data:
        VALID_JSONL = Path(args.valid_data)
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    if args.input:
        # 단건 모드
        out = args.output or args.input.replace(".json", "_cascade.json")
        main_single(args.input, out)
    else:
        # 배치 모드 (평가)
        main_batch()
