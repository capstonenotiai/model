"""
cascade_trigger_sim.py

제품용 cascade 구조 시뮬레이션.

apply_safe_fixes(pred, user_content)
    → 자동 수정해도 안전한 규칙만 적용 (R1, R2)
    → 값을 바꾸고, 어떤 액션이 발생했는지 반환

get_cascade_triggers(pred, fixed_pred, user_content)
    → 위험 신호만 탐지, 값을 직접 바꾸지 않음
    → cascade(GPT fallback) 여부를 결정하기 위한 신호 목록 반환

시뮬레이션 출력:
    - safe fix 후 성능 (118/147 유지 확인)
    - cascade 대상 수 / 비율
    - cascade 대상 중 현재 실패 건수 / 성공 건수
    - trigger별 분포
"""

import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT         = Path(__file__).resolve().parents[1]
EVAL_RESULTS = ROOT / "outputs" / "v1" / "eval_results.json"
VALID_JSONL  = ROOT / "data" / "valid.jsonl"

sys.path.insert(0, str(ROOT / "src"))
from postprocess import postprocess, _START_EXPLICIT_RE
from config import normalize_text

FIELDS = ["title", "start_date", "end_date", "location", "detail"]

# ── Trigger 이름 상수 ──────────────────────────────────────────────────────────
T_NO_DATES          = "날짜_없음"           # start="" AND end=""
T_END_ANNOUNCE      = "end=발표일_의심"      # end가 발표/시상일 날짜
T_YEAR_MISMATCH     = "연도_불일치"          # MM-DD 일치하는 body 날짜가 다른 연도
T_END_NOT_IN_BODY   = "end_본문_미존재"      # end_date가 본문 날짜에 없음 (MD도 없음)
T_START_NOT_IN_BODY = "start_본문_미존재"    # start_date가 본문 날짜에 없음 (MD도 없음)
T_LOC_CANDIDATE     = "location_후보_있음"   # location=""인데 유효 장소 후보 있음
T_TITLE_MISMATCH    = "title_불일치"         # title이 공지 제목과 크게 다름

# ── 날짜 추출 유틸 ─────────────────────────────────────────────────────────────

_FULL_DATE_RE       = re.compile(r"(20\d{2})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})\s*일?")
_PARTIAL_DATE_RE    = re.compile(r"(?<!\d)(\d{1,2})\s*[월./]\s*(\d{1,2})\s*일?(?:\s*[\(\[]\s*[월화수목금토일]\s*[\)\]])?")
_YEAR_RE            = re.compile(r"(20\d{2})\s*[년.\-/]")
_MONTH_ONLY_FULL_RE = re.compile(r"(20\d{2})\s*[년.\-/]\s*(\d{1,2})\s*월(?!\s*[\d./])")
_MONTH_ONLY_PART_RE = re.compile(r"(?<!\d)(\d{1,2})\s*월(?!\s*[\d./])")

_RECEIPT_KW_RE  = re.compile(r"접수\s*기간|신청\s*기간|모집\s*기간|공모\s*기간|지원\s*기간|응모\s*기간|접수\s*일정|신청\s*일정|모집\s*일정|공모\s*일정")
_DEADLINE_KW_RE = re.compile(r"마감|제출\s*기한|응모\s*마감|접수\s*마감|신청\s*마감|마감일|접수\s*종료|제출\s*마감|신청\s*종료|접수\s*기한")
_KAJI_RE        = re.compile(r"까지|이전까지|전까지")
_EVENT_KW_RE    = re.compile(r"행사\s*일(?!\s*정)|대회\s*일시|대회\s*날짜|공연\s*일시|오디션\s*일정|오디션\s*일시|경기\s*일시|시합\s*일시")
_ANNOUNCE_KW_RE = re.compile(r"발표\s*일|결과\s*발표|당첨자\s*발표|시상\s*일|시상\s*식|합격\s*발표|최종\s*발표|합격자\s*발표|수상자\s*발표|당선\s*발표|선정\s*결과|심사\s*결과|발표\s*예정")

# 장소 라벨 (cascade trigger용 — postprocess의 사업장소보다 넓은 범위)
_LOC_LABEL_RE = re.compile(
    r"(?:사업|행사|대회|교육|공연)\s*장소\s*[:：]?\s*"
    r"|오디션\s*장소\s*[:：]?\s*"
    r"|면접\s*장소\s*[:：]?\s*"
    r"|장소\s*[:：]\s*"
    r"|위치\s*[:：]\s*"
)
# 강화된 비장소 제외 — 합격자/추후/예정/발표/안내 등 비장소 표현 추가
_LOC_TRIGGER_INVALID_RE = re.compile(
    r"접수처|문의처|이메일|홈페이지|구글\s*폼|온라인\s*접수|전국|http|@|\.[a-z]"
    r"|합격자|추후|개별|서류|미정|별도|예정|안내|발표|개최|진행|\d{4}|\d{1,2}[./월]\s*\d",
    re.IGNORECASE
)

# 공지 제목 추출 — 실제 포맷: [제목]\n<title>\n\n[본문]
_NOTICE_TITLE_RE = re.compile(r"\[제목\]\s*\n+([^\n]+)")

# title 비교: 제거 가능한 대괄호 내용 (시스템 프롬프트 기준)
_BRACKET_REMOVABLE_RE = re.compile(
    r"^(new|신규|d[-+]?\d+|\d{4}[년]?\s*\d{1,2}[월]?\s*\d{1,2}[일]?)$",
    re.IGNORECASE
)


def _ctx_role(text, s, e):
    pre = text[max(0, s-60):s]; post = text[e:e+25]
    if _ANNOUNCE_KW_RE.search(pre): return "announce"
    if _EVENT_KW_RE.search(pre):    return "event"
    if _RECEIPT_KW_RE.search(pre):  return "receipt"
    if _DEADLINE_KW_RE.search(pre) or _KAJI_RE.search(post): return "deadline"
    return "unknown"


def _extract_dates(text):
    res = defaultdict(list)
    years = _YEAR_RE.findall(text)
    dy = Counter(years).most_common(1)[0][0] if years else "2026"
    spans = []
    for m in _FULL_DATE_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if not (1 <= mo <= 12 and 1 <= d <= 31): continue
        res[f"{y:04d}-{mo:02d}-{d:02d}"].append(_ctx_role(text, m.start(), m.end()))
        spans.append((m.start(), m.end()))
    for m in _PARTIAL_DATE_RE.finditer(text):
        if any(s <= m.start() < e for s, e in spans): continue
        mo, d = int(m.group(1)), int(m.group(2))
        if not (1 <= mo <= 12 and 1 <= d <= 31): continue
        res[f"{dy}-{mo:02d}-{d:02d}"].append(_ctx_role(text, m.start(), m.end()))
    return dict(res)


def _extract_month_ev(text):
    result = set()
    years = _YEAR_RE.findall(text)
    dy = Counter(years).most_common(1)[0][0] if years else "2026"
    for m in _MONTH_ONLY_FULL_RE.finditer(text):
        y, mo = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12: result.add(f"{y:04d}-{mo:02d}")
    for m in _MONTH_ONLY_PART_RE.finditer(text):
        mo = int(m.group(1))
        if 1 <= mo <= 12: result.add(f"{dy}-{mo:02d}")
    return result


def extract_full_dates(text: str) -> set:
    """
    본문에서 연도가 명시된 날짜만 추출 (partial 제외).
    "2026년 3월 5일" / "2026.3.5" / "2026-03-05" 등 → YYYY가 본문에 명시된 경우만.
    채택 조건에서 연도 변경 검증 시 사용.
    """
    result = set()
    for m in _FULL_DATE_RE.finditer(text):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= d <= 31:
            result.add(f"{y:04d}-{mo:02d}-{d:02d}")
    return result


# ── 핵심 함수 1: 안전한 자동 수정 ────────────────────────────────────────────

def apply_safe_fixes(pred: dict, user_content: str) -> tuple:
    """
    자동 수정해도 안전한 규칙만 적용.

    R1: start == end 이고 시작 표현 없음 → start=""
    R2: 마감만 명시된 공고에서 본문 근거 없는 start 제거
        조건: deadline 있고 receipt 없고 MM 근거도 없음

    반환: (fixed_pred, actions[])
    """
    actions = []
    ns, ne = pred["start_date"], pred["end_date"]

    body     = _extract_dates(user_content)
    month_ev = _extract_month_ev(user_content)
    all_d    = set(body)
    dead     = {d for d, rs in body.items() if "deadline" in rs}
    rec      = {d for d, rs in body.items() if "receipt"  in rs}
    has_se   = bool(_START_EXPLICIT_RE.search(user_content))

    # R1: start=end 복사 패턴
    if ns and ns == ne and not has_se:
        ns = ""
        actions.append("R1:start=end→''")

    # R2: 본문 근거 없는 start (마감만 있는 공고)
    pm      = pred["start_date"][5:7] if len(pred["start_date"]) >= 7 else ""
    body_ms = {d[5:7] for d in all_d} | {ym[5:7] for ym in month_ev}
    if (ns and all_d and ns not in all_d
            and not has_se and bool(dead) and not bool(rec)
            and pm not in body_ms):
        ns = ""
        actions.append("R2:start∉본문→''")

    fixed = {**pred, "start_date": ns, "end_date": ne}
    return fixed, actions


# ── 핵심 함수 2: cascade trigger 탐지 (v2 — 4가지 개선) ──────────────────────

def get_cascade_triggers(pred: dict, fixed_pred: dict, user_content: str) -> list:
    """
    위험 신호 탐지. 값을 직접 바꾸지 않음.
    pred       : safe fix 이전 (postprocess 적용 후)
    fixed_pred : safe fix 이후
    반환: trigger 이름 목록 (중복 없음)
    """
    triggers = []
    body = _extract_dates(user_content)
    all_d = set(body)
    ann  = {d for d, rs in body.items() if "announce" in rs}
    dead = {d for d, rs in body.items() if "deadline" in rs}
    rec  = {d for d, rs in body.items() if "receipt"  in rs}

    ns = fixed_pred["start_date"]
    ne = fixed_pred["end_date"]

    # T1: 날짜 없음 (safe fix 후에도 start/end 모두 빈값)
    if not ns and not ne:
        triggers.append(T_NO_DATES)

    # T2: end가 발표/시상일 전용 날짜 (값 삭제 대신 cascade)
    if ne and ne in ann and ne not in dead and ne not in rec:
        triggers.append(T_END_ANNOUNCE)

    # T3: 연도 불일치 (재설계)
    # pred 날짜의 MM-DD가 body 날짜 후보와 일치하지만 YYYY가 다름
    for pd in [ns, ne]:
        if not pd or len(pd) < 10:
            continue
        pred_md = pd[5:]   # "MM-DD"
        pred_yr = pd[:4]   # "YYYY"
        for bd in all_d:
            if bd[5:] == pred_md and bd[:4] != pred_yr:
                if T_YEAR_MISMATCH not in triggers:
                    triggers.append(T_YEAR_MISMATCH)
                break

    # T4: end_date가 본문 날짜에 없음 (MD도 없어야 함 — 연도만 다른 케이스 제외)
    if ne and all_d and ne not in all_d:
        ne_md = ne[5:]
        body_mds = {d[5:] for d in all_d}
        if ne_md not in body_mds:
            triggers.append(T_END_NOT_IN_BODY)

    # T4b: start_date가 본문 날짜에 없음 (MD도 없어야 함 — R2로 해결 안 된 케이스)
    if ns and all_d and ns not in all_d:
        ns_md = ns[5:]
        body_mds = {d[5:] for d in all_d}
        if ns_md not in body_mds:
            triggers.append(T_START_NOT_IN_BODY)

    # T5: location="" 인데 유효 장소 후보 있음 (강화 — false positive 감소)
    if fixed_pred["location"] == "":
        for m in _LOC_LABEL_RE.finditer(user_content):
            after = user_content[m.end():m.end() + 60].split('\n')[0]
            # 구분자에서 자름
            candidate = re.split(r'\s*[-–]\s*|[(,，/]|※', after)[0].strip()
            if (candidate
                    and not _LOC_TRIGGER_INVALID_RE.search(candidate)
                    and re.search(r'[가-힣]{2,}', candidate)   # 한국어 2자 이상 필수
                    and 2 <= len(candidate) <= 20):
                triggers.append(T_LOC_CANDIDATE)
                break

    # T6: title이 공지 제목과 크게 다름 (강화)
    # 포맷: [제목]\n<title>\n\n[본문]
    m_title = _NOTICE_TITLE_RE.search(user_content)
    if m_title:
        notice_t = m_title.group(1).strip()
        pred_t   = fixed_pred["title"]
        fired    = False

        # Check 1: 대괄호 [xxx] 누락 (제거 허용 목록 제외)
        notice_sq = set(re.findall(r'\[([^\]]+)\]', notice_t))
        pred_sq   = set(re.findall(r'\[([^\]]+)\]', pred_t))
        missing_sq = {b for b in (notice_sq - pred_sq)
                      if not _BRACKET_REMOVABLE_RE.match(b.strip())}
        if missing_sq:
            fired = True

        # Check 2: 괄호 부제목 (xxx) 누락
        if not fired:
            notice_rd = set(re.findall(r'\(([^)]{2,})\)', notice_t))
            pred_rd   = set(re.findall(r'\(([^)]{2,})\)', pred_t))
            if notice_rd - pred_rd:
                fired = True

        # Check 3: 긴 한국어 단어 누락 (≥8자 — 기관명 등 고유 복합명사만 대상)
        # ≥5자에서 ≥8자로 상향: 5~7자 일반 복합어의 false positive 제거
        if not fired:
            long_words = re.findall(r'[가-힣]{8,}', notice_t)
            for w in long_words:
                if w not in pred_t:
                    fired = True
                    break

        # Check 4 제거 — recall 기반 fallback은 false positive 과다 (17건 → 제거)

        if fired:
            triggers.append(T_TITLE_MISMATCH)

    return triggers


# ── 유틸 ──────────────────────────────────────────────────────────────────────

def _norm(v): return normalize_text(v) if v else ""

def _cr(pred, gt):
    return all(pred[f] == gt[f] for f in ["title", "start_date", "end_date", "location"])

def _safe_json(text):
    try:    return json.loads(text)
    except: return None


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    with open(EVAL_RESULTS, encoding='utf-8') as f:
        results = json.load(f)
    valid_data = []
    with open(VALID_JSONL, encoding='utf-8') as f:
        for line in f:
            if line.strip():
                valid_data.append(json.loads(line))
    assert len(results) == len(valid_data)

    total = len(results)
    safe_ok = 0                    # safe fix 후 calendar_ready 수
    cascade_total   = 0            # cascade 대상 총 수
    cascade_fail    = 0            # cascade 대상 중 현재 실패
    cascade_pass    = 0            # cascade 대상 중 현재 성공
    trigger_dist    = Counter()    # trigger별 발생 수 (실패/성공 구분)
    trigger_fail_d  = Counter()
    trigger_pass_d  = Counter()
    uncaught_fails  = []           # 실패인데 cascade 안 잡힌 케이스

    case_details = []              # 전체 케이스 상세

    for i, (res, sample) in enumerate(zip(results, valid_data)):
        body   = sample["messages"][1]["content"]
        gt_raw = json.loads(sample["messages"][2]["content"])
        gt     = {f: _norm(gt_raw.get(f,"")) for f in FIELDS}

        # ── 새 postprocess 재적용 (사업장소 포함) ────────────────────────
        raw_json = _safe_json(res.get("pred_json_text",""))
        if raw_json:
            raw_norm = {f: _norm(raw_json.get(f,"")) for f in FIELDS}
            pp_pred  = postprocess(raw_norm, body)
        else:
            pp_pred  = res["pred"]

        # ── Safe fix 적용 ─────────────────────────────────────────────────
        fixed, actions = apply_safe_fixes(pp_pred, body)
        is_ok = _cr(fixed, gt)
        safe_ok += int(is_ok)

        # ── Cascade trigger 탐지 ──────────────────────────────────────────
        triggers = get_cascade_triggers(pp_pred, fixed, body)
        needs_cascade = bool(triggers)

        if needs_cascade:
            cascade_total += 1
            if is_ok:
                cascade_pass += 1
                for t in triggers: trigger_pass_d[t] += 1
            else:
                cascade_fail += 1
                for t in triggers: trigger_fail_d[t] += 1
            for t in triggers: trigger_dist[t] += 1
        else:
            if not is_ok:
                uncaught_fails.append((i, fixed, gt, actions, triggers))

        case_details.append({
            "idx": i, "fixed": fixed, "gt": gt,
            "is_ok": is_ok, "actions": actions, "triggers": triggers
        })

    # ── 출력 ──────────────────────────────────────────────────────────────
    print("=" * 65)
    print("Cascade trigger 시뮬레이션 (v6 + new postprocess 기준)")
    print("=" * 65)
    print(f"\n[Safe fix 후 성능]")
    print(f"  calendar_ready : {safe_ok}/{total}  ({safe_ok/total*100:.1f}%)")
    prev = 118
    print(f"  이전 대비      : {'✅ 유지' if safe_ok >= prev else f'❌ {safe_ok-prev:+d}건'}")

    print(f"\n[Cascade 대상]")
    print(f"  총 대상        : {cascade_total}/{total}  ({cascade_total/total*100:.1f}%)")
    print(f"  현재 실패 건수 : {cascade_fail}건  (trigger coverage: {cascade_fail/(total-safe_ok)*100:.1f}% of {total-safe_ok}건 실패)")
    print(f"  현재 성공 건수 : {cascade_pass}건  (불필요 GPT 호출 비율: {cascade_pass/cascade_total*100:.1f}% of cascade 대상)")

    all_triggers = sorted(set(trigger_dist.keys()))
    print(f"\n[Trigger별 분포]  (총 {cascade_total}건 / 중복 가능)")
    print(f"  {'trigger':<22}  {'총':>4}  {'실패':>4}  {'성공':>4}  {'실패율':>6}")
    print(f"  {'-'*22}  {'----':>4}  {'----':>4}  {'----':>4}  {'------':>6}")
    for t in all_triggers:
        tot  = trigger_dist[t]
        fail = trigger_fail_d[t]
        pass_ = trigger_pass_d[t]
        print(f"  {t:<22}  {tot:>4}  {fail:>4}  {pass_:>4}  {fail/tot*100:>5.1f}%")

    print(f"\n[Cascade가 안 잡은 실패 케이스 — {len(uncaught_fails)}건]")
    for idx, fixed, gt, acts, trigs in uncaught_fails:
        fails = [f for f in ["title","start_date","end_date","location"] if fixed[f] != gt[f]]
        print(f"  [{idx:3d}]  실패필드={fails}")
        for f in fails:
            print(f"         {f}: pred={fixed[f]!r}  gt={gt[f]!r}")

    print(f"\n[Safe fix 액션이 발생한 케이스]")
    for c in case_details:
        if c["actions"]:
            ok_tag = "✅" if c["is_ok"] else "❌"
            print(f"  [{c['idx']:3d}]  {ok_tag}  {c['actions']}  triggers={c['triggers']}")

    print(f"\n{'='*65}")
    print(f"요약:")
    print(f"  safe_fix 후 calendar_ready : {safe_ok}/{total} = {safe_ok/total*100:.1f}%")
    print(f"  cascade 대상               : {cascade_total}건 ({cascade_total/total*100:.0f}%)")
    print(f"  cascade coverage           : {cascade_fail}/{total-safe_ok}건 실패를 cascade가 탐지")
    print(f"  cascade precision          : {cascade_fail}/{cascade_total} = {cascade_fail/cascade_total*100:.1f}% (trigger → 실제 실패)")

    # ── 날짜 trigger만 별도 집계 (고신뢰 subset) ──────────────────────────
    DATE_TRIGGERS = {T_NO_DATES, T_END_ANNOUNCE, T_YEAR_MISMATCH,
                     T_END_NOT_IN_BODY, T_START_NOT_IN_BODY}
    dt_total = dt_fail = dt_pass = 0
    for c in case_details:
        dt = [t for t in c["triggers"] if t in DATE_TRIGGERS]
        if dt:
            dt_total += 1
            if c["is_ok"]: dt_pass += 1
            else:           dt_fail += 1

    n_fail = total - safe_ok
    print(f"\n{'='*65}")
    print(f"[날짜 trigger만 (고신뢰 subset) 비교]")
    print(f"  {'시나리오':<28}  {'cascade':>7}  {'precision':>10}  {'coverage':>9}")
    print(f"  {'-'*28}  {'-------':>7}  {'----------':>10}  {'---------':>9}")
    print(f"  {'날짜 trigger만':<28}  {dt_total:>5}건  "
          f"{dt_fail/dt_total*100:>8.1f}%  "
          f"{dt_fail/n_fail*100:>7.1f}%")
    print(f"  {'전체 (날짜+title+loc)':<28}  {cascade_total:>5}건  "
          f"{cascade_fail/cascade_total*100:>8.1f}%  "
          f"{cascade_fail/n_fail*100:>7.1f}%")
    print(f"\n  → 날짜 trigger만: cascade {dt_total}건 ({dt_total/total*100:.0f}%), "
          f"precision {dt_fail/dt_total*100:.0f}%, coverage {dt_fail/n_fail*100:.0f}%")


if __name__ == "__main__":
    main()
