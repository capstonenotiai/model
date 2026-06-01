import re


# ── 온라인 행사/접수 판별 ─────────────────────────────────────────────────────
_ONLINE_EVENT_RE = re.compile(
    r"온라인\s*(으로\s*)?(진행|개최|운영|실시|수업|강의|특강|교육|행사|설명회|세미나|컨퍼런스|포럼|워크숍|간담회)"
    r"|비대면\s*(으로\s*)?(진행|개최|운영|실시)"
    r"|zoom|웨비나|webinar"
    r"|화상\s*(회의|강의|수업|면접|상담)"
    r"|(강의|행사|교육|수업|면접|상담)\s*(이\s*)?온라인"
)

_ONLINE_APPLY_RE = re.compile(
    r"온라인\s*(접수|신청|제출|등록|참가신청|응모|지원)"
    r"|홈페이지\s*(접수|신청|등록|지원)"
    r"|구글\s*폼|구글폼|google\s*form"
    r"|이메일\s*(접수|신청|제출|지원)"
    r"|sns\s*(업로드|제출|응모)"
    r"|네이버\s*폼|카카오\s*폼"
    r"|웹사이트\s*(접수|신청)"
    r"|\bemail\b.*\b(submit|apply|send)\b"
)

# ── title 주차 괄호 ───────────────────────────────────────────────────────────
_TITLE_WEEK_RE = re.compile(
    r"\s*[\(\[（【]\s*\d{1,2}월\s*\d{1,2}주차\s*[\)\]）】]"
    r"|\s*[\(\[（【]\s*\d{4}년?\s*\d{1,2}월?\s*\d{1,2}주차?\s*[\)\]）】]"
)

# ── location 비장소 판별 ──────────────────────────────────────────────────────
_APPLY_METHOD_LOC_RE = re.compile(
    r"^(온라인|이메일|우편|팩스|홈페이지|구글\s*폼|네이버\s*폼|카카오\s*폼|sns|전산|홈피)?\s*"
    r"(접수|신청|제출|등록|지원|응모)\s*(방법|처|링크|url)?$",
    re.IGNORECASE,
)

# 이메일 주소가 포함된 location (접수처 주소) → 비장소
_EMAIL_IN_LOC_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# ── 사업장소 복원용 제외 패턴 ─────────────────────────────────────────────────
# 날짜·비장소 표현이 candidate에 포함되면 복원 금지
_BUSSINESS_LOC_INVALID_RE = re.compile(
    r"\d{1,2}[./월]\s*\d{1,2}|20\d{2}|합격자|개별|추후|온라인|이메일|홈페이지|전국"
)

# ── 날짜 후처리 ───────────────────────────────────────────────────────────────
# 본문에 접수 시작일이 명시된 표현
_START_EXPLICIT_RE = re.compile(
    r"(접수|신청|모집|공모|지원)\s*기간\s*[:：]\s*\d{4}"
    r"|\d{4}[.\-/년]\s*\d{1,2}[.\-/월]\s*\d{1,2}[일]?\s*[~∼\-]"
    r"|\d{1,2}[./월]\s*\d{1,2}[일]?\s*[~∼]\s*\d{1,2}"
    r"|시작\s*[:：]|개시\s*[:：]|시작일\s*[:：]"
)

# 본문에서 연도 추출 (YYYY년 / YYYY. / YYYY- / YYYY/)
_YEAR_IN_BODY_RE = re.compile(r"(20\d{2})[년.\-/]")

# ── 행정구역 전체명 → 약칭 ────────────────────────────────────────────────────
_ADMIN_FULL_TO_SHORT = {
    "서울특별시": "서울", "서울시": "서울",
    "부산광역시": "부산", "부산시": "부산",
    "대구광역시": "대구", "대구시": "대구",
    "인천광역시": "인천", "인천시": "인천",
    "광주광역시": "광주", "광주시": "광주",
    "대전광역시": "대전", "대전시": "대전",
    "울산광역시": "울산", "울산시": "울산",
    "세종특별자치시": "세종", "세종시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원", "강원도": "강원",
    "충청북도": "충북", "충청남도": "충남",
    "전북특별자치도": "전북", "전라북도": "전북",
    "전라남도": "전남",
    "경상북도": "경북", "경상남도": "경남",
    "제주특별자치도": "제주", "제주도": "제주",
}


def _has_online_event(user_content: str) -> bool:
    return bool(_ONLINE_EVENT_RE.search(user_content))


def _is_apply_method_location(loc: str) -> bool:
    return bool(_APPLY_METHOD_LOC_RE.match(loc))


def postprocess_location(location: str, user_content: str) -> str:
    loc = location.strip()

    # 접수방법 문자열 → ''
    if _is_apply_method_location(loc):
        return ""

    # 이메일 주소 포함 → 접수처 주소로 판단 → ''
    if _EMAIL_IN_LOC_RE.search(loc):
        return ""

    # 불특정·비장소 표현 → ''
    _NONSPECIFIC_LOC_RE = re.compile(
        r"일원$|주요\s*공연장|주요\s*장소|공공장소|여러\s*장소|전국\s*각지|추후\s*공지|추후\s*안내|개별\s*안내|개별\s*공지"
    )
    if loc in ("전국",) or _NONSPECIFIC_LOC_RE.search(loc):
        return ""

    # 행정구역 전체명 → 약칭 (경기도→경기 등)
    loc = _ADMIN_FULL_TO_SHORT.get(loc, loc)

    # '온라인' 양방향 보정
    if loc == "온라인":
        return loc if _has_online_event(user_content) else ""

    if loc == "":
        if _has_online_event(user_content):
            return "온라인"
        # 본문에 "장소: 서울" 명시가 있으면 복원
        if re.search(r"장소\s*[:：]\s*서울", user_content):
            return "서울"
        # "사업장소" 라벨 기반 복원 (좁은 조건, [22] 팔복예술공장 케이스 대상)
        m = re.search(r"사업\s*장소\s*[:：]?\s*([^\n\r|]{2,40})", user_content)
        if m:
            candidate = m.group(1)
            candidate = re.split(r"\s*[-–]\s*|[,，/]|※|문의|접수|신청|제출", candidate)[0].strip()
            if (
                2 <= len(candidate) <= 20
                and not _EMAIL_IN_LOC_RE.search(candidate)
                and not _ONLINE_APPLY_RE.search(candidate)
                and not _BUSSINESS_LOC_INVALID_RE.search(candidate)
            ):
                return candidate

    return loc


def postprocess_dates(start: str, end: str, user_content: str) -> tuple:
    """
    날짜 후처리:
    1. start == end 이고 본문에 시작일 명시 없으면 start=""  (end→start 복사 오류 수정)
    2. pred 연도가 본문에 전혀 없는 연도면 본문 연도로 교정
    """
    # 1. start=end 복사 패턴 제거
    if start and start == end:
        if not _START_EXPLICIT_RE.search(user_content):
            start = ""

    # 2. 연도 검증 (본문에 명시된 연도와 다르면 교정)
    years_in_body = set(_YEAR_IN_BODY_RE.findall(user_content))
    if years_in_body:
        if start and len(start) >= 4 and start[:4] not in years_in_body:
            closest = min(years_in_body, key=lambda y: abs(int(y) - int(start[:4])))
            start = closest + start[4:]
        if end and len(end) >= 4 and end[:4] not in years_in_body:
            closest = min(years_in_body, key=lambda y: abs(int(y) - int(end[:4])))
            end = closest + end[4:]

    return start, end


def postprocess_detail(detail: str, max_len: int = 120) -> str:
    if len(detail) <= max_len:
        return detail
    cut = detail[:max_len]
    min_pos = int(max_len * 0.6)
    for sep in [". ", ", ", " "]:
        idx = cut.rfind(sep)
        if idx >= min_pos:
            return cut[: idx + len(sep)].rstrip()
    return cut.rstrip()


def postprocess_title(title: str) -> str:
    return _TITLE_WEEK_RE.sub("", title).strip()


def postprocess(pred: dict, user_content: str) -> dict:
    result = dict(pred)
    result["title"] = postprocess_title(pred.get("title", ""))
    result["location"] = postprocess_location(pred.get("location", ""), user_content)
    result["detail"] = postprocess_detail(pred.get("detail", ""))
    start, end = postprocess_dates(
        pred.get("start_date", ""), pred.get("end_date", ""), user_content
    )
    result["start_date"] = start
    result["end_date"] = end
    return result
