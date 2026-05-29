import re


# 실제 온라인 행사/강의가 진행되는 경우 (location="온라인" 유지)
_ONLINE_EVENT_RE = re.compile(
    r"온라인\s*(으로\s*)?(진행|개최|운영|실시|수업|강의|특강|교육|행사|설명회|세미나|컨퍼런스|포럼|워크숍|간담회)"
    r"|비대면\s*(으로\s*)?(진행|개최|운영|실시)"
    r"|zoom|웨비나|webinar"
    r"|화상\s*(회의|강의|수업|면접|상담)"
    r"|(강의|행사|교육|수업|면접|상담)\s*(이\s*)?온라인"
)

# 온라인 접수/신청만 있는 경우 (location="" 처리)
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

# title에서 주차 표기 괄호만 제거 (예: "(11월 2주차)")
_TITLE_WEEK_RE = re.compile(
    r"\s*[\(\[（【]\s*\d{1,2}월\s*\d{1,2}주차\s*[\)\]）】]"
    r"|\s*[\(\[（【]\s*\d{4}년?\s*\d{1,2}월?\s*\d{1,2}주차?\s*[\)\]）】]"
)

# location 필드에 접수 방법이 적힌 경우 → '' 처리
_APPLY_METHOD_LOC_RE = re.compile(
    r"^(온라인|이메일|우편|팩스|홈페이지|구글\s*폼|네이버\s*폼|카카오\s*폼|sns|전산|홈피)?\s*"
    r"(접수|신청|제출|등록|지원|응모)\s*(방법|처|링크|url)?$",
    re.IGNORECASE,
)



def _has_online_event(user_content: str) -> bool:
    return bool(_ONLINE_EVENT_RE.search(user_content))


def _has_only_online_apply(user_content: str) -> bool:
    return bool(_ONLINE_APPLY_RE.search(user_content))


def _is_apply_method_location(loc: str) -> bool:
    return bool(_APPLY_METHOD_LOC_RE.match(loc))


def postprocess_location(location: str, user_content: str) -> str:
    """
    양방향 location 보정:
    - pred='온라인' + 실제 온라인 행사 근거 없음 → ''
    - pred=''       + 실제 온라인 행사 근거 있음 → '온라인'
    - pred=접수방법  → ''
    물리적 장소가 명시된 경우는 건드리지 않음.
    """
    loc = location.strip()

    if _is_apply_method_location(loc):
        return ""

    if loc == "온라인":
        if _has_online_event(user_content):
            return loc
        return ""

    if loc == "":
        if _has_online_event(user_content) and not _has_only_online_apply(user_content):
            return "온라인"

    return loc


def postprocess_detail(detail: str, max_len: int = 120) -> str:
    """detail 길이를 max_len 이하로 자름 (문장 경계 우선)."""
    if len(detail) <= max_len:
        return detail
    cut = detail[:max_len]
    # 마지막 구분자(., ,) 위치에서 자름 (60% 이상 위치에서만)
    min_pos = int(max_len * 0.6)
    for sep in [". ", ", ", " "]:
        idx = cut.rfind(sep)
        if idx >= min_pos:
            return cut[: idx + len(sep)].rstrip()
    return cut.rstrip()


def postprocess_title(title: str) -> str:
    """title에서 주차 표기 괄호만 제거."""
    return _TITLE_WEEK_RE.sub("", title).strip()


def postprocess(pred: dict, user_content: str) -> dict:
    """title(주차괄호) + location + detail 후처리를 적용한 새 dict 반환."""
    result = dict(pred)
    result["title"] = postprocess_title(pred.get("title", ""))
    result["location"] = postprocess_location(pred.get("location", ""), user_content)
    result["detail"] = postprocess_detail(pred.get("detail", ""))
    return result
