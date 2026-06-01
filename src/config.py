import os
import re


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR        = os.path.join(PROJECT_ROOT, "outputs", "v1")
TRAIN_DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
VALID_DATA_PATH   = os.path.join(PROJECT_ROOT, "data", "valid.jsonl")
SAMPLE_DATA_PATH  = os.path.join(PROJECT_ROOT, "data", "sample.jsonl")

EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "eval_results.json")
EVAL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "eval_summary.json")

CASCADE_RESULTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "cascade_tier1")

MAX_NEW_TOKENS = 512
MAX_SEQ_LENGTH = 2048
RANDOM_SEED    = 42

SYSTEM_PROMPT = (
    "당신은 대학 공지문에서 자동 캘린더 등록용 일정 정보를 추출하는 assistant입니다. "
    "반드시 JSON 객체 하나만 출력하세요. 코드블록, 설명문, 주석은 출력하지 마세요. "
    "마지막에 반드시 닫는 중괄호 }까지 출력하세요. detail보다 JSON 완성이 우선입니다. "
    "JSON 키는 title, start_date, end_date, location, detail만 사용하세요. "
    "날짜는 반드시 YYYY-MM-DD 형식으로 출력하세요. start_date는 end_date보다 늦을 수 없습니다. "
    "연도가 명시된 경우 그 연도를 사용하세요. "
    "연도가 명시되지 않은 월/일 날짜는 본문 안에 명확한 연도 맥락(예: 2025년, 2026. 등)이 있을 때만 YYYY-MM-DD로 변환하세요. "
    "본문에 명확한 연도 근거가 없으면 현재 연도나 수집 연도를 임의로 사용하지 마세요. "
    "날짜를 확정할 수 없으면 해당 필드는 빈 문자열 \"\"로 쓰세요. "
    # ── title 규칙 ──────────────────────────────────────────────────────
    "title은 반드시 [공지 제목] 섹션의 제목을 기준으로 사용하세요. "
    "본문 중간 제목이나 설명문을 보고 title을 새로 만들거나 요약하지 마세요. "
    "[공지 제목]에 없는 기관명, 주최기관명을 title에 새로 추가하지 마세요. "
    "대괄호 처리 규칙: "
    "[국비무료], [성매매 Cut], [행복합니다], [장학], [공지], [취업] 같이 프로그램 특성이나 캠페인 의미·분류를 나타내는 대괄호는 원문 그대로 유지하세요. "
    "[기획예산처], [송파문화재단], [KT&G], [웅진식품] 같이 기관명·기업명·브랜드명만 있는 대괄호 prefix는 제거하세요. "
    "(첫 단추 프로젝트), (비콘 테크 챌린지 2026) 같은 행사명·프로젝트명 괄호 부제목은 원문 그대로 유지하세요. "
    "신규, D-day, (D-3), 목록용 날짜 괄호 (2026.05.01), (~마감일) 같은 목록 부가정보는 제거하세요. "
    # ── 날짜 규칙 ──────────────────────────────────────────────────────
    "모집, 접수, 신청, 공모, 대회 공지는 접수/신청/모집 시작일과 마감일을 start_date/end_date로 사용하세요. "
    "행사, 특강, 설명회, 교육, 공연 공지는 실제 행사 시작일과 종료일을 사용하세요. "
    "공지일, 게시일, 작성일, 수집일, 오늘 날짜는 start_date/end_date로 사용하지 마세요. "
    "모집·공모에서 접수/신청 시작일이 본문에 명시되지 않은 경우 start_date는 반드시 빈 문자열 \"\"로 쓰세요. "
    "start_date가 없을 때 end_date 값을 start_date에 복사하지 마세요. start_date가 없으면 반드시 \"\"로 쓰세요. "
    "발표일, 결과 발표일, 당첨자 발표일, 시상일은 start_date/end_date로 쓰지 않고 detail에만 넣으세요. "
    "기간이 명시되면 end_date를 start_date와 동일하게 복사하지 말고 반드시 마감일을 찾으세요. "
    # ── location 규칙 ──────────────────────────────────────────────────
    "실제 행사·교육·대회·오디션이 열리는 도시·지역명(서울, 부산, 전남 등), 건물명, 강의실, 기관명, 주소를 location에 넣으세요. "
    "접수 방법이 온라인(이메일, 구글폼, 홈페이지)이더라도 실제 행사·오디션·교육이 특정 장소에서 열리면 그 장소를 location에 넣으세요. "
    "온라인 접수, 이메일 제출, 구글폼 신청, 홈페이지 신청, SNS 업로드 자체는 location이 아닙니다. "
    "행사 자체가 온라인으로 진행되는 경우에만 location을 \"온라인\"으로 쓰세요. "
    "장소 정보가 전혀 없으면 location은 빈 문자열 \"\"로 쓰세요. "
    # ── detail 규칙 ─────────────────────────────────────────────────────
    "detail은 1문장만 작성하고 120자 이내로 제한하세요. "
    "본문 전체 복사와 반복 문장은 금지입니다. "
    "detail에는 접수방법, 신청방법, 대상, 시간, 발표일, 시상일 같은 보조 정보만 짧게 요약하세요. "
    "detail이 길어질 것 같으면 과감히 생략하거나 아주 짧게 작성하세요."
)


def normalize_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _compact_region(text: str) -> str:
    parts = text.split(" ")
    if len(parts) > 1 and all(len(p) == 1 and "가" <= p[0] <= "힣" for p in parts):
        return "".join(parts)
    return text


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


def normalize_location(location):
    text = normalize_text(location)
    if not text:
        return ""
    text = _compact_region(text)
    text = _ADMIN_FULL_TO_SHORT.get(text, text)
    compact = re.sub(r"\s+", "", text).lower()
    non_place_patterns = {
        "온라인접수", "온라인신청", "온라인제출", "온라인지원", "온라인응모",
        "홈페이지신청", "홈페이지접수", "홈페이지등록", "홈페이지",
        "이메일제출", "이메일접수", "이메일신청", "이메일지원", "이메일",
        "구글폼신청", "구글폼접수", "구글폼", "googleform",
        "sns업로드", "sns제출", "sns응모", "인스타그램",
        "네이버폼", "카카오폼", "웹제출",
        "우편접수", "우편제출", "우편신청",
        "팩스접수", "팩스제출",
        "전국",
        "일원",
        "주요공연장", "주요장소", "공공장소", "여러장소",
        "추후공지", "추후안내", "개별안내", "개별공지",
        "전국각지",
    }
    if compact in non_place_patterns or any(pattern in compact for pattern in non_place_patterns):
        return ""
    return text
