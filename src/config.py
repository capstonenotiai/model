import os
import re


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "v1")
TRAIN_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
VALID_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "valid.jsonl")
SAMPLE_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "sample.jsonl")

EVAL_RESULTS_PATH = os.path.join(OUTPUT_DIR, "eval_results.json")
EVAL_SUMMARY_PATH = os.path.join(OUTPUT_DIR, "eval_summary.json")

MAX_NEW_TOKENS = 512
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "당신은 대학 공지문에서 자동 캘린더 등록용 일정 정보를 추출하는 assistant입니다. "
    "반드시 JSON 객체 하나만 출력하세요. 코드블록, 설명문, 주석은 출력하지 마세요. "
    "마지막에 반드시 닫는 중괄호 }까지 출력하세요. detail보다 JSON 완성이 우선입니다. "
    "JSON 키는 title, start_date, end_date, location, detail만 사용하세요. "
    "날짜는 반드시 YYYY-MM-DD 형식으로 출력하세요. start_date는 end_date보다 늦을 수 없습니다. "
    "연도가 명시된 경우 그 연도를 사용하세요. 연도가 없으면 본문에서 가장 가까운 연도 맥락을 참고하세요. "
    "title은 반드시 [공지 제목] 섹션의 제목을 우선 사용하세요. "
    "본문 중간 제목이나 설명문을 보고 title을 새로 만들거나 확장하지 마세요. "
    "신규, D-day, 목록용 날짜 괄호, 말줄임은 제거할 수 있지만 제목의 핵심 단어를 임의로 추가하거나 삭제하지 마세요. "
    "모집, 접수, 신청, 공모, 대회 공지는 접수/신청/모집 시작일과 마감일을 start_date/end_date로 사용하세요. "
    "행사, 특강, 설명회, 교육, 공연 공지는 실제 행사 시작일과 종료일을 사용하세요. "
    "공지일, 게시일, 작성일, 수집일, 오늘 날짜는 start_date/end_date로 사용하지 마세요. "
    "공모전, 문학상, 대외활동 등 응모 시작일이 본문에 명시되지 않은 경우 start_date는 빈 문자열 \"\"로 쓰세요. "
    "발표일, 결과 발표일, 당첨자 발표일, 시상일은 start_date/end_date로 쓰지 않고 detail에만 넣으세요. "
    "기간이 명시되면 end_date를 start_date와 동일하게 복사하지 말고 반드시 마감일을 찾으세요. "
    "실제 행사, 방문, 제출 장소만 location에 넣으세요. "
    "온라인 접수, 이메일 제출, 구글폼 신청, 홈페이지 신청, SNS 업로드는 location으로 보지 마세요. "
    "단순 온라인 접수만 있으면 location은 빈 문자열 \"\"로 쓰세요. "
    "실제 행사나 강의가 온라인으로 진행되는 경우에만 location은 \"온라인\"으로 쓰세요. "
    "오프라인 장소, 건물명, 강의실, 기관명, 주소가 명시되어 있으면 location에 넣으세요. "
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


def normalize_location(location):
    text = normalize_text(location)
    if not text:
        return ""
    compact = re.sub(r"\s+", "", text).lower()
    non_place_patterns = {
        "온라인접수", "온라인신청", "온라인제출", "온라인지원", "온라인응모",
        "온라인접수", "온라인접수",
        "홈페이지신청", "홈페이지접수", "홈페이지등록", "홈페이지",
        "이메일제출", "이메일접수", "이메일신청", "이메일지원", "이메일",
        "구글폼신청", "구글폼접수", "구글폼", "googleform",
        "sns업로드", "sns제출", "sns응모", "인스타그램",
        "네이버폼", "카카오폼", "웹제출",
        "우편접수", "우편제출", "우편신청",
        "팩스접수", "팩스제출",
    }
    if compact in non_place_patterns or any(pattern in compact for pattern in non_place_patterns):
        return ""
    return text
