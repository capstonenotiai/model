# 대학 공지문 캘린더 자동 추출 모델

공지문 텍스트를 입력하면 캘린더 등록에 필요한 JSON을 자동으로 추출합니다.  
Llama-3.1-8B-Instruct 기반 QLoRA 파인튜닝 + GPT-4o-mini Cascade 구조.

---

## 입출력 형식

### 입력

```
[제목]
2026 캡스톤 설계 경진대회 참가팀 모집

[본문]
주관: 공과대학 학생처
접수 기간: 2026년 5월 1일 ~ 6월 30일
행사 장소: 공학관 101호
접수 방법: 이메일 제출 (cse@univ.ac.kr)
시상: 대상 100만원, 최우수상 50만원 (7월 15일 발표)
```

### 출력

```json
{
  "title":      "2026 캡스톤 설계 경진대회 참가팀 모집",
  "start_date": "2026-05-01",
  "end_date":   "2026-06-30",
  "location":   "공학관 101호",
  "detail":     "학부생 팀 대상, 이메일 접수. 대상 100만원, 7월 15일 시상."
}
```

### 필드 설명

| 필드 | 타입 | 설명 |
|------|------|------|
| `title` | string | 공지 제목 (기관명 대괄호 제거, 의미 태그 유지) |
| `start_date` | YYYY-MM-DD or `""` | 접수/행사 시작일. 본문에 없으면 `""` |
| `end_date` | YYYY-MM-DD or `""` | 접수 마감일 또는 행사 종료일 |
| `location` | string or `""` | 실제 행사 장소. 온라인 접수만이면 `""` |
| `detail` | string | 대상·접수방법·시상 등 보조정보 (120자 이내) |

---

## 성능

| 기준 | calendar_ready | 비고 |
|------|---------------|------|
| 공정 평가셋 (75건) | **78.7%** | 튜닝에 미사용 |
| 개발 평가셋 (147건) | 83.7% | 후처리/cascade 튜닝 포함 |

**calendar_ready** = title AND start_date AND end_date AND location 모두 정확한 비율

### 자동 등록 판단

| 상태 | 건수 | precision |
|------|------|-----------|
| auto (자동 등록) | 130건 중 | **85.4%** |
| needs_review (검토 후 등록) | 17건 | — |

현재 배포 상태: **검토 포함 MVP** — 자동 등록 precision 90%+ 미달로 완전 자동 배포 보류

---

## 데이터셋 구축

### 크롤링

- 소스:
  - **ContestKorea** (`crawler/contestkorea.py`) — 공모전·대외활동
  - **Wevity** (`crawler/wevity.py`) — 대외활동·서포터즈·인턴십
  - **충북대 SW학부** (`crawler/cbnu.py`) — 학사공지·장학·취업/진로 (sub0401/0402/0403)
- 수집 필드: 공지 제목, 본문 전문, 출처 URL

### 자동 라벨링

- GPT-4o-mini로 5개 필드 자동 추출
- 라벨링 기준: 시스템 프롬프트와 동일한 정책 적용

### 수작업 정제

- title: 기관명 대괄호 제거 정책, 의미 태그 유지 정책 적용
- location: 비장소 문자열(온라인 접수, 이메일, 구글폼 등) 제거
- start_date: 시작일 명시 없는 케이스 `""` 처리
- 최종 학습 데이터: **train.jsonl 729건 / valid.jsonl 147건**

---

## 파인튜닝 설계

| 항목 | 내용 |
|------|------|
| 베이스 모델 | Llama-3.1-8B-Instruct |
| 학습 방식 | QLoRA 4-bit (Unsloth, gradient checkpointing) |
| LoRA 설정 | r=32, alpha=64, dropout=0.05 |
| 적용 모듈 | q/k/v/o/gate/up/down_proj (7개) |
| 입출력 | Chat template (system/user/assistant) |
| 출력 형식 | JSON 구조체 1개 (5개 필드) |
| 학습 설정 | 5 epochs, lr=5e-5, batch=1, grad_accum=4, bf16 |
| 최대 시퀀스 | 2048 tokens |

---

## 개선 과정

| 버전 | calendar_ready | 주요 변경 |
|------|---------------|---------|
| v1 | 67.0% | 672건 기반 첫 학습. title·날짜·location 기초 규칙 수립 |
| v2 | 70.7% | GT 라벨 수작업 정제. postprocess 추가 (location 비장소 제거, 연도 검증) |
| v3 | 72.1% | epochs 3→5. 과소학습 완화 |
| v4 | 74.8% | start_date="" 케이스 보강 (마감만 있는 공모·스포츠 행사). valid GT 추가 수정 |
| v5 | 76.9% | 재학습 없음. 날짜 검증기(safe_fix R1/R2) + 사업장소 복원 규칙 추가 |
| **v6** | **83.7%** | **Cascade 도입**: 날짜 불확실 케이스(~10%)에만 GPT-4o-mini fallback. 회귀 0건 |

---

## 시스템 프롬프트 설계

### 핵심 원칙

출력 형식, title, 날짜, location, detail 5개 섹션으로 구성.

**출력 형식**
- JSON 객체 하나만 출력. 코드블록·주석 금지.
- `}` 까지 반드시 완성 (detail보다 JSON 완성 우선).

**title 규칙**
- 공지 제목 섹션 그대로 사용. 요약·재작성 금지.
- `[기관명]`, `[기업명]` 대괄호 prefix → 제거
- `[국비무료]`, `[행복합니다]` 의미·캠페인 태그 → 유지
- D-day, 날짜 괄호, 기간연장 표시 → 제거

**날짜 규칙**
- 공지 유형 구분: 모집·공모 → 접수기간, 행사·교육 → 행사기간
- 시작일 본문 미명시 시 `start_date: ""` (end 값 복사 금지)
- 발표일·시상일·평가일 → detail 전용, start/end 사용 금지
- 연도 근거 없으면 임의 추론 금지

**location 규칙**
- 온라인 접수(이메일·구글폼·홈페이지) ≠ 온라인 행사
- 불특정 표현(전국, 일원, 추후공지) → `""`
- 행사 자체가 온라인이면 → `"온라인"`

---

## 파이프라인 구조

```
공지 입력
    ↓
[1단계] Fine-tuned Llama 8B 추론  (src/infer.py)
    ↓  postprocess.py 자동 적용
[2단계] Cascade — 날짜 불확실 케이스만 GPT-4o-mini 재처리
    ↓  trigger 탐지 → GPT fallback → 채택 검증
[3단계] 자동 등록 판단 (auto / needs_review)
```

- **1단계만 사용**: `src/infer.py`
- **전체 파이프라인**: `scripts/cascade_infer.py`

---

## 모델 가중치

- **베이스**: `meta-llama/Llama-3.1-8B-Instruct`
- **LoRA 어댑터**: HuggingFace `yunnjj72/capstone` (private)

```bash
huggingface-cli login
huggingface-cli download yunnjj72/capstone --local-dir outputs/v1
```

`.env` 파일 필요:
```
HF_TOKEN=hf_...
OPENAI_API_KEY=sk-proj-...   # cascade 실행 시
```

---

## 사용법

### 학습

```bash
$env:PYTHONUTF8 = "1"   # Windows
python src/train.py
```

### 평가

```bash
python src/eval.py
# 다른 데이터셋으로 평가 시
python src/eval.py --valid-data data/other_valid.jsonl --output-dir outputs/eval_other
```

결과: `outputs/v1/eval_results.json`, `outputs/v1/eval_summary.json`

### 단일 공지 추론

```bash
python src/infer.py --title "공지제목" --body "공지내용..."
```

### Cascade 파이프라인 (배치)

```bash
python scripts/cascade_infer.py

# GPT 호출 없이 trigger만 확인
python scripts/cascade_infer.py --dry-run

# 다른 eval 결과/데이터로 실행
python scripts/cascade_infer.py \
  --eval-results outputs/v2/eval_results.json \
  --valid-data data/other_valid.jsonl \
  --out-dir outputs/cascade_v2
```

### 회귀 테스트

```bash
python scripts/test_baseline.py
```

---

## 폴더 구조

```
model/
├── src/
│   ├── config.py           # 경로, 하이퍼파라미터, 시스템 프롬프트
│   ├── train.py            # QLoRA 파인튜닝 (Unsloth)
│   ├── eval.py             # 검증 세트 평가
│   ├── infer.py            # 단일 공지 추론
│   └── postprocess.py      # title/location/날짜 후처리
├── scripts/
│   ├── cascade_trigger_sim.py  # 날짜 trigger 탐지
│   ├── cascade_infer.py        # Cascade 파이프라인 (배치/단건)
│   ├── test_baseline.py        # 회귀 방지 테스트
│   └── fix_labels.py           # GT 라벨 정제
├── crawler/
│   ├── base.py
│   ├── contestkorea.py     # ContestKorea 크롤러
│   ├── wevity.py           # Wevity 크롤러
│   ├── cbnu.py             # 충북대 SW학부 크롤러
│   └── runner.py           # 전체 크롤 실행
├── data/
│   └── sample.jsonl        # 테스트용 샘플 10건 (git 포함)
│   # train.jsonl / valid.jsonl 은 .gitignore 제외 (별도 공유)
└── requirements.txt
```

---

## 환경 설정

```bash
pip install -r requirements.txt

# Windows + CUDA 128
pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# Windows 실행 시 필수
$env:PYTHONUTF8 = "1"
```

CUDA GPU 필수 (VRAM 16GB+ 권장).

---

## 후처리 (postprocess.py)

| 대상 | 처리 내용 |
|------|---------|
| title | 주차 표기 괄호 제거 (예: `(11월 2주차)`) |
| start_date | start==end이고 본문에 시작일 명시 없으면 `""` |
| start/end | pred 연도가 본문 연도와 다르면 본문 연도로 교정 |
| location `"온라인"` | 본문에 실제 온라인 행사 근거 없으면 `""` |
| location `""` | 본문에 온라인 행사 근거 있으면 `"온라인"` |
| location | 접수 방법 문자열, 이메일 주소, `"전국"` → `""` |
| location | 행정구역 전체명 → 약칭 (경기도→경기 등) |
| location | `"사업장소:"` 라벨 기반 복원 (좁은 조건) |
| detail | 120자 초과 시 문장 경계에서 잘라냄 |
