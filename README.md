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

평가셋: 개발셋 75건(튜닝과 분리), 최종 테스트셋 75건은 봉인.

### 운영 안전성 (실사용 핵심 지표)

| 지표 | 값 |
|------|------|
| **마감일 안전성** | **100%** — 자동등록 중 마감일을 실제보다 늦게 표시한 사고 **0건** |
| end_date 정확도 | 94.7% |
| date_safe_ready | 92.0% |

캘린더 자동등록에서 가장 치명적인 오류는 *마감일을 실제보다 늦게 알려 사용자가 기회를 놓치는 것*이다.
이 위험 방향(`pred_end > 실제`) 오류율 = **0%**. 이것이 이 모델의 가장 중요한 보증이다.

### 추출 정확도 (개발셋 75건)

| 지표 | 값 | 설명 |
|------|-----|------|
| strict_calendar_ready | 81.3% | 4필드(title·start·end·location) 모두 exact — 회귀 가드 |
| realuse_ready | 82.7% | title 정규화 + 날짜 exact + location loose — 실사용 품질 |
| location_quality | exact 67 / loose 69 / missing 4 / 오등록 2 | 5버킷 진단 |

**왜 strict 하나로 보지 않나**: 필드마다 틀렸을 때의 피해가 다르다. end_date(마감일)는 치명적,
location은 상위지역(loose)이라도 사용 가능, title은 식별만 되면 됨. 그래서 회귀 감시용 strict와
실사용 품질용 real-use를 함께 본다. (전체 지표는 `cascade_infer.py` 실행 시 생성되는 `outputs/cascade_tier1/cascade_summary.json`에 저장된다)

### 자동 등록 판단

| 상태 | 건수 | 비고 |
|------|------|-----------|
| auto (자동 등록) | 72건 | 마감일 위험 오류 0건 |
| needs_review (검토 후 등록) | 3건 | 위험 케이스는 여기로 격리 |

배포 상태: **검토 보조형 MVP** — 자동등록 + 검토 큐 형태로 사용 가능(마감일 위험 0%).
완전 무인 배포는 실제 서비스 도메인(캠퍼스 공지) 측정 후 판단.

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
- 최종 데이터: **train.jsonl 729건 / valid.jsonl 75건(개발셋) / test.jsonl 75건(봉인)**
  - dev/test는 튜닝에 사용하지 않음 (GT 수정은 라벨 정책 오류일 때만)
  - test.jsonl은 최종 평가 시 1회만 사용

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

### 1. 베이스 모델 선택 (탐색)

같은 데이터·프롬프트로 후보 모델을 비교해 Llama-3.1-8B로 고정.

| 모델 | calendar_ready (탐색 36건) | 판단 |
|------|------|------|
| Kanana-1.5-8B | ~0% | 한국어 8B지만 JSON 형식 준수가 약함 |
| Llama-3.2-1B | 2.8% | 너무 작음 |
| Llama-3.2-3B | 11.1% | 부족 |
| **Llama-3.1-8B** | **19.4%** | **채택** — 형식 준수·추출력 균형 |

### 2. 데이터·후처리·학습 개선

데이터를 178 → 424 → 672건으로 확장하며 프롬프트·라벨·후처리를 반복 개선.

| 단계 | calendar_ready | 주요 변경 |
|------|---------------|---------|
| 첫 정식 학습 (672건) | 67.0% | title·날짜·location 기초 규칙 수립 |
| GT 라벨 수작업 정제 | 70.7% | postprocess 추가 (비장소 location 제거, 행정구역 약칭, 연도 검증) |
| epochs 3→5 | 72.1% | 과소학습 완화 |
| start_date="" 보강 | 74.8% | "마감만 있는 공모" 등 시작일 없는 케이스 학습 + 라벨 정정 |
| 날짜 검증기 + 장소 복원 | 76.9% | 재학습 없이 후처리만: start==end 교정, 연도 교정, 사업장소 복원 |
| **Cascade 도입** | **83.7%\*** | 날짜 불확실 케이스(~10%)만 GPT-4o-mini fallback, 회귀 0건 |

### 3. 학습 안정화 (Unsloth)

체크포인트 저장 중 GPU 드라이버 크래시(TDR/BSOD)로 학습이 step 200에서 반복 중단됨.
**Unsloth 4-bit + gradient checkpointing**으로 VRAM 사용을 크게 줄여 해결 → 5 epochs 완주.

### 4. 한계 발견과 평가 재정비 (정직화)

높아 보이던 점수를 검증하는 과정에서 더 중요한 교훈들을 얻음.

| 발견 | 내용 |
|------|------|
| 데이터 단순 증가의 역효과 | 자동라벨 수백 건 추가 시 title −4~5%p 회귀. **양보다 라벨 정책 일관성이 중요** |
| 과적합 발견 | 83.7%는 튜닝에 반복 사용한 평가셋 수치. 미사용 holdout에선 72% |
| 평가체계 재구축 | dev/test(각 75건) 분리, dev/test는 튜닝 금지 → 공정 재측정 **81.3%** |
| location 천장 | 추가 재학습 3회 모두 location 회귀 — 가용 데이터의 장소-양성 케이스 부족이 근본 원인 |
| 측정 재정의 | strict 한 줄 대신 필드 우선순위 도입 → **real-use 82.7%, 마감일 안전성 100%** |

\* **67% → 83.7%는 개발 평가셋(구 147건) 기준 진행 수치**이며, 이 셋은 튜닝에 반복 사용되어 과적합됨.
이후 dev/test를 분리하고 GT를 오딧해 공정하게 재측정한 정직한 값이 **81.3%(strict) / 82.7%(real-use)**,
운영상 가장 중요한 지표는 **마감일 안전성 100%**다. 더 올리려면 실제 서비스 도메인 데이터가 필요하다.

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
│   # train.jsonl(729) / valid.jsonl(75, 개발셋) / test.jsonl(75, 봉인)
│   #   은 .gitignore 제외
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
