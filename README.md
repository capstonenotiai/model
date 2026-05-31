# model

대학 공지문에서 캘린더 일정 정보를 자동으로 추출하는 LLM 파인튜닝 프로젝트입니다.

## 개요

공지문 텍스트를 입력하면 캘린더 등록에 필요한 JSON을 출력합니다.

```json
{
  "title": "2026 캡스톤 설계 경진대회",
  "start_date": "2026-05-01",
  "end_date": "2026-06-30",
  "location": "공학관 101호",
  "detail": "학부생 팀 대상, 지도교수 서명 필요. 수상팀 장학금 지급."
}
```

## 성능 이력 (2026-06-01 기준)

| 버전 | calendar_ready | 방법 |
|------|---------------|------|
| v1 | 66.7% | 초기 학습 (672건, 3 에폭) |
| v2 | 70.7% | 5 에폭, GT 라벨 정제, start_date="" 보강 |
| v3 | 73.5% | GT 추가 정제 (title 대괄호, location), Unsloth 도입 |
| v4 | 76.9% | 후처리 강화 — 재학습 없음 |
| **v5** | **83.0% 🏆** | **Cascade Tier 1 (GPT-4o-mini fallback) 추가** |

- 평가 세트: `data/valid.jsonl` 147건
- calendar_ready = title AND 날짜쌍(start+end) AND location 모두 정확한 비율

### 자동 등록 판단 (v5 기준)

Cascade 이후 각 결과에 자동 등록 가능 여부를 판단합니다.

| 상태 | 건수 | precision |
|------|------|-----------|
| auto (자동 등록) | 132건 | **86.4%** |
| needs_review (검토 후 등록) | 15건 | — |

현재 배포 상태: **검토 포함 MVP** — 자동 등록 precision 90%+ 미달로 완전 자동 배포 보류

---

## 파이프라인 구조 (v5)

```
공지 입력
    ↓
[1단계] Fine-tuned Llama 8B 추론  (src/infer.py)
    ↓  후처리 (postprocess.py)
[2단계] Cascade — 날짜 불확실 케이스만 GPT-4o-mini 재처리  (scripts/cascade_infer.py)
    ↓  cascade trigger 탐지, safe fix, GPT fallback, 채택 검증
[3단계] 자동 등록 판단  (auto / needs_review)
```

- **1단계만 사용**: `src/infer.py` (모델 로드 + 단건 추론)
- **전체 파이프라인**: `scripts/cascade_infer.py` (eval_results.json 기반 배치/단건)

---

## 버전별 변경사항

### v4 — 후처리 강화 (재학습 없음)

`src/postprocess.py` 대폭 업데이트:

- **날짜 검증기 추가** (`postprocess_dates()`):
  - start == end이고 본문에 시작일 명시 없으면 start=""
  - pred 연도가 본문에 없는 연도면 본문 연도로 교정
- **location 후처리 강화**:
  - 이메일 주소 포함 location → "" (접수처 주소 오인식 제거)
  - '전국' → "" (특정 장소 없음)
  - 사업장소 라벨 기반 복원 (좁은 조건, "팔복예술공장" 유형)
  - 행정구역 전체명 → 약칭 (경기도→경기 등)
- **GT 정제 3회차**: validation 세트 명백 오류 수정

### v5 — Cascade Tier 1 (GPT-4o-mini)

날짜 추출이 불확실한 케이스에만 GPT-4o-mini를 fallback으로 사용합니다.

**Trigger 조건** (하나라도 해당 시 GPT 호출):
- start, end 모두 빈값 (날짜를 전혀 못 찾음)
- end가 발표/시상일 전용 날짜
- pred 날짜의 연도가 본문 날짜와 불일치
- end_date가 본문 날짜 후보에 없음
- start_date가 본문 날짜 후보에 없음

**채택 조건** (GPT 결과 검증):
- end="" 인데 본문에 마감일 있으면 거부
- start > end 이면 거부
- end가 발표일 전용이면 거부
- start가 본문에 없는 날짜면 거부
- 연도 변경 시 full date 근거 필요

**v5 결과**:

| 지표 | 값 |
|------|---|
| GPT 호출 | 15건 (전체의 10%) |
| GPT 채택 | 12건 |
| GPT 거부 | 3건 |
| 개선 (❌→✅) | **5건** |
| 회귀 (✅→❌) | **0건** |

---

## 모델

- **베이스 모델**: `meta-llama/Llama-3.1-8B-Instruct`
- **학습 방법**: QLoRA (4-bit, Unsloth, r=32, lora_alpha=64, bf16)
- **학습 데이터**: `data/train.jsonl` 729건 (대학 공지, 대외활동, 공모전)
- **학습 설정**: 5 에폭, lr=5e-5, batch=1, grad_accum=4, max_seq_len=2048
- **학습된 가중치**: [HuggingFace `yunnjj72/capstone`](https://huggingface.co/yunnjj72/capstone) (private)

---

## 학습된 가중치 다운로드

학습된 LoRA 어댑터 가중치는 HuggingFace private 저장소에 있습니다.

### 사전 조건

1. HuggingFace 계정 생성 후 저장소 접근 권한 요청
2. HuggingFace 토큰 발급: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → `New token` → `Read` 권한
3. 프로젝트 루트에 `.env` 파일 생성:

```
HF_TOKEN=hf_여기에토큰값
OPENAI_API_KEY=sk-proj-...   # cascade 실행 시 필요
```

### 다운로드

```bash
huggingface-cli login   # 토큰 입력
huggingface-cli download yunnjj72/capstone --local-dir outputs/v1
```

다운로드 후 `outputs/v1/` 안에 `adapter_model.safetensors`, `adapter_config.json` 등이 있으면 정상입니다.

---

## 폴더 구조

```
model/
├── src/                    # 모델 학습·추론 코드
│   ├── config.py           # 경로, 하이퍼파라미터, 시스템 프롬프트
│   ├── train.py            # QLoRA 파인튜닝 (Unsloth)
│   ├── eval.py             # 검증 세트 평가
│   ├── infer.py            # 단일 공지 추론
│   └── postprocess.py      # title/location/날짜 후처리
├── scripts/
│   ├── cascade_trigger_sim.py  # 날짜 trigger 탐지 로직
│   ├── cascade_infer.py        # Cascade 파이프라인 (배치/단건)
│   ├── test_baseline.py        # 회귀 방지 테스트
│   └── fix_labels.py           # GT 라벨 정제
├── crawler/                # 공지 크롤러
│   ├── base.py
│   ├── cbnu.py
│   ├── contestkorea.py
│   ├── wevity.py
│   └── runner.py
├── data/
│   └── sample.jsonl        # 추론 테스트용 샘플 (10건, git 포함)
│   # train.jsonl / valid.jsonl 은 .gitignore 제외 (별도 공유)
├── run.py
└── requirements.txt
```

---

## 환경 설정

```bash
pip install -r requirements.txt

# Windows에서 Unsloth 사용 시 CUDA torch 재설치 필요
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128

# 실행 시 반드시 PYTHONUTF8=1 환경변수 설정 (Windows cp949 인코딩 오류 방지)
```

CUDA GPU 필수 (VRAM 16GB+ 권장).

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
```

결과: `outputs/v1/eval_results.json`, `outputs/v1/eval_summary.json`

### 단일 공지 추론 (모델만)

```bash
python src/infer.py --auto-user "[제목]\n공지제목\n\n[본문]\n공지내용..."
# 또는
python src/infer.py --title "공지제목" --body "공지내용..."
```

### Cascade 파이프라인 (배치)

eval_results.json이 있어야 합니다 (평가 먼저 실행).

```bash
# .env에 OPENAI_API_KEY 필요
python scripts/cascade_infer.py

# GPT 호출 없이 trigger만 확인
python scripts/cascade_infer.py --dry-run
```

결과: `outputs/cascade_tier1/cascade_results.json`, `outputs/cascade_tier1/cascade_summary.json`

### Cascade 단건 모드

```bash
python scripts/cascade_infer.py --input input.json --output result.json
```

`input.json` 형식:
```json
{
  "body": "[제목]\n공지제목\n\n[본문]\n공지내용...",
  "pred_json_text": "{\"title\": \"...\", \"start_date\": \"...\"}"
}
```

### 회귀 테스트

```bash
python scripts/test_baseline.py
```

---

## 데이터 형식

`data/train.jsonl`의 각 줄:

```json
{
  "messages": [
    {"role": "system",    "content": "...시스템 프롬프트..."},
    {"role": "user",      "content": "[제목]\n...\n\n[본문]\n..."},
    {"role": "assistant", "content": "{\"title\": ..., \"start_date\": ..., ...}"}
  ]
}
```

---

## 후처리 (postprocess.py)

모델 출력에 자동 적용:

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

---

## 향후 개선 방향

- [x] cascade 추론 도입 (v5, 83.0% 달성)
- [x] start_date="" 케이스 추가 학습
- [ ] 날짜 훈련 보강 (v6): 역할 혼재 케이스 + 연도 규칙 강화 → auto precision 90%+ 목표
- [ ] title 라벨 정책 확정 및 재학습: 기관명/대괄호 유지 여부 통일
- [ ] 학습 데이터 추가 크롤링 (현재 729건 → 목표 1,000건 이상)
