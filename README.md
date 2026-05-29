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

## 현재 성능 (v3, 2026-05-30 기준)

| 지표 | v1 | v2 | **v3** |
|------|-----|-----|--------|
| **calendar_ready** (title+날짜+location 모두 정확) | 66.7% | 70.7% | **73.5%** |
| title 정확도 | 93.2% | 94.6% | 93.9% |
| title 정규화 정확도 | 94.6% | 95.9% | 95.2% |
| start_date 정확도 | — | — | **87.1%** |
| 날짜쌍 정확도 | 83.0% | 86.4% | **85.7%** |
| location 정확도 | 83.7% | 85.0% | **89.8%** |
| 유효 JSON 생성율 | 100% | 100% | 100% |

- 평가 세트: `data/valid.jsonl` 147건
- calendar_ready = title AND 날짜쌍 AND location 모두 정확한 비율
- location 정규화 적용 시 calendar_ready **74.1%**

### v3 주요 변경사항 (v2 대비)

- **학습 프레임워크**: 기존 peft+transformers → **Unsloth** (VRAM 40~70% 절감, GPU 드라이버 크래시 해결)
- **정밀도**: fp16 → **bf16** (Ampere GPU 최적화)
- **시스템 프롬프트 개선**:
  - 대괄호 표현 원문 유지 규칙 추가 (`[국비무료]`, `(첫 단추 프로젝트)` 등)
  - `start_date=""` 규칙 강화 (접수 시작일 미명시 시 빈 문자열 명시)
  - `end_date` 복사 금지 명시
- **GT 라벨 정제**: validation 세트 명백 오류 수정, start_date="" 케이스 보강
- **후처리 추가**: `postprocess_title()` — 주차 표기 괄호 제거

## 모델

- **베이스 모델**: `meta-llama/Llama-3.1-8B-Instruct`
- **학습 방법**: QLoRA (4-bit quantization, Unsloth, r=32, lora_alpha=64)
- **학습 데이터**: `data/train.jsonl` 672건 (대학 공지, 대외활동, 공모전 등)
- **학습 설정**: 5 에폭, lr=5e-5, batch=1, grad_accum=4, max_seq_len=2048, bf16
- **학습된 가중치**: [HuggingFace `yunnjj72/capstone`](https://huggingface.co/yunnjj72/capstone) (private)

## 학습된 가중치 다운로드

학습된 LoRA 어댑터 가중치는 HuggingFace private 저장소에 있습니다.

### 사전 조건

1. HuggingFace 계정 생성 후 저장소 접근 권한 요청 (관리자에게 GitHub ID 또는 HF 계정명 알려주기)
2. HuggingFace 토큰 발급: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → `New token` → `Read` 권한
3. 프로젝트 루트에 `.env` 파일 생성:

```
HF_TOKEN=hf_여기에토큰값
```

### 다운로드 방법

```python
import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()
snapshot_download(
    repo_id="yunnjj72/capstone",
    local_dir="outputs/v1",
    token=os.environ["HF_TOKEN"],
)
```

또는 CLI로:

```bash
huggingface-cli login   # 토큰 입력
huggingface-cli download yunnjj72/capstone --local-dir outputs/v1
```

다운로드 후 `outputs/v1/` 안에 `adapter_model.safetensors`, `adapter_config.json` 등이 있으면 정상입니다.
평가·추론은 기존대로 `python src/eval.py`, `python src/infer.py` 실행하면 됩니다.

## 폴더 구조

```
model/
├── src/              # 학습·평가·추론 코드
│   ├── config.py     # 경로, 하이퍼파라미터, 시스템 프롬프트
│   ├── train.py      # QLoRA 파인튜닝 (Unsloth)
│   ├── eval.py       # 검증 세트 평가
│   ├── infer.py      # 단일 샘플 추론
│   └── postprocess.py # title/location/detail 후처리
├── data/
│   └── sample.jsonl  # 추론 테스트용 샘플 (10건, git 포함)
│   # train.jsonl / valid.jsonl 은 .gitignore 제외 (별도 공유)
├── crawler/          # 공지 크롤러
│   ├── base.py
│   ├── cbnu.py
│   ├── contestkorea.py
│   ├── wevity.py
│   └── runner.py
├── scripts/
│   ├── fix_labels.py    # GT 라벨 정제 스크립트
│   └── simulate_fixes.py # 코드/GT 수정 효과 시뮬레이션
├── run.py            # 학습 → 평가 순차 실행
└── requirements.txt
```

## 환경 설정

```bash
pip install -r requirements.txt
# Windows에서 Unsloth 사용 시 CUDA torch 재설치 필요
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
# 실행 시 반드시 PYTHONUTF8=1 환경변수 설정 (Windows cp949 인코딩 오류 방지)
```

베이스 모델은 Hugging Face에서 다운로드하거나 로컬 경로를 `src/config.py`의 `BASE_MODEL_NAME`에 지정하세요.
CUDA GPU 필수 (VRAM 16GB+ 권장).

## 사용법

### 학습 + 평가 한 번에 실행

```bash
# Windows
$env:PYTHONUTF8 = "1"
python run.py
```

로그는 `outputs/v1/train_run.log`, `outputs/v1/eval_run.log`에 저장됩니다.

### 학습만 실행

```bash
cd src
python train.py
```

### 평가만 실행 (학습된 모델 필요)

```bash
cd src
python eval.py
```

결과: `outputs/v1/eval_results.json`, `outputs/v1/eval_summary.json`

### 단일 공지 추론

```bash
cd src
python infer.py --auto-user "[제목]\n공지제목\n\n[본문]\n공지내용..."
```

## 데이터 형식

`data/train.jsonl`의 각 줄은 다음 형식입니다:

```json
{
  "messages": [
    {"role": "system", "content": "...시스템 프롬프트..."},
    {"role": "user",   "content": "[제목] ...\n[본문] ..."},
    {"role": "assistant", "content": "{\"title\": ..., \"start_date\": ..., ...}"}
  ]
}
```

## 데이터 레이블 정제

GT 라벨에 오류가 있을 때 `scripts/fix_labels.py`를 실행합니다:

```bash
python scripts/fix_labels.py
```

## 후처리 (postprocess.py)

모델 출력에 자동으로 적용되는 후처리:

- `title` — 주차 표기 괄호 제거 (예: `(11월 2주차)`)
- `location = "온라인"` → 본문에 실제 온라인 행사 근거 없으면 `""`
- `location = ""` → 본문에 온라인 행사 근거 있으면 `"온라인"`
- `location`이 접수 방법 문자열이면 `""`
- `detail` 120자 초과 시 문장 경계에서 잘라냄

## 향후 개선 방향

- [ ] 학습 데이터 추가 크롤링 (목표 1,000건 이상)
- [ ] cascade 추론 도입 (신뢰도 낮은 케이스 → GPT-4o-mini fallback, 목표 ~83%)
- [ ] start_date="" 케이스 추가 학습 (현재 25건 → 50건 이상)
