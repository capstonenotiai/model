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

## 현재 성능 (v1, 2026-05-25 기준)

| 지표 | 점수 |
|------|------|
| **calendar_ready** (title+날짜+location 모두 정확) | **66.7%** |
| title 정확도 | 93.2% |
| title 정규화 정확도 | 94.6% |
| 날짜쌍 정확도 | 83.0% |
| location 정확도 | 83.7% |
| 유효 JSON 생성율 | 100% |

- 평가 세트: `data/valid.jsonl` 147건
- calendar_ready = title AND 날짜쌍 AND location 모두 정확한 비율

## 모델

- **베이스 모델**: `meta-llama/Llama-3.1-8B-Instruct`
- **학습 방법**: QLoRA (4-bit quantization, r=16, lora_alpha=32)
- **학습 데이터**: `data/train.jsonl` 587건 (대학 공지, 대외활동, 공모전 등)
- **학습 설정**: 4 에폭, lr=5e-5, batch=1, grad_accum=4, max_seq_len=2048

## 폴더 구조

```
model/
├── src/              # 학습·평가·추론 코드
│   ├── config.py     # 경로, 하이퍼파라미터, 시스템 프롬프트
│   ├── train.py      # QLoRA 파인튜닝
│   ├── eval.py       # 검증 세트 평가
│   ├── infer.py      # 단일 샘플 추론
│   └── postprocess.py # location/detail 후처리
├── data/
│   ├── train.jsonl   # 학습 데이터 (587건)
│   ├── valid.jsonl   # 검증 데이터 (147건)
│   └── sample.jsonl  # 추론 테스트용 샘플 (10건)
├── crawler/          # 공지 크롤러
│   ├── base.py
│   ├── cbnu.py
│   ├── contestkorea.py
│   ├── wevity.py
│   └── runner.py
├── scripts/
│   └── fix_labels.py # GT 라벨 정제 스크립트
├── run.py            # 학습 → 평가 순차 실행
└── requirements.txt
```

## 환경 설정

```bash
pip install -r requirements.txt
```

베이스 모델은 Hugging Face에서 다운로드하거나 로컬 경로를 `src/config.py`의 `BASE_MODEL_NAME`에 지정하세요.
CUDA GPU 필수 (VRAM 16GB+ 권장, RTX 5070 Ti에서 테스트).

## 사용법

### 학습 + 평가 한 번에 실행

```bash
python run.py
```

로그는 `outputs/v1/train_run.log`, `outputs/v1/eval_run.log`에 저장됩니다.
학습된 모델 웨이트와 평가 결과는 `outputs/v1/`에 저장됩니다.

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
python infer.py
```

`src/infer.py`에서 입력 텍스트를 수정하거나 함수로 호출하세요.

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

- '온라인 접수', '이메일 접수' 등 접수 방법이 location에 잘못 기재된 경우 → `""` 로 수정
- 불가시 특수문자 제거
- detail 120자 초과 자동 잘라내기

## 후처리 (postprocess.py)

모델 출력에 자동으로 적용되는 후처리:

- `location = "온라인"` → 본문에 실제 온라인 행사 근거 없으면 `""`
- `location = ""` → 본문에 온라인 행사 근거 있으면 `"온라인"`
- `location`이 접수 방법 문자열이면 `""`
- `detail` 120자 초과 시 문장 경계에서 잘라냄

## 향후 개선 방향

- [ ] 날짜 정확도 83% → 90%+ (접수시작일/행사일 혼동, 연도 오인 케이스 데이터 보강)
- [ ] 학습 데이터 추가 레이블링 (200~300건 목표)
- [ ] 신뢰도 기반 선택적 자동 등록 기능
