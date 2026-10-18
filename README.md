# slim-v9 일정 추출 모델

Built with Llama

Llama-3.1-8B-Instruct + LoRA로 한국어 공지 1건을 일정 목록 JSON으로 변환한다. 서비스·계속학습 코드만 제공하며 기반 모델, LoRA 가중치, 학습 데이터와 평가 정답은 배포하지 않는다. 기반 모델은 Meta의 이용 조건에 동의하고 별도로 준비한다.

## 일정 추출 정책

- 학생이 직접 신청·지원·제출하거나 참석해야 하는 날짜만 추출한다. 결과·서류·합격·수상 발표, 시상 포함 행사, 시상품 수령은 제외한다.
- 대외활동의 모집 마감, 날짜가 명시된 면접, 발대식·OT·위촉식은 포함한다. 수료식·해단식은 날짜가 명시된 필수 참석 행사만 포함한다. 몇 달짜리 활동·교육 기간은 제외하고 하루짜리 필수 사전교육·세션은 포함한다.
- 접수와 본행사가 각각 명시되고 서로 다르면 분리한다. 없는 단계는 만들지 않는다. 면접은 `interview`, 발대식 등은 `event`다. 면접은 서비스에서 캘린더 자동 등록 대상에서 제외해야 한다. 면접과 합격 발표가 같은 날로만 적힌 경우 면접은 추출하지 않는다.
- 등록금·참가비·생활관비·학자금 대출 실행 등 납부 마감은 `submission`이다. 선택 참여 설문, 시스템 중단·시설 이용 변경 안내는 제외한다.
- 신청·제출 장소는 방문·현장 접수처가 명시된 경우만 기록한다. 제출처로 명시된 사무실은 방문 제출 장소로 본다. 대회 지역·본행사장을 접수 장소에 붙이지 않는다. 시각은 해당 일정에 명시된 경우만 기록한다.
- 연도 없는 날짜는 게시일 `reference_time` 이후 12개월 안에 해당 월·일이 처음 오는 연도를 사용한다. 게시일이 없으면 연도를 추정하지 않는다. 기간은 종료일 기준으로 연도를 정하고, 시작 월이 종료 월보다 크면 시작일만 전년으로 둔다. 현재 날짜나 다른 글로 보충하지 않는다.
- 요일과 숫자 날짜가 다르면 숫자 날짜를 따른다. 사이트 상단 요약보다 본문을 우선하며, 본문에 날짜가 없고 상시·선착순이면 상단 날짜를 쓴다. 사용자가 추출 결과를 확인·수정할 수 있어야 한다.

## 설치와 서버 실행

Python 3.11과 CUDA 환경을 사용한다. `requirements.txt`는 검증 환경의 서비스·학습 관련 패키지 버전을 고정한다. 추론은 CUDA와 4bit bitsandbytes, 학습은 BF16 지원 GPU가 필요하다. 다른 OS·드라이버 환경은 별도 호환성 확인이 필요하다.

```powershell
python -m pip install -r requirements.txt
python -m model_server.slim_server --prompt-mode slim-v9 --interview-type passthrough --model-dir <adapter> --host 127.0.0.1 --port 8765
```

`<adapter>`는 `adapter_config.json`, `adapter_model.safetensors`와 토크나이저가 있는 로컬 디렉터리다. 기반 모델도 로컬에 준비하며 서버는 다운로드 없이 로드한다. 외부 인터페이스에서는 `--host 0.0.0.0`을 사용하고 `MODEL_API_TOKEN`을 32자 이상으로 환경변수에 설정해야 한다. 토큰이 설정되면 모든 요청에 `X-Model-Token` 헤더가 필요하다. 토큰 값은 명령·문서·로그에 기록하지 않는다. CLI의 면접 기본값은 `map-to-event`이므로 위 명령처럼 `passthrough`를 명시한다.

## API와 입출력

`GET /health`는 준비·사용 중 여부와 모델/프롬프트 해시를 반환한다. `POST /extract`는 `Content-Type: application/json`으로 다음 요청을 받는다.

```json
{"title":"공지 제목","body":"공지 본문","reference_time":"2026-10-07T09:00:00+09:00"}
```

`title`, `body`는 필수 문자열이며 둘 다 공백이면 422다. 제목 4,000자·본문 80,000자 초과 또는 요청 크기 400,000바이트 초과는 413이다. `user_content`는 호환용 선택 필드로 받지만 사용하지 않는다. 예상 밖 필드는 422다. 첨부 열람이나 수집 시각 보충은 하지 않는다.

`reference_time`은 선택 게시일이다. `YYYY-MM-DD` 또는 `YYYY-MM-DDTHH:MM[:SS[.fraction]][Z 또는 ±HH:MM]`을 받는다. 누락·null은 null이고 잘못된 날짜·시각·타입은 null로 바꾸며 `metadata.reference_time_warning="invalid_reference_time"`을 기록한다. 현재 날짜로 대체하지 않는다.

```json
{"schema_version":"notiai-model-candidate-v1","status":"candidate","candidate":{"events":[{"event_type":"interview","title":"면접","start_date":"2026-10-15","end_date":"2026-10-15","start_time":null,"end_time":null,"location":null,"attendance_mode":"online","schedule_status":"unknown"}]},"raw_output":"모델 원시 출력","validation_errors":[],"review_required":true,"metadata":{"prompt_mode":"slim-v9","reference_time":null,"finish_reason":"stop","elapsed_seconds":1.0}}
```

| 필드 | 형식 |
|---|---|
| event_type | application, submission, event, interview |
| title | 일정 제목 문자열 |
| start_date, end_date | YYYY-MM-DD 또는 null |
| start_time, end_time | HH:MM 또는 null |
| location | 물리 장소 문자열 또는 null |
| attendance_mode | offline, online, hybrid, unknown, not_applicable |
| schedule_status | 서버가 추가하는 unknown |

학습·모델 원시 JSON은 `schedule_status`를 제외한 8개 필드이며 서버 후보는 9개 필드다. 일정이 없으면 `events=[]`이다. 공통 검증기는 호환 목적으로 `service_change`도 허용하지만 추출 정책상 제외 대상이다. 관계 정보는 생성하지 않는다.

온라인 일정에 실제 물리 장소가 있으면 장소를 보존하고 참여 방식을 `unknown`으로 바꾼다. 온라인 매체·placeholder 장소는 null로 바꾼다. 잘못된 달력 날짜는 해당 필드만 null, 역전 기간은 종료일만 null로 바꾸고 날짜가 사라진 필드의 시각도 제거한다. 양쪽 날짜가 모두 null인 일정은 제외한다. 원시 출력은 보존하고 대체 날짜·장소를 생성하지 않는다.

잘못된 JSON·스키마나 생성 한도·시간 초과는 `status=invalid`이며 후보는 null 또는 진단용일 수 있다. HTTP 200은 의미 정확성을 보증하지 않으며 `review_required`는 항상 true다. `metadata`에는 모델·프롬프트·입력 해시, 토큰 수, 종료 사유, 지연, 게시일과 경고, `raw_status`, `raw_validation_errors`, `postprocess_repairs`, `review_required_events`, 보정·제외 건수와 면접 변환 정보가 포함된다. 수정 사유에는 `online_physical_location_uncertain`, `invalid_calendar_date`, `end_date_before_start_date`가 있다.

출력 한도 1,024토큰, 문맥 예산 12,288토큰, 생성 시간 60초다. 입력은 조용히 자르지 않는다. 생성 시간 제한은 엄밀한 HTTP 응답 시간 상한이 아니다. 인증 실패 401, 잘못된 요청·문맥 초과 422, 동시 두 번째 요청 503, 내부 추론 실패 500이다.

## 계속 학습

학습 JSONL은 한 줄에 `messages`(system/user/assistant 문자열 대화)와 `source_group_ids`(원문 그룹 ID 목록)를 갖는다. 학습과 검증 원문 그룹은 분리해야 하며 assistant 정답은 8필드 일정 JSON 문자열이다. 실제 데이터는 제공하지 않는다. 시스템 프롬프트는 `model_server/prompts/system_prompt.txt`, 사용자 입력 생성은 `model_server/prompt_input.py`를 사용한다.

```powershell
python training/training_runner.py --train-data <train-jsonl> --valid-data <validation-jsonl> --tokenizer <v10-adapter> --init-adapter <v10-adapter> --max-length 4096 --epochs 3 --report <new-report-json> --output-dir <new-output-dir> --save-model-only --chunked-head-loss-tokens 256 --bf16-frozen-io --memory-efficient-sdpa --train
```

초기 어댑터는 로컬 v10 LoRA이며 새 옵티마이저로 계속 학습한다. 기반 모델 경로가 어댑터 설정에 존재해야 한다. 원본 어댑터는 변경하지 않으며 출력 디렉터리는 새 경로여야 한다. `--train` 없이 실행하면 CPU 토큰화·assistant-only loss 마스킹·길이·중복 검사를 수행한다. `--prepare-only`는 GPU 모델과 Trainer를 준비하므로 CPU 검사와 다르다. 입력을 절단하지 않으며 4,096토큰 초과는 실패한다. batch 1, gradient accumulation 4, 학습률 5e-5, seed 42, BF16 QLoRA를 사용한다. 체크포인트의 eval loss와 실제 생성 품질은 별도로 평가해야 한다.

서비스 후처리는 `model_server/slim_v1.py`에 포함된다. 과거 단일 일정 5필드 후처리와 실험용 손실·분석 코드는 포함하지 않는다.

## 성능과 평가

135건의 개발 평가셋에서 slim-v9 마지막 epoch 결과는 다음과 같다. 이 평가셋은 모델 개선에 반복 사용했으므로 독립 최종 검증 성능과 구분한다.

| 지표(%) | 원시 출력 | 후처리 |
|---|---:|---:|
| 일정 재현율 | 87.7 | 87.7 |
| 일정 정밀도 | 93.4 | 95.5 |
| 날짜 정확도 | 90.1 | 90.1 |
| 마감 정확도 | 93.2 | 93.2 |
| 장소 정확도 | 94.7 | 94.7 |
| 시각 정확도 | 97.1 | 97.1 |
| 완벽 일치 | 68.9 | 71.1 |
| 공지 단위 날짜 일치 | 72.6 | 74.8 |
| 허구 일정 | 0.7 | 0.7 |
| 빈 정답 오탐 | 4.2 | 0.0 |
| 무효 출력 | 0.0 | 0.0 |

평균 지연 6.6초, p95 16.5초, 최대 32.2초, 완료율 100%였다. 봉인 test의 날짜+장소 정확도는 73.7%다. 지연은 측정 환경에 따라 달라진다.

평가 정답은 모델 출력을 보지 않고 원문을 읽어 작성했고 채점 규칙과 채택 기준은 평가 전에 고정했다. 원시 출력과 후처리 출력을 구분하여 일정 단위 재현·정밀, 필드 정확도와 공지 단위 일치를 측정한다. 후처리는 구조·날짜 유효성 보정이며 의미 정확도 개선을 보장하지 않는다. 평가 데이터·정답·채점 산출물은 이 저장소에 포함하지 않는다.

## 검증과 한계

```powershell
python -B -m unittest discover -s tests -v
python -B tools/check_release.py
```

테스트는 합성 입력과 가짜 runtime을 사용하므로 가중치·GPU가 필요 없다. 실제 모델 로딩·GPU 호환성·추론 품질을 검증하는 테스트는 아니다. 누락·허구 일정, 장소 연결 오류, 연도 해석 오류가 남아 있고 첨부 자료는 입력하지 않는다. 서비스에서는 원문과 함께 결과를 확인·수정할 수 있어야 한다.

## 라이선스

Built with Llama. 기반 모델 및 파생 모델 사용에는 [Llama 3.1 Community License](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/LICENSE)와 해당 이용 정책이 적용된다. 기반 모델·LoRA 가중치는 이 저장소에서 배포하지 않는다. 이 고지는 저장소 코드에 별도 라이선스를 부여하는 선언이 아니다.
