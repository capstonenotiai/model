# 모델 서버 API

[README](../README.md)로 돌아가기

## 요청과 응답

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

