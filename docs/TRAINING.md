# 계속 학습

[README](../README.md)로 돌아가기


학습 JSONL은 한 줄에 `messages`(system/user/assistant 문자열 대화)와 `source_group_ids`(원문 그룹 ID 목록)를 갖는다. 학습과 검증 원문 그룹은 분리해야 하며 assistant 정답은 8필드 일정 JSON 문자열이다. 실제 데이터는 제공하지 않는다. 시스템 프롬프트는 `model_server/prompts/system_prompt.txt`, 사용자 입력 생성은 `model_server/prompt_input.py`를 사용한다.

```powershell
python training/training_runner.py --train-data <train-jsonl> --valid-data <validation-jsonl> --tokenizer <v10-adapter> --init-adapter <v10-adapter> --max-length 4096 --epochs 3 --report <new-report-json> --output-dir <new-output-dir> --save-model-only --chunked-head-loss-tokens 256 --bf16-frozen-io --memory-efficient-sdpa --train
```

초기 어댑터는 로컬 v10 LoRA이며 새 옵티마이저로 계속 학습한다. 기반 모델 경로가 어댑터 설정에 존재해야 한다. 원본 어댑터는 변경하지 않으며 출력 디렉터리는 새 경로여야 한다. `--train` 없이 실행하면 CPU 토큰화·assistant-only loss 마스킹·길이·중복 검사를 수행한다. `--prepare-only`는 GPU 모델과 Trainer를 준비하므로 CPU 검사와 다르다. 입력을 절단하지 않으며 4,096토큰 초과는 실패한다. batch 1, gradient accumulation 4, 학습률 5e-5, seed 42, BF16 QLoRA를 사용한다. 체크포인트의 eval loss와 실제 생성 품질은 별도로 평가해야 한다.

서비스 후처리는 `model_server/slim_v1.py`에 포함된다. 과거 단일 일정 5필드 후처리와 실험용 손실·분석 코드는 포함하지 않는다.

