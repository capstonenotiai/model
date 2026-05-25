"""학습 -> 평가 순차 실행 스크립트."""
import subprocess
import sys
import time
from pathlib import Path

ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "src"
LOG_DIR   = ROOT / "outputs" / "v1"
LOG_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

TRAIN_LOG = LOG_DIR / "train_run.log"
EVAL_LOG  = LOG_DIR / "eval_run.log"


def run(script, log_path):
    print(f"\n{'='*50}", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] start: {script}", flush=True)
    print(f"log: {log_path}", flush=True)
    print(f"python: {PYTHON}", flush=True)

    start = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run(
            [PYTHON, script],
            cwd=MODEL_DIR,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - start
    mins, secs = divmod(int(elapsed), 60)
    status = "done" if proc.returncode == 0 else f"FAILED (code={proc.returncode})"
    print(f"[{time.strftime('%H:%M:%S')}] {status} ({mins}m {secs}s)", flush=True)
    return proc.returncode == 0


if __name__ == "__main__":
    ok = run("train.py", TRAIN_LOG)
    if ok:
        run("eval.py", EVAL_LOG)
    else:
        print(f"\nTraining failed. Check log: {TRAIN_LOG}")
