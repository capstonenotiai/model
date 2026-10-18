"""Check release files without reading model weights or private datasets."""
import ast
import fnmatch
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ('*.safetensors', '*.bin', '*.pt', '*.pth', 'adapter_*', '*.jsonl',
             '.env', '.env.*', 'HANDOFF*', 'HISTORY*', 'TASK*', 'RESULT*', 'AGENTS.md', 'CLAUDE.md')
PATTERNS = {
    'personal_path': re.compile(r'[A-Za-z]:[\\/]+Users[\\/]', re.I),
    'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'private_ip': re.compile(r'\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b'),
    'tailnet': re.compile(r'\b[\w.-]+\.ts\.net\b', re.I),
    'secret': re.compile(r'\b(?:hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[A-Z0-9]{16})\b|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    'assigned_secret': re.compile(r'''(?i)\b(?:api_key|access_token|password|secret_key|model_api_token)\s*[:=]\s*["'][A-Za-z0-9_+/=-]{16,}["']'''),
}

def main():
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    untracked = subprocess.check_output(['git', 'ls-files', '--others', '--exclude-standard', '-z'], cwd=ROOT).decode().split('\0')
    files = sorted({p for p in tracked + untracked if p and (ROOT / p).is_file()})
    issues = []
    for name in files:
        path = ROOT / name
        if any(fnmatch.fnmatch(path.name, pattern) for pattern in FORBIDDEN):
            issues.append({'file': name, 'kind': 'forbidden_file'})
            continue
        if name.startswith(('data/', 'outputs/', 'checkpoints/')):
            issues.append({'file': name, 'kind': 'private_artifact_directory'})
        value = path.read_text(encoding='utf-8')
        for kind, pattern in PATTERNS.items():
            for number, line in enumerate(value.splitlines(), 1):
                if pattern.search(line):
                    issues.append({'file': name, 'line': number, 'kind': kind})
        if path.suffix == '.py':
            ast.parse(value, filename=name)
    print(json.dumps({'passed': not issues, 'files_checked': len(files), 'findings': issues}, indent=2))
    return bool(issues)

if __name__ == '__main__':
    sys.exit(main())
