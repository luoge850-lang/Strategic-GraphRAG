"""Reject runtime files and obvious credentials in explicitly staged delivery."""
import json
import re
import subprocess
from pathlib import Path


def main():
    paths = subprocess.check_output(['git', 'diff', '--cached', '--name-only'], text=True).splitlines()
    problems = []
    patterns = [r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY',
                r'gh[pousr]_[A-Za-z0-9]{30,}', r'sk-[A-Za-z0-9_-]{24,}',
                r'"(?:password|api_key|secret)"\s*:\s*"[^\"]+"']
    for name in paths:
        path = Path(name)
        if any(x in path.parts for x in ('configs', 'tmp', 'node_modules', '.venv', 'archive')) or path.suffix in ('.sqlite3', '.bin', '.log'):
            problems.append(name)
        elif path.suffix != '.jpg' and any(re.search(pattern, path.read_text(encoding='utf-8')) for pattern in patterns):
            problems.append(name)
    print(json.dumps({'reviewed_files': len(paths), 'flagged_files': problems,
                     'scope': 'heuristic check, not proof of absence of all secrets'}))
    return 1 if problems else 0


if __name__ == '__main__':
    raise SystemExit(main())
