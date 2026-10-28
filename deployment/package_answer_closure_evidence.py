"""Seal selected public artifacts, excluding private runtimes and server logs."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'experiments/answer-correctness-2026-10-03'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-tests', action='store_true')
    args = parser.parse_args()
    if args.record_tests:
        result = subprocess.run([sys.executable, '-m', 'pytest', '-q'], cwd=ROOT,
                                capture_output=True, text=True, encoding='utf-8')
        with (PUBLIC / 'engineering-tests.txt').open('x', encoding='utf-8') as output:
            output.write(result.stdout + result.stderr)
        with (PUBLIC / 'engineering-tests.json').open('x', encoding='utf-8') as output:
            json.dump({'command': 'python -m pytest -q', 'exit_code': result.returncode,
                       'scope': 'engineering regression, not answer accuracy'}, output, indent=2)
        if result.returncode:
            raise SystemExit(result.returncode)
    files = {}
    for path in sorted(PUBLIC.rglob('*')):
        if path.is_file() and path.suffix in {'.json', '.jsonl', '.md', '.jpg', '.txt'} and path.name != 'public_sha256.json':
            files[path.relative_to(PUBLIC).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    with (PUBLIC / 'public_sha256.json').open('x', encoding='utf-8') as output:
        json.dump(files, output, indent=2)
    print(json.dumps({'sealed_files': len(files)}))


if __name__ == '__main__':
    main()
