"""Offline byte-exact checks and summary recomputation; no database required."""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'experiments/answer-correctness-2026-10-03'


def main():
    errors = []
    checked = 0
    for manifest, base in [('frozen_inputs_sha256.json', ARTIFACTS),
                           ('frozen_scoring_sha256.json', ROOT), ('public_sha256.json', ARTIFACTS)]:
        for name, expected in json.loads((ARTIFACTS / manifest).read_text(encoding='utf-8')).items():
            target = base / name
            checked += 1
            if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                errors.append('hash mismatch: ' + str(target.relative_to(ROOT)))
    summaries = ['development-summary-v1.json', 'development-summary-v2.json', 'validation-summary-v1.json']
    runs = ['development-run-v1', 'development-run-v2', 'validation-run-v1']
    with tempfile.TemporaryDirectory(prefix='graphrag-closure-verify-') as temporary:
        for run, summary in zip(runs, summaries):
            output = Path(temporary) / summary
            completed = subprocess.run([sys.executable, '-m', 'deployment.summarize_answer_closure',
                '--run', str(ARTIFACTS / run), '--output', str(output)], cwd=ROOT,
                capture_output=True, text=True, encoding='utf-8')
            if completed.returncode or not output.exists():
                errors.append('recompute failed: ' + summary)
            elif json.loads(output.read_text(encoding='utf-8')) != json.loads((ARTIFACTS / summary).read_text(encoding='utf-8')):
                errors.append('summary differs: ' + summary)
    for document in [ARTIFACTS / 'README.md', ROOT / 'README.md']:
        for link in re.findall(r'\]\(([^)]+)\)', document.read_text(encoding='utf-8')):
            if '://' in link or link.startswith('#'):
                continue
            if not (document.parent / link.split('#', 1)[0]).exists():
                errors.append('missing link: ' + link)
    main_score = json.loads((ARTIFACTS / 'validation-summary-v1.json').read_text(encoding='utf-8'))['modes']['question_only']
    core = main_score['metrics']['core_semantic']
    reliable = core['denominator'] > 0 and core['numerator'] / core['denominator'] >= .9 and main_score['wrong_pass'] == 0
    print(json.dumps({'integrity': 'PASS' if not errors else 'FAIL', 'hash_entries': checked,
        'exact_summaries': len(summaries), 'errors': errors,
        'reliable_answer_quality_gate': 'PASS' if reliable else 'FAIL',
        'classification': 'restricted research prototype'}, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
