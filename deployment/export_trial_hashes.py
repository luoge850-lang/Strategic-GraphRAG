"""Export byte hashes for selected public records and their scoring inputs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    output = args.directory / 'public_hashes.json'
    files = [p for p in args.directory.rglob('*') if p.is_file() and p.suffix in {'.json', '.jsonl', '.svg', '.md'} and p != output]
    files += [ROOT / 'deployment' / name for name in ['live_acceptance.py', 'run_local_candidate_trial.py', 'summarize_candidate_trial.py']]
    files += [ROOT / 'experiments/financial-evidence-qa-2026-09-28/financial_qa_dev_source_review_20260924_v3.jsonl']
    payload = {'schema': 'public-candidate-record-hashes/v1', 'algorithm': 'SHA256 byte-exact',
               'files': {p.resolve().relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}}
    with output.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream, indent=2)
    print(json.dumps({'files': len(files), 'output': str(output)}))


if __name__ == '__main__':
    main()
