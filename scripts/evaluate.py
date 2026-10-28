"""Recompute the exposed 25-question baseline or replay it against the current API."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
import requests

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'evaluation/current'
spec=importlib.util.spec_from_file_location('current_score',DATA/'score.py')
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url');parser.add_argument('--output',type=Path,default=ROOT/'.local/results/current.jsonl')
    args=parser.parse_args()
    manifest=json.loads((DATA/'manifest.json').read_text())
    baseline=[json.loads(x) for x in (DATA/'baseline.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[]
    for old in baseline:
        started=time.perf_counter()
        if args.url:
            response=requests.post(args.url.rstrip('/')+'/query',json=dict(question=old['label']['question'],retrieval_mode='auto',synthesize=False,use_cache=False,cross_filing=True,max_paths=10),timeout=90)
            response.raise_for_status(); response=response.json()
            build=response['metadata']['build_id']
        else:
            response=old['response'];build=manifest['reference_build_id']
        result=scorer.score(response,old['label'],build)
        rows.append(dict(id=old['id'],response=response,score=result,seconds=time.perf_counter()-started))
        print(old['id'],result['core'],flush=True)
    summary={k:dict(numerator=sum(r['score'][k] is True for r in rows),denominator=sum(r['score'][k] is not None for r in rows),not_applicable=sum(r['score'][k] is None for r in rows)) for k in ['core','numeric','identity','citation_identity','citation_reference','request_joint','false_refusal','severe_wrong_success']}
    summary.update(mode='LIVE_REPLAY' if args.url else 'BASELINE_RECOMPUTATION',review=manifest['review'],questions=len(rows),failed=[r['id'] for r in rows if r['score']['core'] is not True])
    if not args.url:
        assert summary['core']['numerator']==manifest['expected_core']
        assert summary['request_joint']['numerator']==manifest['expected_joint']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
