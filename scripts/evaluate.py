"""Recompute the exposed 25-question baseline or replay it against the current API."""
import argparse
import importlib.util
import json
from pathlib import Path
import time
from datetime import datetime, timezone
from uuid import uuid4
import requests
import sys
from pydantic import ValidationError

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from strategic_graphrag.api.server import QueryResponse
DATA=ROOT/'evaluation/current'
spec=importlib.util.spec_from_file_location('current_score',DATA/'score.py')
scorer=importlib.util.module_from_spec(spec);spec.loader.exec_module(scorer)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url');parser.add_argument('--output',type=Path,default=ROOT/'.local/results/current.jsonl')
    parser.add_argument('--build-id')
    parser.add_argument('--build-manifest',type=Path,default=ROOT/'.local/build/build_identity.json')
    args=parser.parse_args()
    expected_build=(args.build_id or json.loads(args.build_manifest.read_text())['build_id']) if args.url else None
    if args.output==ROOT/'.local/results/current.jsonl':
        args.output=ROOT/'.local/results'/('run-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]+'.jsonl')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((DATA/'manifest.json').read_text())
    baseline=[json.loads(x) for x in (DATA/'baseline.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=[]
    with args.output.open('x',encoding='utf-8') as raw_stream:
        for old in baseline:
            started=time.perf_counter()
            http_status=None; error=None; category='EXECUTED'
            if args.url:
                build=expected_build
                try:
                    transport=requests.post(args.url.rstrip('/')+'/query',json=dict(question=old['label']['question'],retrieval_mode='auto',synthesize=False,use_cache=False,cross_filing=True,max_paths=10),timeout=90)
                    http_status=transport.status_code
                    try: response=transport.json()
                    except requests.exceptions.JSONDecodeError:
                        response={'raw_text':transport.text}; category='INVALID_RESPONSE'
                    if http_status>=400: category='HTTP_ERROR'
                    elif category=='EXECUTED':
                        try:
                            validated=QueryResponse.model_validate(response)
                            if validated.execution_status != "SUCCEEDED":
                                category="DEPENDENCY_ERROR" if validated.execution_status=="DEPENDENCY_ERROR" else "EXECUTION_ERROR"
                        except ValidationError as exc:
                            category='INVALID_RESPONSE';error=str(exc)
                except requests.Timeout as exc:
                    response={};error=str(exc);category='TIMEOUT'
                except requests.ConnectionError as exc:
                    response={};error=str(exc);category='TRANSPORT_ERROR'
            else:
                response=old['response'];build=manifest['reference_build_id']
            result=scorer.score(response if category=='EXECUTED' else {},old['label'],build)
            if category!='EXECUTED':
                result.update(core=False,request_joint=False,false_refusal=False,severe_wrong_success=False)
            elif response.get('answer_status')=='ABSTAINED': category='FALSE_REFUSAL' if result['false_refusal'] else 'REFUSAL'
            elif response.get('answer_status')=='NOT_REQUESTED':category='NOT_EXECUTED'
            elif not result['core']:category='WRONG_ANSWER'
            row=dict(id=old['id'],response=response,score=result,seconds=time.perf_counter()-started,expected_build_id=build,http_status=http_status,error=error,category=category)
            rows.append(row)
            raw_stream.write(json.dumps(row,ensure_ascii=False)+'\n');raw_stream.flush()
            print(old['id'],result['core'],flush=True)
    summary={k:dict(numerator=sum(r['score'][k] is True for r in rows),denominator=sum(r['score'][k] is not None for r in rows),not_applicable=sum(r['score'][k] is None for r in rows)) for k in ['core','numeric','identity','citation_identity','citation_reference','request_joint','false_refusal','severe_wrong_success']}
    summary.update(mode='LIVE_REPLAY' if args.url else 'BASELINE_RECOMPUTATION',review=manifest['review'],questions=len(rows),failed=[r['id'] for r in rows if r['score']['core'] is not True])
    if not args.url:
        assert summary['core']['numerator']==manifest['expected_core']
        assert summary['request_joint']['numerator']==manifest['expected_joint']
    summary['categories']={k:sum(r['category']==k for r in rows) for k in sorted({r['category'] for r in rows})}
    summary['expected_build_id']=expected_build
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
