"""Replay exact historical development requests; never count as holdout."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.live_acceptance import http

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8000');a=p.parse_args()
    old=ROOT/'experiments/service-validity-2026-10-02'
    rows=[json.loads(x) for x in (old/'run-three-modes-v2/natural_language.jsonl').read_text(encoding='utf-8').splitlines()]
    selected=[r for r in rows if r['item_id'] in ['V03','V11','V12']]
    boundary=[json.loads(x) for x in (old/'boundary-probes/requests.jsonl').read_text(encoding='utf-8').splitlines()]
    selected += [r for r in boundary if r.get('mode')=='natural_language']
    records=[]
    for r in selected:
        request=r['request'];new=http(a.base,'/query',request);new.pop('_bytes',None)
        records.append(dict(item_id=r.get('item_id'),partition='exposed_historical_development_only',
            historical_record=r,identical_request=request,new_result=new,
            new_build_id=(new.get('response') or {}).get('metadata',{}).get('build_id')))
    with a.output.open('x',encoding='utf-8') as f:json.dump(dict(records=records,
        conditions='same exact request fields and graph/no-generation/cache-off mode; different isolated build, warm/OS cache not equivalent speed evidence',
        quality_claim='development regression only; preserve all wrong historical PASS'),f,ensure_ascii=False,indent=2)
    print(json.dumps([(r['item_id'],(r['new_result'].get('response') or {}).get('calculation',{}).get('status')) for r in records]))

if __name__=='__main__':main()
