"""Offline standard-library score recomputation; output must be a new file."""
import json,math,hashlib,argparse
from pathlib import Path
from collections import defaultdict,Counter
from score import score
P=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def rate(n,d):
 if not d:return dict(n=n,d=d,value=None,wilson95=None)
 z=1.95996398454;p=n/d;den=1+z*z/d;mid=(p+z*z/(2*d))/den;half=z*math.sqrt(p*(1-p)/d+z*z/(4*d*d))/den
 return dict(n=n,d=d,value=p,wilson95=[max(0,mid-half),min(1,mid+half)])

def summarize(rows):
 a=[r for r in rows if r['answerable']];n=[r for r in a if r['numeric']];s=[r for r in n if r['score']['successful']]
 return dict(total=len(rows),source_answerable_core=rate(sum(r['core'] for r in a),len(a)),successful_numeric_precision=rate(sum(r['core'] for r in s),len(s)),false_refusal=rate(sum(r['response'].get('answer_status')=='ABSTAINED' for r in a),len(a)),nonanswer_not_refusal=sum(r['response'].get('answer_status')=='NOT_REQUESTED' for r in a),all_unfulfilled=rate(sum(not r['core'] for r in a),len(a)),numeric_and_unit=rate(sum(r['score'].get('numeric') is True for r in n),len(n)),complete_request_and_mechanical_metadata=rate(sum(r['joint'] for r in a),len(a)),successful_numeric_requested_source_support=rate(sum(r['score']['citation_reference'] and r['core'] for r in s),len(s)),severe_wrong_success=sum(r['score'].get('severe_wrong_success') is True for r in rows),boundary_controls=rate(sum(r['core'] for r in rows if not r['answerable']),sum(not r['answerable'] for r in rows)),http=dict(Counter(str(r['http_status']) for r in rows)),partial_relation_core=sum(r['label']['kind']=='relationship' and r['core'] for r in rows))

def run():
 identity=read(P/'implementation-freeze.json');assert sha(P/'score.py')==identity['scorer_sha256']
 reference=read(P/'reserved-freeze-v2.json');assert sha(P/reference['labels'])==reference['reference_sha256']
 cohorts={};ledger=[]
 for round_ in [1,2]:
  build=read(P/f'round{round_}/package/build_identity.json')['build_id']
  names=['development28','development37','exposed10','causal-controls']+(['reserved'] if round_==2 else [])
  for name in names:
   labels=read(P/(reference['labels'] if name=='reserved' else name+'-labels.json'));by={l['id']:l for l in labels};rows=[]
   for raw in (P/f'round{round_}/{name}-raw.jsonl').read_text(encoding='utf8').splitlines():
    r=json.loads(raw);l=by[r['id']];sc=score(r['response'],l,build);assert sc==r['score'],r['id']
    relation=l['kind']=='relationship'
    # Bounded relation core is separately AI/PDF reviewed. Partial remains
    # incomplete and never passes full-request joint completeness.
    core=(r['response'].get('answer_status')=='PARTIALLY_ANSWERED' and bool(r['response'].get('citations'))) if relation else sc['core']
    rows.append(dict(r,label=l,score=sc,answerable=bool(l.get('operands')) or relation,numeric=bool(l.get('operands')),core=bool(core),joint=False if relation else bool(sc['request_joint'])))
   strata={}
   for dim in ['family','company','file','kind']:
    g=defaultdict(list)
    for r in rows:
     l=r['label'];ops=l.get('operands',[]);key=l.get(dim) if dim in ['family','kind'] else '+'.join(sorted({o['company_id' if dim=='company' else 'disclosure_version'] for o in ops})) or 'RELATION_OR_CONTROL';g[key].append(r)
    strata[dim]={k:summarize(v) for k,v in g.items()}
   cohorts[f'round{round_}/{name}']=dict(metrics=summarize(rows),strata=strata,unjudged_core=0,review='Mechanical plus separately recorded AI/PDF relation review; human0')
   if round_==2:
    ledger.extend(dict(partition=name,id=r['id'],question=r['label']['question'],family=r['label']['family'],expected=r['label'],state=r['response'].get('answer_status'),score=r['score'],ai_core=r['core'],query_plan=r['response'].get('metadata',{}).get('query_plan'),calculation=r['response'].get('calculation'),citations=r['response'].get('citations'),error=r['error'],raw=f'round2/{name}-raw.jsonl') for r in rows)
 multi=read(P/'round2/multi-raw.json');labels=read(P/'multi-labels-frozen.json')
 for r,l in zip(multi,labels):assert score(r['response'],l,identity['build_id'])==r['score']
 return dict(identity=identity,cohorts=cohorts,multi=dict(n=len(multi),core=sum(r['score']['core'] for r in multi),joint=sum(r['score']['request_joint'] for r in multi),wrong_success=sum(r['score']['severe_wrong_success'] for r in multi),partition='Exposed development, not independent validation'),reserved_limits=reference['limitation'],reserved_family_count=reference['family_count'],model=dict(actual_generation_calls=0,cache=False,synthesis=False,actual_model=None,tokens=None,cost=None),human_reviewed=0,interval_limits='Wilson intervals are descriptive for a small selected correlated sample; no claim of population/unseen-family accuracy.'),ledger

if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--ledger');args=a.parse_args();s,l=run()
 with Path(args.output).open('x',encoding='utf8') as f:json.dump(s,f,ensure_ascii=False,indent=2)
 if args.ledger:
  with Path(args.ledger).open('x',encoding='utf8') as f:json.dump(l,f,ensure_ascii=False,indent=2)
 print(json.dumps({k:v['metrics'] for k,v in s['cohorts'].items() if k.startswith('round2')},ensure_ascii=False,indent=2))
