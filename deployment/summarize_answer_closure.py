"""Recompute all closure metrics from immutable raw HTTP records, offline."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
from deployment.answer_closure_scoring import score

def fraction(values):
    known=[v for v in values if v is not None]
    return dict(numerator=sum(v is True for v in known),denominator=len(known),unjudged=len(values)-len(known))

def wilson(n,d):
    if not d:return None
    z=1.96;p=n/d;den=1+z*z/d
    center=(p+z*z/(2*d))/den
    half=z*math.sqrt(p*(1-p)/d+z*z/(4*d*d))/den
    return [max(0,center-half),min(1,center+half)]

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=dict(schema='answer-correctness-summary/v1',modes={},records=[],raw_hashes={})
    for path in sorted(a.run.glob('*.jsonl')):
        rows=[json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
        scored=[]
        for row in rows:
            judged=score(row.get('response') or {},row['reference'])
            scored.append(dict(item_id=row['item_id'],family_id=row['family_id'],mode=row['mode'],
                operation=row['reference']['operation'],build_id=row['build_id'],http_status=row['http_status'],
                elapsed_ms=row.get('elapsed_ms'),score=judged))
        facts=[x for x in scored if x['operation']!='SAFE_REFUSAL'];refusals=[x for x in scored if x['operation']=='SAFE_REFUSAL']
        metrics={k:fraction([x['score'][k] for x in facts]) for k in ['core_semantic','numeric_and_unit','citation_support','physical_page','metadata_complete','joint','false_refusal']}
        core=metrics['core_semantic']
        lat=sorted(x['elapsed_ms'] for x in scored if x.get('elapsed_ms') is not None)
        groups={}
        for op in sorted({x['operation'] for x in scored}):
            chosen=[x for x in scored if x['operation']==op]
            groups[op]=dict(questions=len(chosen),families=len({x['family_id'] for x in chosen}),
                core_semantic=fraction([x['score']['core_semantic'] for x in chosen]),
                wrong_pass=sum(x['score']['wrong_pass'] for x in chosen))
        result['modes'][path.stem]=dict(build_id=rows[0]['build_id'],questions=len(rows),families=len({x['family_id'] for x in rows}),
            answerable=len(facts),refusal_boundaries=len(refusals),label_tier='AI/PDF diagnostic; no human or second review',
            metrics=metrics,core_semantic_wilson95=wilson(core['numerator'],core['denominator']),
            interval_unit='one frozen operation+metric family per question; developmental quarter variants correlated, descriptive only',
            safe_refusal=fraction([x['score']['safe_refusal'] for x in refusals]),
            wrong_pass=sum(x['score']['wrong_pass'] for x in scored),http=fraction([x['http_status']==200 for x in scored]),
            successful_execution_subset=dict(questions=sum(x['http_status']==200 for x in scored),
                core_semantic=fraction([x['score']['core_semantic'] for x in facts if x['http_status']==200])),
            latency_ms=dict(p50=statistics.median(lat) if lat else None,p95=lat[math.ceil(.95*len(lat))-1] if lat else None),
            conditions='graph only, no generation/query cache; warm DB, restarted HTTP per partition/condition; OS cache uncontrolled',
            groups=groups,failed_items=[x['item_id'] for x in scored if x['score']['core_semantic'] is not True])
        result['records']+=scored
        result['raw_hashes'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    result['limits']=['No tuning on frozen validation responses.',
        'Operation+metric families overlap earlier development metrics; not unseen-domain or human Gold accuracy.',
        'Physical-page score measures reference linkage, not actual browser page rendering.',
        'Unjudged evidence remains unjudged; background pages cannot satisfy direct fact labels.']
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['modes'],ensure_ascii=False))

if __name__=='__main__':main()
