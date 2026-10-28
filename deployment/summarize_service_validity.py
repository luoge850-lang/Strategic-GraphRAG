"""Recompute disjoint condition scores and legacy citation denominators, offline."""
import argparse,hashlib,json,math,statistics
from collections import Counter
from pathlib import Path
from complete_fact_scoring import FIELDS,score_fact,citation_page_score

def read_rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]
def fraction(values):
    judged=[x for x in values if x is not None]
    return dict(numerator=sum(x is True for x in judged),denominator=len(judged),unjudged=len(values)-len(judged))

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.root
    labels={x['item_id']:x for name in ['development_fact_labels.json','validation_fact_labels.json'] for x in json.loads((root/name).read_text())}
    result=dict(schema='service-validity-summary/v1',build_id='build_c3b1d950e8acce78',label_tier='AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW',modes={},records=[])
    for mode in ['natural_language','user_scope','reference_scope_diagnostic']:
        rows=read_rows(root/'run-three-modes-v2'/f'{mode}.jsonl')
        scored=[]
        for row in rows:
            fact=score_fact(row.get('response') or {},labels[row['item_id']]) if row['item_id'] in labels else None
            page=citation_page_score(row.get('response') or {},row['relevance_grades'])
            item=dict(item_id=row['item_id'],family_id=row['family_id'],partition=row['partition'],mode=mode,http_status=row['http_status'],fact=fact,pages=page)
            scored.append(item);result['records'].append(item)
        partitions={}
        for part in sorted({r['partition'] for r in rows}):
            selected=[s for s in scored if s['partition']==part]
            facts=[s['fact'] for s in selected if s['fact'] is not None]
            matching=[r for r in rows if r['partition']==part]
            lat=sorted(r['elapsed_ms'] for r in matching if r.get('elapsed_ms') is not None)
            partitions[part]=dict(questions=len(selected),families=len({s['family_id'] for s in selected}),
                scored_fact_questions=len(facts),scored_fact_families=len({s['family_id'] for s in selected if s['fact'] is not None}),
                fact_fields={k:fraction([f[k] for f in facts]) for k in (*FIELDS,'joint')},
                unscored_semantic_questions=len(selected)-len(facts),
                pages={k:fraction([s['pages'][k] for s in selected]) for k in ['direct_page_hit','background_page_hit','positive_page_hit_legacy']},
                http=fraction([r['http_status']==200 for r in matching]),http_statuses=dict(Counter(str(r['http_status']) for r in matching)),
                latency_p50_ms=statistics.median(lat) if lat else None,
                latency_p95_ms=lat[math.ceil(.95*len(lat))-1] if lat else None,
                build_id=result['build_id'],mode=mode,label_tier=result['label_tier'])
        result['modes'][mode]=partitions
    legacy=Path(__file__).resolve().parents[1]/'experiments/project-closure-2026-09-30'
    historic={}
    for dirname in ['trial-20261002-development','trial-20261002-repair1']:
        rows=json.loads((legacy/dirname/'development-answers.json').read_text())['records']
        judged=[dict(item_id=r['item_id'],family_id=r['family_id'],fact=score_fact(r['response'],labels[r['item_id']]) if r['item_id'] in labels else None,
            pages=citation_page_score(r['response'],r['relevance_grades'])) for r in rows]
        historic[dirname]=dict(build_id=rows[0]['build_id'],mode='reference_scope_diagnostic',questions=len(rows),families=len({r['family_id'] for r in rows}),
            numeric=fraction([x['fact']['number'] for x in judged if x['fact']]),joint=fraction([x['fact']['joint'] for x in judged if x['fact']]),
            pages={k:fraction([x['pages'][k] for x in judged]) for k in ['direct_page_hit','background_page_hit','positive_page_hit_legacy']},records=judged)
    result['historical_rescore']=historic
    result['limits']=['Page hit is not fragment support; unjudged pages are not negative labels.',
        'Full joint requires specific statement context; FINANCIAL_STATEMENTS is not a substitute.',
        'New adjacent facts share some existing metric families; not independent human Gold.',
        'No relation/conditional/comparison answer accuracy assigned without complete source labels/adjudication.',
        'Graph-only sequential runs; no direct speed comparison with embedding-inclusive historical retrieval.']
    inputs=[root/name for name in ['development_fact_labels.json','validation_fact_labels.json','protocol.json','frozen_inputs_sha256.json']]
    inputs += list((root/'run-three-modes-v2').glob('*.json*'))
    result['raw_hashes']={x.relative_to(root).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(inputs)}
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps({m:{part:dict(number=s['fact_fields']['number'],joint=s['fact_fields']['joint'],http=s['http'],direct=s['pages']['direct_page_hit']) for part,s in parts.items()} for m,parts in result['modes'].items()}))

if __name__=='__main__':main()
