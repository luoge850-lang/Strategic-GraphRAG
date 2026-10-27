"""Read-only integrity and exact replay scoring, not independent accuracy."""
import hashlib
import json
import sys
from deployment.request_contract_20261015 import ROOT, read, hashes, score
from deployment.score_five_acceptance import summarize, metric
from deployment.user_qa_20261015 import fact_score
from strategic_graphrag.build_identity import source_fingerprint


def main():
    out=ROOT/'experiments/request-contract-20261015/round2'
    errors=[]
    identity=read(out/'package/build_identity.json')
    if source_fingerprint()!=identity['source_tree_sha256']:
        errors.append('current source differs from candidate')
    if hashes(out/'package')!=read(out/'package-frozen-hashes.json'):
        errors.append('immutable package differs')
    first=out.parent
    if hashes(first/'package')!=read(out/'previous-package-hashes.json'):
        errors.append('previous immutable package differs')
    raw=read(out/'baseline-52-raw.json')
    if summarize(raw)!=read(out/'baseline-52-scored.json'):
        errors.append('52-question summary differs')
    partitions={'previous_52_development':{'questions':len(raw),'metrics':read(out/'baseline-52-scored.json')['metrics']}}
    for name in ['baseline-17','baseline-20']:
        records=read(out/(name+'-raw-scored.json'))
        if any(fact_score(r)!=r['score'] for r in records):
            errors.append(name+' per-question scoring differs')
        partitions[name]={'questions':len(records),'core_semantic':metric([r['score']['core_semantic'] for r in records])}
    records=read(out/'development-raw-scored.json')
    if any(score(r)!=r['score'] for r in records):
        errors.append('request-completeness scoring differs')
    summary=read(out/'development-summary.json')
    for k,v in summary['metrics'].items():
        if metric([r['score'][k] for r in records])!=v:
            errors.append('request-completeness summary differs: '+k)
    partitions['request_completeness_development']=summary
    result=dict(integrity='FAIL' if errors else 'PASS',build_id=identity['build_id'],
                errors=errors,partitions=partitions,
                independent_quality_acceptance='NOT_ESTABLISHED',
                graph_semantic_quality_acceptance='NOT_ESTABLISHED_FOR_THIS_BUILD',
                large_scale_acceptance='NOT_ESTABLISHED_FIVE_DOCUMENTS_ONLY')
    sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if errors else 2


if __name__=='__main__':
    raise SystemExit(main())
