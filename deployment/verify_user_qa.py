"""Read-only source/artifact/recalculation checks. Exit 2: global acceptance unproven."""
import hashlib,json,sys
from deployment.user_qa_20261015 import OUT as FIRST,read,hashes
from strategic_graphrag.build_identity import source_fingerprint
from deployment.user_qa_20261015 import fact_score
from deployment.score_five_acceptance import summarize,metric

def main():
    out=FIRST/'round2';errors=[]
    identity=read(out/'package/build_identity.json')
    if source_fingerprint()!=identity['source_tree_sha256']:errors.append('production source changed')
    if hashes(out/'package')!=read(out/'package-frozen-hashes.json'):errors.append('final package changed')
    if hashes(FIRST/'package')!=read(FIRST/'package-frozen-hashes.json'):errors.append('first package changed')
    old=FIRST.parent/'optimization-20261015/package'
    if hashes(old)!=read(FIRST/'previous-package-hashes.json'):errors.append('previous candidate package changed')
    for name,digest in read(FIRST/'public-manifest.json')['files'].items():
        p=FIRST/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append('artifact mismatch '+name)
    dev=summarize(read(out/'development-52-raw.json'))
    if dev!=read(out/'development-52-scored.json'):errors.append('52 scoring changed')
    others=[]
    for name in ['development-17-raw-scored.json','fresh-raw-scored.json']:
        for row in read(out/name):
            if fact_score(row)!=row['score']:errors.append('fact scoring changed '+row['label']['item_id'])
            others.append(row)
    combined=dev['per_question']+others
    summary=read(out/'combined-summary.json')
    if metric([r['score']['core_semantic'] for r in combined])!=summary['core_semantic']:errors.append('summary changed')
    result=dict(integrity='PASS' if not errors else 'FAIL',build_id=identity['build_id'],core=[summary['core_semantic']['numerator'],len(combined)],answer_precision=[summary['answer_precision']['numerator'],summary['answer_precision']['denominator']],general_accuracy_acceptance='NOT_ESTABLISHED_DEVELOPMENT_AND_TRANSPORT_REPLAY',errors=errors)
    print(json.dumps(result,ensure_ascii=False));return 1 if errors else 2

if __name__=='__main__':sys.exit(main())
