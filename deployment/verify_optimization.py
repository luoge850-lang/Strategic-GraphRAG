"""Read-only verification: exit 2 means intact research delivery, acceptance FAIL."""
import hashlib,json,sys
from deployment.optimize_20261015 import OUT,PACKAGE,PREVIOUS,read,hashes
from strategic_graphrag.build_identity import source_fingerprint
from deployment.score_five_acceptance import summarize

def main():
    errors=[]
    for name,folder in [('previous-package-hashes.json',PREVIOUS/'five-package-round2'),('package-frozen-hashes.json',PACKAGE)]:
        if read(OUT/name)!=hashes(folder):errors.append(name+' mismatch')
    if read(PACKAGE/'build_identity.json')['source_tree_sha256']!=source_fingerprint():errors.append('production source mismatch')
    if summarize(read(OUT/'regression-raw.json'))!=read(OUT/'regression-scored.json'):errors.append('regression recomputation mismatch')
    if hashlib.sha256((OUT/'new-family-labels.json').read_bytes()).hexdigest()!=read(OUT/'new-family-freeze.json')['label_sha256']:errors.append('new labels changed')
    manifest=read(OUT/'public-manifest.json')
    for name,digest in manifest['files'].items():
        p=OUT/name
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:errors.append('artifact mismatch: '+name)
    fresh=read(OUT/'new-family-raw-scored.json');summary=read(OUT/'new-family-summary.json')
    numerator=sum(r['score']['core_semantic'] for r in fresh)
    if (numerator,len(fresh))!=(summary['metrics']['core_semantic']['numerator'],summary['metrics']['core_semantic']['denominator']):errors.append('new summary mismatch')
    result=dict(integrity='PASS' if not errors else 'FAIL',acceptance='FAIL',regression=[52,52],new_family=[numerator,len(fresh)],errors=errors)
    print(json.dumps(result,ensure_ascii=False));return 1 if errors else 2

if __name__=='__main__':sys.exit(main())
