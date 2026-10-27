"""Read-only hash verification and exact recomputation of frozen QA scores.

Exit 0 = integrity and acceptance pass; 2 = intact but failed acceptance;
1 = missing/mutated artifact or recomputation mismatch.
"""
import hashlib,json
from deployment.acceptance_20261015 import ROOT,OUT,hashes
from deployment.score_five_acceptance import summarize
from strategic_graphrag.build_identity import source_fingerprint
def main():
    errors=[]
    manifest=json.loads((OUT/'delivery-manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['files'].items():
        path=ROOT/name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:errors.append('hash mismatch: '+name)
    identity=json.loads((OUT/'five-package-round2/build_identity.json').read_text())
    if source_fingerprint()!=identity['source_tree_sha256']:errors.append('current production source fingerprint differs')
    raw=json.loads((OUT/'validation-round2-raw.json').read_text(encoding='utf-8'))
    recomputed=summarize(raw)
    saved=json.loads((OUT/'validation-scored.json').read_text(encoding='utf-8'))
    if recomputed!=saved:errors.append('frozen automatic summary recomputation differs')
    for name,digest in json.loads((OUT/'final-scoring-freeze.json').read_text()).items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:errors.append('frozen scorer changed: '+name)
    final_hashes=json.loads((OUT/'package-round2-frozen-hashes.json').read_text())
    if hashes(OUT/'five-package-round2')!=final_hashes:errors.append('final candidate inventory/content differs')
    if hashes(ROOT/'reports/isolated_staging/build_ff4e5bcf11c71930')!=json.loads((OUT/'old-package-before.json').read_text()):errors.append('old immutable package differs')
    result=dict(integrity='FAIL' if errors else 'PASS',checked_files=len(manifest['files']),exact_frozen_summary_recomputed=not errors,
       acceptance='FAIL',classification='restricted reproducible research prototype',
       automatic_core='45/52',ai_source_audit_core='43/52',answerable_false_refusal='7/42',errors=errors)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 1 if errors else 2
if __name__=='__main__':raise SystemExit(main())
