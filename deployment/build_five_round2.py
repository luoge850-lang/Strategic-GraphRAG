"""Final repair build: hash-checked parse/extraction cache, fresh vector namespace.

Broad rule relation candidates remain archived for review. Only numeric table
facts and narrow source-disclosed company dependencies enter this candidate.
"""
import copy,json,time,shutil,hashlib
from deployment.acceptance_20261015 import ROOT,OUT,save,hashes
from deployment.build_five_corpus import event
from scripts import run_isolated_staging as staging
from strategic_graphrag.pipeline.company_disclosures import extract

def main():
    import pymupdf
    pdfs=list(staging.DEFAULT_PDFS)+[OUT/f'sources/TSMC-{y}-Annual-Report.pdf' for y in [2023,2024]]
    old=OUT/'five-package'; package=OUT/'five-package-round2'
    package.mkdir(exist_ok=False)
    identity=staging.identity_for(pdfs,'two-company-five-document-20261015-final-round2').to_dict()
    bid=identity['build_id'];staging.phase_identity(package,identity,bid)
    reg=json.loads((package/'document_registry.json').read_text())
    for r in reg['documents']:
        if r['filename'].startswith('TSMC'):r['company_id']='TSMC'
    staging.json_write(package/'document_registry.json',reg)
    results=[];quarantine=[]
    summary=json.loads((old/'document_layer_summary.json').read_text())
    for doc in summary['files']:
        src=old/doc['snapshot']
        if staging.file_hash(src)!=doc['snapshot_sha256']:raise ValueError('parse cache hash mismatch')
        data=json.loads(src.read_text(encoding='utf-8'));data['build_id']=bid
        # Page/table identities remain source identities, not invented new text.
        staging.json_write(package/doc['snapshot'],data)
        doc['snapshot_sha256']=staging.file_hash(package/doc['snapshot'])
    summary['build_id']=bid;staging.json_write(package/'document_layer_summary.json',summary)
    raw=json.loads((old/'candidate_extraction.json').read_text(encoding='utf-8'))
    for row in raw['files']:
        result=copy.deepcopy(row);name=row['filename']
        if row['document_sha256']!=identity['pdfs'][name]['sha256']:raise ValueError('extraction cache mismatch')
        retained=[]
        for triple in result['accepted_triples']:
            if triple['relation']=='REPORTS_METRIC':retained.append(triple)
            else:quarantine.append(dict(filename=name,triple=triple,reason='BROAD_RULE_RELATION_REQUIRES_INDEPENDENT_SEMANTIC_REVIEW',status='PENDING'))
        pdf=next(p for p in pdfs if p.name==name);company='TSMC' if name.startswith('TSMC') else 'NVIDIA_CORPORATION'
        start=time.perf_counter()
        with pymupdf.open(pdf) as d:
            for i,page in enumerate(d):
                for triple in extract(page.get_text(),company):
                    triple.update(source_page=i+1,page=i+1,source_filing=name,document_sha256=row['document_sha256'],filing_year=row['year'])
                    retained.append(triple)
        result['accepted_triples']=retained;results.append(result)
        event(package,file=name,stage='relation_source_patterns',status='COMPLETED',seconds=time.perf_counter()-start,
              cache='financial extraction and canonical page snapshots reused with hashes; not a new parse measurement',candidates=len(retained))
    staging.json_write(package/'candidate_extraction.json',dict(build_id=bid,files=results,cache_provenance=str(old),quarantined=quarantine))
    staging.phase_accepted(package,bid)
    accepted=json.loads((package/'accepted_facts.json').read_text(encoding='utf-8'))
    accepted['quarantined_rule_relations']=quarantine
    staging.json_write(package/'accepted_facts.json',accepted)
    started=time.perf_counter();staging.phase_indexes(package,bid)
    event(package,stage='fresh_vector_build',status='COMPLETED',seconds=time.perf_counter()-started,cost=None,embedding_token_count=None)
    event(package,stage='build',status='COMPLETED',build_id=bid,files=5,llm_calls=0,llm_tokens=0,cost=None)
    staging.json_write(package/'artifact_ledger.json',staging.package_artifacts(package,bid))
    save('final-source-fingerprint.json',dict(build_id=bid,source_tree_sha256=identity['source_tree_sha256'],
         runtime_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in list((ROOT/'strategic_graphrag').rglob('*.py'))+list((ROOT/'frontend/src').rglob('*.tsx'))+list((ROOT/'frontend/src').rglob('*.ts'))},
         protocol='No further production changes or validation-led tuning permitted. Two development repair rounds exhausted.'))
    print(bid,flush=True)
if __name__=='__main__':main()
