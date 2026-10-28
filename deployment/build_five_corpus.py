"""Isolated five-document build with per-file durable checkpoints and retry."""
import argparse,json,time,os,hashlib,traceback
from pathlib import Path
from deployment.acceptance_20261015 import ROOT, OUT
from scripts import run_isolated_staging as staging
from strategic_graphrag.document_layer import DocumentLayerReader,ValidatedSnapshotReader
from strategic_graphrag.pipeline.pipeline import KnowledgeGraphPipeline,PipelineConfig
from strategic_graphrag.pipeline.consolidated_ntd_table import extract

def event(out,**row):
    import psutil
    row.update(timestamp=time.time(),rss_bytes=psutil.Process().memory_info().rss)
    with (out/'journal.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
def build(single=False):
    import pdfplumber
    pdfs=list(staging.DEFAULT_PDFS)+[OUT/f'sources/TSMC-{y}-Annual-Report.pdf' for y in [2023,2024]]
    if single:pdfs=[pdfs[-1]]
    identity=staging.identity_for(pdfs,'two-company-five-document-20261015-round1'+('-single' if single else '')).to_dict()
    package=OUT/('single-package' if single else 'five-package')
    package.mkdir(parents=True,exist_ok=True)
    if (package/'build_identity.json').exists():
        frozen=json.loads((package/'build_identity.json').read_text())
        if identity!=frozen:raise RuntimeError('source changed after build start; use new package')
    else:
        staging.phase_identity(package,identity,identity['build_id'])
        reg=json.loads((package/'document_registry.json').read_text())
        for r in reg['documents']:
            if r['filename'].startswith('TSMC-'):r['company_id']='TSMC'
        staging.json_write(package/'document_registry.json',reg)
    bid=identity['build_id']; results=[]; docs=[]; failed=[]
    order=list(enumerate(pdfs))
    for idx,pdf in order:
        checkpoint=package/'checkpoints'/f'{pdf.name}.json'
        if checkpoint.exists():
            item=json.loads(checkpoint.read_text(encoding='utf-8'));results.append(item['result']);docs.append(item['document'])
            event(package,file=pdf.name,status='SKIPPED_CHECKPOINT_HASH_VERIFIED',pdf_sha256=staging.file_hash(pdf));continue
        if idx==2 and not (package/'injected-third-failure.json').exists():
            staging.json_write(package/'injected-third-failure.json',dict(file=pdf.name,kind='INJECTED_BEFORE_PARSE',reason='recovery test, not parser failure'))
            event(package,file=pdf.name,status='FAILED_INJECTED',attempt=1);failed.append(pdf);continue
        process_file(package,pdf,bid,results,docs)
    for pdf in failed:
        event(package,file=pdf.name,status='RETRY',attempt=2)
        process_file(package,pdf,bid,results,docs)
    staging.json_write(package/'document_layer_summary.json',dict(build_id=bid,files=docs,total_pages=sum(d['total_pages'] for d in docs),page_conservation=all(d['coverage']['conservation_holds'] for d in docs)))
    staging.normalize_no_llm_candidate_metadata(results)
    staging.json_write(package/'candidate_extraction.json',dict(build_id=bid,files=results,mode='no_llm; NVIDIA canonical pipeline; TSMC conservative consolidated NTD summary parser'))
    start=time.perf_counter();staging.phase_accepted(package,bid);event(package,stage='accepted',status='COMPLETED',seconds=time.perf_counter()-start)
    start=time.perf_counter();staging.phase_indexes(package,bid);event(package,stage='vectors',status='COMPLETED',seconds=time.perf_counter()-start)
    staging.json_write(package/'artifact_ledger.json',staging.package_artifacts(package,bid))
    event(package,stage='build',status='COMPLETED',build_id=bid,files=len(pdfs),llm_calls=0,llm_tokens=0,embedding_tokens=None,cost=None)
    print(bid,flush=True)

def process_file(package,pdf,bid,results,docs):
    import pdfplumber
    start=time.perf_counter();event(package,file=pdf.name,stage='parse',status='STARTED')
    document=DocumentLayerReader().read(pdf,build_id=bid)
    snapshot=package/'documents'/f'{pdf.name}.json';document.write_json(snapshot)
    record=dict(filename=pdf.name,pdf_sha256=document.pdf_sha256,snapshot=str(snapshot.relative_to(package)),snapshot_sha256=staging.file_hash(snapshot),total_pages=document.total_pages,coverage=document.coverage(),status='PASS' if document.coverage()['failed']==0 else 'FAIL')
    elapsed=time.perf_counter()-start;event(package,file=pdf.name,stage='parse',status=record['status'],seconds=elapsed,pages=document.total_pages,pages_per_second=document.total_pages/elapsed)
    start=time.perf_counter()
    if pdf.name.startswith('TSMC-'):
        triples=[]
        with pdfplumber.open(pdf) as d:
            for i,page in enumerate(d.pages):
                text=page.extract_text() or ''
                if 'Financial Performance' not in text or 'Unit: NT$ thousands' not in text:continue
                rows=extract(page,company_id='TSMC',filename=pdf.name,filing_year=int(pdf.name[5:9]))
                for r in rows:r.update(source_page=i+1,page=i+1)
                triples.extend(rows)
        result=dict(filename=pdf.name,document_sha256=document.pdf_sha256,year=int(pdf.name[5:9]),status='completed',accepted_triples=triples,llm=dict(calls=0,network_calls=0))
    else:
        config=PipelineConfig(pdf_dir=str(pdf.parent),allow_multiple_pdfs=True,dry_run=True,capture_triples=True,
            company_id='NVIDIA_CORPORATION',document_registry_path=str(package/'document_registry.json'),
            pending_table_queue_path=str(package/'pending_table_candidates.jsonl'),build_id=bid)
        pipeline=KnowledgeGraphPipeline(config)
        pipeline.document_reader=ValidatedSnapshotReader({pdf.name:dict(path=snapshot,snapshot_sha256=record['snapshot_sha256'])},pipeline.document_reader)
        result=pipeline.process_batch(pdf_paths=[str(pdf)])[0]
        if result.get('status')!='completed':raise RuntimeError(str(result))
    event(package,file=pdf.name,stage='extract',status='COMPLETED',seconds=time.perf_counter()-start,candidates=len(result.get('accepted_triples') or []),llm_calls=0,token_count=0,cost=None)
    staging.json_write(package/'checkpoints'/f'{pdf.name}.json',dict(document=record,result=result))
    results.append(result);docs.append(record)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--single',action='store_true');a=p.parse_args();build(a.single)
