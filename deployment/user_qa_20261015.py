"""Real-user QA campaign: isolated build, source-derived facts, fresh holdout.

Old 52+17 exposed questions are development, never new validation. Costs and
unmeasured tokens stay null. No old result or release pointer is changed.
"""
import argparse,copy,json,hashlib,os,shutil,time
from pathlib import Path
from deployment.optimize_20261015 import ROOT,PREVIOUS,inventory,read,hashes
from scripts import run_isolated_staging as staging
OUT=ROOT/'experiments/user-qa-20261015';PACKAGE=OUT/'package';BASE=ROOT/'experiments/optimization-20261015'

def save(name,value):
    OUT.mkdir(exist_ok=True)
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def build():
    import pymupdf,pdfplumber
    from strategic_graphrag.pipeline.consolidated_ntd_table import extract
    from strategic_graphrag.pipeline.financial_cache import extraction_contract,reusable,refresh_nvidia
    save('previous-package-hashes.json',hashes(BASE/'package'))
    old=BASE/'package';PACKAGE.mkdir(exist_ok=False)
    pdfs=list(staging.DEFAULT_PDFS)+[PREVIOUS/f'sources/TSMC-{y}-Annual-Report.pdf' for y in (2023,2024)]
    identity=staging.identity_for(pdfs,'real-user-qa-five-source-20261015').to_dict();bid=identity['build_id']
    staging.phase_identity(PACKAGE,identity,bid)
    registry=read(PACKAGE/'document_registry.json')
    for d in registry['documents']:
        if d['filename'].startswith('TSMC'):d['company_id']='TSMC'
    staging.json_write(PACKAGE/'document_registry.json',registry)
    summary=read(old/'document_layer_summary.json');summary['build_id']=bid
    for doc in summary['files']:
        src=old/doc['snapshot']
        if staging.file_hash(src)!=doc['snapshot_sha256']:raise ValueError('parse cache changed')
        data=read(src);data['build_id']=bid;staging.json_write(PACKAGE/doc['snapshot'],data)
        doc['snapshot_sha256']=staging.file_hash(PACKAGE/doc['snapshot'])
    staging.json_write(PACKAGE/'document_layer_summary.json',summary)
    extraction=read(old/'candidate_extraction.json');extraction['build_id']=bid;records=[]
    for row in extraction['files']:
        name=row['filename']
        if row['document_sha256']!=identity['pdfs'][name]['sha256']:raise ValueError('PDF changed')
        pdf=next(p for p in pdfs if p.name==name);start=time.perf_counter()
        company='TSMC' if name.startswith('TSMC') else 'NVIDIA_CORPORATION'
        contract=extraction_contract(document_sha256=row['document_sha256'],company_id=company,filing_year=row['year'],filename=name)
        row['build_id']=bid
        if not name.startswith('TSMC'):
            if reusable(row,contract):
                records.append(dict(filename=name,status='COMPLETED',mode='CONTRACT_VERIFIED_FINANCIAL_CACHE'))
            else:
                records.append(refresh_nvidia(row,pdf,contract=contract))
            continue
        with pymupdf.open(pdf) as d:
            selected=[i for i,p in enumerate(d) if any(t in p.get_text() for t in ['Financial Status','Financial Performance','Cash Flow'])]
        retained=[t for t in row['accepted_triples'] if t['relation']!='REPORTS_METRIC']
        with pdfplumber.open(pdf) as d:
            for i in selected:
                for t in extract(d.pages[i],company_id='TSMC',filename=name,filing_year=row['year']):
                    t.update(source_page=i+1,page=i+1,document_sha256=row['document_sha256'])
                    retained.append(t)
        row['accepted_triples']=retained
        row['financial_extraction_contract']=contract
        records.append(dict(filename=name,status='COMPLETED',physical_pages_examined=[i+1 for i in selected],seconds=time.perf_counter()-start,accepted_candidates=len(retained),llm_calls=0,token_count=0))
    # Do not present inherited pipeline counters as this extraction's results.
    # Keep them for provenance, while current candidate counts come from rows.
    legacy_fields=('triples_extracted','triples_ingested','extraction_method_counts',
                   'table_quality','page_stats','coverage_ledger','relations','entities',
                   'pages_with_strict_triples','pending_table_candidates','llm','llm_cache')
    for row in extraction['files']:
        inherited={key:row.pop(key) for key in legacy_fields if key in row}
        if inherited:row.setdefault('inherited_pipeline_metadata',[]).append(inherited)
        row['triples_extracted']=len(row['accepted_triples'])
        row['triples_ingested']=0
        row['counter_scope']='Current candidate rows before acceptance guard; no graph writes in extraction phase'
    extraction['counts']=dict(files=len(extraction['files']),candidate_triples=sum(len(row['accepted_triples']) for row in extraction['files']),llm_calls=0)
    staging.json_write(PACKAGE/'candidate_extraction.json',extraction)
    staging.phase_accepted(PACKAGE,bid)
    start=time.perf_counter();staging.phase_indexes(PACKAGE,bid)
    save('build-record.json',dict(build_id=bid,source_tree_sha256=identity['source_tree_sha256'],extraction_records=records,fresh_vector_seconds=time.perf_counter()-start,source='Financial candidates require document/company/year/extractor/dependency contract; missing or stale contract forces PDF extraction. Page snapshots and unchanged prose retain historical lineage.',cost=None,embedding_tokens=None,llm_calls=0))
    save('package-frozen-hashes.json',hashes(PACKAGE))
    print(bid,flush=True)

def import_graph():
    from neo4j import GraphDatabase
    from deployment.import_candidate import prepare,write_empty
    c=read(OUT/'neo4j/private_credentials.json')
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            counts=s.execute_write(write_empty,prepare(read(PACKAGE/'graph.json')))
            s.run('CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]').consume();s.run('CALL db.awaitIndexes(30)').consume()
    save('import.json',counts);save('inventory.json',inventory(OUT/'neo4j'))

def serve():
    runtime=OUT/'runtime';identity=read(PACKAGE/'build_identity.json');manifest=read(PACKAGE/'vector_index_manifest.json');c=read(OUT/'neo4j/private_credentials.json')
    if not runtime.exists():
        runtime.mkdir();shutil.copytree(PACKAGE/'vector_index',runtime/'data/chroma_db');shutil.copytree(ROOT/'frontend/dist',runtime/'frontend-dist')
    sources=read(BASE/'runtime/source-registry.json');sources['build_id']=identity['build_id'];registry=runtime/'source-registry.json'
    if not registry.exists():registry.write_text(json.dumps(sources),encoding='utf-8')
    os.environ.update(NEO4J_URI=c['uri'],NEO4J_USERNAME=c['username'],NEO4J_PASSWORD=c['password'],NEO4J_DATABASE=c['database'],GRAPHRAG_BUILD_ID=identity['build_id'],GRAPH_VECTOR_COLLECTION=manifest['collection'],GRAPH_EMBEDDING_BACKEND='chroma_onnx',GRAPHRAG_SOURCE_REGISTRY=str(registry),GRAPHRAG_CANDIDATE_GRAPH=str(PACKAGE/'graph.json'),GRAPHRAG_FRONTEND_DIST=str(runtime/'frontend-dist'),API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0')
    os.chdir(runtime)
    import uvicorn
    uvicorn.run('strategic_graphrag.api.server:app',host='127.0.0.1',port=8005)

def freeze():
    rows=[]
    references=[('long-term investments','LONG_TERM_INVESTMENTS',149040373,129442117),('property, plant and equipment','PROPERTY_PLANT_EQUIPMENT',3234980070,3064474984),('right-of-use assets','RIGHT_OF_USE_ASSETS',40128391,40424830),('intangible assets','INTANGIBLE_ASSETS',26282520,22766744),('other assets','OTHER_ASSETS',153154526,81229630),('noncurrent liabilities','NONCURRENT_LIABILITIES',1103837171,1135525052),('capital stock','CAPITAL_STOCK',259327332,259320710),('capital surplus','CAPITAL_SURPLUS',73260765,69876381),('others equity','OTHER_EQUITY',38705047,-28314256),('equity attributable to shareholders of the parent','PARENT_EQUITY',4288545167,3458913627)]
    for alias,metric,a,b in references:
        for y,v in [(2024,a),(2023,b)]:
            rows.append(dict(item_id=f'F-{metric}-{y}',family_id='TSMC-'+metric,company_id='TSMC',metric_id=metric,year=y,value=v,unit='TWD thousands',business_scope='CONSOLIDATED',nature='BALANCE',source_filing='TSMC-2024-Annual-Report.pdf',page=63,question=f'What were TSMC consolidated {alias} at the end of fiscal {y} as disclosed in its 2024 annual report?',source_row=alias,review='AI_PDF_NOT_HUMAN_GOLD'))
    save('fresh-labels.json',rows)
    save('fresh-freeze.json',dict(sha256=hashlib.sha256((OUT/'fresh-labels.json').read_bytes()).hexdigest(),questions=20,families=10,source='Original TSMC 2024 PDF physical page 63 consolidated left half; independent visual reference, not system outputs',excluded_development_families='All prior 52+17 families and known UI defects; no paraphrase split across sets',scope='Narrow financial-position metric-family holdout, not global accuracy',production_source_sha256=read(PACKAGE/'build_identity.json')['source_tree_sha256'],no_post_validation_tuning=True))

def query(label):
    from deployment.live_acceptance import http
    req=dict(question=label['question'],max_paths=10,retrieval_mode='auto',synthesize=False,use_cache=False)
    r=http('http://127.0.0.1:8005','/query',req);r.pop('_bytes',None)
    return dict(label=label,request=req,result=r)

def fact_score(row):
    label=row['label'];resp=row['result'].get('response') or {};c=resp.get('calculation') or {};obs=c.get('observations') or []
    semantic=c.get('status')=='PASS' and c.get('operation')=='fact' and any(o.get('company_id')==label['company_id'] and o.get('metric_id')==label['metric_id'] and o.get('fact_period')==f"FY{label['year']}" and o.get('business_scope')=='CONSOLIDATED' and o.get('measurement_nature')==label['nature'] and o.get('source_filing')==label['source_filing'] for o in obs)
    numeric=isinstance(c.get('value'),(int,float)) and abs(c['value']-label['value'])<.5 and c.get('unit')==label['unit']
    page=any(cite.get('source_filing')==label['source_filing'] and cite.get('page')==label['page'] for cite in resp.get('citations',[])) if c.get('status')=='PASS' else None
    return dict(core_semantic=semantic,numeric_and_unit=numeric,joint=semantic and numeric and page is True,physical_page=page,false_refusal=c.get('status')!='PASS',wrong_pass=c.get('status')=='PASS' and not semantic,request_success=row['result'].get('http_status')==200 and resp.get('execution_status')=='SUCCEEDED')

def evaluate():
    from deployment.score_five_acceptance import summarize,metric
    labels=read(PREVIOUS/'validation-labels.json');rows=[query(l) for l in labels]
    save('development-52-raw.json',rows);save('development-52-scored.json',summarize(rows))
    exposed=read(BASE/'new-family-labels.json');rows=[]
    for l in exposed:
        r=query(l);r['score']=fact_score(r);rows.append(r)
    save('development-17-raw-scored.json',rows)
    freeze=read(OUT/'fresh-freeze.json')
    assert hashlib.sha256((OUT/'fresh-labels.json').read_bytes()).hexdigest()==freeze['sha256']
    rows=[]
    for l in read(OUT/'fresh-labels.json'):
        r=query(l);r['score']=fact_score(r);rows.append(r);print(l['item_id'],r['score'],flush=True)
    save('fresh-raw-scored.json',rows)
    metrics={k:metric([r['score'][k] for r in rows]) for k in rows[0]['score']}
    metrics['answer_precision']=metric([r['score']['joint'] for r in rows if (r['result'].get('response') or {}).get('answer_status')=='ANSWERED'])
    save('fresh-summary.json',dict(metrics=metrics,questions=20,families=10,review='AI_PDF_NOT_HUMAN_GOLD',human_gold=0,scope=freeze['scope']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','import','serve','freeze','evaluate']);a=p.parse_args()
    dict(build=build,import_graph=import_graph,serve=serve,freeze=freeze,evaluate=evaluate).get(a.phase if a.phase!='import' else 'import_graph')()
