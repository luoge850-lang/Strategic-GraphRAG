"""Isolated implementation of the exposed real-user findings.

All scenarios here are development regressions. Old frozen/exposed families
are not relabelled as independent validation. Every run writes a new artifact.
"""
import argparse,hashlib,json,os,shutil,time
from pathlib import Path
from deployment.optimize_20261015 import ROOT,PREVIOUS,read,hashes,inventory
from scripts import run_isolated_staging as staging
OUT=ROOT/'experiments/request-contract-20261015';PACKAGE=OUT/'package'
BASE=ROOT/'experiments/user-qa-20261015/round2'
PORT=8007

def save(name,value):
    OUT.mkdir(exist_ok=True)
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def build():
    save('previous-package-hashes.json',hashes(BASE/'package'))
    PACKAGE.mkdir(exist_ok=False)
    pdfs=list(staging.DEFAULT_PDFS)+[PREVIOUS/f'sources/TSMC-{y}-Annual-Report.pdf' for y in (2023,2024)]
    identity=staging.identity_for(pdfs,'financial-request-contract-20261015').to_dict();bid=identity['build_id']
    staging.phase_identity(PACKAGE,identity,bid)
    reg=read(BASE/'package/document_registry.json');reg['build_id']=bid;staging.json_write(PACKAGE/'document_registry.json',reg)
    summary=read(BASE/'package/document_layer_summary.json');summary['build_id']=bid
    for doc in summary['files']:
        src=BASE/'package'/doc['snapshot']
        if staging.file_hash(src)!=doc['snapshot_sha256']:raise ValueError('snapshot hash changed')
        data=read(src);data['build_id']=bid;staging.json_write(PACKAGE/doc['snapshot'],data)
        doc['snapshot_sha256']=staging.file_hash(PACKAGE/doc['snapshot'])
    staging.json_write(PACKAGE/'document_layer_summary.json',summary)
    extraction=read(BASE/'package/candidate_extraction.json');extraction['build_id']=bid
    staging.json_write(PACKAGE/'candidate_extraction.json',extraction)
    staging.phase_accepted(PACKAGE,bid);t=time.perf_counter();staging.phase_indexes(PACKAGE,bid)
    save('build-record.json',dict(build_id=bid,source_tree_sha256=identity['source_tree_sha256'],fresh_vector_seconds=time.perf_counter()-t,cached_extraction=True,financial_values_changed=False,llm_calls=0,cost=None,embedding_tokens=None))
    save('package-frozen-hashes.json',hashes(PACKAGE));print(bid,flush=True)

def import_graph():
    from neo4j import GraphDatabase
    from deployment.import_candidate import prepare,write_empty
    c=read(OUT/'neo4j/private_credentials.json')
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            counts=s.execute_write(write_empty,prepare(read(PACKAGE/'graph.json')))
            s.run('CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]').consume();s.run('CALL db.awaitIndexes(30)').consume()
    save('import.json',counts);save('inventory.json',inventory(OUT/'neo4j'))

def serve(port=8007):
    runtime=OUT/'runtime';identity=read(PACKAGE/'build_identity.json');manifest=read(PACKAGE/'vector_index_manifest.json');c=read(OUT/'neo4j/private_credentials.json')
    if not runtime.exists():
        runtime.mkdir();shutil.copytree(PACKAGE/'vector_index',runtime/'data/chroma_db');shutil.copytree(ROOT/'frontend/dist',runtime/'frontend-dist')
    sources=read(BASE/'runtime/source-registry.json');sources['build_id']=identity['build_id'];registry=runtime/'source-registry.json'
    if not registry.exists():registry.write_text(json.dumps(sources),encoding='utf-8')
    os.environ.update(NEO4J_URI=c['uri'],NEO4J_USERNAME=c['username'],NEO4J_PASSWORD=c['password'],NEO4J_DATABASE=c['database'],GRAPHRAG_BUILD_ID=identity['build_id'],GRAPH_VECTOR_COLLECTION=manifest['collection'],GRAPH_EMBEDDING_BACKEND='chroma_onnx',GRAPHRAG_SOURCE_REGISTRY=str(registry),GRAPHRAG_CANDIDATE_GRAPH=str(PACKAGE/'graph.json'),GRAPHRAG_FRONTEND_DIST=str(runtime/'frontend-dist'),API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0')
    os.chdir(runtime)
    import uvicorn
    uvicorn.run('strategic_graphrag.api.server:app',host='127.0.0.1',port=port)

def labels():
    cases=[]
    def add(cid,question,kind,values=None,**extra):cases.append(dict(item_id=cid,question=question,kind=kind,values=values or {},partition='EXPOSED_DEVELOPMENT_NOT_VALIDATION',review='AI_PDF_NOT_HUMAN_GOLD',**extra))
    multi=dict(REVENUE=[130497,'USD millions'],NET_INCOME=[72880,'USD millions'])
    add('D01','What were NVIDIA revenue and net income in fiscal 2025?','multi',multi)
    add('D02','What was NVIDIA revenue in fiscal 2025 in USD billions?','scalar',dict(REVENUE=[130.497,'USD billions']))
    add('D03','英伟达2025财年收入是多少？','scalar',dict(REVENUE=[130497,'USD millions']))
    add('D04','How much cash did TSMC spend on investing in 2024?','clarify',reason='investing_cash_flow_or_capital_expenditure')
    add('D05','What was TSMC consolidated net cash used in investing activities in fiscal 2024 in its 2024 annual report?','scalar',dict(INVESTING_CASH_FLOW=[-864842769,'TWD thousands']))
    add('D06','Convert NVIDIA fiscal 2025 revenue from USD millions to USD billions.','scalar',dict(REVENUE=[130.497,'USD billions']))
    add('D07','How many USD millions did NVIDIA lose because of TSMC earthquakes in fiscal 2024?','abstain')
    add('D08','What were TSMC consolidated total assets in fiscal 2024?','scope',filing='2025-10-K.pdf',reason='selected_filing_conflicts_with_requested_company')
    add('D09','What were NVIDIA revenue and earnings per share in fiscal 2025?','partial',dict(REVENUE=[130497,'USD millions']),missing='EARNINGS_PER_SHARE')
    add('D10','What were NVIDIA revenue and employee satisfaction in fiscal 2025?','clarify',reason='additional_requested_item_not_resolved')
    add('D11','What were NVIDIA revenue and net income in fiscal 2025 in USD billions?','multi',dict(REVENUE=[130.497,'USD billions'],NET_INCOME=[72.88,'USD billions']))
    add('D12','英伟达2025财年营收和净利润分别是多少？','multi',multi)
    add('D13','Calculate NVIDIA net income divided by revenue in fiscal 2025.','ratio',dict(RATIO=[72880/130497,'ratio']))
    add('D14','What percentage of revenue were research and development expenses for NVIDIA fiscal 2022 in its 2023 10-K?','scalar',dict(R_AND_D_RATIO=[19.6,'percent']))
    add('D15','What was NVIDIA revenue in fiscal 2030?','abstain')
    add('D16','What were TSMC unconsolidated total assets in fiscal 2024?','abstain')
    add('D17','What was TSMC revenue in fiscal 2024 in USD millions?','abstain')
    add('D18','What was TSMC consolidated cash balance at December 31, 2023, in its 2024 annual report?','scalar',dict(CASH_AND_CASH_EQUIVALENTS=[1465427753,'TWD thousands']))
    add('D19','What were NVIDIA net income and revenue in fiscal 2025?','multi',multi)
    add('D20','Compare NVIDIA revenue and net income between fiscal 2024 and fiscal 2025.','clarify',reason='multiple_metrics_operation_not_supported')
    add('D21','What was NVIDIA revenue in fiscal 2025 in billions of USD?','scalar',dict(REVENUE=[130.497,'USD billions']))
    add('D22','What was NVIDIA revenue in fiscal 2025 in millions?','scalar',dict(REVENUE=[130497,'USD millions']))
    add('D23','台积电2024年火星工厂收入是多少？','abstain')
    add('D24','What were TSMC consolidated total assets and total liabilities in fiscal 2024?','multi',dict(TOTAL_ASSETS=[6691938000,'TWD thousands'],TOTAL_LIABILITIES=[2368362135,'TWD thousands']))
    save('development-labels.json',cases)
    save('development-protocol.json',dict(questions=len(cases),partition='All exposed real-user failures and semantic-family variants are development only.',source='Independent original PDF references from previous source packets; no production reference values.',new_independent_validation=False,no_global_accuracy_claim=True))

def request(label):
    from deployment.live_acceptance import http
    time.sleep(1.5)
    req=dict(question=label['question'],max_paths=10,retrieval_mode='auto',synthesize=False,use_cache=False,cross_filing=not bool(label.get('filing')))
    if label.get('filing'):req['source_filing']=label['filing']
    r=http(f'http://127.0.0.1:{PORT}','/query',req);r.pop('_bytes',None)
    print(label['item_id'],r.get('http_status'),(r.get('response') or {}).get('answer_status'),flush=True)
    return dict(label=label,request=req,result=r)

def score(row):
    l=row['label'];response=row['result'].get('response') or {};c=response.get('calculation') or {};meta=response.get('metadata') or {}
    request_ok=row['result'].get('http_status')==200 and response.get('execution_status')=='SUCCEEDED'
    values=l['values'];kind=l['kind'];numerical=None;support=None
    if kind in {'multi','partial'}:
        results={i['metric_id']:i for i in c.get('items') or []}
        numerical=all(m in results and results[m]['status']=='PASS' and abs(results[m]['value']-v[0])<=1e-6 and results[m]['unit']==v[1] for m,v in values.items())
        state=(c.get('status')=='PASS' and response.get('answer_status')=='ANSWERED' and c.get('answered_count')==len(values) and c.get('required_count')==len(values)) if kind=='multi' else (c.get('status')=='PARTIAL' and response.get('answer_status')=='PARTIALLY_ANSWERED' and c.get('value') is None and l['missing'] in c.get('missing_metrics',[]) and response.get('grounding_status')!='VERIFIED')
        observed={o.get('metric_id') for o in c.get('observations') or []}
        state=state and set(values)<=observed
    elif kind in {'scalar','ratio'}:
        v=next(iter(values.values()));numerical=isinstance(c.get('value'),(int,float)) and abs(c['value']-v[0])<=1e-6 and c.get('unit')==v[1]
        state=c.get('status')=='PASS' and response.get('answer_status')=='ANSWERED'
        if kind=='scalar':state=state and next(iter(values)) in {o.get('metric_id') for o in c.get('observations') or []}
    elif kind in {'clarify','scope'}:
        state=response.get('answer_status')=='ABSTAINED' and c.get('reason_code')==l['reason'] and c.get('value') is None
        if l['item_id']=='D04':state=state and len(meta.get('clarification_suggestions') or [])==2
        if kind=='scope':state=state and meta.get('scope_suggestion')=='all'
    else:state=response.get('answer_status')=='ABSTAINED' and c.get('value') is None and c.get('status')!='PASS'
    if numerical is not None:
        locations={(x['evidence_id'],x['source_filing'],x['page']) for x in response.get('citations') or []}
        support=bool(c.get('observations')) and all((o['claim_id'],o['source_filing'],o['page']) in locations for o in c.get('observations') or [])
    return dict(request_success=request_ok,status_contract_correct=request_ok and state,numeric_and_unit=numerical,source_identity_coverage=support,
                scenario_check=request_ok and state and numerical is not False and support is not False,
                fully_answered=response.get('answer_status')=='ANSWERED',partition='DEVELOPMENT')

def evaluate():
    from deployment.score_five_acceptance import summarize,metric
    from deployment.user_qa_20261015 import fact_score
    raw=[request(l) for l in read(PREVIOUS/'validation-labels.json')]
    save('baseline-52-raw.json',raw);save('baseline-52-scored.json',summarize(raw))
    for name,source in [('baseline-17',ROOT/'experiments/optimization-20261015/new-family-labels.json'),('baseline-20',ROOT/'experiments/user-qa-20261015/round2/fresh-labels.json')]:
        records=[]
        for label in read(source):
            r=request(label);r['score']=fact_score(r);records.append(r)
        save(name+'-raw-scored.json',records)
    records=[]
    for label in read(OUT/'development-labels.json'):
        r=request(label);r['score']=score(r);records.append(r)
    save('development-raw-scored.json',records)
    save('development-summary.json',dict(metrics={k:metric([r['score'][k] for r in records]) for k in ['request_success','status_contract_correct','numeric_and_unit','source_identity_coverage','scenario_check']},questions=len(records),review='AI_PDF_NOT_HUMAN_GOLD',new_validation=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','import','serve','labels','evaluate']);a=p.parse_args()
    dict(build=build,**{'import':import_graph},serve=serve,labels=labels,evaluate=evaluate)[a.phase]()
