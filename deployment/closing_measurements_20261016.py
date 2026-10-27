"""Same-candidate measurements. No inferred human judgments or mock models."""
import collections,json,math,shutil,statistics,time
from concurrent.futures import ThreadPoolExecutor
from deployment.finalization_20261016 import ROOT,OUT,save,http

def performance():
    import psutil
    identity=json.loads((OUT/'package/build_identity.json').read_text())
    samples=[];groups=[]
    for concurrency in [1,2,4]:
        for category,question in [('retrieval_calculation','What were NVIDIA revenue in fiscal 2024 and net income in fiscal 2025?'),('clarification','What was NVIDIA revenue excluding China in fiscal 2025?')]:
            before=time.perf_counter()
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                results=list(pool.map(lambda _:http(question,8009),range(8)))
            after=time.perf_counter();lat=sorted(r['seconds'] for r in results)
            groups.append(dict(concurrency=concurrency,category=category,n=8,success=sum(r.get('http_status')==200 for r in results),timeouts=sum('timed out' in r.get('error','') for r in results),errors=sum(r.get('http_status')!=200 for r in results),p50_seconds=statistics.median(lat),p95_seconds=lat[math.ceil(.95*len(lat))-1],wall_seconds=after-before,raw=results))
            samples.append(dict(timestamp=time.time(),system_available_memory=psutil.virtual_memory().available,
                processes=[dict(pid=p.pid,name=p.info['name'],rss=p.info['memory_info'].rss) for p in psutil.process_iter(['name','memory_info']) if p.info['name'] and p.info['name'].lower() in {'python.exe','java.exe'}]))
            # Keep transport rate limiting separate from measured concurrency.
            time.sleep(18)
    save('performance.json',dict(build_id=identity['build_id'],groups=groups,resources=samples,
        cold_request='First development query after new server process; not OS page-cache cold',warm_cache='Application response cache disabled',model_calls=0,tokens=None,cost=None))

def retrieval():
    import chromadb
    from chromadb.utils import embedding_functions
    from scripts.run_financial_retrieval_matrix import BM25,_ranked,_aggregate_pages,_select_mode_hits,_page_key
    package=OUT/'package';target=OUT/'retrieval-runtime'
    shutil.copytree(package/'vector_index',target)
    manifest=json.loads((package/'vector_index_manifest.json').read_text())
    collection=chromadb.PersistentClient(path=str(target)).get_collection(manifest['collection'],embedding_function=embedding_functions.DefaultEmbeddingFunction())
    raw=collection.get(include=['documents','metadatas'])
    docs=[dict(id=ident,document=raw['documents'][i],metadata=raw['metadatas'][i]) for i,ident in enumerate(raw['ids'])]
    bm=BM25(docs);by_page={};edges=collections.defaultdict(list)
    for item in docs:by_page.setdefault(_page_key(item['metadata'],item['id']),item)
    for edge in json.loads((package/'graph.json').read_text())['edges']:edges[f"{edge['source_filing']}#{edge['page']}"].append(edge)
    modes={'keyword_bm25':'关键词检索（BM25）','dense_semantic':'语义向量检索','hybrid_rrf':'关键词与语义向量融合检索','graph_expansion':'融合检索加图扩展','graph_time':'融合检索加图扩展和时间条件','time_only':'融合检索加时间条件'}
    # Use exactly the existing selector mode identifiers, recorded with full names.
    from scripts.run_financial_retrieval_matrix import MODES,METHOD_NAMES_ZH
    modes=METHOD_NAMES_ZH
    print('retrieval identifiers',MODES,flush=True)
    labels=json.loads((OUT/'development-slot-labels.json').read_text())
    labels=[l for l in labels if len(l['slots'])==1 and len(l['slots'][0]['operands'])==1]
    records=[]
    for label in labels:
        bm_hits=_ranked(_aggregate_pages(bm.search(label['question'],50)))[:50]
        result=collection.query(query_texts=[label['question']],n_results=50,include=['documents','metadatas','distances'])
        chunks=[dict(id=ident,document=result['documents'][0][i],metadata=result['metadatas'][0][i],score=1/(1+max(0,result['distances'][0][i]))) for i,ident in enumerate(result['ids'][0])]
        dense=_ranked(_aggregate_pages(chunks))[:50]
        for mode in MODES:
            start=time.perf_counter();hits,plan,status,counts=_select_mode_hits(mode,label['question'],bm_hits,dense,edges,by_page)
            operand=label['slots'][0]['operands'][0];ranks=[]
            for rank,hit in enumerate(hits,1):
                meta=hit.get('metadata') or {};text=(hit.get('document') or '').lower()
                if any(meta.get('source_filing')==s['file'] and meta.get('page')==s['page'] and any(row.lower() in text for row in s['rows']) for s in operand['sources']) and str(operand['value']) in text.replace(',',''):ranks.append(rank)
            records.append(dict(item_id=label['item_id'],family=label['family'],method_identifier=mode,method_name=modes.get(mode,mode),status=status,selector_latency_ms=(time.perf_counter()-start)*1000,hits=hits,positive_covered=bool(ranks),reciprocal_rank=1/min(ranks) if ranks else 0,unjudged_hits=max(0,len(hits)-len(ranks))))
    save('retrieval.json',dict(build_id=manifest['build_id'],records=records,questions=len(labels),final_evidence_budget=10,initial_arm_budget=50,full_recall=None,ndcg=None,generation='NOT_MEASURED',judgments='Only frozen source-positive rows; others unjudged. Ranking latency excludes shared keyword/embedding computations, so not full end-to-end latency.',independent_validation=False))

def graph_audit():
    from neo4j import GraphDatabase
    from deployment.optimize_20261015 import read
    c=read(OUT/'neo4j/private_credentials.json');identity=read(OUT/'package/build_identity.json')
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as driver:
        with driver.session() as session:
            observations=[r['o'] for r in session.run('MATCH(o:FinancialObservation) RETURN properties(o) AS o')]
            relations=[r.data() for r in session.run('MATCH(a)-[r]->(b) WHERE type(r) <> "REPORTS_METRIC" AND type(r) <> "HAS_OBSERVATION" RETURN a.id AS source,type(r) AS relation,b.id AS target,properties(r) AS properties')]
    labels=read(OUT/'development-slot-labels.json');expected={}
    for label in labels:
        for slot in label['slots']:
            for op in slot['operands']:expected[(op['company_id'],op['metric_id'],op['fact_period'])]=op
    rows=[]
    for key,want in expected.items():
        candidates=[o for o in observations if (o.get('company_id'),o.get('metric_id'),o.get('fiscal_period'))==key]
        judged=[]
        for obs in candidates:
            match=any(obs.get('source_filing')==s['file'] and obs.get('page')==s['page'] and obs.get('row_label','').lower() in [v.lower() for v in s['rows']] for s in want['sources'])
            judged.append(dict(observation=obs,source_row_supported=match and obs['value']==want['value'] and obs['unit']==want['unit'] and obs['business_scope']==want['business_scope']))
        rows.append(dict(expected=want,candidates=judged,extraction_expected_fact_found=any(x['source_row_supported'] for x in judged)))
    save('graph-source-audit.json',dict(build_id=identity['build_id'],source_independent_expected_list=rows,all_financial_observations=len(observations),other_relationships=relations,
        scope='Convenience slice of independently labelled source rows from two companies; not a representative stratified whole-graph audit. Amounts and balance/flow covered; percentage and broader relationship semantics NOT_ESTABLISHED.',review='AI_PDF_NOT_HUMAN_GOLD'))

if __name__=='__main__':
    import sys
    {'performance':performance,'retrieval':retrieval,'graph':graph_audit}[sys.argv[1]]()
