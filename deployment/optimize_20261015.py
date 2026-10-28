"""New optimization campaign. Old acceptance outputs are immutable inputs.

Run inventory, build, import, serve, evaluate, report in this order. The known
52 questions are development regression data, never a new validation set.
"""
import argparse,copy,hashlib,json,os,shutil,time
from pathlib import Path
from deployment.acceptance_20261015 import ROOT, OUT as PREVIOUS, hashes
from scripts import run_isolated_staging as staging

OUT=ROOT/'experiments/optimization-20261015'
PACKAGE=OUT/'package'

def read(path): return json.loads(path.read_text(encoding='utf-8'))
def save(name,data):
    OUT.mkdir(exist_ok=True)
    with (OUT/name).open('x',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False,indent=2)

def inventory(state):
    from neo4j import GraphDatabase
    c=read(state/'private_credentials.json')
    queries={
      'node_labels':'MATCH(n) RETURN labels(n) AS labels,count(*) AS count',
      'relationship_types':'MATCH()-[r]->() RETURN type(r) AS type,count(*) AS count',
      'financial_scope':'MATCH(n:FinancialObservation) RETURN n.company_id AS company,collect(DISTINCT n.metric_id) AS metrics,collect(DISTINCT n.fiscal_period) AS periods,count(*) AS observations',
      'documents':'MATCH(n:Document) RETURN properties(n) AS document',
      'totals':'MATCH(n) WITH count(n) AS nodes MATCH()-[r]->() RETURN nodes,count(r) AS relationships'}
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            return {name:[r.data() for r in s.run(q)] for name,q in queries.items()}

def before():
    save('previous-package-hashes.json',hashes(PREVIOUS/'five-package-round2'))
    save('inventory-before.json',inventory(PREVIOUS/'neo4j-round2'))
    save('campaign-protocol.json',dict(target=.9,previous_validation='All 52 exposed questions promoted to development regression; no fresh holdout claim.',review='AI_PDF_NOT_HUMAN_GOLD',production_changes='one optimization round',release_pointer_changed=False,github_upload=False))

def build():
    from strategic_graphrag.pipeline.financial_candidate_guard import rejection_reason
    old=PREVIOUS/'five-package-round2';PACKAGE.mkdir(exist_ok=False)
    pdfs=list(staging.DEFAULT_PDFS)+[PREVIOUS/f'sources/TSMC-{y}-Annual-Report.pdf' for y in (2023,2024)]
    identity=staging.identity_for(pdfs,'optimization-five-source-20261015').to_dict();bid=identity['build_id']
    staging.phase_identity(PACKAGE,identity,bid)
    registry=read(PACKAGE/'document_registry.json')
    for d in registry['documents']:
        if d['filename'].startswith('TSMC'):d['company_id']='TSMC'
    staging.json_write(PACKAGE/'document_registry.json',registry)
    summary=read(old/'document_layer_summary.json');summary['build_id']=bid
    for doc in summary['files']:
        src=old/doc['snapshot']
        if staging.file_hash(src)!=doc['snapshot_sha256']:raise ValueError('snapshot changed')
        data=read(src);data['build_id']=bid
        staging.json_write(PACKAGE/doc['snapshot'],data)
        doc['snapshot_sha256']=staging.file_hash(PACKAGE/doc['snapshot'])
    staging.json_write(PACKAGE/'document_layer_summary.json',summary)
    extraction=read(old/'candidate_extraction.json');extraction['build_id']=bid;removed=[]
    for f in extraction['files']:
        if f['document_sha256']!=identity['pdfs'][f['filename']]['sha256']:raise ValueError('source changed')
        keep=[]
        for t in f['accepted_triples']:
            reason=rejection_reason(t)
            if reason:removed.append(dict(filename=f['filename'],candidate=t,reason=reason))
            else:keep.append(t)
        f['accepted_triples']=keep
    staging.json_write(PACKAGE/'candidate_extraction.json',extraction)
    save('quarantine.json',removed)
    staging.phase_accepted(PACKAGE,bid)
    start=time.perf_counter();staging.phase_indexes(PACKAGE,bid)
    save('build-record.json',dict(build_id=bid,source_tree_sha256=identity['source_tree_sha256'],cache='hash-checked prior parsed pages and extraction candidates; context filtering only; no new parse claim',fresh_vector_seconds=time.perf_counter()-start,quarantined_candidates=len(removed),cost=None,embedding_tokens=None,llm_calls=0))
    save('package-frozen-hashes.json',hashes(PACKAGE));print(bid,flush=True)

def import_graph():
    from neo4j import GraphDatabase
    from deployment.import_candidate import prepare,write_empty
    c=read(OUT/'neo4j/private_credentials.json');graph=read(PACKAGE/'graph.json')
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            counts=s.execute_write(write_empty,prepare(graph))
            s.run('CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]').consume()
            s.run('CALL db.awaitIndexes(30)').consume()
    save('import.json',counts);save('inventory-after.json',inventory(OUT/'neo4j'))

def serve():
    runtime=OUT/'runtime';identity=read(PACKAGE/'build_identity.json');manifest=read(PACKAGE/'vector_index_manifest.json');c=read(OUT/'neo4j/private_credentials.json')
    if not runtime.exists():
        runtime.mkdir();shutil.copytree(PACKAGE/'vector_index',runtime/'data/chroma_db')
    sources=read(PREVIOUS/'runtime-round2/source-registry.json');sources['build_id']=identity['build_id']
    registry=runtime/'source-registry.json'
    if not registry.exists():registry.write_text(json.dumps(sources),encoding='utf-8')
    os.environ.update(NEO4J_URI=c['uri'],NEO4J_USERNAME=c['username'],NEO4J_PASSWORD=c['password'],NEO4J_DATABASE=c['database'],GRAPHRAG_BUILD_ID=identity['build_id'],GRAPH_VECTOR_COLLECTION=manifest['collection'],GRAPH_EMBEDDING_BACKEND='chroma_onnx',GRAPHRAG_SOURCE_REGISTRY=str(registry),GRAPHRAG_CANDIDATE_GRAPH=str(PACKAGE/'graph.json'),API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0')
    os.chdir(runtime)
    import uvicorn
    uvicorn.run('strategic_graphrag.api.server:app',host='127.0.0.1',port=8004)

def evaluate():
    from deployment.live_acceptance import http
    from deployment.score_five_acceptance import summarize
    labels=read(PREVIOUS/'validation-labels.json');rows=[]
    save('regression-labels.json',labels)
    for label in labels:
        request=dict(question=label['question'],max_paths=10,retrieval_mode='graph',synthesize=False,use_cache=False)
        result=http('http://127.0.0.1:8004','/query',request);result.pop('_bytes',None)
        rows.append(dict(label=label,request=request,result=result));print(label['item_id'],(result.get('response') or {}).get('calculation',{}).get('status'),flush=True)
    save('regression-raw.json',rows);save('regression-scored.json',summarize(rows))
    save('integrity-after.json',dict(old_package_unchanged=read(OUT/'previous-package-hashes.json')==hashes(PREVIOUS/'five-package-round2'),new_package_unchanged=read(OUT/'package-frozen-hashes.json')==hashes(PACKAGE)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['before','build','import','serve','evaluate']);a=p.parse_args()
    {'before':before,'build':build,'import':import_graph,'serve':serve,'evaluate':evaluate}[a.phase]()
