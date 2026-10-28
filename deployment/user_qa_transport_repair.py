"""Second repair: API/static quota isolation, unchanged answer semantics.

Preserves first-attempt 429 records. Same frozen questions are transport replay,
not claimed as a newly unseen holdout. The request quota stays at 60/minute.
"""
import argparse,hashlib,json,os,shutil,time
from deployment import user_qa_20261015 as campaign
from scripts import run_isolated_staging as staging
ROOT=campaign.ROOT;FIRST=campaign.OUT;OUT=FIRST/'round2';PACKAGE=OUT/'package'
campaign.OUT=OUT;campaign.PACKAGE=PACKAGE
read=campaign.read;save=campaign.save

def build():
    PACKAGE.mkdir(parents=True,exist_ok=False)
    pdfs=list(staging.DEFAULT_PDFS)+[campaign.PREVIOUS/f'sources/TSMC-{y}-Annual-Report.pdf' for y in (2023,2024)]
    identity=staging.identity_for(pdfs,'real-user-qa-transport-repair-20261015').to_dict();bid=identity['build_id']
    staging.phase_identity(PACKAGE,identity,bid)
    reg=read(FIRST/'package/document_registry.json');reg['build_id']=bid;staging.json_write(PACKAGE/'document_registry.json',reg)
    summary=read(FIRST/'package/document_layer_summary.json');summary['build_id']=bid
    for doc in summary['files']:
        src=FIRST/'package'/doc['snapshot']
        if staging.file_hash(src)!=doc['snapshot_sha256']:raise ValueError('snapshot mismatch')
        data=read(src);data['build_id']=bid;staging.json_write(PACKAGE/doc['snapshot'],data)
        doc['snapshot_sha256']=staging.file_hash(PACKAGE/doc['snapshot'])
    staging.json_write(PACKAGE/'document_layer_summary.json',summary)
    extraction=read(FIRST/'package/candidate_extraction.json');extraction['build_id']=bid
    staging.json_write(PACKAGE/'candidate_extraction.json',extraction)
    staging.phase_accepted(PACKAGE,bid)
    t=time.perf_counter();staging.phase_indexes(PACKAGE,bid)
    save('build-record.json',dict(build_id=bid,source_tree_sha256=identity['source_tree_sha256'],fresh_vector_seconds=time.perf_counter()-t,cached_financial_extraction='unchanged from first repair; original source identities preserved',cost=None,embedding_tokens=None))
    save('package-frozen-hashes.json',campaign.hashes(PACKAGE))
    for name in ['fresh-labels.json','fresh-freeze.json']:shutil.copyfile(FIRST/name,OUT/name)
    save('replay-protocol.json',dict(partition='TRANSPORT_REPLAY_NOT_NEW_UNSEEN_VALIDATION',initial_run='all 20 frozen requests HTTP429; originals preserved in parent directory',semantic_code_changes=False,repair='static assets exempt; read/query separate quotas; query quota unchanged 60/min',pacing_seconds=1.5,no_post_replay_production_changes=True))
    print(bid,flush=True)

def serve():
    runtime=OUT/'runtime';identity=read(PACKAGE/'build_identity.json');manifest=read(PACKAGE/'vector_index_manifest.json');c=read(OUT/'neo4j/private_credentials.json')
    if not runtime.exists():
        runtime.mkdir();shutil.copytree(PACKAGE/'vector_index',runtime/'data/chroma_db');shutil.copytree(ROOT/'frontend/dist',runtime/'frontend-dist')
    sources=read(FIRST/'runtime/source-registry.json');sources['build_id']=identity['build_id'];registry=runtime/'source-registry.json'
    if not registry.exists():registry.write_text(json.dumps(sources),encoding='utf-8')
    os.environ.update(NEO4J_URI=c['uri'],NEO4J_USERNAME=c['username'],NEO4J_PASSWORD=c['password'],NEO4J_DATABASE=c['database'],GRAPHRAG_BUILD_ID=identity['build_id'],GRAPH_VECTOR_COLLECTION=manifest['collection'],GRAPH_EMBEDDING_BACKEND='chroma_onnx',GRAPHRAG_SOURCE_REGISTRY=str(registry),GRAPHRAG_CANDIDATE_GRAPH=str(PACKAGE/'graph.json'),GRAPHRAG_FRONTEND_DIST=str(runtime/'frontend-dist'),API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0')
    os.chdir(runtime)
    import uvicorn
    uvicorn.run('strategic_graphrag.api.server:app',host='127.0.0.1',port=8006)

def evaluate():
    from deployment.live_acceptance import http
    def paced(label):
        time.sleep(1.5)
        req=dict(question=label['question'],max_paths=10,retrieval_mode='auto',synthesize=False,use_cache=False)
        r=http('http://127.0.0.1:8006','/query',req);r.pop('_bytes',None)
        print(label['item_id'],r.get('http_status'),flush=True)
        return dict(label=label,request=req,result=r)
    campaign.query=paced
    campaign.evaluate()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['build','import','serve','evaluate']);a=p.parse_args()
    dict(build=build,**{'import':campaign.import_graph},serve=serve,evaluate=evaluate)[a.phase]()
