"""Run an owned isolated HTTP server; never change a release pointer."""
import json,os,shutil,sys,argparse
from pathlib import Path
from deployment.acceptance_20261015 import ROOT,OUT,hashes
def main():
    p=argparse.ArgumentParser();p.add_argument('--round',default='round1');p.add_argument('--port',type=int,default=8002);a=p.parse_args()
    package=OUT/('five-package' if a.round=='round1' else 'five-package-round2')
    runtime=OUT/f'runtime-{a.round}'
    identity=json.loads((package/'build_identity.json').read_text())
    manifest=json.loads((package/'vector_index_manifest.json').read_text())
    c=json.loads((OUT/f'neo4j-{a.round}/private_credentials.json').read_text())
    if not runtime.exists():
        runtime.mkdir()
        shutil.copytree(package/'vector_index',runtime/'data/chroma_db')
    sources=[]
    for name,record in identity['pdfs'].items():
        source=(OUT/'sources'/name) if name.startswith('TSMC') else ROOT/'data'/('pdfs' if name.startswith('2025') else 'pdfs_other')/name
        sources.append(dict(filename=name,path=str(source),sha256=record['sha256']))
    registry=runtime/'source-registry.json'
    if not registry.exists():registry.write_text(json.dumps(dict(build_id=identity['build_id'],documents=sources)),encoding='utf-8')
    os.environ.update(NEO4J_URI=c['uri'],NEO4J_USERNAME=c['username'],NEO4J_PASSWORD=c['password'],NEO4J_DATABASE=c['database'],
        GRAPHRAG_BUILD_ID=identity['build_id'],GRAPH_VECTOR_COLLECTION=manifest['collection'],GRAPH_EMBEDDING_BACKEND='chroma_onnx',
        GRAPHRAG_SOURCE_REGISTRY=str(registry),GRAPHRAG_CANDIDATE_GRAPH=str(package/'graph.json'),
        API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0')
    os.chdir(runtime)
    import uvicorn
    uvicorn.run('strategic_graphrag.api.server:app',host='127.0.0.1',port=a.port)
if __name__=='__main__':main()
