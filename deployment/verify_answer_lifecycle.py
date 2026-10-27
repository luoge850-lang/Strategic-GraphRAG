"""Bounded isolated lifecycle with real dependency-error HTTP and recovery."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.live_acceptance import http,run

def save(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser()
    for n in ['candidate','credentials','runtime','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--keep-demo',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from scripts.run_isolated_staging import verify_package
    before=verify_package(a.candidate)
    if before['status']!='PASS':raise RuntimeError('candidate rejected')
    c=json.loads(a.credentials.read_text());identity=json.loads((a.candidate/'build_identity.json').read_text())
    m=json.loads((a.candidate/'vector_index_manifest.json').read_text())
    env={k:v for k,v in os.environ.items() if not k.endswith('API_KEY')}
    env.update(NEO4J_URI=c['uri'],NEO4J_USERNAME=c['username'],NEO4J_PASSWORD=c['password'],NEO4J_DATABASE=c['database'],
        GRAPHRAG_BUILD_ID=identity['build_id'],GRAPH_VECTOR_COLLECTION=m['collection'],GRAPH_EMBEDDING_BACKEND='chroma_onnx',
        API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0',PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1')
    def start(name,overrides=None):
        with (a.output/(name+'.log')).open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-m','uvicorn','strategic_graphrag.api.server:app','--host','127.0.0.1','--port','8000'],
                cwd=a.runtime.resolve(),env=dict(env,**(overrides or {})),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError('HTTP startup exited')
            if http('http://127.0.0.1:8000','/health/live',timeout=1)['http_status']==200:return proc
            time.sleep(.5)
        proc.terminate();proc.wait(timeout=30);raise RuntimeError('startup deadline exceeded')
    def preflight(name,overrides):
        proc=subprocess.run([sys.executable,str(ROOT/'deployment/preflight.py'),'--candidate-dir',str(a.candidate.resolve()),
            '--data-root',str(a.runtime.resolve()/'data'),'--output',str((a.output/(name+'.json')).resolve())],
            env=dict(env,**overrides),cwd=ROOT,capture_output=True,timeout=90)
        return dict(name=name,exit_code=proc.returncode)
    process=None
    try:
        checks=[preflight('wrong-build',dict(GRAPHRAG_BUILD_ID='build_0000000000000000')),
                preflight('unavailable-preflight',dict(NEO4J_URI='bolt://127.0.0.1:17999')),
                preflight('restored-preflight',{})]
        process=start('dependency-http',dict(NEO4J_URI='bolt://127.0.0.1:17999'))
        failed=http('http://127.0.0.1:8000','/query',dict(question='What was NVIDIA revenue for fiscal 2025?',
            retrieval_mode='graph',synthesize=False,use_cache=False,max_paths=10),timeout=45)
        failed.pop('_bytes',None);save(a.output/'dependency-http.json',failed)
        process.terminate();process.wait(timeout=30);process=None
        process=start('restored');initial=run('http://127.0.0.1:8000',a.output/'initial-five')
        process.terminate();process.wait(timeout=30);process=None
        process=start('restart');restart=run('http://127.0.0.1:8000',a.output/'restart-five')
        result=dict(build_id=identity['build_id'],source_tree_sha256=identity['source_tree_sha256'],
            vector_collection=m['collection'],database_uri=c['uri'],checks=checks,
            dependency_http_status=failed['http_status'],dependency_response=failed.get('response'),initial=initial,restart=restart,
            immutable_before=before,immutable_after=verify_package(a.candidate),demo_pid=process.pid if a.keep_demo else None,
            scope='isolated HTTP restart and dependency-config recovery; not OS/crash recovery or production SLO',production_mutations=0)
        save(a.output/'lifecycle.json',result)
        print(json.dumps(dict(build=result['build_id'],dependency_http=failed['http_status'],initial=initial,restart=restart,demo_pid=result['demo_pid'])))
    finally:
        if process is not None and not a.keep_demo:process.terminate();process.wait(timeout=30)

if __name__=='__main__':main()
