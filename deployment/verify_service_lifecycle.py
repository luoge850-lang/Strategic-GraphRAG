"""Isolated HTTP restart, dependency/build rejection and recovery; no release switch."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.live_acceptance import http,run

def main():
    p=argparse.ArgumentParser()
    for name in ['candidate','credentials','runtime','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--keep-demo',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    c=json.loads(a.credentials.read_text());ident=json.loads((a.candidate/'build_identity.json').read_text())
    manifest=json.loads((a.candidate/'vector_index_manifest.json').read_text())
    env={k:v for k,v in os.environ.items() if not k.endswith('API_KEY')}
    env.update(NEO4J_URI=c['uri'],NEO4J_USERNAME=c['username'],NEO4J_PASSWORD=c['password'],NEO4J_DATABASE=c['database'],
        GRAPHRAG_BUILD_ID=ident['build_id'],GRAPH_VECTOR_COLLECTION=manifest['collection'],GRAPH_EMBEDDING_BACKEND='chroma_onnx',
        API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0',PYTHONPATH=str(ROOT),PYTHONUNBUFFERED='1')
    def preflight(name,override):
        t=time.perf_counter()
        proc=subprocess.run([sys.executable,str(ROOT/'deployment/preflight.py'),'--candidate-dir',str(a.candidate.resolve()),
          '--data-root',str(a.runtime.resolve()/'data'),'--output',str((a.output/(name+'.json')).resolve())],
          cwd=ROOT,env=dict(env,**override),capture_output=True,timeout=90)
        return dict(name=name,exit_code=proc.returncode,elapsed_ms=(time.perf_counter()-t)*1000)
    checks=[preflight('wrong-build',{'GRAPHRAG_BUILD_ID':'build_0000000000000000'}),
            preflight('dependency-unavailable',{'NEO4J_URI':'bolt://127.0.0.1:17999'}),preflight('recovery',{})]
    proc=None
    def start(name):
        with (a.output/(name+'.log')).open('xb') as log:
            proc=subprocess.Popen([sys.executable,'-m','uvicorn','strategic_graphrag.api.server:app','--host','127.0.0.1','--port','8000'],
              cwd=a.runtime.resolve(),env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        deadline=time.monotonic()+90
        while time.monotonic()<deadline:
            if proc.poll() is not None:raise RuntimeError('HTTP exited')
            if http('http://127.0.0.1:8000','/health/live',timeout=1)['http_status']==200:return proc
            time.sleep(.5)
        raise RuntimeError('HTTP startup timeout')
    try:
        proc=start('initial');initial=run('http://127.0.0.1:8000',a.output/'initial-five')
        proc.terminate();proc.wait(timeout=30);proc=None
        proc=start('restart');restart=run('http://127.0.0.1:8000',a.output/'restart-five')
        from scripts.run_isolated_staging import verify_package
        result=dict(build_id=ident['build_id'],checks=checks,initial=initial,restart=restart,
            rollback=dict(status='NOT_EXECUTED',reason='No production or candidate publication pointer changed; HTTP restart/recovery is not cross-build database rollback'),
            immutable_after=verify_package(a.candidate),demo_pid=proc.pid if a.keep_demo else None,
            production_mutations=0,scope='restart is HTTP process restart, not operating-system or Neo4j crash recovery')
        with (a.output/'lifecycle.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
        print(json.dumps(dict(checks=checks,initial=initial,restart=restart,demo_pid=result['demo_pid'])),flush=True)
    finally:
        if proc and not a.keep_demo:proc.terminate();proc.wait(timeout=30)

if __name__=='__main__':main()
