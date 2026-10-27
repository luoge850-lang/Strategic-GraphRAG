"""Real HTTP closure: frozen question-only, user fixture, diagnostic conditions.

No answer or reference metric/year/evidence is sent to the HTTP API.
User-scope is an explicit synthetic file-selection fixture, not natural input.
Restart HTTP per partition/condition to avoid changing the default rate limit;
each block has at most 40 questions. No evaluation-led tuning is performed.
"""
import argparse,hashlib,json,os,subprocess,sys,time,shutil,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.live_acceptance import http

def save(path,data):
    with path.open('x',encoding='utf-8') as f:json.dump(data,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser()
    for n in ['candidate','credentials','inputs','runtime','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--partition',choices=['development','validation'],required=True)
    p.add_argument('--port',type=int,default=8002)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from scripts.run_isolated_staging import verify_package
    import psutil
    before=verify_package(a.candidate);save(a.output/'immutable-before.json',before)
    if before['status']!='PASS':raise RuntimeError('immutable candidate rejected')
    identity=json.loads((a.candidate/'build_identity.json').read_text())
    manifest=json.loads((a.candidate/'vector_index_manifest.json').read_text())
    frozen=json.loads((a.inputs/'frozen_inputs_sha256.json').read_text())
    for name,digest in frozen.items():
        if hashlib.sha256((a.inputs/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('frozen input changed')
    scoring_files=['deployment/complete_fact_scoring.py','deployment/answer_closure_scoring.py','deployment/run_answer_closure.py']
    scorer_hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in scoring_files}
    frozen_scorer=a.inputs/'frozen_scoring_sha256.json'
    if frozen_scorer.exists():
        if json.loads(frozen_scorer.read_text())!=scorer_hashes:raise RuntimeError('scoring contract changed')
    else:save(frozen_scorer,scorer_hashes)
    labels=json.loads((a.inputs/f'{a.partition}_labels.json').read_text(encoding='utf-8'))
    if not a.runtime.exists():
        a.runtime.mkdir(parents=True)
        shutil.copytree(a.candidate/'vector_index',a.runtime/'data/chroma_db')
        for name,record in identity['pdfs'].items():
            source=ROOT/'data'/('pdfs' if name.startswith('2025') else 'pdfs_other')/name
            if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('source hash mismatch')
            target=a.runtime/'data'/source.parent.name/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
    creds=json.loads(a.credentials.read_text())
    if not creds['uri'].startswith('bolt://127.0.0.1:'):raise RuntimeError('loopback store required')
    env={k:v for k,v in os.environ.items() if not k.endswith('API_KEY')}
    env.update(NEO4J_URI=creds['uri'],NEO4J_USERNAME=creds['username'],NEO4J_PASSWORD=creds['password'],
        NEO4J_DATABASE=creds['database'],GRAPHRAG_BUILD_ID=identity['build_id'],GRAPH_VECTOR_COLLECTION=manifest['collection'],
        GRAPH_EMBEDDING_BACKEND='chroma_onnx',API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0',
        PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT))
    check=subprocess.run([sys.executable,str(ROOT/'deployment/preflight.py'),'--candidate-dir',str(a.candidate.resolve()),
        '--data-root',str(a.runtime.resolve()/'data'),'--output',str((a.output/'preflight.json').resolve())],
        cwd=ROOT,env=env,capture_output=True,timeout=90)
    if check.returncode:raise RuntimeError('real source/store/vector preflight rejected')
    process=None
    try:
        for mode in ['question_only','explicit_user_scope_fixture','reference_scope_diagnostic']:
            started=time.perf_counter();peak=0
            with (a.output/f'{mode}-server.log').open('xb') as log:
                process=subprocess.Popen([sys.executable,'-m','uvicorn','strategic_graphrag.api.server:app',
                    '--host','127.0.0.1','--port',str(a.port)],cwd=a.runtime.resolve(),env=env,
                    stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            base=f'http://127.0.0.1:{a.port}';deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('HTTP startup exited')
                if http(base,'/health/live',timeout=1)['http_status']==200:break
                time.sleep(.5)
            else:raise RuntimeError('HTTP startup deadline exceeded')
            startup=(time.perf_counter()-started)*1000
            with (a.output/f'{mode}.jsonl').open('x',encoding='utf-8') as stream:
                for label in labels:
                    request=dict(question=label['question'],max_paths=10,retrieval_mode='graph',synthesize=False,use_cache=False)
                    if mode!='question_only':request.update(source_filing=label['source_filing'],cross_filing=False)
                    measured=http(base,'/query',request);measured.pop('_bytes',None)
                    response=measured.get('response') or {};metadata=response.get('metadata') or {}
                    row=dict(item_id=label['item_id'],family_id=label['family_id'],partition=label['partition'],mode=mode,
                        question=label['question'],request=request,query_plan=metadata.get('query_plan'),
                        actual_filter=metadata.get('source_filing'),reference=label,label_hashes=frozen,scorer_hashes=scorer_hashes,
                        build_id=identity['build_id'],source_fingerprint=identity.get('source_fingerprint'),
                        code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        input_provenance='question only' if mode=='question_only' else 'explicit artificial user file-selection fixture' if mode=='explicit_user_scope_fixture' else 'known reference file, diagnostic upper-bound only',**measured)
                    stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                    try:peak=max(peak,psutil.Process(process.pid).memory_info().rss)
                    except psutil.Error:pass
            save(a.output/f'{mode}-resources.json',dict(requests=len(labels),startup_ms=startup,
                sampled_http_rss_bytes=peak,sampling_scope='HTTP process sampled between requests, not peak system memory',
                hardware=platform.processor(),platform=platform.platform(),cpu_count=os.cpu_count(),
                memory_total_bytes=psutil.virtual_memory().total,database='real Neo4j warm/uncontrolled OS cache',
                retrieval_mode='graph',synthesize=False,use_cache=False,rate_limit='unchanged default; <=40 queries per block',
                llm_calls=0,embedding_calls='not instrumented; graph-only route',cost=None))
            process.terminate();process.wait(timeout=30);process=None
            print(a.partition,mode,'RECORDED',len(labels),flush=True)
    finally:
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=30)
        save(a.output/'immutable-after.json',verify_package(a.candidate))

if __name__=='__main__':main()
