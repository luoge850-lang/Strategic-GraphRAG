"""Three disjoint input-condition trials against an existing isolated real store."""
import argparse,hashlib,json,os,subprocess,sys,time,platform,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from deployment.live_acceptance import http

def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','credentials','inputs','dataset','output','runtime']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--port',type=int,default=8002)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    from scripts.run_isolated_staging import verify_package
    import psutil
    before=verify_package(a.candidate);save(a.output/'immutable-before.json',before)
    if before['status']!='PASS':raise RuntimeError('immutable package rejected')
    identity=json.loads((a.candidate/'build_identity.json').read_text())
    manifest=json.loads((a.candidate/'vector_index_manifest.json').read_text())
    if not a.runtime.exists():
        a.runtime.mkdir(parents=True)
        shutil.copytree(a.candidate/'vector_index',a.runtime/'data/chroma_db')
        for name,record in identity['pdfs'].items():
            source=ROOT/'data'/('pdfs' if name.startswith('2025') else 'pdfs_other')/name
            if hashlib.sha256(source.read_bytes()).hexdigest()!=record['sha256']:raise RuntimeError('PDF hash mismatch')
            target=a.runtime/'data'/source.parent.name/name
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    frozen=json.loads((a.inputs/'frozen_inputs_sha256.json').read_text())
    for name,digest in frozen.items():
        if hashlib.sha256((a.inputs/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('frozen inputs changed')
    protocol=json.loads((a.inputs/'protocol.json').read_text())
    dev=[]
    for line in a.dataset.read_text(encoding='utf-8').splitlines():
        family=json.loads(line)
        for item in [family,*family.get('candidate_variants',[])]:
            dev.append(dict(item_id=item['item_id'],family_id=family['family_id'],question=item['question'],
                question_type=item.get('question_type'),partition='development_previously_used',
                diagnostic_filing=item.get('candidate_filing') or family.get('candidate_filing'),
                relevance_grades=item.get('relevance_grades',{})))
    validation=json.loads((a.inputs/'validation_fact_labels.json').read_text())
    for label in validation:
        dev.append(dict(item_id=label['item_id'],family_id=label['metric'],question=label['question'],
            question_type='adjacent_fact_validation',partition='frozen_adjacent_diagnostic_no_tuning',
            diagnostic_filing=label['source_filing'],relevance_grades={f"{x['file']}#{x['page']}":3 for x in label['direct_locations']}))
    creds=json.loads(a.credentials.read_text())
    if not creds['uri'].startswith('bolt://127.0.0.1:'):raise ValueError('only loopback trial allowed')
    env={k:v for k,v in os.environ.items() if not k.endswith('API_KEY')}
    env.update(NEO4J_URI=creds['uri'],NEO4J_USERNAME=creds['username'],NEO4J_PASSWORD=creds['password'],
        NEO4J_DATABASE=creds['database'],GRAPHRAG_BUILD_ID=identity['build_id'],
        GRAPH_VECTOR_COLLECTION=manifest['collection'],GRAPH_EMBEDDING_BACKEND='chroma_onnx',
        API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0',PYTHONUNBUFFERED='1',PYTHONPATH=str(ROOT))
    check=subprocess.run([sys.executable,str(ROOT/'deployment/preflight.py'),'--candidate-dir',str(a.candidate.resolve()),
        '--data-root',str(a.runtime.resolve()/'data'),'--output',str((a.output/'preflight.json').resolve())],cwd=ROOT,env=env,capture_output=True,timeout=90)
    if check.returncode:raise RuntimeError('real-store preflight rejected')
    process=None
    try:
        for mode in protocol['modes']:
            start=time.perf_counter();peak=0
            with (a.output/f'{mode}-server.log').open('xb') as log:
                process=subprocess.Popen([sys.executable,'-m','uvicorn','strategic_graphrag.api.server:app','--host','127.0.0.1','--port',str(a.port)],
                    cwd=a.runtime.resolve(),env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            base=f'http://127.0.0.1:{a.port}';deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('HTTP process exited')
                if http(base,'/health/live',timeout=1)['http_status']==200:break
                time.sleep(.5)
            else:raise RuntimeError('startup timed out')
            startup=(time.perf_counter()-start)*1000
            with (a.output/f'{mode}.jsonl').open('x',encoding='utf-8') as stream:
                for item in dev:
                    request=dict(question=item['question'],max_paths=10,retrieval_mode='graph',synthesize=False,use_cache=False)
                    if mode=='user_scope':request.update(source_filing=protocol['user_scope']['source_filing'],cross_filing=False)
                    if mode=='reference_scope_diagnostic':request.update(source_filing=item['diagnostic_filing'],cross_filing=False)
                    result=http(base,'/query',request);result.pop('_bytes',None)
                    response=result.get('response') or {};metadata=response.get('metadata') or {}
                    record=dict(**item,mode=mode,request=request,query_plan=metadata.get('query_plan'),
                        actual_scope=metadata.get('source_filing'),build_id=identity['build_id'],
                        label_hashes=frozen,code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                        **result)
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
                    try:
                        parent=psutil.Process(process.pid)
                        peak=max(peak,sum(x.memory_info().rss for x in [parent,*parent.children(recursive=True)]))
                    except psutil.Error:pass
            save(a.output/f'{mode}-resources.json',dict(requests=len(dev),startup_ms=startup,
                peak_rss_bytes_between_requests=peak,sampling_scope='HTTP tree only, sampled after requests, not true transient peak',
                database='preserved real Neo4j, already warm; restarted HTTP process per condition',
                os_cache='uncontrolled',route='graph',synthesize=False,query_cache=False,
                generation_calls=0,embedding_inference='not measured separately; graph route',platform=platform.platform()))
            process.terminate();process.wait(timeout=30);process=None
            print(mode,'RECORDED',len(dev),flush=True)
    finally:
        if process is not None and process.poll() is None:process.terminate();process.wait(timeout=30)
        save(a.output/'immutable-after.json',verify_package(a.candidate))

if __name__=='__main__':main()
