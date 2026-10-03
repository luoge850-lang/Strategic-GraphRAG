"""Isolated old-source/database/vector/config rollback, no release pointer writes.

Old database must be offline. Copy it to a NEW trial directory; never restart,
overwrite, delete or migrate the preserved original database. Two profiles are
verified against their own source fingerprints and source-specific preflight.
"""
import argparse,hashlib,json,os,shutil,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.live_acceptance import http

def save(path,obj):
    with path.open('x',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2)

def main():
    p=argparse.ArgumentParser()
    for n in ['old-source','old-state','old-candidate','new-candidate','new-credentials','new-runtime','distribution','java-home','trial','output']:
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    original=a.old_state.resolve();trial=a.trial.resolve()
    if trial.exists() or original==trial or original in trial.parents:raise RuntimeError('fresh distinct trial required')
    c=json.loads((original/'private_credentials.json').read_text());port=int(c['uri'].rsplit(':',1)[1])
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('old database online; offline copy refused')
    if original.is_symlink() or (original/'data').is_symlink():raise RuntimeError('linked database path refused')
    config_before=hashlib.sha256((original/'conf/neo4j.conf').read_bytes()).hexdigest()
    shutil.copytree(original/'data',trial/'data')
    shutil.copytree(original/'conf',trial/'conf')
    config=trial/'conf/neo4j.conf'
    settings={'server.directories.data':(trial/'data').as_posix(),'server.directories.logs':(trial/'logs').as_posix(),
        'server.directories.run':(trial/'run').as_posix(),'server.bolt.listen_address':'127.0.0.1:17692',
        'server.bolt.advertised_address':'127.0.0.1:17692','server.http.listen_address':'127.0.0.1:17479',
        'server.http.advertised_address':'127.0.0.1:17479'}
    lines=[line for line in config.read_text().splitlines() if line.split('=',1)[0].strip() not in settings]
    config.write_text('\n'.join(lines+[k+'='+v for k,v in settings.items()])+'\n',encoding='utf-8')
    c.update(uri='bolt://127.0.0.1:17692',scope='isolated copied offline rollback database')
    save(trial/'private_credentials.json',c)
    restart=subprocess.run([sys.executable,str(ROOT/'deployment/restart_owned_neo4j.py'),'--state',str(trial),
        '--distribution',str(a.distribution.resolve()),'--java-home',str(a.java_home.resolve()),
        '--record',str((a.output/'copied-database-start.json').resolve())],cwd=ROOT,capture_output=True,timeout=110)
    if restart.returncode:raise RuntimeError('copied database startup rejected')
    runtime=trial/'runtime';shutil.copytree(a.old_candidate/'vector_index',runtime/'data/chroma_db')
    old_id=json.loads((a.old_candidate/'build_identity.json').read_text())
    for name,digest in old_id['pdfs'].items():
        source=ROOT/'data'/('pdfs' if name.startswith('2025') else 'pdfs_other')/name
        if hashlib.sha256(source.read_bytes()).hexdigest()!=digest['sha256']:raise RuntimeError('old PDF identity mismatch')
        target=runtime/'data'/source.parent.name/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    profiles=[('new-before',ROOT,a.new_candidate,a.new_runtime,json.loads(a.new_credentials.read_text())),
              ('old-rollback',a.old_source.resolve(),a.old_candidate,runtime,c),
              ('new-restored',ROOT,a.new_candidate,a.new_runtime,json.loads(a.new_credentials.read_text()))]
    records=[]
    for name,source,candidate,data,creds in profiles:
        ident=json.loads((candidate/'build_identity.json').read_text());manifest=json.loads((candidate/'vector_index_manifest.json').read_text())
        env={k:v for k,v in os.environ.items() if not k.endswith('API_KEY')}
        env.update(PYTHONPATH=str(source),NEO4J_URI=creds['uri'],NEO4J_USERNAME=creds['username'],NEO4J_PASSWORD=creds['password'],
            NEO4J_DATABASE=creds['database'],GRAPHRAG_BUILD_ID=ident['build_id'],GRAPH_VECTOR_COLLECTION=manifest['collection'],
            GRAPH_EMBEDDING_BACKEND='chroma_onnx',API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0')
        checked=subprocess.run([sys.executable,str(source/'deployment/preflight.py'),'--candidate-dir',str(candidate.resolve()),
            '--data-root',str(data.resolve()/'data'),'--output',str((a.output/(name+'-preflight.json')).resolve())],
            cwd=source,env=env,capture_output=True,timeout=90)
        if checked.returncode:
            records.append(dict(profile=name,status='REJECTED',preflight_exit=checked.returncode));continue
        with (a.output/(name+'.log')).open('xb') as log:
            process=subprocess.Popen([sys.executable,'-m','uvicorn','strategic_graphrag.api.server:app','--host','127.0.0.1','--port','8003'],
                cwd=data.resolve(),env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline=time.monotonic()+90
            while time.monotonic()<deadline:
                if process.poll() is not None:raise RuntimeError('profile HTTP exited')
                if http('http://127.0.0.1:8003','/health/live',timeout=1)['http_status']==200:break
                time.sleep(.5)
            else:raise RuntimeError('profile startup timeout')
            response=http('http://127.0.0.1:8003','/query',dict(question='What revenue for fiscal 2024 was disclosed in the 2025 filing?',
                retrieval_mode='graph',synthesize=False,use_cache=False,max_paths=10));response.pop('_bytes',None)
            calc=(response.get('response') or {}).get('calculation') or {}
            records.append(dict(profile=name,status='PASS' if response['http_status']==200 and calc.get('status')=='PASS' and calc.get('value')==60922 else 'FAIL',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip(),
                source_tree_sha256=ident['source_tree_sha256'],build_id=ident['build_id'],vector_collection=manifest['collection'],
                database_uri=creds['uri'],database_scope='copied old database' if name=='old-rollback' else 'new isolated candidate database',
                configuration_identity_sha256=hashlib.sha256(json.dumps({k:v for k,v in env.items() if k in ['GRAPHRAG_BUILD_ID','NEO4J_URI','NEO4J_DATABASE','GRAPH_VECTOR_COLLECTION','GRAPH_EMBEDDING_BACKEND']},sort_keys=True).encode()).hexdigest(),
                request_result=response))
        finally:process.terminate();process.wait(timeout=30)
    from scripts.run_isolated_staging import verify_package
    save(a.output/'rollback.json',dict(status='PASS' if len(records)==3 and all(r['status']=='PASS' for r in records) else 'FAIL',
        records=records,old_original_config_unchanged=hashlib.sha256((original/'conf/neo4j.conf').read_bytes()).hexdigest()==config_before,
        old_candidate_after=verify_package(a.old_candidate),new_candidate_after=verify_package(a.new_candidate),
        production_pointer_mutations=0,original_database_mutations=0,
        scope='three isolated source+store+vector+configuration profiles, restored new profile; not live production failover'))
    print(json.dumps([(r['profile'],r['status']) for r in records]))

if __name__=='__main__':main()
