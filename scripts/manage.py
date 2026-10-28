"""Single local entry point: init, build, db-start, import, serve, stop."""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LOCAL = ROOT / '.local'
BUILD = LOCAL / 'build'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def identity_for(paths, corpus_id):
    from strategic_graphrag.build_identity import make_build_identity
    return make_build_identity(paths, corpus_id=corpus_id,
        extraction_provider='local_rules_and_tables', extraction_model='not_applicable',
        query_model='disabled_for_local_acceptance', report_model='disabled_for_local_acceptance',
        embedding_backend='chroma_onnx', embedding_model='all-MiniLM-L6-v2',
        dependency_lock=ROOT / 'requirements-lock.txt')


def build():
    import pdfplumber
    import pymupdf
    import chromadb
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from strategic_graphrag.build_identity import sha256_file
    from strategic_graphrag.pipeline.financial_cache import refresh_nvidia, extraction_contract
    from strategic_graphrag.pipeline.company_disclosures import extract
    from strategic_graphrag.pipeline.consolidated_ntd_table import extract as extract_ntd
    from strategic_graphrag.pipeline.accepted_graph import accept_candidates
    if BUILD.exists():
        raise FileExistsError('Current build exists. Stop services and remove it explicitly before rebuilding.')
    docs = read(ROOT / 'data/corpus.json')['documents']
    paths = [ROOT / 'data/pdfs' / d['filename'] for d in docs]
    for d, p in zip(docs, paths):
        if sha256_file(p) != d['sha256']:
            raise ValueError('PDF identity mismatch: ' + d['filename'])
    identity = identity_for(paths, 'five-report-current-v1').to_dict()
    bid = identity['build_id']
    write(BUILD / 'build_identity.json', identity)
    splitter = RecursiveCharacterTextSplitter(chunk_size=2400, chunk_overlap=300)
    client = chromadb.PersistentClient(path=str(BUILD / 'vector_index'))
    collection = client.create_collection(name='current_' + bid, embedding_function=DefaultEmbeddingFunction())
    files, registry, timings = [], [], []
    for d, path in zip(docs, paths):
        started = time.perf_counter()
        company, year, filename = d['company_id'], d['filing_year'], d['filename']
        row = dict(filename=filename, document_sha256=d['sha256'], year=year, accepted_triples=[])
        if company == 'NVIDIA_CORPORATION':
            refresh_nvidia(row, path, contract=extraction_contract(document_sha256=d['sha256'], company_id=company, filing_year=year, filename=filename))
        with pymupdf.open(path) as pdf:
            texts = [p.get_text() for p in pdf]
        if company == 'TSMC':
            with pdfplumber.open(path) as pdf:
                for n, text in enumerate(texts, 1):
                    if any(s in text for s in ('Financial Status', 'Financial Performance', 'Cash Flow')):
                        for fact in extract_ntd(pdf.pages[n-1], company_id=company, filename=filename, filing_year=year):
                            fact.update(source_page=n, page=n, document_sha256=d['sha256'])
                            row['accepted_triples'].append(fact)
                        pdf.pages[n-1].close()
        ids, chunks, metas = [], [], []
        # Page text comes directly from the original PDF. Physical page is always 1-based.
        for n, text in enumerate(texts, 1):
            for fact in extract(text, company):
                fact.update(source_page=n, page=n, source_filing=filename, document_sha256=d['sha256'])
                row['accepted_triples'].append(fact)
            for i, chunk in enumerate(splitter.split_text(text)):
                cid = f'{filename}:{n}:{i}'
                ids.append(cid); chunks.append(chunk)
                metas.append(dict(build_id=bid, source_filing=filename, doc_id=path.stem,
                                  page=n, chunk_id=cid, chunk_index=i, pdf_sha256=d['sha256']))
        for offset in range(0, len(ids), 64):
            collection.add(ids=ids[offset:offset+64], documents=chunks[offset:offset+64], metadatas=metas[offset:offset+64])
        registry.append(dict(filename=filename, path=path.relative_to(ROOT).as_posix(), sha256=d['sha256'], pages=len(texts)))
        files.append(row)
        timings.append(dict(filename=filename, pages=len(texts), chunks=len(ids), candidates=len(row['accepted_triples']), seconds=time.perf_counter()-started, model_calls=0, tokens=0))
        print(json.dumps(timings[-1]), flush=True)
    write(BUILD / 'candidate_extraction.json', dict(files=files, build_id=bid))
    accept_candidates(BUILD, bid)
    (BUILD / 'candidate_extraction.json').unlink()
    write(BUILD / 'source-registry.json', dict(build_id=bid, documents=registry))
    write(BUILD / 'vector_manifest.json', dict(build_id=bid, collection=collection.name, chunks=collection.count(), embedding_backend='chroma_onnx', embedding_model='all-MiniLM-L6-v2'))
    write(BUILD / 'build-run.json', dict(build_id=bid, files=timings, mode='FRESH_PDF_EXTRACTION', generation='DISABLED'))
    print(bid)


def init(args):
    if (LOCAL / 'config.json').exists():
        raise FileExistsError('Local configuration already exists')
    home, java = Path(args.neo4j_home).resolve(), Path(args.java_home).resolve()
    if not (home / 'bin/neo4j.bat').is_file() or not (java / 'bin/java.exe').is_file():
        raise ValueError('Neo4j distribution and JDK required')
    write(LOCAL / 'config.json', dict(neo4j_home=str(home), java_home=str(java),
        uri='bolt://127.0.0.1:7687', username='neo4j', password=secrets.token_hex(24), database='neo4j', http_port=7474))


def db_start(config):
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable
    home, state = Path(config['neo4j_home']), LOCAL / 'neo4j'
    conf = state / 'conf'
    env = {**os.environ, 'JAVA_HOME': config['java_home'], 'NEO4J_CONF': str(conf)}
    if not conf.exists():
        shutil.copytree(home / 'conf', conf)
        settings = {'server.default_listen_address':'127.0.0.1',
            'server.bolt.listen_address':'127.0.0.1:7687','server.http.listen_address':'127.0.0.1:7474',
            'server.https.enabled':'false','server.memory.heap.initial_size':'256m',
            'server.memory.heap.max_size':'512m','server.memory.pagecache.size':'256m',
            'server.directories.data':(state/'data').as_posix(), 'server.directories.logs':(state/'logs').as_posix(),
            'server.directories.run':(state/'run').as_posix(),'dbms.security.auth_enabled':'true'}
        p = conf / 'neo4j.conf'
        lines = [x for x in p.read_text().splitlines() if x.split('=',1)[0].strip() not in settings]
        p.write_text('\n'.join(lines+[f'{k}={v}' for k,v in settings.items()])+'\n')
        result = subprocess.run([str(home/'bin/neo4j-admin.bat'),'dbms','set-initial-password',config['password']], env=env, capture_output=True)
        if result.returncode:
            raise RuntimeError('Neo4j password initialization failed')
    # Windows daemon commands require an installed OS service. Keep this
    # project-local instance in console mode, and record its owned launcher.
    with (state/'console.log').open('ab') as log:
        process = subprocess.Popen([str(home/'bin/neo4j.bat'),'console'], cwd=home, env=env,
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
    write(state/'process.json',dict(pid=process.pid))
    deadline = time.monotonic()+90
    while time.monotonic()<deadline:
        if process.poll() is not None:
            raise RuntimeError('Neo4j exited; inspect .local/neo4j/console.log')
        try:
            with GraphDatabase.driver(config['uri'], auth=(config['username'],config['password']), connection_timeout=2) as driver:
                driver.verify_connectivity()
            print('Neo4j READY'); return
        except ServiceUnavailable:
            time.sleep(1)
    raise TimeoutError('Neo4j startup timed out')


def check_build():
    from strategic_graphrag.build_identity import source_fingerprint, sha256_file, BuildIdentityMismatch
    identity = read(BUILD / 'build_identity.json')
    if identity['source_tree_sha256'] != source_fingerprint(ROOT):
        raise BuildIdentityMismatch('Running sources differ from current build')
    registry = read(BUILD / 'source-registry.json')
    if registry['build_id'] != identity['build_id']:
        raise BuildIdentityMismatch('Registry/build mismatch')
    for doc in registry['documents']:
        if sha256_file(ROOT / doc['path']) != doc['sha256']:
            raise BuildIdentityMismatch('PDF identity mismatch: '+doc['filename'])
    return identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['init','build','db-start','import','serve','stop'])
    parser.add_argument('--neo4j-home'); parser.add_argument('--java-home')
    parser.add_argument('--port',type=int,default=8000)
    args=parser.parse_args(); os.chdir(ROOT)
    if args.command=='init': init(args); return
    if args.command=='build': build(); return
    config=read(LOCAL/'config.json')
    if args.command=='db-start': db_start(config); return
    if args.command=='stop':
        pid=int(read(LOCAL/'neo4j/process.json')['pid'])
        query=f"Get-CimInstance Win32_Process -Filter 'ProcessId={pid}' | Select-Object -ExpandProperty CommandLine"
        command=subprocess.run(['powershell','-NoProfile','-Command',query],capture_output=True,text=True,check=True).stdout
        if str(Path(config['neo4j_home'])) not in command:
            raise RuntimeError('Recorded launcher is absent or belongs to another process; refusing to stop it')
        subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],check=True,capture_output=True)
        print('Owned local Neo4j console process tree stopped; recovery is checked on restart')
        return
    identity=check_build(); bid=identity['build_id']
    if args.command=='import':
        from neo4j import GraphDatabase
        from strategic_graphrag.pipeline.graph_import import prepare,write_empty
        plan=prepare(read(BUILD/'graph.json'))
        with GraphDatabase.driver(config['uri'],auth=(config['username'],config['password'])) as driver:
            with driver.session(database=config['database']) as session:
                result=session.execute_write(write_empty,plan)
                session.run('CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]').consume()
                session.run('CALL db.awaitIndexes(30)').consume()
        write(BUILD/'import-run.json',dict(build_id=bid,**result)); print(result); return
    os.environ.update(NEO4J_URI=config['uri'],NEO4J_USERNAME=config['username'],NEO4J_PASSWORD=config['password'],
        NEO4J_DATABASE=config['database'],GRAPHRAG_BUILD_ID=bid,GRAPH_VECTOR_COLLECTION='current_'+bid,
        GRAPH_EMBEDDING_BACKEND='chroma_onnx',GRAPHRAG_SOURCE_REGISTRY=str(BUILD/'source-registry.json'),
        GRAPHRAG_CANDIDATE_GRAPH=str(BUILD/'graph.json'),GRAPHRAG_FRONTEND_DIST=str(ROOT/'frontend/dist'),
        GRAPHRAG_VECTOR_PATH=str(BUILD/'vector_index'),
        API_AUTH_ENABLED='false',QUERY_CACHE_TTL_SECONDS='0',LLM_PROVIDER='deepseek',
        DEEPSEEK_API_KEY='',GROQ_API_KEY='',GEMINI_API_KEY='')
    import uvicorn
    uvicorn.run('strategic_graphrag.api.server:app',host='127.0.0.1',port=args.port)


if __name__=='__main__': main()
