"""Read identities, test replay rejection and external cancellation boundaries."""
import json,sys,subprocess,time,hashlib,os
from collections import defaultdict
from pathlib import Path
from deployment.acceptance_20261015 import ROOT,OUT,save,hashes
from deployment.import_candidate import prepare,write_empty
def main():
    import chromadb
    from neo4j import GraphDatabase
    from strategic_graphrag.document_layer import DocumentLayerReader
    from strategic_graphrag.engine.graph_rag_engine import CausalPathFinder
    from strategic_graphrag.build_identity import source_fingerprint
    package=OUT/'five-package-round2';identity=json.loads((package/'build_identity.json').read_text())
    c=json.loads((OUT/'neo4j-round2/private_credentials.json').read_text())
    graph=json.loads((package/'graph.json').read_text(encoding='utf-8'));plan=prepare(graph)
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            before=s.run('MATCH (n) RETURN count(n) AS count').single()['count']
            try:s.execute_write(write_empty,plan);replay='UNEXPECTED_SUCCESS'
            except ValueError as exc:replay=dict(status='REJECTED_NONEMPTY_STORE',reason=str(exc))
            after=s.run('MATCH (n) RETURN count(n) AS count').single()['count']
            documents=[dict(r['o']) for r in s.run('MATCH (o:Document) RETURN properties(o) AS o')]
            obs=[dict(r['o']) for r in s.run('MATCH (o:FinancialObservation) RETURN properties(o) AS o')]
            conflicts=[r.data() for r in s.run('''MATCH (o:FinancialObservation)
              WITH o.company_id AS company,o.metric_id AS metric,o.fiscal_period AS period,o.unit AS unit,
                   collect(DISTINCT o.value) AS values,collect({claim:o.claim_id,file:o.source_filing,page:o.page,value:o.value}) AS sources
              WHERE size(values)>1 RETURN company,metric,period,unit,values,sources''')]
        wrong=CausalPathFinder(d).find_metric_disclosures('REVENUE',company_id='NVIDIA_CORPORATION',build_id='build_ff4e5bcf11c71930')
    expected=identity['pdfs'];company_for=lambda name:'TSMC' if name.startswith('TSMC') else 'NVIDIA_CORPORATION'
    failures=[]
    for doc in documents:
        if doc['document_sha256']!=expected[doc['filename']]['sha256'] or doc['build_id']!=identity['build_id']:failures.append('Neo4j document mismatch')
    for o in obs:
        if o['company_id']!=company_for(o['source_filing']) or o['build_id']!=identity['build_id']:failures.append('observation company/build mismatch')
    manifest=json.loads((package/'vector_index_manifest.json').read_text())
    client=chromadb.PersistentClient(path=str(OUT/'retrieval-runtime'));col=client.get_collection(manifest['collection'])
    vectors=col.get(include=['metadatas'])
    for meta in vectors['metadatas']:
        if meta['build_id']!=identity['build_id'] or meta['pdf_sha256']!=expected[meta['source_filing']]['sha256']:failures.append('vector source mismatch')
    wrong_vector=col.get(where={'build_id':'build_ff4e5bcf11c71930'},include=[])
    by_measure=defaultdict(set)
    for o in obs:by_measure[(o['company_id'],o['metric_id'],o['fiscal_period'],o['unit'])].add((o['claim_id'],o['source_filing'],o['value']))
    multis=[dict(company=k[0],metric=k[1],period=k[2],unit=k[3],sources=[dict(claim=v[0],file=v[1],value=v[2]) for v in sorted(values)]) for k,values in by_measure.items() if len(values)>1]
    testdir=OUT/'engineering-boundaries';testdir.mkdir(exist_ok=False)
    bad=testdir/'not-a-pdf.pdf';bad.write_bytes(b'not a valid PDF\n')
    try:DocumentLayerReader().read(bad,build_id=identity['build_id']);bad_result=dict(status='UNEXPECTED_ACCEPT')
    except Exception as exc:bad_result=dict(status='EXPLICIT_EXCEPTION',error_type=type(exc).__name__,message=str(exc))
    script='from strategic_graphrag.document_layer import DocumentLayerReader; DocumentLayerReader().read(__import__("sys").argv[1],build_id="boundary-test")'
    command=[sys.executable,'-c',script,str(OUT/'sources/TSMC-2023-Annual-Report.pdf')]
    started=time.perf_counter()
    try:subprocess.run(command,cwd=ROOT,capture_output=True,timeout=.01,creationflags=subprocess.CREATE_NO_WINDOW);timeout=dict(status='UNEXPECTED_COMPLETION')
    except subprocess.TimeoutExpired:timeout=dict(status='EXTERNAL_HARNESS_TIMEOUT',seconds=time.perf_counter()-started,native_timeout_state='NOT_IMPLEMENTED_OR_VERIFIED')
    process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,creationflags=subprocess.CREATE_NO_WINDOW)
    time.sleep(.2);process.terminate();process.communicate(timeout=15)
    cancel=dict(status='EXTERNAL_PROCESS_CANCELLED',exit_code=process.returncode,native_cancel_state='NOT_IMPLEMENTED_OR_VERIFIED',resume_after_mid_parse_kill='NOT_MEASURED')
    checkpoints=[]
    for checkpoint in sorted((OUT/'five-package/checkpoints').glob('*.json')):
        row=json.loads(checkpoint.read_text(encoding='utf-8'));doc=row['document'];name=doc['filename']
        checkpoints.append(dict(file=name,pdf_sha256=doc['pdf_sha256'],matches=doc['pdf_sha256']==expected[name]['sha256'],status='COMPLETED_CHECKPOINT_READ_IN_NEW_PROCESS'))
    journal=[json.loads(x) for x in (OUT/'five-package/journal.jsonl').read_text().splitlines()]
    single=[json.loads(x) for x in (OUT/'single-package/journal.jsonl').read_text().splitlines()]
    resources=dict(batch_files=5,batch_physical_pages=815,batch_seconds=journal[-1]['timestamp']-journal[0]['timestamp'],
        batch_sampled_max_rss_bytes=max(x['rss_bytes'] for x in journal),true_peak_rss_bytes=None,
        single_files=1,single_physical_pages=94,single_seconds=single[-1]['timestamp']-single[0]['timestamp'],single_sampled_max_rss_bytes=max(x['rss_bytes'] for x in single),
        total_input_bytes=sum(x['bytes'] for x in expected.values()),candidate_disk_bytes=sum(p.stat().st_size for p in package.rglob('*') if p.is_file()),
        final_vector_chunks=manifest['chunks_indexed'],batch_vector_seconds=next(x['seconds'] for x in journal if x.get('stage')=='vectors'),
        single_vector_chunks=json.loads((OUT/'single-package/vector_index_manifest.json').read_text())['chunks_indexed'],
        actual_llm_network_calls=0,llm_tokens=0,embedding_build_calls=None,embedding_build_tokens=None,cost=None,
        concurrency='one document ingest job; internal library thread count not measured',memory_budget_bytes=8*1024**3,hard_memory_limit_enforcement='NOT_IMPLEMENTED')
    resources['batch_chunks_per_second']=resources['final_vector_chunks']/resources['batch_vector_seconds']
    old=ROOT/'reports/isolated_staging/build_ff4e5bcf11c71930'
    old_before=json.loads((OUT/'old-package-before.json').read_text())
    final_before=json.loads((OUT/'package-round2-frozen-hashes.json').read_text())
    save('engineering-results.json',dict(replay=replay,node_count_before=before,node_count_after=after,unexpected_duplicates=after-before,
        identity=dict(neo4j_documents=len(documents),neo4j_observations=len(obs),chroma_chunks=len(vectors['ids']),failures=failures,
            wrong_build_neo4j_hits=len(wrong),wrong_build_chroma_hits=len(wrong_vector['ids'])),
        multi_source_observations=multis,conflicts=conflicts,
        bad_pdf=bad_result,cancellation=cancel,timeout=timeout,checkpoints=checkpoints,
        injected_third_failure=[x for x in journal if x.get('status') in {'FAILED_INJECTED','RETRY'}],
        resources=resources,old_package_unchanged=hashes(old)==old_before,final_package_unchanged=hashes(package)==final_before,
        final_source_fingerprint_matches=source_fingerprint()==identity['source_tree_sha256'],
        batch_scope='Five-document ingestion and one-document independent ingestion. No large-scale test.',
        limitations=['Replay rejected atomically rather than idempotent successful re-ingestion.','Injected failure is before third-file parse; later files complete and third retries in same process. New process verifies checkpoints; true crash-mid-write restart not measured.',
           'Cancellation and timeout are external harness actions, not native ingestion status APIs.','No re-run after immutable freeze; raw package journals and caches retained.']))
    print(json.dumps(dict(replay=replay,identity_failures=failures,conflict_groups=len(conflicts),multi_source_groups=len(multis),resources=resources),indent=2))
if __name__=='__main__':main()
