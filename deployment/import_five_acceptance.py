"""Import new five-file candidate only to an empty authenticated loopback store."""
import argparse,json,time
from deployment.acceptance_20261015 import OUT,hashes,save
from deployment.import_candidate import prepare,write_empty
def main():
    from neo4j import GraphDatabase
    p=argparse.ArgumentParser();p.add_argument('--round',default='round1');a=p.parse_args()
    package=OUT/('five-package' if a.round=='round1' else 'five-package-round2')
    before=hashes(package)
    graph=json.loads((package/'graph.json').read_text(encoding='utf-8'))
    identity=json.loads((package/'build_identity.json').read_text(encoding='utf-8'))
    if graph['build_id']!=identity['build_id']:raise ValueError('build mismatch')
    if len(identity['pdfs'])!=5:raise ValueError('expected five real documents')
    for edge in graph['edges']:
        if edge['document_sha256']!=identity['pdfs'][edge['source_filing']]['sha256']:raise ValueError('PDF mismatch')
    c=json.loads((OUT/f'neo4j-{a.round}/private_credentials.json').read_text())
    if not c['uri'].startswith('bolt://127.0.0.1:'):raise ValueError('not local')
    plan=prepare(graph);started=time.perf_counter()
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            counts=s.execute_write(write_empty,plan)
            s.run('CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS FOR (n:Entity) ON EACH [n.name]').consume()
            s.run('CALL db.awaitIndexes(30)').consume()
    save(f'import-{a.round}.json',dict(build_id=graph['build_id'],counts=counts,seconds=time.perf_counter()-started,
         integrity_unchanged=before==hashes(package),scope='new empty loopback store; no remote mutations'))
    save(f'package-{a.round}-frozen-hashes.json',before)
    print(counts)
if __name__=='__main__':main()
