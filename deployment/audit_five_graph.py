"""Real-store stratified sample with original PDF review packet; no auto Gold."""
import json,random,re
from collections import defaultdict
from deployment.acceptance_20261015 import ROOT,OUT,save
def main():
    import pymupdf
    from neo4j import GraphDatabase
    c=json.loads((OUT/'neo4j-round2/private_credentials.json').read_text())
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            observations=[dict(r['o']) for r in s.run('MATCH (o:FinancialObservation) RETURN properties(o) AS o ORDER BY o.source_filing,o.id')]
            relations=[r.data() for r in s.run("MATCH (a:Company)-[r:DEPENDS_ON_FOUNDRY|EXPOSED_TO]->(b:Entity) MATCH (c:EvidenceClaim {id:r.evidence_id}) RETURN a.id AS source,b.id AS target,type(r) AS relation,properties(c) AS claim,properties(r) AS properties")]
            source_identity=[r.data() for r in s.run('MATCH (d:Document) RETURN properties(d) AS document')]
    grouped=defaultdict(list)
    for o in observations:grouped[o['source_filing']].append(o)
    rng=random.Random(20261015);packet=[]
    graph=json.loads((OUT/'five-package-round2/graph.json').read_text())
    for name,items in sorted(grouped.items()):
        pdf=OUT/'sources'/name if name.startswith('TSMC') else ROOT/'data'/('pdfs' if name.startswith('2025') else 'pdfs_other')/name
        with pymupdf.open(pdf) as doc:
            for o in rng.sample(items,min(8,len(items))):
                evidence=next(e['evidence'] for e in graph['edges'] if e['claim_id']==o['claim_id'])
                page=doc[o['page']-1].get_text()
                packet.append(dict(index=len(packet),observation=o,claim_text=evidence,source_page_text=page,
                    review_grade='PENDING_AI_PDF_REVIEW_NOT_HUMAN_GOLD'))
    save('graph-review-packet.json',dict(seed=20261015,frame='8 observations per each of five primary files, real Neo4j. All retained nonnumeric company relations enumerated separately.',observations=packet,relations=relations,documents=source_identity))
    for item in packet:
        o=item['observation'];print(item['index'],o['source_filing'],o['page'],o['fiscal_period'],o['metric_id'],o['value'],o['unit'],o['statement_type'],repr(item['claim_text'][:220]),flush=True)
    print('RELATIONS',json.dumps(relations,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
