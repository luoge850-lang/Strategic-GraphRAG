"""Read-only failure traces from raw runs and actual Neo4j projection."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from deployment.answer_closure_scoring import score

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--credentials',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.credentials.read_text())
    from neo4j import GraphDatabase
    from strategic_graphrag.schema.financial_observation import FinancialObservation
    rows=[]
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            for r in s.run("MATCH (o:FinancialObservation) WHERE o.metric_id IN ['R_AND_D_RATIO','SG_AND_A_RATIO'] RETURN properties(o) AS observation"):
                obs=dict(r['observation'])
                try:FinancialObservation(**obs);error=None
                except TypeError as exc:error=str(exc)
                rows.append(dict(observation=obs,construction_error=error))
    failures=[]
    for run in ['development-run-v1','development-run-v2','validation-run-v1']:
        path=a.inputs/run/'question_only.jsonl'
        for line in path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);judged=score(r.get('response') or {},r['reference'])
            if judged['core_semantic'] is not True:
                failures.append(dict(run=run,item_id=r['item_id'],question=r['question'],reference=r['reference'],
                    query_plan=r['query_plan'],actual_filter=r['actual_filter'],paths=(r.get('response') or {}).get('paths'),
                    calculation=(r.get('response') or {}).get('calculation'),status=(r.get('response') or {}).get('outcome'),
                    score=judged,root_cause=('Qualified table metric rejected by old entity-presence filter' if run=='development-run-v1' and r['item_id'] in ['D-CF-AR','D-DC'] else
                        'Neo4j omits null currency property; required nullable dataclass constructor argument missing; real adapter skips ratio observation' if r['item_id'] in ['D-RD-PERCENT','D-SGA-PERCENT'] else
                        'income before income tax not planned as PRETAX_INCOME; generic income resolves NET_INCOME' if 'PRETAX' in r['item_id'] else
                        'divided by operation not recognized; unsupported ratio incorrectly becomes FACT' if r['item_id']=='H-REFUSE-computed-ratio' else
                        'Evidence or metric planning incomplete; see exact raw trace, not inferred pass'),
                    fix_status='NO_POST_FREEZE_FIX; development two-round limit reached; preserve failure'))
    result=dict(real_nullable_observation_diagnostics=rows,failures=failures,
        scope='AI/PDF diagnosis and actual-store reads only; no database or validation-label mutation',production_mutations=0)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(dict(failures=len(failures),ratio_records=len(rows),construction_errors=sum(bool(r['construction_error']) for r in rows))))

if __name__=='__main__':main()
