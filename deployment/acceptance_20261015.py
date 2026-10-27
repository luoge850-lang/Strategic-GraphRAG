"""Append-only evidence collection for the October 15 acceptance effort."""
import hashlib, json, os, platform, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/acceptance-20261015'
def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/name).open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
def hashes(base):
    return {str(p.relative_to(base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(base.rglob('*')) if p.is_file()}
def initial():
    import psutil
    from deployment.live_acceptance import http
    from neo4j import GraphDatabase
    candidate=ROOT/'reports/isolated_staging/build_ff4e5bcf11c71930'
    original=Path(os.environ.get('GRAPHRAG_ORIGINAL_WORKSPACE', str(Path.home()/'OneDrive/Desktop/Nvidia-GraphRAG-Engine')))
    stores=original/'reports/local_neo4j_trial_20260930/state_answer_closure_20261003_v2'
    c=json.loads((stores/'private_credentials.json').read_text())
    workspaces=[]
    for p in [ROOT,original]:
        workspaces.append(dict(path=str(p),commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),status=subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True)))
    with GraphDatabase.driver(c['uri'],auth=(c['username'],c['password'])) as d:
        with d.session(database=c['database']) as s:
            inventory=[r.data() for r in s.run('MATCH (n) RETURN labels(n) AS labels, n.build_id AS build_id, count(*) AS count')]
            ratios=[r.data() for r in s.run("MATCH (n:FinancialObservation) WHERE n.metric_id ENDS WITH '_RATIO' RETURN properties(n) AS observation")]
    import chromadb
    client=chromadb.PersistentClient(path=str(ROOT/'reports/answer-closure-runtime-v2/data/chroma_db'))
    vectors=[dict(name=x.name,count=x.count()) for x in client.list_collections()]
    save('initial-inventory.json',dict(workspaces=workspaces,neo4j_uri=c['uri'],neo4j=inventory,ratios=ratios,chroma=vectors,
        hardware=dict(platform=platform.platform(),cpus=os.cpu_count(),ram=psutil.virtual_memory().total),
        processes=[dict(pid=p.pid,name=p.info['name']) for p in psutil.process_iter(['name']) if p.info['name'] in ['python.exe','java.exe','node.exe']],
        budget=dict(concurrency=1,memory_limit_bytes=8*1024**3,documents=5,cost=None,llm='disabled; no provider configured')))
    save('old-package-before.json',hashes(candidate))
    qs=["What was NVIDIA income before income tax for fiscal 2025 in its 2025 10-K?",
        "What was NVIDIA pretax profit for fiscal 2025 in its 2025 10-K?",
        "Calculate NVIDIA net income divided by revenue for fiscal 2025 in its 2025 10-K.",
        "What percentage of revenue were research and development expenses for NVIDIA fiscal 2022 in its 2023 10-K?",
        "What was NVIDIA revenue in Q4 FY2025?", "What was AMD revenue in FY2025?"]
    rows=[]
    for q in qs:
        request=dict(question=q,max_paths=10,retrieval_mode='graph',synthesize=False,use_cache=False)
        result=http('http://127.0.0.1:8000','/query',request); result.pop('_bytes',None)
        rows.append(dict(question=q,partition='development_exposed',request=request,result=result))
    save('development-before.json',rows)
if __name__=='__main__': initial()
