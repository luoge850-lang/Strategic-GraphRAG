"""Six fixed retrieval arms, same frozen corpus/questions and 10-chunk budget.

Only independently source-labelled positive rows are judged. Unjudged hits
are never called negatives; full Recall and nDCG remain unmeasured.
"""
import json,time,shutil,random,statistics,re,hashlib
from collections import defaultdict
from deployment.acceptance_20261015 import ROOT,OUT,save,hashes
from scripts.run_financial_retrieval_matrix import BM25,_ranked,_aggregate_pages,_select_mode_hits,_page_key,MODES
def main():
    import chromadb
    from chromadb.utils import embedding_functions
    package=OUT/'five-package-round2';before=hashes(package)
    frozen=json.loads((OUT/'package-round2-frozen-hashes.json').read_text())
    if before!=frozen:raise ValueError('candidate integrity mismatch')
    target=OUT/'retrieval-runtime';shutil.copytree(package/'vector_index',target)
    m=json.loads((package/'vector_index_manifest.json').read_text())
    client=chromadb.PersistentClient(path=str(target));collection=client.get_collection(m['collection'],embedding_function=embedding_functions.DefaultEmbeddingFunction())
    raw=collection.get(include=['documents','metadatas'])
    docs=[dict(id=ident,document=raw['documents'][i],metadata=raw['metadatas'][i]) for i,ident in enumerate(raw['ids'])]
    by_page={}
    for item in docs:by_page.setdefault(_page_key(item['metadata'],item['id']),item)
    bm=BM25(docs);edges=defaultdict(list)
    for edge in json.loads((package/'graph.json').read_text())['edges']:edges[f"{edge['source_filing']}#{edge['page']}"].append(edge)
    labels=[x for x in json.loads((OUT/'validation-labels.json').read_text()) if x['operation']!='SAFE_REFUSAL']
    schedule=[]
    for label in labels:
        modes=list(MODES);random.Random(label['item_id']+':20261015').shuffle(modes)
        schedule.extend((label,mode) for mode in modes)
    warm=time.perf_counter();collection.query(query_texts=[labels[0]['question']],n_results=1);warm=time.perf_counter()-warm
    save('retrieval-protocol-freeze.json',dict(build_id=m['build_id'],questions=len(labels),modes=list(MODES),arm_budget=50,final_budget=10,
        runner_sha256=hashlib.sha256(__import__('pathlib').Path(__file__).read_bytes()).hexdigest(),
        baseline_module_sha256=hashlib.sha256((ROOT/'scripts/run_financial_retrieval_matrix.py').read_bytes()).hexdigest(),
        labels_sha256=hashlib.sha256((OUT/'validation-labels.json').read_bytes()).hexdigest(),warmup_seconds=warm,seed='item_id:20261015',
        judgments='positive rows from independent PDF only; all other evidence UNJUDGED',generation=False))
    results=[];embedding_calls=1
    with (OUT/'retrieval-raw.jsonl').open('x',encoding='utf-8') as f:
        for label,mode in schedule:
            start=time.perf_counter();error=None;bm_hits=[];dense=[]
            try:
                if mode!='dense_semantic':bm_hits=_ranked(_aggregate_pages(bm.search(label['question'],50)))[:50]
                if mode!='keyword_bm25':
                    embedding_calls+=1
                    result=collection.query(query_texts=[label['question']],n_results=min(50,len(docs)),include=['documents','metadatas','distances'])
                    chunks=[dict(id=ident,document=result['documents'][0][i],metadata=result['metadatas'][0][i],score=1/(1+max(0,result['distances'][0][i]))) for i,ident in enumerate(result['ids'][0])]
                    dense=_ranked(_aggregate_pages(chunks))[:50]
                hits,plan,status,counts=_select_mode_hits(mode,label['question'],bm_hits,dense,edges,by_page)
            except Exception as exc:
                hits=[];error=f'{type(exc).__name__}: {exc}';status='EXECUTION_FAILURE';counts={};plan=None
            elapsed=time.perf_counter()-start
            references=label.get('required_facts') or [label]
            supported=[];known_page=[]
            for reference in references:
                allowed={(x['file'],x['page']) for x in reference['direct_locations']}
                ranks=[];pranks=[]
                for i,hit in enumerate(hits,1):
                    meta=hit.get('metadata') or {}
                    if (meta.get('source_filing'),meta.get('page')) not in allowed:continue
                    pranks.append(i);text=hit.get('document') or ''
                    if any(anchor.lower() in text.lower() for anchor in reference['row_anchors']) and re.search(r'(?<![\d.])'+re.escape(str(reference['source_value']))+r'(?![\d.])',text.replace(',','')):ranks.append(i)
                supported.append(min(ranks) if ranks else None);known_page.append(min(pranks) if pranks else None)
            record=dict(item_id=label['item_id'],family_id=label['family_id'],method=mode,build_id=m['build_id'],question=label['question'],latency_ms=elapsed*1000,
                status=status,error=error,query_plan=plan,counts=counts,hits=hits,reference_row_ranks=supported,reference_page_ranks=known_page,
                judged_positive_count=len(references),support_covered=sum(x is not None for x in supported),
                reciprocal_rank=1/min([x for x in supported if x is not None]) if any(supported) else 0)
            f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();results.append(record)
    report={}
    for mode in MODES:
        rows=[r for r in results if r['method']==mode];lat=sorted(r['latency_ms'] for r in rows)
        report[mode]=dict(request_success=sum(r['error'] is None for r in rows),requests=len(rows),
            known_positive_row_coverage=dict(numerator=sum(r['support_covered'] for r in rows),denominator=sum(r['judged_positive_count'] for r in rows),unjudged_retrieved_hits=sum(len(r['hits']) for r in rows)-sum(r['support_covered'] for r in rows)),
            mean_known_positive_reciprocal_rank=statistics.mean(r['reciprocal_rank'] for r in rows),
            latency_ms=dict(median=statistics.median(lat),p95=lat[math_ceil(.95*len(lat))-1]),full_recall=None,ndcg=None)
    save('retrieval-summary.json',dict(methods=report,embedding_calls=embedding_calls,embedding_token_count=None,cost=None,llm_calls=0,
        immutable_package_unchanged=before==hashes(package),scope='Finite page/chunk source-positive coverage and reciprocal-rank lower bound; not full relevance judgments. No nDCG or full recall claim.'))
    print(json.dumps(report,indent=2),flush=True)
def math_ceil(x):
    import math
    return math.ceil(x)
if __name__=='__main__':main()
