"""Source-authored unseen-family coverage probe, frozen before requests.

Purposive expansion into TSMC balance/cash-flow families. This is not a random
sample of all user questions, and cannot estimate corpus-wide accuracy.
"""
import argparse,hashlib,json,re
from deployment.optimize_20261015 import OUT,PREVIOUS,read,save
from deployment.live_acceptance import http
from deployment.score_five_acceptance import metric

def freeze():
    rows=[]
    values=[('total current assets','TOTAL_CURRENT_ASSETS',3088352120,2194032910),
      ('total assets','TOTAL_ASSETS',6691938000,5532371215),
      ('total current liabilities','TOTAL_CURRENT_LIABILITIES',1264524964,913583316),
      ('total liabilities','TOTAL_LIABILITIES',2368362135,2049108368),
      ('retained earnings','RETAINED_EARNINGS',3917252023,3158030792),
      ('total equity','TOTAL_EQUITY',4323575865,3483262847)]
    for alias,mid,a,b in values:
        for y,v in [(2024,a),(2023,b)]:
            rows.append(dict(item_id=f'H-{mid}-{y}',family_id=f'TSMC-{mid}',question=f'What were TSMC consolidated {alias} at the end of fiscal {y} as disclosed in its 2024 annual report?',company_id='TSMC',metric_id=mid,year=y,value=v,page=63,nature='BALANCE'))
    for y,v in [(2024,2127627043),(2023,1465427753)]:
        rows.append(dict(item_id=f'H-CASH-{y}',family_id='TSMC-CASH-BALANCE',question=f'What was TSMC consolidated cash balance at December 31, {y}, in its 2024 annual report?',company_id='TSMC',metric_id='CASH_AND_CASH_EQUIVALENTS',year=y,value=v,page=65,nature='BALANCE'))
    for alias,mid,v in [('net cash provided by operating activities','OPERATING_CASH_FLOW',1826177068),('net cash used in investing activities','INVESTING_CASH_FLOW',-864842769),('net cash used in financing activities','FINANCING_CASH_FLOW',-346300910)]:
        rows.append(dict(item_id=f'H-{mid}',family_id=f'TSMC-{mid}',question=f'What was TSMC consolidated {alias} in fiscal 2024 in its 2024 annual report? Use the signed cash-flow amount.',company_id='TSMC',metric_id=mid,year=2024,value=v,page=65,nature='PERIOD_MOVEMENT'))
    for r in rows:r.update(source_filing='TSMC-2024-Annual-Report.pdf',unit='TWD thousands',business_scope='CONSOLIDATED',review_grade='AI_PDF_VISUAL_NOT_HUMAN_GOLD')
    save('new-family-labels.json',rows)
    save('new-family-freeze.json',dict(label_sha256=hashlib.sha256((OUT/'new-family-labels.json').read_bytes()).hexdigest(),pdf_sha256=hashlib.sha256((PREVIOUS/'sources/TSMC-2024-Annual-Report.pdf').read_bytes()).hexdigest(),families=10,questions=len(rows),source_physical_pages=[63,65],rendered_source_review=True,production_frozen_before_labels=True,scope='Purposive unseen TSMC balance and cash-flow family coverage probe; not a representative estimate. No tuning after this run.'))

def run():
    freeze=read(OUT/'new-family-freeze.json')
    assert hashlib.sha256((OUT/'new-family-labels.json').read_bytes()).hexdigest()==freeze['label_sha256']
    rows=[]
    for label in read(OUT/'new-family-labels.json'):
        request=dict(question=label['question'],max_paths=10,retrieval_mode='graph',synthesize=False,use_cache=False)
        result=http('http://127.0.0.1:8004','/query',request);result.pop('_bytes',None)
        response=result.get('response') or {};c=response.get('calculation') or {};obs=c.get('observations') or []
        semantic=c.get('status')=='PASS' and c.get('operation')=='fact' and any(o.get('company_id')==label['company_id'] and o.get('metric_id')==label['metric_id'] and o.get('fact_period')==f"FY{label['year']}" and o.get('business_scope')=='CONSOLIDATED' for o in obs)
        numeric=isinstance(c.get('value'),(int,float)) and abs(c['value']-label['value'])<.5 and c.get('unit')==label['unit']
        location=any(x.get('page')==label['page'] and x.get('source_filing')==label['source_filing'] for x in response.get('citations',[])) if c.get('status')=='PASS' else None
        scores=dict(core_semantic=semantic,numeric_and_unit=numeric,joint=semantic and numeric and location is True,physical_page=location,false_refusal=c.get('status') in {'INSUFFICIENT_EVIDENCE','AMBIGUOUS','OPERATION_UNSUPPORTED','NOT_APPLICABLE'},severe_wrong_pass=c.get('status')=='PASS' and not semantic,request_success=result.get('http_status')==200 and response.get('execution_status')=='SUCCEEDED')
        rows.append(dict(label=label,request=request,result=result,score=scores));print(label['item_id'],c.get('status'),c.get('value'),flush=True)
    save('new-family-raw-scored.json',rows)
    save('new-family-summary.json',dict(metrics={k:metric([r['score'][k] for r in rows]) for k in rows[0]['score']},questions=len(rows),families=10,human_gold=0,review='AI_PDF_NOT_HUMAN_GOLD',accuracy_gate='PASS' if sum(r['score']['core_semantic'] for r in rows)/len(rows)>=.9 else 'FAIL',strata={str(p):dict(correct=sum(r['score']['core_semantic'] for r in rows if r['label']['page']==p),total=sum(r['label']['page']==p for r in rows)) for p in [63,65]}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','run']);a=p.parse_args()
    {'freeze':freeze,'run':run}[a.phase]()
