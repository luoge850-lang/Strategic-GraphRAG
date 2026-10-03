"""Supplemental source-fixed percentage and annual/quarter boundary diagnostics."""
import argparse,hashlib,json
from pathlib import Path
from live_acceptance import http
from complete_fact_scoring import score_fact

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--base',default='http://127.0.0.1:8000');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for ident,metric,value,year,anchor in [('P01','R_AND_D_RATIO',19.6,2022,'research and development'),('P02','SG_AND_A_RATIO',9.1,2023,'sales, general and administrative')]:
        label=dict(item_id=ident,company='NVIDIA_CORPORATION',metric=metric,fact_period=f'FY{year}',source_filing='2023-10-K.pdf',
          currency=None,source_scale=None,source_value=value,answer_value=value,answer_unit='percent',tolerance='0.05',
          tolerance_basis='half of 0.1 printed percentage point',direct_locations=[dict(file='2023-10-K.pdf',page=43,statement='MD_AND_A')],
          row_anchors=[anchor],label_tier='AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW')
        cases.append(dict(item_id=ident,question=f'What percentage of revenue were {anchor} expenses for NVIDIA fiscal year {year} in its 2023 10-K?',label=label))
    cases.append(dict(item_id='Q01',question='What was NVIDIA revenue in fiscal 2024 Q4 as reported in its 2024 10-K?',label=None,
      rubric='No independently checked quarterly source reference established. UNSCORED, never count a matching annual number as quarterly correctness.'))
    with (a.output/'frozen_cases.json').open('x',encoding='utf-8') as f:json.dump(cases,f,indent=2)
    digest=hashlib.sha256((a.output/'frozen_cases.json').read_bytes()).hexdigest()
    with (a.output/'requests.jsonl').open('x',encoding='utf-8') as f:
        for mode in ['natural_language','user_scope','reference_scope_diagnostic']:
            for case in cases:
                request=dict(question=case['question'],max_paths=10,retrieval_mode='graph',synthesize=False,use_cache=False)
                if mode=='user_scope':request.update(source_filing='2025-10-K.pdf',cross_filing=False)
                if mode=='reference_scope_diagnostic':request.update(source_filing='2024-10-K.pdf' if case['item_id']=='Q01' else '2023-10-K.pdf',cross_filing=False)
                result=http(a.base,'/query',request);result.pop('_bytes',None)
                response=result.get('response') or {}
                row=dict(item_id=case['item_id'],mode=mode,request=request,label_sha256=digest,build_id='build_c3b1d950e8acce78',
                    score=score_fact(response,case['label']) if case['label'] else None,**result)
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
    print('RECORDED 9 supplemental requests; no application tuning')
if __name__=='__main__':main()
