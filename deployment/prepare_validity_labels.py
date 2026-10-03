"""Publish the fixed, PDF-authored labels and protocol before querying new families."""
import argparse,hashlib,json
from pathlib import Path

TIER='AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW'
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    # Values transcribed from original physical PDF pages, not engine output.
    dev=[
      ('FQA-459504a0c8aab868b348','R_AND_D_EXPENSE',5268,2022,2023,43,'MD_AND_A','research and development'),
      ('FQA-841546b4d92198e1bb46','MARKETABLE_SECURITIES',19218,2022,2023,56,'BALANCE_SHEET','marketable securities'),
      ('FQA-9ebad1d7a48a48c4c012','TOTAL_CURRENT_ASSETS',28829,2022,2023,56,'BALANCE_SHEET','total current assets'),
      ('FQA-e4a8908ea7e94a8710a1','ACCOUNTS_RECEIVABLE',4650,2022,2023,56,'BALANCE_SHEET','accounts receivable'),
      ('FQA-DEV-d687185a6ce263bdbd63','REVENUE',60922,2024,2024,79,'FINANCIAL_NOTES','total revenue'),
      ('FQA-DEV-3c57ba0a364db4d47d0b','TOTAL_CURRENT_ASSETS',44345,2024,2024,52,'BALANCE_SHEET','total current assets'),
      ('FQA-DEV-2ad79e78ab860ac3675c','REVENUE',130497,2025,2025,80,'FINANCIAL_NOTES','total revenue'),
      ('FQAV-DEV-USD-BILLIONS-2025-REVENUE','REVENUE',130497,2025,2025,80,'FINANCIAL_NOTES','total revenue')]
    outside=[
      ('V01','TOTAL_ASSETS',44187,2022,2023,56,'BALANCE_SHEET','total assets'),
      ('V02','INVENTORIES',2605,2022,2023,56,'BALANCE_SHEET','inventories'),
      ('V03','ACCOUNTS_PAYABLE',1783,2022,2023,56,'BALANCE_SHEET','accounts payable'),
      ('V04','OPERATING_COST',7434,2022,2023,43,'MD_AND_A','total operating expenses'),
      ('V05','SALES_GENERAL_ADMIN_EXPENSE',2166,2022,2023,43,'MD_AND_A','sales, general and administrative'),
      ('V06','R_AND_D_EXPENSE',8675,2024,2024,50,'INCOME_STATEMENT','research and development'),
      ('V07','NET_INCOME',29760,2024,2024,50,'INCOME_STATEMENT','net income'),
      ('V08','TOTAL_ASSETS',65728,2024,2024,52,'BALANCE_SHEET','total assets'),
      ('V09','ACCOUNTS_RECEIVABLE',9999,2024,2024,52,'BALANCE_SHEET','accounts receivable'),
      ('V10','REVENUE',26974,2023,2024,50,'INCOME_STATEMENT','revenue'),
      ('V11','CASH_FLOW_ACCOUNTS_RECEIVABLE',-2215,2022,2023,58,'CASH_FLOW_STATEMENT','accounts receivable'),
      ('V12','DATA_CENTER_REVENUE',115186,2025,2025,80,'FINANCIAL_NOTES','data center')]
    def make(row):
        ident,metric,value,year,filing,page,statement,anchor=row
        return dict(item_id=ident,family_id=ident,company='NVIDIA_CORPORATION',metric=metric,
            fact_period=f'FY{year}',source_filing=f'{filing}-10-K.pdf',currency='USD',source_scale='millions',
            source_value=value,answer_value=value,answer_unit='USD millions',tolerance='0.5',
            tolerance_basis='half of one printed million; fixed before prediction',
            direct_locations=[dict(file=f'{filing}-10-K.pdf',page=page,statement=statement)],
            row_anchors=[anchor],label_tier=TIER,human_review='NOT_PERFORMED')
    labels=[make(r) for r in dev]
    labels[-1].update(answer_value=130.497,answer_unit='USD billions',tolerance='0.0005',
                     tolerance_basis='printed million half-unit divided by 1000; exact scale conversion')
    labels[4]['direct_locations'].append(dict(file='2024-10-K.pdf',page=50,statement='INCOME_STATEMENT'))
    for label in labels[6:]:
        label['direct_locations'].append(dict(file='2025-10-K.pdf',page=52,statement='INCOME_STATEMENT'))
    valid=[make(r) for r in outside]
    question_names=['total assets','inventories balance','accounts payable balance','total operating expenses',
        'sales, general and administrative expenses','research and development expense','net income',
        'total assets','accounts receivable balance','total revenue',
        'cash-flow adjustment for accounts receivable','Data Center revenue']
    for label,name in zip(valid,question_names):
        label['question']=f"What {name} did NVIDIA report for fiscal year {label['fact_period'][2:]} in its {label['source_filing'][:4]} 10-K?"
        label['partition']='frozen_outside_development_no_tuning'
    protocol=dict(schema='service-validity/v1',build_id='build_c3b1d950e8acce78',
        modes=['natural_language','user_scope','reference_scope_diagnostic'],
        user_scope=dict(source_filing='2025-10-K.pdf',provenance='predeclared benchmark file-selector input, not actual human selection; independent of references'),
        natural_language='question only; transport flags graph/max_paths=10/synthesize=false/use_cache=false; no scope fields',
        reference_scope_diagnostic='candidate_filing from historical development data, or source label for new validation; upper bound only',
        route='graph',generation=False,cache=False,evidence_budget=10,
        stop_condition='one frozen run per mode; no application repair or tuning on new validation results',
        semantic_rules={'comparison':'all required periods and operations must match source labels; absent complete labels remains UNSCORED',
          'relation':'direct citation must express requested relationship; semantic adjudication required, not automatic PASS',
          'conditional_risk':'preserve modal possibility and condition; cannot convert to realized causal claim; adjudication required',
          'unanswerable':'refusal plus no asserted answer; corpus absence requires independently bounded source review',
          'ambiguity':'explicit ambiguity/refusal instead of arbitrary observation selection; adjudication required'},
        outside_coverage='12 adjacent monetary facts; balance vs movement, total vs components and disclosure covered; percentage and quarterly references not established in this set',
        research_hypothesis=None,stability_claim=False)
    for name,payload in [('development_fact_labels.json',labels),('validation_fact_labels.json',valid),('protocol.json',protocol)]:
        with (a.output/name).open('x',encoding='utf-8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.glob('*.json')}
    with (a.output/'frozen_inputs_sha256.json').open('x') as f:json.dump(hashes,f,indent=2)
    print(json.dumps(hashes))

if __name__=='__main__':main()
