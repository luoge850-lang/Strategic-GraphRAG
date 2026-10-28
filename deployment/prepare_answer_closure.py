"""Freeze source-authored closure inputs before any new HTTP evaluation.

The catalogue below was transcribed from the PDFs, not graph predictions.
This is AI/PDF engineering diagnosis, not human Gold. Operations define
families together with metric and measurement identity; metric overlap with
previous development is explicitly retained, not claimed as unseen domains.
"""
import argparse, hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# FY2025 / FY2024 printed values in the FY2025 filing, physical pages 52,54,80.
CATALOGUE = [
    ('COST_OF_REVENUE', 'cost of revenue', 32639, 16621, 52, 'INCOME_STATEMENT'),
    ('GROSS_PROFIT', 'gross profit', 97858, 44301, 52, 'INCOME_STATEMENT'),
    ('NET_INCOME', 'net income', 72880, 29760, 52, 'INCOME_STATEMENT'),
    ('PRETAX_INCOME', 'income before income tax', 84026, 33818, 52, 'INCOME_STATEMENT'),
    ('INCOME_TAX_EXPENSE', 'income tax expense', 11146, 4058, 52, 'INCOME_STATEMENT'),
    ('TOTAL_CURRENT_LIABILITIES', 'total current liabilities', 18047, 10631, 54, 'BALANCE_SHEET'),
    ('TOTAL_CURRENT_ASSETS', 'total current assets', 80126, 44345, 54, 'BALANCE_SHEET'),
    ('TOTAL_ASSETS', 'total assets', 111601, 65728, 54, 'BALANCE_SHEET'),
    ('TOTAL_LIABILITIES', 'total liabilities', 32274, 22750, 54, 'BALANCE_SHEET'),
    ('TOTAL_SHAREHOLDERS_EQUITY', "total shareholders' equity", 79327, 42978, 54, 'BALANCE_SHEET'),
    ('CASH_AND_CASH_EQUIVALENTS', 'cash and cash equivalents', 8589, 7280, 54, 'BALANCE_SHEET'),
    ('MARKETABLE_SECURITIES', 'marketable securities', 34621, 18704, 54, 'BALANCE_SHEET'),
    ('INVENTORIES', 'inventories', 10080, 5282, 54, 'BALANCE_SHEET'),
    ('GAMING_REVENUE', 'Gaming', 11350, 10447, 80, 'FINANCIAL_NOTES'),
    ('AUTOMOTIVE_REVENUE', 'Automotive', 1694, 1091, 80, 'FINANCIAL_NOTES'),
    ('PROFESSIONAL_VISUALIZATION_REVENUE', 'Professional Visualization', 1878, 1553, 80, 'FINANCIAL_NOTES'),
]

def save(path, value):
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)

def fact(metric, anchor, value, year, page, statement, filing='2025-10-K.pdf'):
    from strategic_graphrag.schema.financial_observation import measurement_identity
    return dict(company='NVIDIA_CORPORATION', metric=metric, fact_period=f'FY{year}',
        source_filing=filing, currency='USD', source_scale='millions', source_value=value,
        answer_value=value, answer_unit='USD millions', tolerance='0.5',
        tolerance_basis='half of one printed million; source precision, not prediction error',
        direct_locations=[dict(file=filing,page=page,statement=statement)],
        row_anchors=[anchor], label_tier='AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW',
        human_review='NOT_PERFORMED', independent_human_review='NOT_PERFORMED',
        **measurement_identity(metric,f'FY{year}','USD millions'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    from strategic_graphrag.schema.financial_observation import measurement_identity
    root=a.output; root.mkdir(parents=True,exist_ok=True)
    old=ROOT/'experiments/service-validity-2026-10-02'
    raw=[json.loads(x) for x in (old/'run-three-modes-v2/natural_language.jsonl').read_text(encoding='utf-8').splitlines()]
    questions={r['item_id']:r['question'] for r in raw}
    dev=[]
    for label in json.loads((old/'development_fact_labels.json').read_text(encoding='utf-8')):
        label.update(question=questions[label['item_id']],partition='exposed_development',operation='FACT',
                     **measurement_identity(label['metric'],label['fact_period'],label['answer_unit']))
        dev.append(label)
    failures=[
        ('D-CF-AR','What cash-flow adjustment for accounts receivable did NVIDIA report for fiscal 2022 in its 2023 10-K?',
         fact('CASH_FLOW_CHANGE_ACCOUNTS_RECEIVABLE','Accounts receivable',-2215,2022,58,'CASH_FLOW_STATEMENT','2023-10-K.pdf')),
        ('D-DC','What was NVIDIA Data Center revenue in fiscal 2025 as disclosed in its 2025 10-K?',
         fact('DATA_CENTER_REVENUE','Data Center',115186,2025,80,'FINANCIAL_NOTES')),
        ('D-AP','What accounts payable balance did NVIDIA report for fiscal 2022 in its 2023 10-K?',
         fact('ACCOUNTS_PAYABLE','Accounts payable',1783,2022,56,'BALANCE_SHEET','2023-10-K.pdf')),
    ]
    for item,q,label in failures:
        dev.append(dict(label,item_id=item,family_id=item,question=q,partition='exposed_development',operation='FACT'))
    for item,metric,q,value in [
        ('D-RD-PERCENT','R_AND_D_RATIO','What percentage of revenue were research and development expenses for NVIDIA fiscal 2022 in its 2023 10-K?',19.6),
        ('D-SGA-PERCENT','SG_AND_A_RATIO','What percentage of revenue were sales, general and administrative expenses for NVIDIA fiscal 2023 in its 2023 10-K?',9.1)]:
        year=2022 if item=='D-RD-PERCENT' else 2023
        label=fact(metric,'Research and development' if year==2022 else 'Sales, general and administrative',value,year,43,'MD_AND_A','2023-10-K.pdf')
        label.update(currency=None,source_scale='percent',answer_unit='percent',tolerance='0.05',
                     tolerance_basis='half of one printed tenth of a percentage point',
                     **measurement_identity(metric,f'FY{year}','percent'))
        dev.append(dict(label,item_id=item,family_id=item,question=q,partition='exposed_development',operation='FACT'))
    for i,wording in enumerate(['Q4 FY2024','FY2024 Q4','fiscal 2024 Q4','fourth quarter of fiscal 2024']):
        for j,suffix in enumerate(['',' as disclosed in its 2025 10-K']):
            dev.append(dict(item_id=f'D-Q-{i}-{j}',family_id='D-quarter-paraphrases',
                question=f'What was NVIDIA revenue in {wording}{suffix}?',partition='exposed_development',
                operation='SAFE_REFUSAL',source_filing='2025-10-K.pdf',
                accepted_statuses=['INSUFFICIENT_EVIDENCE','AMBIGUOUS','OPERATION_UNSUPPORTED'],
                refusal_basis='annual-only extracted fact scope; never substitute annual for quarter',label_tier='AI_PDF_CAPABILITY_BOUNDARY'))
    validation=[]
    for metric,anchor,current,prior,page,statement in CATALOGUE:
        scope=anchor+' revenue' if metric.endswith('_REVENUE') else anchor
        label=fact(metric,anchor,current,2025,page,statement)
        validation.append(dict(label,item_id='H-FACT-'+metric,family_id='FACT:'+metric,
            question=f'What was NVIDIA {scope} for fiscal 2025, as disclosed in the 2025 10-K?',operation='FACT'))
    for metric,anchor,current,prior,page,statement in CATALOGUE[:10]:
        label=fact(metric,anchor,current,2025,page,statement)
        label.update(answer_value=current/1000,answer_unit='USD billions',tolerance='0.0005',
                     tolerance_basis='printed-million rounding propagated through exact division by 1000')
        validation.append(dict(label,item_id='H-CONVERT-'+metric,family_id='UNIT_CONVERSION:'+metric,
            question=f'Convert NVIDIA {anchor} for fiscal 2025, reported in the 2025 10-K, from USD millions to USD billions.',operation='UNIT_CONVERSION'))
    for metric,anchor,current,prior,page,statement in CATALOGUE[:7]:
        label=fact(metric,anchor,current,2025,page,statement)
        label.update(answer_value=current-prior,tolerance='1',
                     tolerance_basis='sum of two printed-million rounding bounds',
                     required_facts=[fact(metric,anchor,prior,2024,page,statement),fact(metric,anchor,current,2025,page,statement)])
        validation.append(dict(label,item_id='H-DELTA-'+metric,family_id='ABSOLUTE_CHANGE:'+metric,
            question=f'Calculate the absolute difference in NVIDIA {anchor} between fiscal 2024 and fiscal 2025 using its 2025 10-K.',operation='ABSOLUTE_CHANGE'))
    rejected=[
        ('quarter-gaming','What was NVIDIA Gaming revenue in Q1 FY2025 as reported in its 2025 10-K?'),
        ('quarter-net-income','What was NVIDIA net income in second quarter of fiscal 2025 as reported in its 2025 10-K?'),
        ('future-assets','What were NVIDIA total assets in fiscal 2026?'),
        ('unknown-region','What was NVIDIA regional revenue in fiscal 2025?'),
        ('missing-year-inventory','Compare NVIDIA inventories between years.'),
        ('computed-ratio','Calculate NVIDIA research and development expenses divided by revenue in fiscal 2025.'),
        ('percentage-point','What was NVIDIA gross margin percentage-point change between fiscal 2024 and fiscal 2025?'),
    ]
    for name,q in rejected:
        validation.append(dict(item_id='H-REFUSE-'+name,family_id='SAFE_REFUSAL:'+name,
            question=q,source_filing='2025-10-K.pdf',operation='SAFE_REFUSAL',
            accepted_statuses=['INSUFFICIENT_EVIDENCE','AMBIGUOUS','OPERATION_UNSUPPORTED'],
            refusal_basis='declared capability boundary or unresolved request; no broader fact may PASS',
            label_tier='AI_PDF_CAPABILITY_BOUNDARY'))
    for label in validation:label['partition']='frozen_unexposed_forms_no_tuning'
    assert len(validation)==40 and len({l['family_id'] for l in validation})==40
    save(root/'development_labels.json',dev);save(root/'validation_labels.json',validation)
    sources={}
    for name in ['2023-10-K.pdf','2024-10-K.pdf','2025-10-K.pdf']:
        path=ROOT/'data'/('pdfs' if name.startswith('2025') else 'pdfs_other')/name
        sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    save(root/'source_manifest.json',dict(pdfs=sources,catalogue=CATALOGUE,
        source_method='PDF physical-page text read before HTTP; catalogue independently transcribed; no prediction/graph export',
        families='metric + operation + measurement identity, not paraphrase count',
        independence_limit='Adjacent metrics overlap previous development. Frozen unexposed forms are engineering holdout, not entirely unseen metric families or human Gold.'))
    names=['protocol.json','development_labels.json','validation_labels.json','source_manifest.json']
    save(root/'frozen_inputs_sha256.json',{n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names})
    print(json.dumps(dict(development_questions=len(dev),validation_questions=len(validation),families=40)))

if __name__=='__main__':main()
