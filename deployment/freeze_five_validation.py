"""Source-authored holdout, AI/PDF review only. Never derives truth from graph."""
import json,hashlib
from deployment.acceptance_20261015 import ROOT,OUT,save
from strategic_graphrag.schema.financial_observation import measurement_identity

# Independently read consolidated (not parent-only) performance tables:
# TSMC 2023 physical p67, 2024 physical p64. Values are TWD thousands.
CATALOGUE=[
 ('COST_OF_REVENUE','cost of revenue','Cost of Revenue',1269954135,986625213,915536486),
 ('GROSS_PROFIT','gross profit','Gross Profit',1624353564,1175110628,1348354806),
 ('OPERATING_COST','operating expenses','Operating Expenses',301070315,253833716,226707552),
 ('OPERATING_INCOME','operating income','Income from Operations',1322053050,921465606,1121278851),
 ('PRETAX_INCOME','income before income tax','Income before Income Tax',1405838635,979171324,1144190718),
 ('INCOME_TAX_EXPENSE','income tax expense','Income Tax Expenses',233406876,141403807,127290203),
 ('NET_INCOME','net income','Net Income',1172431759,837767517,1016900515)]
def fact(metric,anchor,value,year,filing_year):
    filing=f'TSMC-{filing_year}-Annual-Report.pdf'
    return dict(company='TSMC',metric=metric,fact_period=f'FY{year}',source_filing=filing,
       currency='TWD',source_scale='thousands',source_value=value,answer_value=value,answer_unit='TWD thousands',
       tolerance='0.5',tolerance_basis='half of one printed thousand',row_anchors=[anchor],
       direct_locations=[dict(file=filing,page=64 if filing_year==2024 else 67,statement='INCOME_STATEMENT')],
       label_tier='AI_PDF_SOURCE_REVIEW_NOT_HUMAN_GOLD',human_review='NOT_PERFORMED',
       **measurement_identity(metric,f'FY{year}','TWD thousands'))
def main():
    labels=[]
    for metric,name,anchor,v24,v23,v22 in CATALOGUE:
        for year,filing,value in [(2024,2024,v24),(2023,2024,v23),(2023,2023,v23),(2022,2023,v22)]:
            labels.append(dict(fact(metric,anchor,value,year,filing),item_id=f'V-{metric}-{year}-{filing}',family_id=f'TSMC:{metric}',operation='FACT',
                 question=f'What was TSMC consolidated {name} for fiscal {year}, as disclosed in its {filing} annual report?'))
        base=fact(metric,anchor,v24,2024,2024)
        labels.append(dict(base,item_id=f'V-CONVERT-{metric}',family_id=f'TSMC:{metric}',operation='UNIT_CONVERSION',
            question=f'Convert TSMC consolidated {name} for fiscal 2024 in its 2024 annual report from TWD thousands to TWD millions.',
            answer_value=v24/1000,answer_unit='TWD millions',tolerance='0.0005'))
        labels.append(dict(base,item_id=f'V-DELTA-{metric}',family_id=f'TSMC:{metric}',operation='ABSOLUTE_CHANGE',
            question=f'Calculate the absolute difference in TSMC consolidated {name} between fiscal 2023 and fiscal 2024 using its 2024 annual report.',
            answer_value=v24-v23,tolerance='1',required_facts=[fact(metric,anchor,v23,2023,2024),base]))
    safety=[
       ('future','What was TSMC consolidated revenue in fiscal 2030?','future period absent'),
       ('wrong-company','What was Apple consolidated revenue in fiscal 2024?','company absent from primary corpus'),
       ('undisclosed-scope','What was TSMC NVIDIA customer revenue in fiscal 2024?','named customer revenue not established'),
       ('fx','Convert TSMC fiscal 2024 revenue in its 2024 annual report from TWD thousands to USD millions.','no authorized exchange-rate protocol'),
       ('cross-currency','Compare NVIDIA and TSMC consolidated revenue for fiscal 2024.','different currencies and fiscal dates; no direct comparable aggregate'),
       ('cross-definition','Compare NVIDIA gross profit and TSMC gross margin for fiscal 2024.','different metric definitions, amount versus ratio'),
       ('reverse','Did NVIDIA manufacture semiconductor wafers for TSMC in fiscal 2025?','reverse of disclosed foundry direction'),
       ('missing-chain','How did TSMC supply disruption cause Apple revenue losses in fiscal 2024?','Apple financial/loss intermediate evidence absent'),
       ('unreported-loss','How many USD millions did NVIDIA lose because of TSMC earthquakes in fiscal 2025?','no source supports NVIDIA-specific realized loss amount'),
       ('time-mismatch','Did the 2025 NVIDIA 10-K prove that TSMC supplied NVIDIA in fiscal 2030?','disclosure cannot establish future realized relationship'),
    ]
    for key,q,basis in safety:
        labels.append(dict(item_id='V-SAFE-'+key,family_id='SAFE:'+key,operation='SAFE_REFUSAL',question=q,
            accepted_statuses=['INSUFFICIENT_EVIDENCE','AMBIGUOUS','OPERATION_UNSUPPORTED','NOT_APPLICABLE'],
            refusal_basis=basis,label_tier='AI_PDF_CAPABILITY_BOUNDARY_NOT_HUMAN_GOLD',source_filing='ALL'))
    for row in labels:row['partition']='frozen_validation'
    save('validation-labels.json',labels)
    inputs={str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [OUT/'validation-labels.json',OUT/'sources/TSMC-2023-Annual-Report.pdf',OUT/'sources/TSMC-2024-Annual-Report.pdf']}
    save('validation-freeze.json',dict(hashes=inputs,questions=len(labels),financial_questions=42,safety_questions=10,
        family_split='All historically exposed NVIDIA questions are development; TSMC revenue is development. Seven other TSMC metric families, all paraphrases, years and operations held out together.',
        limits='52 supported questions, not padded to 60. Two-page consolidated table frame; not general table or relation accuracy. AI/PDF review, no human Gold. No tuning after validation.'))
if __name__=='__main__':main()
