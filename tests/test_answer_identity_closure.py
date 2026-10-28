from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.schema.financial_observation import measurement_identity
import pytest

@pytest.mark.parametrize('wording',['Q4 FY2024','FY2024 Q4','fiscal 2024 Q4','fourth quarter of fiscal 2024'])
def test_quarter_scope_never_drops(wording):
    for suffix in ['', ' as disclosed in its 2025 10-K']:
        plan=parse_query('What was NVIDIA revenue in '+wording+suffix+'?').to_dict()
        assert plan['fact_period']=='Q4 FY2024'
        assert plan['period_granularity']=='QUARTERLY'
        if suffix:assert plan['document_scope']=='2025-10-K.pdf'

@pytest.mark.parametrize('question,metric',[
 ('What accounts payable balance did NVIDIA report for fiscal 2022 in its 2023 10-K?','ACCOUNTS_PAYABLE'),
 ('What cash-flow adjustment for accounts receivable did NVIDIA report for fiscal 2022?','CASH_FLOW_CHANGE_ACCOUNTS_RECEIVABLE'),
 ('What was NVIDIA Data Center revenue for fiscal 2025?','DATA_CENTER_REVENUE'),
 ('What percentage of revenue were research and development expenses for NVIDIA fiscal 2022?','R_AND_D_RATIO'),
 ('What percentage of revenue were sales, general and administrative expenses for NVIDIA fiscal 2023?','SG_AND_A_RATIO')])
def test_metric_semantics(question,metric):
    q=parse_query(question);assert q.target_metric==metric;assert not q.ambiguity;assert q.calculation is None

def test_computed_ratio_is_not_disclosed_lookup():
    assert parse_query('Calculate research and development expense as a percentage of NVIDIA revenue in FY2024').calculation=='RATIO'
    assert parse_query('What was NVIDIA revenue growth from FY2023 to FY2025?').calculation=='PERCENT_CHANGE'
    assert parse_query('What was NVIDIA gross margin percentage-point change between FY2023 and FY2024?').calculation=='PERCENTAGE_POINT_CHANGE'

def test_unknown_scope_refuses_and_annual_still_resolves():
    assert parse_query('What is NVIDIA regional revenue in FY2025?').ambiguity
    q=parse_query('What was NVIDIA revenue in fiscal 2025 as disclosed in its 2025 10-K?')
    assert q.fact_period=='FY2025' and not q.ambiguity
    assert measurement_identity('CASH_FLOW_CHANGE_ACCOUNTS_PAYABLE','FY2022','USD millions')['measurement_nature']=='PERIOD_MOVEMENT'

def test_disclosed_child_ratio_keeps_parent_and_two_periods():
    from strategic_graphrag.pipeline.financial_table_extractor import extract_financial_table_triples
    class Page:
        def extract_tables(self):
            return [[['Research and development expenses','$ 7,339','$ 5,268','39 %'],
                     ['% of revenue','27.2 %','19.6 %']]]
    text = 'Operating Expenses\nYear Ended\nJanuary 29, January 30,\n2023 2022 Change\n($ in millions)\nResearch and development expenses $ 7,339 $ 5,268 39 %\n% of revenue 27.2 % 19.6 %'
    rows=extract_financial_table_triples(Page(),text,2023,company_id='NVIDIA_CORPORATION')
    ratio=next(r for r in rows if r['target']=='R_AND_D_RATIO')
    import json
    assert ratio['metric_unit']=='percent'
    assert len(json.loads(ratio['metric_values_json']))==2
    assert 'Research and development' in ratio['evidence_sentence']

@pytest.mark.parametrize('metric,row,unit,statement,context',[
    ('CASH_FLOW_CHANGE_ACCOUNTS_RECEIVABLE','Accounts receivable (2,215)','USD millions','CASH_FLOW_STATEMENT','Changes in operating assets and liabilities'),
    ('DATA_CENTER_REVENUE','Data Center 115,186','USD millions','FINANCIAL_NOTES','The following table summarizes revenue by specialized markets'),
    ('R_AND_D_RATIO','Research and development 27.2 19.6','percent','MD_AND_A','expressed as a percentage of revenue')])
def test_qualified_table_metric_requires_source_context(metric,row,unit,statement,context):
    from strategic_graphrag.pipeline.extractor import TripleExtractor
    triple=dict(target=metric,evidence_sentence=row,metric_unit=unit,statement_type=statement,table_context=context)
    assert TripleExtractor._typed_table_measurement_support(triple,context)
    triple['table_context']='unrelated'
    assert not TripleExtractor._typed_table_measurement_support(triple,'unrelated')
