from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.engine.financial_request import answer_financial_request
from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine
from test_public_calculation_contract import observation,path,calculate,plan
import pytest

@pytest.mark.parametrize('words,expected',[
 ('revenue and net income',['REVENUE','NET_INCOME']),
 ('net income and revenue',['NET_INCOME','REVENUE']),
 ('gross profit and gross margin',['GROSS_PROFIT','GROSS_MARGIN']),
 ('revenue, net income, and earnings per share',['REVENUE','NET_INCOME','EARNINGS_PER_SHARE']),
 ('cash and cash equivalents',['CASH_AND_CASH_EQUIVALENTS']),
])
def test_all_explicit_metrics_survive_planning(words,expected):
    assert parse_query(f'What were NVIDIA {words} in FY2025?').requested_metrics==expected

@pytest.mark.parametrize('unit',['USD billions','billions of USD','billions'])
def test_requested_scale_does_not_need_convert_verb(unit):
    q=parse_query(f'What was NVIDIA revenue in FY2025 in {unit}?')
    assert q.calculation=='UNIT_CONVERSION' and q.calculation_target_scale=='billions'

def test_unknown_conjunct_and_scope_qualifiers_cannot_disappear():
    assert 'additional_requested_item_not_resolved' in parse_query('What were NVIDIA revenue and employee satisfaction in FY2025?').ambiguity
    assert parse_query('台积电2024年火星工厂收入是多少？').ambiguity
    assert parse_query('What were TSMC unconsolidated revenue and net income in FY2024?').ambiguity

def test_common_chinese_and_ambiguous_spending():
    assert parse_query('英伟达2025财年收入是多少？').target_metric=='REVENUE'
    assert parse_query('英伟达2025财年营收和净利润分别是多少？').requested_metrics==['REVENUE','NET_INCOME']
    q=parse_query('How much cash did TSMC spend on investing in 2024?')
    assert 'investing_cash_flow_or_capital_expenditure' in q.ambiguity
    assert len(q.clarification_options)==2

def test_balance_date_and_disclosure_version_are_separate():
    q=parse_query('What was TSMC cash balance at December 31, 2023, in its 2024 annual report?')
    assert q.fact_period=='FY2023' and q.document_scope=='TSMC-2024-Annual-Report.pdf' and q.fiscal_years==[2023]

def test_partial_facts_never_get_full_pass_and_each_metric_is_retrieved():
    class Finder:
        def __init__(self,missing=False):self.metrics=[];self.missing=missing
        def find_metric_disclosures(self,metric,**kwargs):
            self.metrics.append(metric)
            return [] if self.missing and metric=='NET_INCOME' else [path(observation(120 if metric=='REVENUE' else 30,metric=metric,fact_year=2025))]
    question='What were NVIDIA revenue and net income in FY2025?'
    plan=parse_query(question).to_dict();plan['build_id']='build_test'
    engine=GraphRAGEngine.__new__(GraphRAGEngine)
    for missing in [False,True]:
        finder=Finder(missing)
        result=answer_financial_request(question,plan,finder,'build_test',None,engine._serialize_path)
        assert finder.metrics==['REVENUE','NET_INCOME']
        assert result['calculation']['answered_count']==(1 if missing else 2)
        assert result['answer_status']==('PARTIALLY_ANSWERED' if missing else 'ANSWERED')
        assert result['calculation']['status']==('PARTIAL' if missing else 'PASS')
        if missing:
            assert result['calculation']['value'] is None and result['grounding_status']!='VERIFIED'

def test_duplicate_source_candidates_remain_auditable_without_duplicate_answer_facts():
    a=observation(120,evidence_id='claim_a');b=observation(120,evidence_id='claim_b')
    result=calculate([path(a),path(b)],plan())
    assert result['status']=='PASS' and len(result['observations'])==1
    assert {o['claim_id'] for o in result['candidate_observations']}=={'claim_a','claim_b'}
