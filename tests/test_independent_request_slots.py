"""Metamorphic request binding checks; no reference production amounts."""
import pytest
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.engine.financial_request import answer_financial_request
from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine
from test_public_calculation_contract import observation,path

@pytest.mark.parametrize('first,second',[(2024,2025),(2025,2024),(2023,2025)])
def test_metric_local_years_and_executor(first,second):
    question=f'What were NVIDIA revenue in fiscal {first} and net income in fiscal {second}?'
    plan=parse_query(question).to_dict();plan['build_id']='build_test'
    assert [(s['metric_id'],s['fact_period']) for s in plan['requested_facts']]==[('REVENUE',f'FY{first}'),('NET_INCOME',f'FY{second}')]
    calls=[]
    class Finder:
        def find_metric_disclosures(self,metric,**kwargs):
            calls.append((metric,kwargs['year_start'],kwargs['year_end']))
            return [path(observation(12,metric=metric,fact_year=kwargs['year_start']))]
    result=answer_financial_request(question,plan,Finder(),'build_test',None,GraphRAGEngine.__new__(GraphRAGEngine)._serialize_path)
    assert calls==[('REVENUE',first,first),('NET_INCOME',second,second)]
    assert result['answer_status']=='ANSWERED'

def test_metric_local_units_and_repeated_metric_are_not_deduplicated():
    q=parse_query('What were NVIDIA revenue in USD billions and net income in USD millions in fiscal 2025?')
    assert [s['output_scale'] for s in q.requested_facts]==['billions','millions']
    q=parse_query('What were NVIDIA revenue in fiscal 2024 and revenue in fiscal 2025?')
    assert [s['fact_period'] for s in q.requested_facts]==['FY2024','FY2025']
    assert len(q.requested_facts)==2
    q=parse_query('What were NVIDIA revenue in fiscal 2024 and fiscal 2025?')
    assert len(q.requested_facts)==2

@pytest.mark.parametrize('join',[', ',' and ',' plus '])
def test_unknown_items_survive_list_separators(join):
    q=parse_query('What were NVIDIA revenue'+join+'employee satisfaction in fiscal 2025?')
    assert 'additional_requested_item_not_resolved' in q.ambiguity

@pytest.mark.parametrize('restriction',['excluding China','only in Europe','adjusted','non-GAAP'])
def test_restrictions_cannot_be_answered_by_total(restriction):
    q=parse_query(f'What was NVIDIA revenue {restriction} in fiscal 2025?')
    assert 'unresolved_restrictive_qualifier' in q.ambiguity

def test_scope_and_company_minimal_changes():
    for company,canonical in [('NVIDIA','NVIDIA_CORPORATION'),('TSMC','TSMC')]:
        q=parse_query(f'What were {company} revenue and net income in fiscal 2025?')
        assert all(s['company_id']==canonical for s in q.requested_facts)
    q=parse_query('What were NVIDIA data center revenue in fiscal 2024 and gaming revenue in fiscal 2025?')
    assert len(q.requested_facts)==2
    assert [s['business_scope'] for s in q.requested_facts]==['DATA_CENTER','GAMING']
