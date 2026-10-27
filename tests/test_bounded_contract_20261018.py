"""Metamorphic slot execution, context and identity checks, synthetic values."""
import pytest
from types import SimpleNamespace
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.engine.financial_request import answer_financial_request,slot_plan
from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine
from strategic_graphrag.pipeline.financial_context import describe,dated_columns
from strategic_graphrag.engine.vector_rag_baseline import VectorRAGBaseline
from strategic_graphrag.build_identity import BuildIdentityMismatch
from strategic_graphrag.pipeline.financial_candidate_guard import rejection_reason
from test_public_calculation_contract import observation,path

@pytest.mark.parametrize('verb',['What was','How much was','Give','Report'])
@pytest.mark.parametrize('tail',['in the 2025 filing',', using the 2025 annual report',', as disclosed in its 2025 10-K'])
def test_surface_wording_same_typed_executor(verb,tail):
    q=f'{verb} NVIDIA interest income during FY2024 {tail}?';p=parse_query(q).to_dict()
    class Finder:
        def find_metric_disclosures(self,metric,**kw):return [path(observation(17,metric=metric,fact_year=2024))]
    r=answer_financial_request(q,p,Finder(),'build_test',None,GraphRAGEngine.__new__(GraphRAGEngine)._serialize_path)
    assert r['answer_status']=='ANSWERED' and r['calculation']['value']==17

def test_slot_plan_does_not_copy_another_slots_scope_or_unit():
    p=parse_query('Give NVIDIA revenue in FY2024 in USD billions and net income in FY2025 in USD millions.').to_dict()
    locals=[slot_plan(x,'build_test') for x in p['requested_facts']]
    assert [(x['fact_period'],x['calculation_target_scale']) for x in locals]==[('FY2024','billions'),('FY2025','millions')]

def test_source_context_unknown_stays_unknown():
    assert describe('Unknown heading\nRevenue 17','Unknown','Revenue 17','FINANCIAL_TABLE')['financial_basis']=='UNKNOWN'
    c=describe('Consolidated Balance Sheets\nJanuary 31, 2025 January 31, 2024','Consolidated Balance Sheets','Goodwill 17','BALANCE_SHEET')
    assert c['financial_basis']=='ACTUAL_REPORTED' and c['source_section']=='FINANCIAL_STATEMENTS'
    assert dated_columns('January 31, 2025 January 31, 2024',[2025,2024])[2025]['period_start']=='2024-02-01'

def test_vector_identity_uses_stored_metadata():
    v=VectorRAGBaseline.__new__(VectorRAGBaseline);v.build_id='expected'
    v.collection=SimpleNamespace(get=lambda **kw:{'metadatas':[{'build_id':'other'}]})
    with pytest.raises(BuildIdentityMismatch):v._validate_identity()

def test_adjacent_unknown_expense_is_not_total_operating_cost():
    q=parse_query('How much contribution expense did NVIDIA disclose for retirement plans in FY2025?')
    assert q.target_metric=='RETIREMENT_CONTRIBUTION_EXPENSE'
    assert parse_query('Give NVIDIA deferred income tax assets in FY2024.').target_metric=='DEFERRED_TAX_ASSETS'
    assert 'unsupported_financial_component' not in parse_query('Give NVIDIA deferred income tax assets in FY2024.').ambiguity

def test_segment_columns_stay_pending():
    assert rejection_reason(dict(relation='REPORTS_METRIC',column_semantics='SEGMENT_COLUMNS_UNRESOLVED'))=='segment_columns_cannot_substitute_for_period_columns'

def test_missing_slot_never_completes_request():
    q='Give NVIDIA revenue in FY2024 and goodwill in FY2025.';p=parse_query(q).to_dict()
    class Finder:
        def find_metric_disclosures(self,metric,**kw):
            return [path(observation(17,metric=metric,fact_year=kw['year_start']))] if metric=='REVENUE' else []
    r=answer_financial_request(q,p,Finder(),'build_test',None,GraphRAGEngine.__new__(GraphRAGEngine)._serialize_path)
    assert r['answer_status']=='PARTIALLY_ANSWERED' and r['calculation']['missing_slots']==['s2']
