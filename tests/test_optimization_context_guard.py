from dataclasses import replace
from strategic_graphrag.engine.financial_calculation import _comparability_assessment
from strategic_graphrag.pipeline.financial_candidate_guard import rejection_reason
from test_public_calculation_contract import observation


def test_same_table_requires_same_claim_page_scope_and_unit():
    a=replace(observation(100),comparability_status='SAME_TABLE')
    b=replace(a,id='second',value=120,fiscal_year=2023,fiscal_period='FY2023')
    assert _comparability_assessment([a,b])[1] is None
    for changes in ({'claim_id':'other'},{'page':51},{'company_id':'TSMC'}, {'unit':'TWD thousands'}, {'table_name':'other'}, {'business_scope':'GRAPHICS'}):
        assert _comparability_assessment([a,replace(b,**changes)])[1]


def test_unresolved_context_is_quarantined_without_reference_values():
    base=dict(relation='REPORTS_METRIC',table_name='Consolidated balance sheets',metric_unit='USD millions')
    assert rejection_reason(base) is None
    for title in ['Purchase price allocation','Networking Graphics All Other Consolidated','Income as a percentage of revenue']:
        assert rejection_reason(dict(base,table_name=title))
    assert rejection_reason(dict(base,table_name='Income as a percentage of revenue',metric_unit='percent')) is None
