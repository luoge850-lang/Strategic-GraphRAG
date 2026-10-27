"""Source-independent metamorphic checks for the first bounded repair round."""
import pytest
from dataclasses import replace
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.engine.financial_calculation import calculate_public
from test_public_calculation_contract import observation,path

@pytest.mark.parametrize('metric,expected', [('interest income','INTEREST_INCOME'),('interest expense','INTEREST_EXPENSE'),('net property and equipment','PROPERTY_PLANT_EQUIPMENT')])
def test_disclosure_prefix_does_not_add_obligation(metric,expected):
    a=parse_query(f'In the 2025 10-K, what was NVIDIA {metric} during FY2024?')
    b=parse_query(f'What was NVIDIA {metric} during FY2024 in the 2025 filing?')
    assert a.target_metric==b.target_metric==expected
    assert a.fiscal_years==b.fiscal_years==[2024]
    assert a.document_scope==b.document_scope=='2025-10-K.pdf'
    assert not a.ambiguity and not b.ambiguity
    assert not a.comparison and not b.comparison

def test_exclusion_is_not_requested_metric():
    p=parse_query('Report NVIDIA FY2025 interest expense, not interest income. Use the 2025 annual report.')
    assert p.requested_metrics==['INTEREST_EXPENSE']
    assert len(p.requested_facts)==1 and not p.ambiguity
    p=parse_query('From the 2025 report, retrieve NVIDIA Automotive revenue for FY2023, not company-wide sales.')
    assert p.target_metric=='AUTOMOTIVE_REVENUE'
    assert p.fiscal_years==[2023] and not p.comparison and not p.ambiguity

def test_unknown_comma_output_remains_unresolved():
    p=parse_query('What were NVIDIA revenue, employee satisfaction in FY2025?')
    assert 'additional_requested_item_not_resolved' in p.ambiguity

def test_proforma_never_substitutes_actual_even_when_number_matches():
    obs=replace(observation(17),table_name='Unaudited Pro Forma',financial_basis='PRO_FORMA')
    p=parse_query('What was NVIDIA revenue in FY2025?').to_dict();p['build_id']='build_test'
    assert calculate_public([path(obs)],query_plan=p)['status']!='PASS'
