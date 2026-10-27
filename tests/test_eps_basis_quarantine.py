import pytest
from strategic_graphrag.pipeline.financial_candidate_guard import rejection_reason

@pytest.mark.parametrize('row',['Basic earnings per share','Diluted earnings per share','Net income per diluted share'])
def test_generic_eps_identity_cannot_claim_a_basis(row):
    fact=dict(relation='REPORTS_METRIC',target='EARNINGS_PER_SHARE',row_label=row,metric_unit='USD per share')
    assert rejection_reason(fact)=='earnings_per_share_basis_not_in_fact_identity'

def test_quarantine_does_not_remove_ordinary_financial_facts():
    assert rejection_reason(dict(relation='REPORTS_METRIC',target='REVENUE',table_name='Income statement',unit='USD millions')) is None
