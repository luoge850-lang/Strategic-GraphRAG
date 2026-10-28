import pytest
from strategic_graphrag.engine.query_understanding import parse_query

@pytest.mark.parametrize('adjunct',['as disclosed in','as reported in','as stated in','according to'])
@pytest.mark.parametrize('metric',['cost of revenue','gross profit','net income','pretax income'])
def test_disclosure_adjunct_is_not_an_unknown_metric(adjunct,metric):
    q=parse_query(f'What was TSMC consolidated {metric} for fiscal 2024, {adjunct} its 2024 annual report?')
    assert not q.ambiguity
    assert q.company=='TSMC' and q.fact_period=='FY2024'
    assert q.document_scope=='TSMC-2024-Annual-Report.pdf'
    assert len(q.requested_facts)==1

@pytest.mark.parametrize('suffix',[', employee satisfaction',' and employee satisfaction',' plus employee satisfaction'])
def test_disclosure_adjunct_does_not_hide_unknown_requirements(suffix):
    q=parse_query('What was TSMC revenue for fiscal 2024, as disclosed in its 2024 annual report'+suffix+'?')
    assert 'additional_requested_item_not_resolved' in q.ambiguity

@pytest.mark.parametrize('question,metric',[
 ('What percentage of revenue were research and development expenses for NVIDIA fiscal 2022 in its 2023 10-K?','R_AND_D_RATIO'),
 ('What were NVIDIA research and development expenses as a percentage of revenue in FY2022?','R_AND_D_RATIO'),
 ('What percentage of revenue was NVIDIA gross profit in FY2025?','GROSS_MARGIN'),
])
def test_disclosed_percentage_is_one_semantic_measure(question,metric):
    q=parse_query(question)
    assert not q.ambiguity
    assert q.requested_metrics==[metric]
    assert len(q.requested_facts)==1
    assert q.requested_facts[0]['metric_id']==metric

def test_two_disclosed_ratios_do_not_silently_become_one():
    q=parse_query('What percentage of revenue were NVIDIA research and development and sales general and administrative expenses in FY2025?')
    assert q.ambiguity

@pytest.mark.parametrize('document',['the Mars report','an unknown supplement'])
def test_disclosure_adjunct_requires_a_resolvable_document(document):
    q=parse_query(f'What was NVIDIA revenue in FY2025, as disclosed in {document}?')
    assert 'additional_requested_item_not_resolved' in q.ambiguity
