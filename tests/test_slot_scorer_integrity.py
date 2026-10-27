import copy
import pytest
from deployment.slot_score_20261016 import score

def fixture():
    operand=dict(company_id='A',metric_id='REVENUE',fact_period='FY2025',business_scope='CONSOLIDATED',value=12,unit='USD millions',sources=[dict(file='report.pdf',page=3,rows=['Revenue'])])
    label=dict(slots=[dict(value=.012,unit='USD billions',operation='unit_conversion',operands=[operand])])
    obs=dict(operand,build_id='build_x',fiscal_period='FY2025',source_filing='report.pdf',disclosure_version='report.pdf',page=3,row_label='Revenue',claim_id='claim_x')
    response=dict(answer_status='ANSWERED',calculation=dict(status='PASS',operation='unit_conversion',value=.012,unit='USD billions',observations=[obs]),citations=[dict(evidence_id='claim_x',source_filing='report.pdf',page=3)])
    return response,label

@pytest.mark.parametrize('field,value,failed',[
 ('company_id','WRONG','fact_identity'),('fact_period','FY1999','fact_identity'),
 ('fiscal_period','FY1999','fact_identity'),('metric_id','NET_INCOME','fact_identity'),
 ('business_scope','GAMING','fact_identity'),('unit','TWD thousands','numeric_unit'),
 ('row_label','Net income','source_support'),('page',4,'source_support'),
 ('source_filing','other.pdf','fact_identity'),('disclosure_version','other.pdf','fact_identity'),
])
def test_identity_unit_and_same_page_wrong_row_fail(field,value,failed):
    response,label=fixture()
    assert score(response,label,build_id='build_x')['full_fact_joint']
    response['calculation']['observations'][0][field]=value
    scored=score(response,label,build_id='build_x')
    assert not scored[failed] and not scored['full_fact_joint']

def test_wrong_citation_and_missing_item_and_operation_fail():
    response,label=fixture();response['citations'][0]['evidence_id']='other'
    assert not score(response,label,build_id='build_x')['citation_identity']
    response,label=fixture();label['slots'].append(copy.deepcopy(label['slots'][0]))
    assert not score(response,label,build_id='build_x')['request_complete']
    response,label=fixture();response['calculation']['operation']='fact'
    assert not score(response,label,build_id='build_x')['core_semantics']

def test_partial_does_not_hide_wrong_returned_part():
    response,label=fixture();response['answer_status']='PARTIALLY_ANSWERED'
    response['calculation']['observations'][0]['company_id']='WRONG'
    scored=score(response,label,build_id='build_x')
    assert not scored['slots'][0]['identity'] and not scored['request_complete']
    assert scored['severe_wrong_success']


def test_source_support_requires_fact_identity():
    response,label=fixture()
    response['calculation']['observations'][0]['company_id']='WRONG'
    assert not score(response,label,build_id='build_x')['source_support']


def test_row_typography_preserves_semantic_qualifiers():
    response,label=fixture()
    label['slots'][0]['operands'][0]['sources'][0]['rows']=['Accounts receivable, net']
    response['calculation']['observations'][0]['row_label']='accounts receivable net'
    assert score(response,label,build_id='build_x')['source_support']
    response['calculation']['observations'][0]['row_label']='accounts receivable gross'
    assert not score(response,label,build_id='build_x')['source_support']
    response['calculation']['observations'][0]['row_label']='total accounts receivable net'
    assert not score(response,label,build_id='build_x')['source_support']
