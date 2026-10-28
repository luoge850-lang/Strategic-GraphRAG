import copy
import pytest
from strategic_graphrag.pipeline.financial_cache import extraction_contract,reusable

def contract():
    return extraction_contract(document_sha256='a'*64,company_id='ISSUER_A',filing_year=2024,filename='report.pdf')

def test_legacy_pdf_only_cache_is_not_reusable():
    c=contract()
    assert not reusable(dict(document_sha256=c['document_sha256']),c)
    assert reusable(dict(financial_extraction_contract=c),c)

@pytest.mark.parametrize('key',['document_sha256','company_id','filing_year','filename','extractor_sha256','pdfplumber_version'])
def test_any_extraction_identity_change_invalidates_cache(key):
    c=contract();old=copy.deepcopy(c);old[key]='different'
    assert not reusable(dict(financial_extraction_contract=old),c)
