"""Fail-closed identity for reusable financial extraction candidates.

A matching PDF alone does not make old extracted rows current. Prose candidate
lineage is separate; this contract applies only to REPORTS_METRIC candidates.
"""
import hashlib
import importlib.metadata
from pathlib import Path


def extraction_contract(*, document_sha256, company_id, filing_year, filename):
    digest=hashlib.sha256()
    folder=Path(__file__).resolve().parent
    for name in ['financial_cache.py','financial_context.py','financial_table_extractor.py','financial_note_extractor.py','consolidated_ntd_table.py','financial_candidate_guard.py']:
        digest.update(name.encode())
        digest.update((folder/name).read_bytes().replace(b'\r\n',b'\n'))
    return dict(schema='financial-extraction-cache/v1',document_sha256=document_sha256,
                company_id=company_id,filing_year=int(filing_year),filename=filename,
                extractor_sha256=digest.hexdigest(),pdfplumber_version=importlib.metadata.version('pdfplumber'))


def reusable(row, contract):
    return row.get('financial_extraction_contract')==contract


def refresh_nvidia(row, pdf, *, contract):
    import time
    import pdfplumber
    from .financial_table_extractor import extract_financial_table_triples, _find_metric, _numeric_values
    from .financial_note_extractor import extract as note_extract
    started=time.perf_counter();facts=[];selected=[]
    with pdfplumber.open(pdf) as document:
        total=len(document.pages)
        for number,page in enumerate(document.pages,1):
            text=page.extract_text() or ''
            extra=note_extract(text,contract['company_id'],contract['filename'],contract['filing_year'])
            for fact in extra:
                fact.update(source_page=number,page=number,document_sha256=contract['document_sha256']);facts.append(fact)
            # The same strict row recognizer selects table work; no fixed page
            # list or reference values are used. Percent rows retain parents.
            if not any(_find_metric(line) and _numeric_values(line) for line in text.splitlines()):
                page.close()
                continue
            selected.append(number)
            for fact in extract_financial_table_triples(page,text,contract['filing_year'],company_id=contract['company_id'],document_id=contract['filename']):
                fact.update(source_page=number,page=number,source_filing=contract['filename'],document_sha256=contract['document_sha256'])
                facts.append(fact)
            page.close()
    retained=[fact for fact in row['accepted_triples'] if fact['relation']!='REPORTS_METRIC']
    row['accepted_triples']=retained+facts
    row['financial_extraction_contract']=contract
    return dict(filename=contract['filename'],status='COMPLETED',mode='FRESH_FINANCIAL_EXTRACTION',
                pages_examined=total,table_pages=selected,financial_candidates=len(facts),
                retained_prose_candidates=len(retained),seconds=time.perf_counter()-started,llm_calls=0,token_count=0)
