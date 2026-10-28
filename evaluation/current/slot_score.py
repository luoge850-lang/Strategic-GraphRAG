"""Independent slot labels are inputs, never derived from a system answer.

Source support is an AI/PDF diagnostic against frozen page/row labels, separate
from citation identity and from browser physical-page checks (unjudged here).
"""
import math
import re

def row_key(value):
    """Normalize typography only; do not erase qualifiers such as net/total."""
    return re.sub(r'\s+', ' ', re.sub(r'[,\u00a0]', ' ', str(value).casefold())).strip()

def equal(actual, expected):
    return isinstance(actual,(int,float)) and not isinstance(actual,bool) and math.isfinite(actual) and abs(actual-expected)<=max(1e-6,abs(expected)*1e-10)

def score(response, label, *, build_id):
    calc=response.get('calculation') or {}
    expected=label['slots']
    if not expected:
        correct=response.get('answer_status')=='ABSTAINED' and calc.get('status')!='PASS' and calc.get('value') is None
        return dict(request_complete=correct,core_semantics=correct,numeric_unit=None,
            fact_identity=None,citation_identity=None,source_support=None,browser_page=None,
            correct_refusal=correct,false_refusal=None,severe_wrong_success=not correct,slots=[])
    items=calc.get('items') or [calc]
    checks=[]
    for index,want in enumerate(expected):
        item=items[index] if index<len(items) else {}
        obs=item.get('observations') or []
        identity_fields=['company_id','metric_id','fact_period','business_scope']
        identity=bool(obs) and len(obs)==len(want['operands'])
        support=identity
        links=bool(obs)
        numeric=item.get('status')=='PASS' and equal(item.get('value'),want['value']) and item.get('unit')==want['unit'] and item.get('operation')==want['operation']
        locations={(c.get('evidence_id'),c.get('source_filing'),c.get('page')) for c in response.get('citations') or []}
        for i,operand in enumerate(want['operands']):
            actual=obs[i] if i<len(obs) else {}
            identity=identity and all(actual.get(k)==operand[k] for k in identity_fields)
            # Both canonical and serialized aliases must agree, not just one.
            identity=identity and actual.get('fiscal_period')==operand['fact_period'] and actual.get('build_id')==build_id
            allowed=operand['sources']
            matched=any(actual.get('source_filing')==s['file'] and actual.get('disclosure_version')==s['file'] and actual.get('page')==s['page'] and row_key(actual.get('row_label','')) in [row_key(x) for x in s['rows']] for s in allowed)
            identity=identity and any(actual.get('source_filing')==s['file'] and actual.get('disclosure_version')==s['file'] for s in allowed)
            operand_numeric=equal(actual.get('value'),operand['value']) and actual.get('unit')==operand['unit']
            numeric=numeric and operand_numeric
            support=support and matched and operand_numeric
            links=links and (actual.get('claim_id'),actual.get('source_filing'),actual.get('page')) in locations
        support=support and identity
        checks.append(dict(identity=bool(identity),numeric_unit=bool(numeric),citation_identity=bool(links),source_support=bool(support),returned=index<len(items) and item.get('status')=='PASS'))
    nested=[o for item in items for o in item.get('observations') or []]
    root_consistent=calc.get('observations',[])==nested if calc.get('items') else True
    complete=len(items)==len(expected) and response.get('answer_status')=='ANSWERED' and calc.get('status')=='PASS' and root_consistent
    identity=all(c['identity'] for c in checks)
    numeric=all(c['numeric_unit'] for c in checks)
    support=all(c['source_support'] for c in checks)
    links=all(c['citation_identity'] for c in checks)
    core=complete and identity and numeric
    return dict(request_complete=core,core_semantics=core,numeric_unit=numeric,
        fact_identity=identity,citation_identity=links,source_support=support,browser_page=None,
        correct_refusal=None,false_refusal=response.get('answer_status')=='ABSTAINED',
        severe_wrong_success=(response.get('answer_status')=='ANSWERED' and not core) or
            (response.get('answer_status')=='PARTIALLY_ANSWERED' and
             (len(items)>len(expected) or any(c['returned'] and not (c['identity'] and c['numeric_unit']) for c in checks))),
        full_fact_joint=core and support and links,slots=checks)
