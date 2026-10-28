"""Fail-closed fact scoring. Labels are source-authored, never prediction-derived."""
from decimal import Decimal, InvalidOperation
import re

FIELDS = ('number', 'unit', 'company', 'metric', 'period', 'disclosure',
          'statement', 'citation_support', 'physical_page')

# Semantic alias, established by entity_registry.py:90-95 at the evaluated commit.
# The frozen reference value/location is unchanged; retain v1 scores for audit.
METRIC_ALIASES = {'SALES_GENERAL_ADMIN_EXPENSE': 'SG_AND_A_EXPENSE'}

def equal_number(actual, expected, tolerance):
    try:
        a, b, t = map(lambda x: Decimal(str(x)), (actual, expected, tolerance))
        return a.is_finite() and b.is_finite() and t.is_finite() and t >= 0 and abs(a-b) <= t
    except (InvalidOperation, TypeError, ValueError):
        return False

def _score_single_observation(response, label, *, layered=False):
    calc = response.get('calculation') or {}
    observations = calc.get('observations') or []
    citations = response.get('citations') or []
    # Do not pick a favourable observation from a conflicting or multi-value answer.
    obs = observations[0] if len(observations) == 1 else {}
    evidence_parts = []
    for path in response.get('paths') or []:
        for i, eid in enumerate(path.get('evidence_ids') or []):
            if eid == obs.get('evidence_id') and i < len(path.get('evidence') or []):
                evidence_parts.append(path['evidence'][i])
    evidence = ' '.join(evidence_parts) or str(obs.get('evidence') or obs.get('raw_row') or '')
    locations = {(x['file'], x['page']): x for x in label['direct_locations']}
    matched = [c for c in citations if (c.get('source_filing'), c.get('page')) in locations]
    linked = [c for c in matched if c.get('evidence_id') == obs.get('evidence_id')
              and c.get('source_filing') == obs.get('source_filing')
              and c.get('page') == obs.get('page')]
    # Exact row semantic anchor and source numeral must occur in the actual observation.
    # This is a diagnostic row-support check, not human entailment adjudication.
    norm = re.sub(r'\s+', ' ', evidence.lower())
    anchors = label['row_anchors']
    source_number = str(label['source_value'])
    numeric_evidence = re.sub(r'\(([\d,]+(?:\.\d+)?)\)',r'-\1',evidence).replace(',', '')
    numeric_token = re.search(r'(?<![\d.])' + re.escape(source_number) + r'(?![\d.])',
                              numeric_evidence) is not None
    support = bool(linked) and numeric_token and any(a.lower() in norm for a in anchors)
    location_judged = bool(matched)
    statement_values = {x['statement'] for x in label['direct_locations']}
    statement = (obs.get('statement_type') == locations[(linked[0]['source_filing'],linked[0]['page'])]['statement']
                 if linked else False if obs.get('statement_type') not in statement_values else None)
    results = {
        'number': equal_number(calc.get('value'), label['answer_value'], label['tolerance']),
        'unit': calc.get('unit') == label['answer_unit'] and obs.get('currency') == label['currency']
                and obs.get('scale') == label['source_scale'],
        'company': obs.get('company_id', obs.get('company')) == label['company'],
        'metric': obs.get('metric_id', obs.get('metric')) == METRIC_ALIASES.get(label['metric'], label['metric']),
        'period': obs.get('fact_period') == label['fact_period'],
        'disclosure': obs.get('source_filing') == label['source_filing'],
        'statement': statement,
        'citation_support': support if location_judged else None,
        'physical_page': bool(linked) if location_judged else None,
    }
    results['status_success'] = calc.get('status') == 'PASS'
    results['joint'] = False if any(v is False for v in results.values()) else None if any(v is None for v in results.values()) else True
    results['label_tier'] = label['label_tier']
    results['support_scope'] = 'source-checked row anchor, numeral, location and evidence-ID linkage; no human Gold'
    results['location_scope'] = 'known independently judged physical-page positions; unmatched pages UNJUDGED, not automatically wrong; not browser navigation score'
    if not layered:
        return results
    for key in ('business_scope','measurement_nature','value_kind','period_granularity'):
        if key in label:results[key]=obs.get(key)==label[key]
    core_keys=('number','unit','company','metric','period','disclosure','status_success')
    results['core_semantic']=all(results[k] for k in core_keys) and all(results.get(k,True) for k in ('business_scope','measurement_nature','value_kind','period_granularity'))
    results['numeric_and_unit']=results['number'] and results['unit']
    results['metadata_complete']=results['statement']
    required=[results[k] for k in (*core_keys,'citation_support','physical_page','statement')]
    required += [results[k] for k in ('business_scope','measurement_nature','value_kind','period_granularity') if k in results]
    results['joint']=False if any(v is False for v in required) else None if any(v is None for v in required) else True
    return results

def score_fact(response,label, *, layered=False):
    observations=(response.get('calculation') or {}).get('observations') or []
    if len(observations)<=1 or not layered:return _score_single_observation(response,label,layered=layered)
    import copy
    keys=('company_id','metric_id','value','currency','scale','fact_period','source_filing',
          'business_scope','measurement_nature','value_kind','period_granularity')
    identities={tuple(str(o.get(k)) for k in keys) for o in observations}
    rows=[]
    for observation in observations:
        one=copy.deepcopy(response);one['calculation']['observations']=[observation]
        rows.append(_score_single_observation(one,label,layered=True))
    out=dict(rows[0]);out['observation_count']=len(observations)
    out['observation_conflict']=len(identities)!=1
    for key in (*FIELDS,'core_semantic','numeric_and_unit','metadata_complete','joint'):
        values=[r[key] for r in rows]
        out[key]=False if any(v is False for v in values) else None if any(v is None for v in values) else True
    if out['observation_conflict']:out['core_semantic']=out['joint']=False
    return out

def citation_page_score(response, grades):
    cited = {f"{c.get('source_filing')}#{c.get('page')}" for c in response.get('citations', [])}
    direct = {p for p, g in grades.items() if g >= 2}
    background = {p for p, g in grades.items() if g == 1}
    return {'direct_page_hit': bool(cited & direct) if direct else None,
            'background_page_hit': bool(cited & background) if background else None,
            'positive_page_hit_legacy': bool(cited & (direct | background)) if grades else None,
            'unjudged_cited_pages': sorted(cited - set(grades)),
            'direct_page_labels': sorted(direct), 'background_page_labels': sorted(background)}
