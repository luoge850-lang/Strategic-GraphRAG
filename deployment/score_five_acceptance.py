"""Frozen source-label scoring, with separate semantic/numeric/support layers."""
import argparse,json,math,re,hashlib,statistics,random
from collections import defaultdict
from deployment.acceptance_20261015 import ROOT,OUT,save
from deployment.complete_fact_scoring import equal_number
KEYS=('company_id','metric_id','fact_period','source_filing','business_scope','measurement_nature','value_kind','period_granularity')
def score(row):
    label=row['label']; result=row['result'];response=result.get('response') or {};calc=response.get('calculation') or {}
    status=calc.get('status');http=result.get('http_status');execution=response.get('execution_status')
    base=dict(request_success=http==200 and execution=='SUCCEEDED',timeout=result.get('error_type')=='TimeoutError' or execution=='TIMEOUT',
              execution_failure=http!=200 or execution not in {'SUCCEEDED'},status=status,review_grade=label.get('label_tier'))
    if label['operation']=='SAFE_REFUSAL':
        safe=base['request_success'] and status in label['accepted_statuses'] and calc.get('value') is None and response.get('answer_status') in {'ABSTAINED','PARTIALLY_ANSWERED'}
        return dict(base,core_semantic=safe,numeric_and_unit=None,joint=safe,citation_support=None,physical_page=None,
                    correct_refusal=safe,false_refusal=None,severe_wrong_pass=status=='PASS',missing_source_success=False)
    expected=label.get('required_facts') or [label];observations=calc.get('observations') or []
    paths=response.get('paths') or [];citations=response.get('citations') or []
    semantic=[];supports=[];locations=[];joint_fields=[]
    for reference in expected:
        identity=dict(company_id=reference['company'],metric_id=reference['metric'],fact_period=reference['fact_period'],source_filing=reference['source_filing'],
             **{k:reference[k] for k in KEYS[4:]})
        matched=[o for o in observations if all(o.get(k)==v for k,v in identity.items())]
        semantic.append(bool(matched))
        evidence_ok=False;page_ok=False;judged=False;fields_ok=False
        for obs in matched:
            fields_ok |= (obs.get('currency')==reference['currency'] and obs.get('scale')==reference['source_scale'] and equal_number(obs.get('value'),reference['source_value'],'0.5'))
            for cite in citations:
                if cite.get('evidence_id')!=obs.get('evidence_id'):continue
                physical={(x['file'],x['page']) for x in reference['direct_locations']}
                if (cite.get('source_filing'),cite.get('page')) not in physical:continue
                judged=True;page_ok |= cite.get('page')==obs.get('page') and cite.get('source_filing')==obs.get('source_filing')
                for path in paths:
                    for i,eid in enumerate(path.get('evidence_ids',[])):
                        if eid!=obs.get('evidence_id') or i>=len(path.get('evidence',[])):continue
                        text=re.sub(r'\s+',' ',path['evidence'][i]);numeric=text.replace(',','')
                        number=bool(re.search(r'(?<![\d.])'+re.escape(str(reference['source_value']))+r'(?![\d.])',numeric))
                        evidence_ok |= number and any(anchor.lower() in text.lower() for anchor in reference['row_anchors'])
        supports.append(evidence_ok if judged else None);locations.append(page_ok if judged else None);joint_fields.append(fields_ok)
    operation=calc.get('operation')=={'FACT':'fact','UNIT_CONVERSION':'unit_conversion','ABSOLUTE_CHANGE':'absolute_change'}[label['operation']]
    semantic_ok=all(semantic) and status=='PASS' and operation and len(observations)>=len(expected)
    numeric=equal_number(calc.get('value'),label['answer_value'],label['tolerance']) and calc.get('unit')==label['answer_unit'] and all(joint_fields)
    def aggregate(values):return False if False in values else None if None in values else True
    support=aggregate(supports);physical=aggregate(locations)
    joint=aggregate([semantic_ok,numeric,support,physical])
    return dict(base,core_semantic=semantic_ok,numeric_and_unit=numeric,joint=joint,citation_support=support,physical_page=physical,
       correct_refusal=None,false_refusal=base['request_success'] and status in {'INSUFFICIENT_EVIDENCE','AMBIGUOUS','OPERATION_UNSUPPORTED'},
       severe_wrong_pass=status=='PASS' and not semantic_ok,missing_source_success=status=='PASS' and not citations)
def metric(values):
    n=sum(x is True for x in values);d=sum(x is not None for x in values);u=len(values)-d
    if not d:return dict(numerator=n,denominator=d,unjudged=u,rate=None,wilson95=None)
    p=n/d;z=1.95996398454;center=(p+z*z/(2*d))/(1+z*z/d);half=z*math.sqrt(p*(1-p)/d+z*z/(4*d*d))/(1+z*z/d)
    return dict(numerator=n,denominator=d,unjudged=u,rate=p,wilson95=[center-half,center+half])
def summarize(rows):
    judged=[dict(row,score=score(row)) for row in rows]
    fields=['core_semantic','numeric_and_unit','joint','citation_support','physical_page','correct_refusal','false_refusal','request_success','timeout','execution_failure','severe_wrong_pass','missing_source_success']
    def group(values):return {k:metric([r['score'][k] for r in values]) for k in fields}
    strata={}
    for name,getkey in [('type',lambda r:r['label']['operation']),('company',lambda r:r['label'].get('company','MULTI_OR_UNSPECIFIED')),('document',lambda r:r['label']['source_filing'])]:
        groups=defaultdict(list)
        for row in judged:groups[getkey(row)].append(row)
        strata[name]={key:group(values) for key,values in groups.items()}
    families=defaultdict(list)
    for row in judged:families[row['label']['family_id']].append(row['score']['core_semantic'])
    rng=random.Random(20261015);samples=[];keys=list(families)
    for _ in range(10000):
        v=[x for key in rng.choices(keys,k=len(keys)) for x in families[key]];samples.append(sum(v)/len(v))
    samples.sort()
    return dict(metrics=group(judged),strata=strata,questions=len(judged),families=len(families),
        family_cluster_bootstrap_core95=[samples[250],samples[9749]],interval_scope='Descriptive finite purposive sample. Repeated questions within 7 financial families are dependent; Wilson is not a generalization guarantee.',
        review_grade='AI_PDF_SOURCE_REVIEW_NOT_HUMAN_GOLD',human_gold_judged=0,per_question=judged)
def main():
    result=summarize(json.loads((OUT/'validation-round2-raw.json').read_text(encoding='utf-8')))
    save('validation-scored.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in {'per_question','strata'}},indent=2))
if __name__=='__main__':main()
