"""Layered closure scoring, including calculations and explicit safe refusal."""
import copy
from deployment.complete_fact_scoring import score_fact, equal_number

def aggregate(values):
    return False if any(v is False for v in values) else None if any(v is None for v in values) else True

def score(response,label):
    calc=response.get('calculation') or {}
    status=calc.get('status')
    if label['operation']=='SAFE_REFUSAL':
        safe=status in label['accepted_statuses'] and calc.get('value') is None
        return dict(core_semantic=safe,numeric_and_unit=None,citation_support=None,
                    physical_page=None,metadata_complete=None,joint=safe,
                    safe_refusal=safe,false_refusal=None,wrong_pass=status=='PASS',status=status)
    if not label.get('required_facts'):
        out=score_fact(response,label,layered=True)
    else:
        observations=calc.get('observations') or []
        required={l['fact_period']:l for l in label['required_facts']}
        individual=[]
        for obs in observations:
            reference=required.get(obs.get('fact_period'))
            if reference is None:
                individual.append(dict(core_semantic=False,numeric_and_unit=False,citation_support=None,
                    physical_page=None,metadata_complete=False,joint=False))
                continue
            one=copy.deepcopy(response)
            one['calculation'].update(observations=[obs],value=obs.get('value'),unit=obs.get('unit'))
            individual.append(score_fact(one,reference,layered=True))
        out={k:aggregate([r[k] for r in individual]) if individual else False
             for k in ('core_semantic','numeric_and_unit','citation_support','physical_page','metadata_complete','joint')}
        covered={o.get('fact_period') for o in observations}
        complete=set(required)<=covered
        numeric=equal_number(calc.get('value'),label['answer_value'],label['tolerance'])
        unit=calc.get('unit')==label['answer_unit']
        out['numeric_and_unit']=numeric and unit
        out['core_semantic']=out['core_semantic'] and complete and numeric and unit and status=='PASS'
        out['joint']=aggregate([out['joint'],out['core_semantic']])
    out.update(status=status,false_refusal=status in {'INSUFFICIENT_EVIDENCE','AMBIGUOUS','OPERATION_UNSUPPORTED'},
               safe_refusal=None,wrong_pass=status=='PASS' and not out['core_semantic'])
    return out
