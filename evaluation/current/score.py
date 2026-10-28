"""Layered, source-reference scorer. No response-derived reference answers."""
import math,re
ALIASES={'TOTAL_COMPREHENSIVE_INCOME':'COMPREHENSIVE_INCOME','CASH_PAID_INTEREST':'CASH_PAID_FOR_INTEREST'}
canonical=lambda x:ALIASES.get(x,x)
norm=lambda x:re.sub(r'[^a-z0-9]','',str(x).casefold())
def eq(a,b,t):return isinstance(a,(float,int)) and not isinstance(a,bool) and math.isfinite(a) and abs(a-b)<=t
def score(r,l,build):
    c=r.get('calculation') or {};claimed=r.get('answer_status')=='ANSWERED' or c.get('status')=='PASS'
    if l.get('items'):
        got={x.get('slot_id'):x for x in c.get('items',[])};checks=[]
        for i,label in enumerate(l['items'],1):
            item=got.get('s'+str(i),{});fake=dict(r,calculation=item,answer_status='ANSWERED' if item.get('status')=='PASS' else 'ABSTAINED')
            checks.append(score(fake,label,build))
        complete=all(x['core'] is True for x in checks) and c.get('answered_count')==len(checks) and c.get('required_count')==len(checks) and len(got)==len(checks)
        core=complete and claimed
        return dict(core=core,successful=claimed,precision_correct=core if claimed else None,numeric=all(x['numeric'] for x in checks),identity=all(x['identity'] for x in checks),citation_identity=all(x['citation_identity'] for x in checks),citation_reference=all(x['citation_reference'] for x in checks),metadata_joint=all(x['metadata_joint'] for x in checks),request_joint=core and all(x['request_joint'] for x in checks),false_refusal=not complete,severe_wrong_success=claimed and not core,items=checks)
    ops=l.get('operands',[])
    if not ops:
        if l['kind']=='relationship':return dict(core=None,successful=claimed,precision_correct=None,numeric=None,identity=None,citation_identity=None,citation_reference=None,metadata_joint=None,request_joint=None,false_refusal=r.get('answer_status')=='ABSTAINED',severe_wrong_success=None,review_required=True)
        ok=r.get('answer_status') in l.get('expected_statuses',['ABSTAINED']) and not claimed and c.get('value') is None
        return dict(core=ok,successful=claimed,precision_correct=ok if claimed else None,numeric=None,identity=None,citation_identity=None,citation_reference=None,metadata_joint=None,request_joint=ok,false_refusal=None,severe_wrong_success=claimed,boundary_status=ok)
    obs=c.get('observations') or [];checks=[];locs={(x.get('evidence_id'),x.get('source_filing'),x.get('page')) for x in r.get('citations') or []}
    for op in ops:
        found=[o for o in obs if o.get('company_id')==op['company_id'] and canonical(o.get('metric_id'))==canonical(op['metric_id']) and (o.get('fact_period') or o.get('fiscal_period'))==op['fact_period']]
        identity=bool(found) and all(o.get('source_filing')==op['disclosure_version'] and o.get('build_id')==build and all(o.get(k)==op[k] for k in ['business_scope','measurement_nature','period_granularity']) and o.get('financial_basis')==op.get('financial_basis','ACTUAL_REPORTED') for o in found)
        amount=bool(found) and all(eq(o.get('value'),op['value'],op.get('tolerance',.5)) and o.get('unit')==op['unit'] for o in found)
        ci=bool(found) and all((o.get('claim_id'),o.get('source_filing'),o.get('page')) in locs for o in found)
        cr=ci and all(any(o['source_filing']==s['file'] and o['page']==s['page'] and norm(o.get('row_label'))==norm(s['anchor']) for s in op['sources']) for o in found)
        metadata=bool(found) and all(all(o.get(k) not in [None,'','UNKNOWN','UNKNOWN_TABLE'] for k in ['document_sha256','source_section','table_name','row_label','column_label','basis_evidence','period_end']) and o.get('statement_type') not in [None,'','UNKNOWN','FINANCIAL_TABLE','FINANCIAL_STATEMENTS'] for o in found)
        checks.append(dict(identity=identity,amount=amount,citation_identity=ci,citation_reference=cr,metadata=metadata,observations=found))
    expected={(o['company_id'],canonical(o['metric_id']),o['fact_period']) for o in ops};extra=any((o.get('company_id'),canonical(o.get('metric_id')),o.get('fact_period') or o.get('fiscal_period')) not in expected for o in obs)
    identity=all(x['identity'] for x in checks) and not extra
    numeric=eq(c.get('value'),l['expected_value'],l.get('answer_tolerance',.5)) and c.get('unit')==l['expected_unit']
    operation=not l.get('expected_operation') or c.get('operation')==l['expected_operation']
    core=claimed and r.get('answer_status')=='ANSWERED' and identity and numeric and operation and all(x['amount'] for x in checks)
    ci=all(x['citation_identity'] for x in checks);cr=all(x['citation_reference'] for x in checks);meta=all(x['metadata'] for x in checks)
    return dict(core=bool(core),successful=claimed,precision_correct=bool(core) if claimed else None,numeric=bool(numeric),identity=bool(identity),operation=bool(operation),citation_identity=ci,citation_reference=cr,metadata_joint=meta,request_joint=bool(core and cr and meta),false_refusal=r.get('answer_status')=='ABSTAINED',severe_wrong_success=bool(claimed and not core),details=checks)
