"""Execute bounded financial answer slots and check request completeness.

This module never supplies reference values. Every returned amount must pass
the existing typed calculator, and every requested slot keeps its own status.
"""
from .financial_calculation import calculate_public
from ..schema.financial_observation import measurement_identity
from ..response_contract import apply_response_contract
import re


def slot_plan(slot, build_id):
    """Construct a calculator contract from this slot, never a global plan copy."""
    period=slot.get('fact_period');years=slot.get('fact_years') or ([int(re.search(r'20\d{2}',period).group())] if period else [])
    operation=slot.get('operation','FACT')
    return dict(company=slot.get('company_id'),target_metric=slot.get('metric_id'),fact_period=period,
        fiscal_years=years,fiscal_year_start=min(years) if years else None,fiscal_year_end=max(years) if years else None,
        task_type='FACT' if operation in {'FACT','UNIT_CONVERSION'} else 'CALCULATION',
        calculation=None if operation=='FACT' else operation,comparison='POINT_TO_POINT' if operation=='CROSS_YEAR' else None,
        ratio_metrics=slot.get('operands',[]) if operation=='RATIO' else [],
        document_scope=slot.get('source_filing'),disclosure_as_of=slot.get('disclosure_as_of'),
        financial_basis=slot.get('financial_basis','ACTUAL_REPORTED'),
        business_scope=slot.get('business_scope'),measurement_nature=slot.get('measurement_nature'),
        value_kind=slot.get('value_kind'),period_granularity=slot.get('period_granularity'),
        calculation_target_currency=slot.get('output_currency'),calculation_target_scale=slot.get('output_scale'),
        ambiguity=slot.get('unresolved_conditions',[]),build_id=build_id)


def execute_slot(slot,finder,build_id,source_filing):
    local=slot_plan(slot,build_id);years=local['fiscal_years'];paths=[]
    metrics=slot.get('operands') if slot.get('operation')=='RATIO' else [slot['metric_id']]
    for metric in metrics or []:
        paths.extend(finder.find_metric_disclosures(metric,year_start=min(years) if years else None,
            year_end=max(years) if years else None,company_id=slot['company_id'],build_id=build_id,
            source_filing=source_filing or slot.get('source_filing')))
    return calculate_public(paths,query_plan=local),paths


def respond(question,plan,build_id,answer,reason,*,suggestions=None,scope_suggestion=None):
    return apply_response_contract(dict(query=question,intent='CLARIFICATION',intent_display='需要明确查询条件',answer=answer,
        paths=[],citations=[],evidence_sentences=[],structured_report=dict(status='CLARIFICATION_REQUIRED',claims=[]),
        calculation=dict(status='AMBIGUOUS',operation='none',value=None,unit=None,display=answer,reason_code=reason),
        metadata=dict(build_id=build_id,query_plan=plan,total_candidates=0,avg_score=0,
                      clarification_suggestions=suggestions or [],scope_suggestion=scope_suggestion)),
        execution_status='SUCCEEDED',answer_status='ABSTAINED',grounding_status='NOT_APPLICABLE',provenance_note='request_scope_clarification')


def answer_financial_request(question,plan,finder,build_id,source_filing,serialize):
    ambiguities=plan.get('ambiguity') or []
    if 'unresolved_financial_identity_modifier' in ambiguities:
        return respond(question,plan,build_id,'The specific financial component or balance modifier is unresolved. A broader metric cannot substitute for it.','unresolved_financial_identity_modifier')
    if 'attributed_financial_outcome_not_disclosed' in ambiguities and plan.get('task_type')!='RELATION':
        return respond(question,plan,build_id,'The requested event-attributed financial amount is not established by the stored disclosures. A reported company total cannot substitute for an attributed loss.','attributed_financial_outcome_not_disclosed')
    if plan.get('task_type')=='RELATION':return None
    if any(x in ambiguities for x in ['unsupported_financial_component','pro_forma_request_not_supported']):
        return respond(question,plan,build_id,'The requested financial component or basis is not supported; consolidated actual amounts cannot substitute for it.','unsupported_financial_component_or_basis')
    if 'investing_cash_flow_or_capital_expenditure' in ambiguities:
        return respond(question,plan,build_id,
            '“投资花了多少现金”可能指投资活动现金净流出，也可能指资本支出。两者口径不同，请选择所需指标。 Investing cash-flow outflow and capital expenditures are different measures.',
            'investing_cash_flow_or_capital_expenditure',suggestions=plan.get('clarification_options'))
    if 'additional_requested_item_not_resolved' in ambiguities:
        return respond(question,plan,build_id,'问题包含尚未识别的其他要求，不能只回答已识别的一项并标记完整。请明确每个指标。 Some requested items were not resolved; a subset cannot be marked complete.','additional_requested_item_not_resolved')
    if 'unresolved_restrictive_qualifier' in ambiguities:
        return respond(question,plan,build_id,'问题包含排除、地区、仅限或调整口径；当前不能可靠满足这些条件，不能以公司总额替代。请明确所需披露口径。','unresolved_restrictive_qualifier')
    if 'slot_period_binding_requires_clarification' in ambiguities:
        return respond(question,plan,build_id,'无法将期间唯一绑定到每项需求，请逐项明确年份。','slot_period_binding_requires_clarification')
    company=plan.get('company')
    numeric=plan.get('task_type') in {'FACT','CALCULATION','COMPARISON'}
    slots=plan.get('requested_facts') or []
    if not numeric or finder is None or not slots or any(not s.get('company_id') for s in slots):return None
    if source_filing and plan.get('document_scope') and source_filing!=plan['document_scope']:
        return respond(question,plan,build_id,f"当前选中 {source_filing}，问题指定的披露版本为 {plan['document_scope']}。请切换到对应文件或全部语料后重新查询。",'selected_filing_conflicts_with_requested_disclosure',scope_suggestion='all')
    if source_filing and hasattr(finder,'driver'):
        with finder.driver.session() as session:
            owners=[r['company'] for r in session.run('MATCH(o:FinancialObservation) WHERE o.build_id=$build AND o.source_filing=$file RETURN DISTINCT o.company_id AS company',build=build_id,file=source_filing)]
        if owners and company not in owners:
            return respond(question,plan,build_id,
                f"当前文件 {source_filing} 中已录入的财务主体为 {', '.join(owners)}，问题指定 {company}。请切换到对应公司的文件或全部语料；不能用另一家公司数据替代。",
                'selected_filing_conflicts_with_requested_company',scope_suggestion='all')
    metrics=plan.get('requested_metrics') or []
    multi=(len(slots)>1 or len(metrics)>1) and plan.get('calculation') not in {'RATIO'}
    if not multi:
        slot=slots[0]
        result,found=execute_slot(slot,finder,build_id,source_filing)
        used={o['id'] for o in result.get('observations') or []}
        accepted=[serialize(p) for p in found if any(o.id in used for o in p.financial_observations)]
        cites=[dict(evidence_id=o['claim_id'],source_filing=o['source_filing'],page=o['page']) for o in result.get('observations') or []]
        passed=result.get('status')=='PASS'
        result.update(required_count=1,answered_count=int(passed),request_slot=slot)
        return apply_response_contract(dict(query=question,intent='FINANCIAL_FACT',intent_display='Financial fact',answer=result.get('display',''),
            calculation=result,paths=accepted,citations=cites,evidence_sentences=[e for p in accepted for e in p['evidence']],
            structured_report=dict(status='TYPED_FINANCIAL_ANSWER' if passed else 'INSUFFICIENT_EVIDENCE',claims=[]),
            metadata=dict(build_id=build_id,query_plan=plan,total_candidates=len(found),avg_score=0,answer_coverage=dict(required=1,answered=int(passed),complete=passed),llm=dict(calls=0,provider='not_used'))),
            execution_status='SUCCEEDED',answer_status='ANSWERED' if passed else 'ABSTAINED',grounding_status='VERIFIED' if passed else 'NOT_APPLICABLE',provenance_note='slot_contract_typed_financial_execution')
    if plan.get('calculation') not in {None,'UNIT_CONVERSION'} or (plan.get('comparison') and len({s['metric_id'] for s in slots})>1):
        return respond(question,plan,build_id,'多个指标同时涉及年度比较或其他运算，请分别指定运算。不会只计算第一项。','multiple_metrics_operation_not_supported')
    if any(not slot.get('fact_period') for slot in slots):
        return respond(question,plan,build_id,'多指标查询还需要明确同一个事实期间，例如 FY2025；披露文件年份不等于事实年份。','multi_fact_period_requires_clarification')
    if len(slots)>8:
        return respond(question,plan,build_id,'一次最多查询8个明确财务指标，请分批查询。','requested_fact_budget_exceeded')
    items=[];paths=[];citations=[];seen=set();observations=[]
    for slot in slots:
        metric=slot['metric_id']; period=slot['fact_period']
        result,found=execute_slot(slot,finder,build_id,source_filing)
        items.append(dict(metric_id=metric,slot_id=slot['slot_id'],request_slot=slot,**result))
        if result['status']!='PASS':continue
        observations.extend(result.get('observations') or [])
        used={o['id'] for o in result.get('observations') or []}
        for path in found:
            if not any(o.id in used for o in path.financial_observations):continue
            entry=serialize(path);entry['evidence_role']='ANSWER_CRITICAL';paths.append(entry)
        for obs in result.get('observations') or []:
            key=(obs['claim_id'],obs['source_filing'],obs['page'])
            if key not in seen:
                seen.add(key);citations.append(dict(evidence_id=key[0],source_filing=key[1],page=key[2]))
    answered=sum(i['status']=='PASS' for i in items)
    complete=answered==len(slots)
    state='ANSWERED' if complete else 'PARTIALLY_ANSWERED' if answered else 'ABSTAINED'
    status='PASS' if complete else 'PARTIAL' if answered else 'INSUFFICIENT_EVIDENCE'
    lines=[f'已回答 {answered}/{len(slots)} 项。'+(' 所有请求指标均已逐项核对。' if complete else ' 请求未完整满足，不能标记整体PASS。')]
    for item in items:
        slot=item['request_slot']
        lines.append(f"{slot['company_id']} · {item['metric_id']} · {slot['fact_period']}: "+(item['display'] if item['status']=='PASS' else '未能提供 / '+item.get('reason_code','unsupported')))
    answer='\n\n'.join(lines)
    calculation=dict(schema='deterministic-calculation/v1',status=status,operation='multi_fact',items=items,
        observations=observations,value=[dict(metric_id=i['metric_id'],value=i.get('value'),unit=i.get('unit')) for i in items] if complete else None,
        unit='per-item',display=answer,reason_code='all_requested_facts_supported' if complete else 'requested_facts_incomplete',
        required_count=len(slots),answered_count=answered,missing_metrics=[i['metric_id'] for i in items if i['status']!='PASS'],missing_slots=[i['slot_id'] for i in items if i['status']!='PASS'])
    return apply_response_contract(dict(query=question,intent='MULTI_FINANCIAL_FACT',intent_display='多指标财务查询',answer=answer,
        calculation=calculation,paths=paths,citations=citations,evidence_sentences=[e for p in paths for e in p['evidence']],
        structured_report=dict(status='TYPED_FINANCIAL_ANSWER' if complete else 'PARTIAL_FACT_COVERAGE',claims=[]),
        metadata=dict(build_id=build_id,query_plan=plan,total_candidates=len(paths),top_paths=len(paths),avg_score=0,
            answer_evidence_status='TYPED_FACTS_CHECKED',answer_coverage=dict(required=len(slots),answered=answered,complete=complete),llm=dict(calls=0,provider='not_used'))),
        execution_status='SUCCEEDED',answer_status=state,grounding_status='VERIFIED' if complete else 'INSUFFICIENT' if answered else 'NOT_APPLICABLE',provenance_note='per_requested_fact_coverage_and_typed_calculation')
