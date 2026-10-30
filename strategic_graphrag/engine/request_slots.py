"""Bind bounded financial clauses without discarding unresolved requirements."""
import re


def bind_slots(question, original, plan, spans, parse):
    spans=sorted(spans)
    # Commas in a source prefix qualify the question, not an extra output.
    # Relation questions have no financial slots; their obligations are checked
    # by the build-scoped relation executor.
    # Restriction words are obligations, not decorative retrieval keywords.
    if re.search(r'\b(excluding|except|only|adjusted|non[- ]gaap|china|chinese|europe|asia|united states|domestic|overseas)\b',question,re.I):
        plan.ambiguity.append('unresolved_restrictive_qualifier')
    if not spans:return []
    # A comma inside an alias or date is not a list separator.
    for join in re.finditer(r',\s*',question):
        if join.start()<spans[0][0] and re.match(r'\s*(?:for|in|using|according to)\b',question,re.I) and re.match(r'(?:what|how much|give|report)\b',question[join.end():],re.I):continue
        if join.start()<spans[0][0] and re.search(r'\b(report|10-k|filing|disclosure|according|consolidated column)\b',question[:join.start()],re.I):continue
        if any(start<=join.start()<end for start,end,_,_ in spans):continue
        tail=question[join.end():]
        if plan.target_metric=='DATA_CENTER_NETWORKING_REVENUE' and re.match(r'how much came from\s+networking\b',tail,re.I):continue
        if not tail.strip(' .?;'):continue
        if re.match(r'(?:not|rather than)\s+(?:the\s+)?(?:company|total|balance|annual|quarter)',tail,re.I):continue
        # Disclosure adjuncts qualify the preceding fact; they are not list
        # items. Unknown conjuncts within/after the adjunct remain checked by
        # the planner and subsequent comma iterations.
        if re.match(r'(?:using|from|in|as\s+(?:disclosed|reported|stated)\s+in|according\s+to)\s+(?:(?:its|the)\s+)?20\d{2}\s+(?:(?:annual\s+)?report|(?:form\s+)?10-k|filing|disclosure)\b',tail,re.I):continue
        if re.match(r'(?:and\b\s*)?(?:in\b|at\b|for\b|(?:20\d{2})\b|fiscal\b|FY\s*20\d{2}\b)',tail,re.I):continue
        if not any(0<=start-join.end()<=4 for start,_,_,_ in spans):
            plan.ambiguity.append('additional_requested_item_not_resolved')
    if plan.calculation not in {None,'UNIT_CONVERSION'} or plan.comparison:
        return [dict(slot_id='s1',company_id=plan.company,metric_id=plan.target_metric,
            fact_period=plan.fact_period,source_filing=plan.document_scope,
            business_scope=plan.business_scope,operation=plan.calculation or 'CROSS_YEAR',
            operands=list(plan.ratio_metrics),original_fragment=original,
            disclosure_as_of=plan.disclosure_as_of,financial_basis=plan.financial_basis,
            measurement_nature=plan.measurement_nature,value_kind=plan.value_kind,
            period_granularity=plan.period_granularity,fact_years=list(plan.fiscal_years),
            unresolved_conditions=list(plan.ambiguity),
            output_currency=plan.calculation_target_currency,output_scale=plan.calculation_target_scale)]
    prefix=question[:spans[0][0]]
    if re.search(r'\b(?:and|plus)\s*$', prefix, re.I):
        plan.ambiguity.append('preceding_requested_item_not_resolved')
    fragments=[]
    for index,(start,end,metric,alias) in enumerate(spans):
        stop=spans[index+1][0] if index+1<len(spans) else len(question)
        text=question[start:stop]
        # The conjunction and any explicit next-company prefix belong to the next item.
        separator=[m for m in re.finditer(r'\s+(?:and|plus|also)\s+|,\s*(?:and\s+)?',text,re.I) if m.start()>=end-start]
        if index+1<len(spans) and separator:
            cut=separator[-1]
            next_prefix=text[cut.end():]
            text=text[:cut.start()]
        else:next_prefix=''
        fragments.append((prefix+text,metric,text))
        if next_prefix.strip():prefix=next_prefix
    # Repeated metric with an explicit second period is a second obligation.
    if len(spans)==1:
        parts=re.split(r'\s+(?:and|plus)\s+(?=(?:in\s+)?(?:fiscal|FY|20\d{2})\b)',fragments[0][0],flags=re.I)
        if len(parts)>1:
            fragments=[(parts[0],spans[0][2],parts[0])]+[(prefix+spans[0][3]+' '+part,spans[0][2],part) for part in parts[1:]]
    locals_=[parse(text,_bind_slots=False) for text,_,_ in fragments]
    periods={p.fact_period for p in locals_ if p.fact_period}
    # Only a unique trailing period or unit can be distributed to an omitted item.
    shared_period=next(iter(periods)) if len(periods)==1 else None
    explicit_units=[(p.calculation_target_currency,p.calculation_target_scale) for p in locals_]
    shared_unit=explicit_units[-1] if explicit_units and all(not any(u) for u in explicit_units[:-1]) else (None,None)
    slots=[]
    for index,((text,metric,fragment),local) in enumerate(zip(fragments,locals_)):
        period=local.fact_period or shared_period
        currency,scale=explicit_units[index]
        if not currency and not scale:currency,scale=shared_unit
        if len(fragments)>1 and not period:
            plan.ambiguity.append('slot_period_binding_requires_clarification')
        local_metric=local.target_metric or metric
        slots.append(dict(slot_id=f's{index+1}',company_id=local.company or plan.company,
            metric_id=local_metric,fact_period=period,
            source_filing=local.document_scope or plan.document_scope,
            disclosure_as_of=local.disclosure_as_of or plan.disclosure_as_of,
            business_scope=local.business_scope,measurement_nature=local.measurement_nature,
            financial_basis=local.financial_basis,
            value_kind=local.value_kind,period_granularity=local.period_granularity,
            output_currency=currency,output_scale=scale,
            operation='UNIT_CONVERSION' if currency or scale else 'FACT',
            operands=[local_metric],fact_years=[int(re.search(r'20\d{2}',period).group())] if period else [],
            original_fragment=fragment if question==original else original,
            normalized_fragment=text,fragment_mapping='exact_normalized_clause' if question==original else 'whole_original_preserved_after_normalization',
            unresolved_conditions=list(local.ambiguity)))
    return slots
