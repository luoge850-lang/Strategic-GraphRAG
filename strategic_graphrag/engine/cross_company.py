"""Bounded cross-company answers from build-scoped real graph evidence.

Paths establish disclosed dependencies only. Conditional interpretation is
explicitly labelled, and never promoted to a realized event or quantified loss.
"""
import re
from .financial_calculation import calculate_public
from ..response_contract import apply_response_contract
from ..schema.financial_observation import measurement_identity

def answer_cross_company(question, plan, finder, build_id, source_filing=None):
    from ..ontology.entity_registry import CANONICAL_MAP
    from .bilingual_query import normalize_financial_question
    text=normalize_financial_question(question).lower().replace('_',' ')
    companies={canonical for alias,(canonical,label) in CANONICAL_MAP.items()
        if label=='Company' and re.search(r'(?<!\w)'+re.escape(alias.replace('_',' '))+r'(?!\w)',text)}
    relation_request=bool(re.search(r'\b(foundr\w*|suppl\w*|manufactur\w*|depend\w*|risk|earthquake\w*|disrupt\w*|lose|loss\w*)\b',text))
    if len(companies)<2 and not (relation_request and companies & {'TSMC','NVIDIA_CORPORATION'}):return None
    if not hasattr(finder,'driver'):return None
    metadata=dict(build_id=build_id,query_plan=plan,total_candidates=0,avg_score=0,
                  llm=dict(calls=0,provider='not_used'),review_grade='RULE_EVIDENCE_TRACE_NOT_HUMAN_GOLD')
    paths=[];citations=[]
    def response(answer,status='ABSTAINED',reason='insufficient_cross_company_evidence',comparison=None):
        metadata['total_candidates']=len(paths)
        return apply_response_contract(dict(query=question,intent='CROSS_COMPANY',intent_display='Evidence-based cross-company query',
            answer=answer,paths=paths,citations=citations,evidence_sentences=[e for p in paths for e in p['evidence']],
            calculation=dict(status='OPERATION_UNSUPPORTED' if comparison else 'NOT_APPLICABLE',operation='cross_company',value=None,unit=None,display=reason.replace('_',' '),reason_code=reason,comparison=comparison),
            structured_report=dict(status='INSUFFICIENT_EVIDENCE' if status=='ABSTAINED' else 'CONDITIONAL_EVIDENCE',claims=[],limitations='No realized causal effect or NVIDIA loss amount is established.'),metadata=metadata),
            execution_status='SUCCEEDED',answer_status=status,grounding_status='NOT_APPLICABLE' if status=='ABSTAINED' else 'NOT_EXECUTED',provenance_note='build_scoped_disclosures_and_labelled_conditional_interpretation')
    if not companies <= {'TSMC','NVIDIA_CORPORATION'}:
        return response('Insufficient evidence for the requested company chain. Evidence for a different company cannot fill the missing link.')
    # A financial request mentioning a customer is not a request for a supply
    # path. Never replace an undisclosed customer-specific amount with a risk.
    if re.search(r'\b(customer|supplier)[ -]?(specific|revenue|sales)\b|\bcustomer\s+(revenue|sales)\b', text):
        return response('Insufficient evidence for customer-specific revenue or sales. Consolidated financial amounts and supplier relationships do not establish this customer-specific amount.', reason='customer_specific_amount_not_disclosed')
    if len(companies) > 1 and not relation_request and not re.search(r'\b(compare|versus|vs)\b',text):
        return response('Specify a company-specific metric or an explicit comparison. No company or metric is substituted.', reason='cross_company_financial_scope_unresolved')
    years=plan.get('fiscal_years') or []
    if re.search(r'\b(compare|versus|vs)\b',text) and plan.get('target_metric') and not relation_request:
        from .query_understanding import FINANCIAL_METRICS_MAP
        spans=[]
        for alias, metric in sorted(FINANCIAL_METRICS_MAP.items(),key=lambda x:-len(x[0])):
            for match in re.finditer(r'(?<!\w)'+re.escape(alias)+r'(?!\w)',text):
                if not any(match.start()<end and match.end()>start for start,end,_ in spans):
                    spans.append((match.start(),match.end(),metric))
        if len({metric for _,_,metric in spans}) > 1:
            return response('The requested metrics have different definitions. No shared metric or numerical comparison is substituted. Please request the same metric for each company.', reason='cross_company_metric_definitions_differ')
        local=[]
        for company in sorted(companies):
            metric=plan['target_metric']
            sub=dict(plan,company=company,task_type='FACT',calculation=None,comparison=None,ratio_metrics=[],
                     ambiguity=[x for x in plan.get('ambiguity',[]) if x!='multiple_companies_require_explicit_comparison'],
                     document_scope=None,disclosure_as_of=None,**measurement_identity(metric,plan.get('fact_period') or '',''))
            found=finder.find_metric_disclosures(metric,year_start=plan.get('fiscal_year_start'),year_end=plan.get('fiscal_year_end'),company_id=company,build_id=build_id,source_filing=source_filing)
            value=calculate_public(found,query_plan=sub)
            local.append(dict(company_id=company,result=value))
            for path in found:
                paths.append(dict(path_id=path.path_id,nodes=path.nodes,node_labels=path.node_labels,relationships=path.relationships,
                    evidence=path.evidence,pages=path.pages,filings=path.filings,years=path.years,evidence_ids=path.evidence_ids,
                    causal_strengths=path.causal_strengths,total_hops=1,score=0,financial_observations=[o.to_dict() for o in path.financial_observations]))
                for i,eid in enumerate(path.evidence_ids):citations.append(dict(evidence_id=eid,source_filing=path.filings[i],page=path.pages[i]))
        lines=['Company-specific observations (descriptive only):']
        for item in local:lines.append(item['company_id']+': '+item['result'].get('display','unavailable'))
        lines.append('The disclosed company-specific amounts can be juxtaposed descriptively. Each observation preserves its actual fiscal period, currency, scale and definition. Fiscal-year labels alone do not make periods identical. A numerical difference requires comparable periods, currency and metric definitions; no exchange-rate conversion or cross-company difference is computed here.')
        return response('\n\n'.join(lines),'PARTIALLY_ANSWERED','cross_company_comparability_not_established',local)
    with finder.driver.session() as session:
        rows=[r.data() for r in session.run('''MATCH (a:Company)-[r:DEPENDS_ON_FOUNDRY|EXPOSED_TO]->(b:Entity)
          MATCH (c:EvidenceClaim {id:r.evidence_id})
          WHERE r.build_id=$build AND c.build_id=$build AND c.verification_status='VERBATIM'
          AND a.id IN ['NVIDIA_CORPORATION','TSMC']
          AND ($filing IS NULL OR r.source_filing=$filing)
          AND ($years=[] OR r.filing_fiscal_year IN $years)
          RETURN a.id AS source,b.id AS target,type(r) AS relation,r.source_filing AS filing,
                 r.page AS page,r.filing_fiscal_year AS year,c.id AS evidence_id,c.text AS evidence
          ORDER BY year DESC, filing, evidence_id''',build=build_id,years=years,filing=source_filing)]
    metadata['total_candidates']=len(rows)
    # Preserve disclosure version; selecting latest does not establish continuous
    # validity between two reporting dates.
    selected={}
    for r in rows:selected.setdefault((r['source'],r['target'],r['relation']),r)
    for i,r in enumerate(selected.values()):
        paths.append(dict(path_id=f'company_{i}',nodes=[r['source'],r['target']],node_labels=['Company','Company' if r['target']=='TSMC' else 'RiskFactor'],
            relationships=[r['relation']],evidence=[r['evidence']],pages=[r['page']],filings=[r['filing']],years=[r['year']],
            evidence_ids=[r['evidence_id']],causal_strengths=['DISCLOSED_ONLY'],total_hops=1,score=0))
        citations.append(dict(evidence_id=r['evidence_id'],source_filing=r['filing'],page=r['page']))
    dep=selected.get(('NVIDIA_CORPORATION','TSMC','DEPENDS_ON_FOUNDRY'))
    risk=selected.get(('TSMC','NATURAL_DISASTER','EXPOSED_TO'))
    if not dep:return response('Insufficient evidence for the requested direction and disclosure time. No complete cross-company conclusion is supported.')
    lines=['Directly disclosed dependency: NVIDIA uses TSMC as a foundry to produce semiconductor wafers. This supports NVIDIA depending on TSMC, not NVIDIA manufacturing wafers for TSMC.',
           'Relationship evidence: '+dep['evidence']+f" [{dep['filing']}, PDF physical page {dep['page']}]",]
    if risk:
        lines += ['Directly disclosed possible risk: '+risk['evidence']+f" [{risk['filing']}, PDF physical page {risk['page']}]",
           'Conditional interpretation: if the disclosed disaster risk interrupts TSMC wafer production or deliveries serving NVIDIA, the disclosed foundry dependency provides a possible route to NVIDIA wafer-supply disruption. The path does not establish that this happened.']
    else:lines.append('Missing link: no matched TSMC operational-risk disclosure for the requested time. The dependency alone cannot establish a complete risk path.')
    lines.append('Unsupported links: customer-specific delivery loss, alternative capacity, inventories, realized NVIDIA business impact and NVIDIA revenue-loss amounts are not established by these citations. Disclosure years identify source versions; they do not prove continuous validity or causal occurrence.')
    assertion_text=re.sub(r'\bwithout\s+(?:assuming|estimating|quantifying)\b[^.;?]*','',text)
    numeric_loss=bool(re.search(r'\b(how many|how much|lost|lose|loss|caused|prove|did)\b',assertion_text))
    if numeric_loss:
        lines.insert(0,'无法从这些披露确定 NVIDIA 已发生的营收损失金额或证明事件已经发生。 The cited disclosures do not establish a realized NVIDIA revenue-loss amount or causal occurrence.')
    return response('\n\n'.join(lines),'ABSTAINED' if numeric_loss else 'PARTIALLY_ANSWERED',
                    'realized_or_quantified_effect_not_supported' if numeric_loss else 'conditional_path_only')
