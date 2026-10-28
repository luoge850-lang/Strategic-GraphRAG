"""Separate AI source audit. Preserve frozen automatic scores verbatim."""
import json,hashlib,re
from deployment.acceptance_20261015 import ROOT,OUT,save
from deployment.score_five_acceptance import metric
_append_save=save
def save(name,data):
    path=OUT/name
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=data:raise ValueError('existing source review differs')
        return
    _append_save(name,data)
def main():
    packet=json.loads((OUT/'graph-review-packet.json').read_text(encoding='utf-8'))
    judgments=[]
    for row in packet['observations']:
        o=row['observation']
        judgments.append(dict(index=row['index'],observation_id=o['id'],company=True,metric=True,business_scope=True,
            value_and_sign=True,unit_currency_scale=True,period_and_disclosure=True,physical_page=True,joint=True,
            review_grade='AI_PDF_SOURCE_READING_NOT_HUMAN_GOLD',
            basis='Read original PDF table heading, column period and row; percentage-of-revenue tables kept as ratios; cash-flow working-capital rows kept as movements; TSMC left consolidated table kept apart from right unconsolidated table.'))
    rel=[]
    for r in packet['relations']:
        rel.append(dict(source=r['source'],target=r['target'],relation=r['relation'],claim_id=r['claim']['id'],entities=True,
            type_and_direction=True,source_direct_support=True,modality_preserved=True,locator=True,joint=True,
            evidence=r['claim']['text'],review_grade='AI_PDF_SOURCE_READING_NOT_HUMAN_GOLD'))
    save('graph-source-judgments.json',dict(financial=judgments,relations=rel,financial_joint=metric([r['joint'] for r in judgments]),
         relation_support=metric([r['source_direct_support'] for r in rel]),
         limits='40 stratified financial observations, not all 355. Three retained company relations enumerated, after broad rules were quarantined. No human review. Cannot infer universal 95% quality from this sample.'))
    labels=json.loads((OUT/'validation-labels.json').read_text())
    # Independent source checklist was transcribed from two original PDF pages.
    expected={}
    for label in labels:
        if label['operation']=='FACT':expected[(label['source_filing'],label['metric'],label['fact_period'])]=label
    for year,v23,v22 in [(2023,2161735841,2263891292),(2024,2894307699,2161735841)]:
        for fy,value in [(year,v23),(year-1,v22)]:
            expected[(f'TSMC-{year}-Annual-Report.pdf','REVENUE',f'FY{fy}')]=dict(source_value=value)
    edges=json.loads((OUT/'five-package-round2/graph.json').read_text())['edges']
    checklist=[]
    for (file,metric_id,period),label in expected.items():
        matches=[e for e in edges if e['source_filing']==file and e['target']==metric_id and f"FY{e['fact_year']}"==period and e['source']=='TSMC' and float(e['metric_value'])==label['source_value'] and e['metric_unit']=='TWD thousands']
        checklist.append(dict(file=file,page=67 if '2023' in file else 64,metric_id=metric_id,fact_period=period,expected_value=label['source_value'],extracted=bool(matches)))
    rel_expected=[('NVIDIA_CORPORATION','TSMC','foundry'),('NVIDIA_CORPORATION','SAMSUNG_ELECTRONICS','foundry'),
       ('NVIDIA_CORPORATION','SK_HYNIX','memory_supplier'),('NVIDIA_CORPORATION','MICRON_TECHNOLOGY','memory_supplier'),
       ('NVIDIA_CORPORATION','SAMSUNG_ELECTRONICS','memory_supplier'),('NVIDIA_CORPORATION','HON_HAI','assembly_testing_packaging'),
       ('NVIDIA_CORPORATION','WISTRON','assembly_testing_packaging'),('NVIDIA_CORPORATION','FABRINET','assembly_testing_packaging'),
       ('TSMC','NATURAL_DISASTER','conditional_operational_risk')]
    relation_checklist=[]
    for source,target,predicate in rel_expected:
        filename='TSMC-2024-Annual-Report.pdf' if source=='TSMC' else '2025-10-K.pdf';page=70 if source=='TSMC' else 9
        relation_checklist.append(dict(source=source,target=target,predicate=predicate,file=filename,page=page,
             extracted=any(e['source']==source and e['target']==target and e['source_filing']==filename and e['relation']!='REPORTS_METRIC' for e in edges)))
    save('independent-extraction-checklist.json',dict(financial=checklist,relationships=relation_checklist,
       financial_recall=metric([r['extracted'] for r in checklist]),relationship_recall=metric([r['extracted'] for r in relation_checklist]),
       review_grade='AI_PDF_SOURCE_CHECKLIST_NOT_HUMAN_GOLD',
       frame='Two original TSMC consolidated tables (32 period-specific financial facts), NVIDIA 2025 p9 supplier paragraph plus TSMC 2024 p70 conditional risk (9 assertions). Chosen from original PDFs, not predicted candidates; this is selected-page recall, not corpus-wide recall.'))
    scores=json.loads((OUT/'validation-scored.json').read_text())
    flags=[dict(item_id='V-SAFE-undisclosed-scope',correct_refusal=False,core_semantic=False,joint=False,
                 reason='Requested TSMC revenue from NVIDIA customers. Response instead supplies a dependency and disaster path. No scalar PASS, but unrelated partial facts do not correctly answer/refuse this fact scope.'),
           dict(item_id='V-SAFE-cross-definition',correct_refusal=False,core_semantic=False,joint=False,
                 reason='Question asks NVIDIA gross profit versus TSMC gross margin. Response displays NVIDIA gross margin instead. It refuses a scalar comparison, but its local observation substitutes the requested metric.')]
    flag_ids={x['item_id'] for x in flags}
    reviewed=[]
    for row in scores['per_question']:
        score=dict(row['score'])
        if row['label']['item_id'] in flag_ids:score.update(correct_refusal=False,core_semantic=False,joint=False)
        reviewed.append(dict(item_id=row['label']['item_id'],automated=row['score'],ai_source_audit=score))
    save('post-freeze-semantic-audit.json',dict(flags=flags,per_question=reviewed,
        core_semantic=metric([r['ai_source_audit']['core_semantic'] for r in reviewed]),
        correct_refusal=metric([r['ai_source_audit']['correct_refusal'] for r in reviewed]),
        protocol='Separate source audit identifies automatic scorer blind spots; frozen labels, scorer, raw responses and 45/52 automatic summary remain unchanged. No code repair, rerun or reference-answer modification.',review_grade='AI_SOURCE_AUDIT_NOT_HUMAN_GOLD'))
    # Correct the supplemental hit-accounting field without altering frozen runs.
    raw=[json.loads(x) for x in (OUT/'retrieval-raw.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup={r['item_id']:r for r in labels};methods={}
    for mode in {r['method'] for r in raw}:
        selected=[r for r in raw if r['method']==mode];unknown=0;judged=0
        for row in selected:
            label=lookup[row['item_id']];refs=label.get('required_facts') or [label]
            for hit in row['hits']:
                meta=hit.get('metadata') or {};text=hit.get('document') or ''
                positive=any((meta.get('source_filing'),meta.get('page')) in {(p['file'],p['page']) for p in ref['direct_locations']} and any(a.lower() in text.lower() for a in ref['row_anchors']) and bool(re.search(r'(?<![\d.])'+re.escape(str(ref['source_value']))+r'(?![\d.])',text.replace(',',''))) for ref in refs)
                judged+=positive;unknown+=not positive
        methods[mode]=dict(judged_positive_hits=judged,unjudged_hits=unknown,negative_hits=0,
            note='Original unjudged_retrieved_hits field subtracts fact coverage from hit counts and is not a valid unique-hit count. Use these explicit hit judgments; coverage/MRR results unchanged.')
    save('retrieval-hit-audit.json',methods)
if __name__=='__main__':main()
