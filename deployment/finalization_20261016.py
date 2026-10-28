"""Bounded closing campaign. All writes are append-only campaign artifacts."""
import copy
import json
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen
from strategic_graphrag.build_identity import source_fingerprint, sha256_file

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/finalization-20261016'
PREVIOUS = ROOT / 'experiments/request-contract-20261015/round2'

def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)

def http(question, port=8008, filing=None):
    payload = dict(question=question, retrieval_mode='auto', synthesize=False,
                   use_cache=False, cross_filing=True, max_paths=10)
    if filing:payload.update(source_filing=filing,cross_filing=False)
    start = time.perf_counter()
    try:
        with urlopen(Request(f'http://127.0.0.1:{port}/query', data=json.dumps(payload).encode(),
                             headers={'Content-Type': 'application/json'}), timeout=60) as response:
            return dict(http_status=response.status, seconds=time.perf_counter()-start,
                        response=json.load(response))
    except Exception as error:
        return dict(http_status=None, seconds=time.perf_counter()-start, error=str(error))

QUESTIONS = [
 'What were NVIDIA revenue in fiscal 2024 and net income in fiscal 2025?',
 'What were NVIDIA revenue in USD billions and net income in USD millions in fiscal 2025?',
 'What were NVIDIA revenue, employee satisfaction in fiscal 2025?',
 'What was NVIDIA revenue excluding China in fiscal 2025?',
]

def baseline():
    from deployment.request_contract_20261015 import score
    from strategic_graphrag.engine.query_understanding import parse_query
    save('protocol-frozen.json', dict(
        campaign='bounded-close-20261016', max_repair_rounds=2,
        corpus='Existing five PDFs only; no new extraction or generation chain',
        supported='Two companies; typed annual financial facts, source-preserving unit conversion, existing arithmetic. Independently bound fact slots; unresolved qualifiers require clarification.',
        development='All exposed 52/17/20/24 questions, all current counterexamples and their semantic families; no independent claim',
        holdout='Created after implementation freeze from original PDFs; family overlap explicitly reported; any use for repair converts it to development',
        gates=dict(core_semantics_min=.90, answerable_false_refusal_max=.10, severe_substitution_success_max=0),
        review='AI_PDF_NOT_HUMAN_GOLD', generation='NOT_MEASURED',
        publication='No commit, push, deletion, historical overwrite or formal release pointer change authorized'))
    identity=json.loads((PREVIOUS/'package/build_identity.json').read_text(encoding='utf-8'))
    save('baseline-identity.json',dict(candidate=identity, source_fingerprint=source_fingerprint(),
        git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),
        package_hashes={p.relative_to(PREVIOUS/'package').as_posix():sha256_file(p) for p in (PREVIOUS/'package').rglob('*') if p.is_file()}))
    # Preserve the exact runtime source before changing the shared checkout.
    import zipfile
    with zipfile.ZipFile(OUT/'baseline-source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for folder in ['strategic_graphrag','deployment','scripts','tests','frontend/src']:
            for path in (ROOT/folder).rglob('*'):
                if path.is_file() and path.suffix in {'.py','.ts','.tsx','.css'}:
                    archive.write(path,path.relative_to(ROOT))
    records=[]
    for question in QUESTIONS:
        records.append(dict(question=question, plan=parse_query(question).to_dict(),result=http(question)))
        time.sleep(1.5)
    save('baseline-counterexamples.json',records)
    rows=json.loads((PREVIOUS/'development-raw-scored.json').read_text(encoding='utf-8'))
    d02=next(row for row in rows if row['label']['item_id']=='D02')
    mutations=[]
    for field,value in [('company_id','WRONG_COMPANY'),('fact_period','FY1999')]:
        changed=copy.deepcopy(d02)
        for obs in changed['result']['response']['calculation']['observations']:obs[field]=value
        mutations.append(dict(field=field,value=value,legacy_score=score(changed)))
    save('baseline-scorer-negative.json',mutations)
    print(json.dumps(dict(counterexamples=[(r['result'].get('response') or {}).get('answer_status') for r in records], scorer_mutations=mutations)))

def campaign():
    from deployment import request_contract_20261015 as module
    module.BASE=PREVIOUS;module.OUT=OUT;module.PACKAGE=OUT/'package';module.PORT=8009
    return module

def build():
    save('implementation-freeze.json',dict(source_fingerprint=source_fingerprint(),
        repair_rounds=2,first_round='Independent clause slots, unknown commas/restrictions, executor uses slot years/units, strict independent scoring; 307 pass, 1 comma-year regression.',
        second_round='Recognize comma-separated FY labels without suppressing unknown comma items; scorer operand identity and source/unit negative checks; 320 tests passed.',
        frozen=True,post_freeze_repairs_allowed=False))
    campaign().build()
    save('frontend-artifact-hashes.json',{p.relative_to(ROOT/'frontend/dist').as_posix():sha256_file(p) for p in (ROOT/'frontend/dist').rglob('*') if p.is_file()})

def reference_labels():
    """References transcribed from the original PDF packet, not observations."""
    import pdfplumber,re
    specs=[('data/pdfs_other/2024-10-K.pdf',None),('data/pdfs/2025-10-K.pdf',[38,75,80]),
        ('experiments/acceptance-20261015/sources/TSMC-2024-Annual-Report.pdf',[63,64])]
    pages=[]
    for filename,selected in specs:
        with pdfplumber.open(ROOT/filename) as pdf:
            for n in selected or range(1,len(pdf.pages)+1):
                page=pdf.pages[n-1]
                if 'TSMC' in filename:page=page.crop((0,0,page.width/2,page.height))
                pages.append(dict(file=Path(filename).name,page=n,text=page.extract_text() or ''))
    save('reference-page-packet.json',pages)
    def operand(company,metric,year,value,aliases,unit=None):
        number=f'{value:,}'
        sources=[]
        for page in pages:
            if ('TSMC' in page['file'])!=(company=='TSMC'):continue
            if any(number in line and any(alias.lower() in line.lower() for alias in aliases) for line in page['text'].splitlines()):
                sources.append(dict(file=page['file'],page=page['page'],rows=aliases))
        if not sources:raise ValueError(('No independently located source row',metric,year,value))
        return dict(company_id=company,metric_id=metric,fact_period=f'FY{year}',business_scope='CONSOLIDATED',value=value,
            unit=unit or ('TWD thousands' if company=='TSMC' else 'USD millions'),sources=sources)
    nv=lambda metric,year,value,aliases:operand('NVIDIA_CORPORATION',metric,year,value,aliases)
    r24=nv('REVENUE',2024,60922,['Revenue']);r25=nv('REVENUE',2025,130497,['Revenue'])
    n24=nv('NET_INCOME',2024,29760,['Net income']);n25=nv('NET_INCOME',2025,72880,['Net income'])
    assets=operand('TSMC','TOTAL_ASSETS',2024,6691938000,['Total Assets'])
    liabilities=operand('TSMC','TOTAL_LIABILITIES',2024,2368362135,['Total Liabilities'])
    def slot(op,value=None,unit=None,operation='fact'):
        return dict(value=op['value'] if value is None else value,unit=unit or op['unit'],operation=operation,operands=[op])
    labels=[]
    def add(question,slots,family):labels.append(dict(item_id=f'C{len(labels)+1:02}',question=question,slots=slots,family=family,partition='DEVELOPMENT',review='AI_PDF_NOT_HUMAN_GOLD'))
    add(QUESTIONS[0],[slot(r24),slot(n25)],'nvidia_income')
    add(QUESTIONS[1],[slot(r25,130.497,'USD billions','unit_conversion'),slot(n25,72880,'USD millions','unit_conversion')],'nvidia_income')
    add(QUESTIONS[2],[],'unknown_item');add(QUESTIONS[3],[],'restricted_scope')
    add('What were NVIDIA revenue in fiscal 2024 and revenue in fiscal 2025?',[slot(r24),slot(r25)],'nvidia_income')
    add('What were NVIDIA revenue in fiscal 2024 and fiscal 2025?',[slot(r24),slot(r25)],'nvidia_income')
    add('What were NVIDIA net income in fiscal 2025 and revenue in fiscal 2024?',[slot(n25),slot(r24)],'nvidia_income')
    add('What were NVIDIA revenue and net income in fiscal 2025 in USD billions?',[slot(r25,130.497,'USD billions','unit_conversion'),slot(n25,72.88,'USD billions','unit_conversion')],'nvidia_income')
    add('What was NVIDIA revenue in fiscal 2025 in USD billions?',[slot(r25,130.497,'USD billions','unit_conversion')],'nvidia_income')
    add('What was NVIDIA net income in fiscal 2024?',[slot(n24)],'nvidia_income')
    add('What were TSMC consolidated total assets and total liabilities in fiscal 2024 in its 2024 annual report?',[slot(assets),slot(liabilities)],'tsmc_balance')
    add('What were TSMC consolidated total assets in fiscal 2024?',[slot(assets)],'tsmc_balance')
    add('Calculate NVIDIA net income divided by revenue in fiscal 2025.',[dict(value=72880/130497,unit='ratio',operation='ratio',operands=[n25,r25])],'nvidia_income')
    for words,family in [('revenue plus employee satisfaction','unknown_item'),('revenue and employee satisfaction','unknown_item'),('revenue only in Europe','restricted_scope'),('adjusted revenue','restricted_scope'),('revenue excluding China','restricted_scope')]:
        add(f'What was NVIDIA {words} in fiscal 2025?',[],family)
    add('What was TSMC revenue in fiscal 2024 in USD millions?',[],'fx_refusal')
    add('What was NVIDIA revenue in fiscal 2030?',[],'future_refusal')
    save('development-slot-labels.json',labels)
    return labels

def evaluate():
    from deployment.slot_score_20261016 import score
    labels=reference_labels();identity=json.loads((OUT/'package/build_identity.json').read_text(encoding='utf-8'))
    records=[]
    for label in labels:
        result=http(label['question'],8009)
        records.append(dict(label=label,result=result,score=score(result.get('response') or {},label,build_id=identity['build_id'])))
        print(label['item_id'],result.get('http_status'),records[-1]['score']['core_semantics'],flush=True)
        time.sleep(1.5)
    save('development-slot-results.json',records)
    # Replays retain their old scorer identity; never presented as independent verification.
    from deployment.request_contract_20261015 import score as old_score
    from deployment.score_five_acceptance import summarize
    from deployment.user_qa_20261015 import fact_score
    sources=[('52',ROOT/'experiments/acceptance-20261015/validation-labels.json'),
        ('17',ROOT/'experiments/optimization-20261015/new-family-labels.json'),
        ('20',ROOT/'experiments/user-qa-20261015/round2/fresh-labels.json'),
        ('24',PREVIOUS/'development-labels.json')]
    for name,path in sources:
        rows=[]
        for label in json.loads(path.read_text(encoding='utf-8')):
            result=http(label['question'],8009,label.get('filing'))
            row=dict(label=label,result=result)
            if name!='52':row['score']=(old_score if name=='24' else fact_score)(row)
            rows.append(row);time.sleep(1.5)
        save('replay-'+name+'.json',rows if name!='52' else dict(raw=rows,summary=summarize(rows)))
        print('replay',name,'done',flush=True)

if __name__=='__main__':
    import sys
    phase=sys.argv[1]
    {'baseline':baseline,'build':build,'import':lambda:campaign().import_graph(),
     'serve':lambda:campaign().serve(8009),'evaluate':evaluate}[phase]()
