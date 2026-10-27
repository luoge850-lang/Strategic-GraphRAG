"""New authorized recovery batch; historical labels/results stay untouched."""
import json,os,subprocess,sys,time
from pathlib import Path
from deployment.finalization_20261016 import ROOT,http
from strategic_graphrag.build_identity import source_fingerprint

OUT=ROOT/'experiments/accuracy-recovery-20261016'
BASE=ROOT/'experiments/finalization-20261016'

def read(path):return json.loads(path.read_text(encoding='utf-8'))
def save(name,value):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)

def campaign():
    from deployment import request_contract_20261015 as module
    module.BASE=BASE;module.OUT=OUT;module.PACKAGE=OUT/'package';module.PORT=8012
    return module

def build():
    from deployment import user_qa_20261015 as module
    module.BASE=BASE;module.OUT=OUT;module.PACKAGE=OUT/'package'
    save('implementation-freeze.json',dict(source_fingerprint=source_fingerprint(),rounds=2,
        round1='Disclosure adjuncts, semantic percentage span, explicit parent-attributable row extraction; 343 tests passed.',
        round2='Two adversarial unknown-document adjuncts failed; narrowed disclosure grammar preserves unknown sources. Full regression log retained.',
        old_reserved_promoted_to_development=True,no_new_independent_accuracy_claim=True,
        scope='Original five PDFs; TSMC candidate extraction rerun from original PDF pages. NVIDIA extraction/page snapshots remain hash-checked cache.'))
    module.build()

def serve():campaign().serve(8012)
def import_graph():campaign().import_graph()

def launch():
    log=(OUT/'server.log').open('xb')
    p=subprocess.Popen([sys.executable,'-m','deployment.accuracy_recovery_20261016','serve'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.CREATE_NEW_PROCESS_GROUP)
    log.close();save('server-launch.json',dict(pid=p.pid,port=8012))

def evaluate():
    from deployment.score_five_acceptance import summarize
    from deployment.user_qa_20261015 import fact_score
    from deployment.request_contract_20261015 import score as old_score
    from deployment.slot_score_20261016 import score as slot_score
    identity=read(OUT/'package/build_identity.json');summary={}
    specifications=[('old52',ROOT/'experiments/acceptance-20261015/validation-labels.json','52'),
        ('old17',ROOT/'experiments/optimization-20261015/new-family-labels.json','fact'),
        ('transport20',ROOT/'experiments/user-qa-20261015/round2/fresh-labels.json','fact'),
        ('contract24',ROOT/'experiments/request-contract-20261015/round2/development-labels.json','contract'),
        ('slot20',BASE/'development-slot-labels.json','slot'),
        ('exposed_reserved12',BASE/'reserved-labels-frozen.json','reserved')]
    for name,path,kind in specifications:
        labels=read(path)
        if kind=='reserved':labels=labels['labels']
        rows=[]
        for label in labels:
            result=http(label['question'],8012,label.get('filing'))
            row=dict(label=label,result=result,current_partition='EXPOSED_DEVELOPMENT_THIS_BATCH')
            if kind in {'slot','reserved'}:row['score']=slot_score(result.get('response') or {},label,build_id=identity['build_id'])
            elif kind!='52':row['score']=(old_score if kind=='contract' else fact_score)(row)
            rows.append(row);print(name,label.get('item_id'),result.get('http_status'),(result.get('response') or {}).get('answer_status'),flush=True)
            time.sleep(1.5)
        if kind=='52':
            scored=summarize(rows);save(name+'-results.json',dict(raw=rows,summary=scored))
            summary[name]=dict(numerator=scored['metrics']['core_semantic']['numerator'],denominator=len(rows),metrics=scored['metrics'])
        else:
            save(name+'-results.json',rows)
            key='core_semantics' if kind in {'slot','reserved'} else 'scenario_check' if kind=='contract' else 'core_semantic'
            summary[name]=dict(numerator=sum(r['score'][key] for r in rows),denominator=len(rows),metric=key,
                transport_success=sum(r['result'].get('http_status')==200 for r in rows))
        save(name+'-summary.json',summary[name])
    save('development-summary.json',dict(build_id=identity['build_id'],partitions=summary,
        independent_validation=False,review='AI_PDF_NOT_HUMAN_GOLD',generation='NOT_MEASURED',
        retained_score_limit='Legacy 24 scorer is weak, and frozen source-row aliases cause known full-fact string-check failures in slot labels. Core recovery is not complete factual or generation acceptance.'))

if __name__=='__main__':
    import sys
    {'build':build,'import':import_graph,'serve':serve,'launch':launch,'evaluate':evaluate}[sys.argv[1]]()
