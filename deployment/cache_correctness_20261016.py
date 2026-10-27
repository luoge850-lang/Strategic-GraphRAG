"""Isolated cache-correctness follow-up; historical candidates stay immutable."""
import json,subprocess,sys
from deployment.accuracy_closure_20261016 import ROOT,read
BASE=ROOT/'experiments/accuracy-closure-20261016/round2'
OUT=ROOT/'experiments/cache-correctness-20261016'
PORT=8015

def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as stream:json.dump(data,stream,ensure_ascii=False,indent=2)

def build():
    from deployment import user_qa_20261015 as campaign
    campaign.BASE=BASE;campaign.OUT=OUT;campaign.PACKAGE=OUT/'package'
    campaign.build()
    from strategic_graphrag.build_identity import source_fingerprint
    save('implementation-freeze.json',dict(source=source_fingerprint(),build_id=read(OUT/'package/build_identity.json')['build_id'],prior_final32_is_development=True))

def import_graph():
    from deployment import request_contract_20261015 as campaign
    campaign.OUT=OUT;campaign.PACKAGE=OUT/'package'
    campaign.import_graph()

def serve():
    from deployment import request_contract_20261015 as campaign
    campaign.BASE=BASE;campaign.OUT=OUT;campaign.PACKAGE=OUT/'package'
    campaign.serve(PORT)

def launch():
    with (OUT/'server.log').open('xb') as stream:
        p=subprocess.Popen([sys.executable,'-m','deployment.cache_correctness_20261016','serve'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    save('server-launch.json',dict(pid=p.pid,port=PORT))

if __name__=='__main__':
    {'build':build,'import':import_graph,'serve':serve,'launch':launch}[sys.argv[1]]()
