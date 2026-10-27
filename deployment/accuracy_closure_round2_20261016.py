"""Second and final repair round of the bounded accuracy closure campaign."""
import subprocess
import sys
from deployment.accuracy_closure_20261016 import ROOT,read
OUT=ROOT/'experiments/accuracy-closure-20261016/round2'
BASE=ROOT/'experiments/accuracy-closure-20261016'
PORT=8014

def build():
    OUT.mkdir(exist_ok=False)
    from deployment import user_qa_20261015 as campaign
    campaign.BASE=BASE
    campaign.OUT=OUT
    campaign.PACKAGE=OUT/'package'
    campaign.build()
    from strategic_graphrag.build_identity import source_fingerprint
    campaign.save('implementation-freeze.json',dict(build_id=read(OUT/'package/build_identity.json')['build_id'],source=source_fingerprint(),round=2,previous_reserved_promoted_to_development=True))

def import_graph():
    from deployment import request_contract_20261015 as campaign
    campaign.OUT=OUT;campaign.PACKAGE=OUT/'package'
    campaign.import_graph()

def serve():
    from deployment import request_contract_20261015 as campaign
    campaign.BASE=BASE;campaign.OUT=OUT;campaign.PACKAGE=OUT/'package'
    campaign.serve(PORT)

def launch():
    from deployment.accuracy_closure_20261016 import save as unused
    with (OUT/'server.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-m','deployment.accuracy_closure_round2_20261016','serve'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    with (OUT/'server-launch.json').open('x',encoding='utf-8') as stream:
        import json
        json.dump(dict(pid=p.pid,port=PORT),stream)

if __name__=='__main__':
    {'build':build,'import':import_graph,'serve':serve,'launch':launch}[sys.argv[1]]()
