"""One pass through exact questions; records transport errors separately."""
import argparse,json,time,hashlib
from deployment.acceptance_20261015 import OUT,save
from deployment.live_acceptance import http
def main():
    p=argparse.ArgumentParser();p.add_argument('--partition',choices=['development','validation'],required=True);p.add_argument('--round',default='round1');p.add_argument('--port',type=int,default=8002);a=p.parse_args()
    if a.partition=='validation':
        frozen=json.loads((OUT/'validation-freeze.json').read_text())
        for name,digest in frozen['hashes'].items():
            if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=digest:raise ValueError('frozen label/source changed')
        labels=json.loads((OUT/'validation-labels.json').read_text())
    else:
        labels=[dict(item_id=f'D-{i}',question=row['question'],request=row['request']) for i,row in enumerate(json.loads((OUT/'development-before.json').read_text()))]
        labels.extend([dict(item_id='D-TSMC-REVENUE',question='What was TSMC consolidated revenue for fiscal 2024 in its 2024 annual report?'),
            dict(item_id='D-NVIDIA-DELTA',question='Calculate the absolute difference in NVIDIA revenue between fiscal 2024 and fiscal 2025 using its 2025 10-K.')])
    rows=[]
    for label in labels:
        request=label.get('request') or dict(question=label['question'],max_paths=10,retrieval_mode='graph',synthesize=False,use_cache=False)
        result=http(f'http://127.0.0.1:{a.port}','/query',request);result.pop('_bytes',None)
        rows.append(dict(label=label,request=request,result=result))
        print(label['item_id'],result.get('http_status'),(result.get('response') or {}).get('calculation',{}).get('status'),flush=True)
    save(f'{a.partition}-{a.round}-raw.json',rows)
if __name__=='__main__':main()
