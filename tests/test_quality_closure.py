import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from strategic_graphrag.engine.query_understanding import parse_query
from scripts.evaluate import scorer
from scripts import evaluate


@pytest.mark.parametrize('payload,status,category', [
    ([],200,'INVALID_RESPONSE'), (None,200,'INVALID_RESPONSE'),
    ({},200,'INVALID_RESPONSE'), ([],503,'HTTP_ERROR'),
])
def test_invalid_http_payload_is_recorded_for_every_question(tmp_path,monkeypatch,payload,status,category):
    class Response:
        status_code=status
        def json(self): return payload
    monkeypatch.setattr(evaluate.requests,'post',lambda *a,**k:Response())
    output=tmp_path/'result.jsonl'
    monkeypatch.setattr(sys,'argv',['evaluate','--url','http://test','--build-id','expected','--output',str(output)])
    evaluate.main()
    rows=[json.loads(x) for x in output.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==25
    assert all(x['category']==category and x['response']==payload and not x['score']['core'] for x in rows)


@pytest.mark.parametrize('failure,category', [('json','INVALID_RESPONSE'),('timeout','TIMEOUT'),('connection','TRANSPORT_ERROR')])
def test_transport_boundaries_continue(tmp_path,monkeypatch,failure,category):
    class Response:
        status_code=200
        text='not json'
        def json(self): raise evaluate.requests.exceptions.JSONDecodeError('invalid',self.text,0)
    def post(*args,**kwargs):
        if failure=='timeout': raise evaluate.requests.Timeout('timeout')
        if failure=='connection': raise evaluate.requests.ConnectionError('offline')
        return Response()
    monkeypatch.setattr(evaluate.requests,'post',post)
    output=tmp_path/'result.jsonl'
    monkeypatch.setattr(sys,'argv',['evaluate','--url','http://test','--build-id','expected','--output',str(output)])
    evaluate.main()
    rows=[json.loads(x) for x in output.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==25
    assert all(x['category']==category and not x['score']['core'] for x in rows)


@pytest.mark.parametrize('alias,metric', [
    ('Income from operations', 'OPERATING_INCOME'),
    ('Operating income', 'OPERATING_INCOME'),
    ('Intangible assets, net', 'INTANGIBLE_ASSETS'),
    ('Intangible assets net', 'INTANGIBLE_ASSETS'),
])
@pytest.mark.parametrize('year', [2021, 2024])
def test_accounting_alias_identity(alias, metric, year):
    plan = parse_query(f'Report NVIDIA {alias} for FY{year}, using the 2025 annual report?')
    assert not plan.ambiguity
    assert plan.requested_facts[0]['metric_id'] == metric
    assert plan.requested_facts[0]['fact_period'] == f'FY{year}'


@pytest.mark.parametrize('metrics', [
    'intangible assets, net, employee satisfaction',
    'employee satisfaction and intangible assets, net',
    'income from operations excluding China',
])
def test_unknown_obligations_remain(metrics):
    assert parse_query(f'Report NVIDIA {metrics} in fiscal 2024?').ambiguity


@pytest.mark.parametrize('field,value', [('build_id','wrong'),('fact_period','FY1999'),
    ('row_label','Wrong row'),('unit','TWD thousands')])
def test_reference_scorer_rejects_mutations(field,value):
    root=Path(__file__).resolve().parents[1]
    row=json.loads((root/'evaluation/current/baseline.jsonl').read_text(encoding='utf-8').splitlines()[0])
    response=copy.deepcopy(row['response'])
    response['calculation']['observations'][0][field]=value
    score=scorer.score(response,row['label'],'build_8a5ce6d74be82a7c')
    assert not score['request_joint']


def test_unavailable_service_keeps_all_questions(tmp_path):
    root=Path(__file__).resolve().parents[1]
    output=tmp_path/'raw.jsonl'
    result=subprocess.run([sys.executable,str(root/'scripts/evaluate.py'),
        '--url','http://127.0.0.1:1','--build-id','expected_build','--output',str(output)],
        capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in output.read_text(encoding='utf-8').splitlines()]
    assert len(rows)==25
    assert all(row['category']=='TRANSPORT_ERROR' and row['score']['core'] is False for row in rows)
    assert all(row['expected_build_id']=='expected_build' for row in rows)
    rerun=subprocess.run([sys.executable,str(root/'scripts/evaluate.py'),
        '--build-id','expected_build','--output',str(output)],capture_output=True)
    assert rerun.returncode!=0
