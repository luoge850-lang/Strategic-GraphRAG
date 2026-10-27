import pytest
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.schema.financial_observation import measurement_identity
from strategic_graphrag.pipeline.consolidated_ntd_table import cash_flow

@pytest.mark.parametrize('question,company,metric',[
 ('台积电2024年的总资产是多少？','TSMC','TOTAL_ASSETS'),
 ('英伟达2025财年的税前利润是多少？','NVIDIA_CORPORATION','PRETAX_INCOME'),
 ('台积电2024年的总负债是多少？','TSMC','TOTAL_LIABILITIES'),
])
def test_bilingual_identity(question,company,metric):
    q=parse_query(question)
    assert (q.company,q.target_metric)==(company,metric)
    assert not q.ambiguity and q.task_type=='FACT'

def test_unknown_chinese_clause_cannot_be_silently_dropped():
    assert parse_query('台积电2024年火星工厂的总资产是多少？').ambiguity
    q=parse_query('What were TSMC unconsolidated total assets in fiscal 2024?')
    assert q.business_scope=='UNCONSOLIDATED' and q.ambiguity
    assert parse_query('What was TSMC net income attributable to shareholders of the parent in FY2024?').target_metric=='PARENT_NET_INCOME'

def test_cash_flow_statement_is_not_a_balance():
    assert measurement_identity('OPERATING_CASH_FLOW','FY2024','TWD thousands')['measurement_nature']=='PERIOD_MOVEMENT'
    assert measurement_identity('TOTAL_EQUITY','FY2024','TWD thousands')['measurement_nature']=='BALANCE'

def test_cash_table_does_not_accept_unknown_columns_or_unconsolidated_only():
    class Half:
        def extract_tables(self):return [[['unknown']*8]]
    assert not cash_flow(Half(),'Cash Flow\nConsolidated\nUnit: NT$ thousands','TSMC','source.pdf',2024,(0,0,100,100))
    assert not cash_flow(Half(),'Cash Flow\nUnconsolidated\nUnit: NT$ thousands','TSMC','source.pdf',2024,(0,0,100,100))

def test_quota_exhaustion_does_not_break_page_assets_or_sources(monkeypatch):
    import asyncio
    from starlette.requests import Request
    from starlette.responses import Response
    from strategic_graphrag.api import server
    monkeypatch.setattr(server,'_API_AUTH_ENABLED',False)
    monkeypatch.setattr(server,'_RATE_LIMIT_PER_MINUTE',1)
    monkeypatch.setattr(server,'_RATE_BUCKETS',__import__('collections').defaultdict(__import__('collections').deque))
    async def next_response(request):return Response('ok')
    def request(path,method='GET'):
        return Request(dict(type='http',method=method,path=path,raw_path=path.encode(),query_string=b'',headers=[],scheme='http',server=('localhost',80),client=('127.0.0.1',1234)))
    async def run():
        assert (await server.request_guard(request('/query','POST'),next_response)).status_code==200
        assert (await server.request_guard(request('/query','POST'),next_response)).status_code==429
        for path in ['/','/assets/app.js','/assets/app.css']:
            assert (await server.request_guard(request(path),next_response)).status_code==200
        assert (await server.request_guard(request('/evaluation/table-quality/source/document.pdf/page/1'),next_response)).status_code==200
    asyncio.run(run())
