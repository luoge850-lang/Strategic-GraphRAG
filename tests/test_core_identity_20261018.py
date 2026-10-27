import pytest
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.pipeline.financial_note_extractor import extract
from strategic_graphrag.schema.financial_observation import measurement_identity

@pytest.mark.parametrize('phrase,metric,nature',[
 ('accumulated other comprehensive income','ACCUMULATED_OTHER_COMPREHENSIVE_INCOME','BALANCE'),
 ('total comprehensive income','COMPREHENSIVE_INCOME','PERIOD_RESULT'),
 ('other comprehensive income','OTHER_COMPREHENSIVE_INCOME_NET','PERIOD_RESULT'),
 ('total operating expenses','OPERATING_COST','PERIOD_RESULT')])
@pytest.mark.parametrize('verb',['What was','How much was','Give','Report'])
def test_adjacent_metric_identity(phrase,metric,nature,verb):
 p=parse_query(f'{verb} NVIDIA {phrase} for FY2024 using the 2025 annual report?')
 assert p.target_metric==metric
 assert p.requested_facts[0]['metric_id']==metric
 assert p.requested_facts[0]['measurement_nature']==nature
 assert not p.ambiguity

@pytest.mark.parametrize('condition',['caused by a fire','due to an earthquake','attributable to export restrictions','resulting from a disaster'])
def test_causal_loss_not_generic_total(condition):
 p=parse_query(f'What was NVIDIA revenue loss {condition} in FY2024?')
 assert 'attributed_financial_outcome_not_disclosed' in p.ambiguity

def test_statement_source_tail_is_not_unknown_item():
 p=parse_query('How much cash did NVIDIA pay for interest in FY2024, according to its 2025 cash flow statement?')
 assert not p.ambiguity
 assert p.requested_facts[0]['source_filing']=='2025-10-K.pdf'

@pytest.mark.parametrize('noun',['deferred revenue','unearned revenue','unrealized net income','accumulated net income'])
def test_unknown_specific_metric_never_degrades_to_total(noun):
 assert 'unresolved_financial_identity_modifier' in parse_query(f'Give NVIDIA {noun} in FY2024.').ambiguity

def test_resolved_cash_adjustment_does_not_keep_unknown_movement_flag():
 assert not parse_query('What stock-based compensation adjustment appears in NVIDIA operating cash flows for FY2025 in the 2025 report?').ambiguity

def test_current_federal_note_row_keeps_current_not_deferred():
 import json
 text='Notes to the Consolidated Financial Statements\nIncome Taxes\nYear Ended\nJan 28, 2024 Jan 29, 2023\n(In millions)\nCurrent income taxes:\nFederal $ 17 $ 11\nDeferred income taxes:\nFederal (7) (3)'
 r=extract(text,'ISSUER','sample.pdf',2024)
 assert len(r)==1 and r[0]['target']=='CURRENT_FEDERAL_INCOME_TAX'
 assert json.loads(r[0]['metric_values_json'])[0]['value']=='17'

def test_note_prose_values_follow_respective_years():
 text='Notes to the Consolidated Financial Statements\nEmployee Retirement Plans\nOur contribution expense for fiscal years 2024, 2023, and 2022 was $19 million, $13 million, and $11 million, respectively.'
 rows=extract(text,'ISSUER','sample.pdf',2024)
 assert len(rows)==1 and '19.0' in rows[0]['metric_values_json']

def test_cash_outflow_and_missing_zero_columns_keep_position():
 text='Consolidated Statements of Income\n(In millions)\nYear Ended\nJan 28, 2024 Jan 29, 2023 Jan 30, 2022\nAcquisition termination cost — — 17'
 rows=extract(text,'ISSUER','sample.pdf',2024)
 import json
 assert [v['value'] for v in json.loads(rows[0]['metric_values_json'])]==['0','0','17']
 assert measurement_identity('DIVIDENDS_PAID','FY2024','USD millions')['measurement_nature']=='PERIOD_MOVEMENT'
