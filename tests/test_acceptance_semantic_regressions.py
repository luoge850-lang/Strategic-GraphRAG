"""Exposed development families, never holdout evidence."""
import pytest
from strategic_graphrag.engine.query_understanding import parse_query

@pytest.mark.parametrize('label', ['income before income tax','income before taxes','profit before tax','pre-tax profit','pretax profit','pre-tax income'])
def test_pretax_aliases_preserve_metric(label):
    assert parse_query(f'What was NVIDIA {label} in FY2025?').target_metric == 'PRETAX_INCOME'

@pytest.mark.parametrize('expression', ['net income divided by revenue','revenue / net income','the quotient of revenue and net income'])
def test_arithmetic_never_becomes_fact(expression):
    q=parse_query(f'Calculate NVIDIA {expression} in FY2025')
    assert q.calculation == 'RATIO' and q.task_type == 'CALCULATION'

def test_companies_are_not_defaulted_to_nvidia():
    assert parse_query('What was AMD revenue in FY2025?').company == 'ADVANCED_MICRO_DEVICES'
    assert parse_query('What was TSMC revenue in FY2024?').company == 'TSMC'
    assert parse_query('Compare NVIDIA and TSMC revenue in FY2024').ambiguity

@pytest.mark.parametrize('period', ['Q4 FY2025','FY2025 Q4','fourth quarter of fiscal 2025'])
def test_quarter_is_not_annual(period):
    q=parse_query(f'What was NVIDIA revenue in {period}?')
    assert q.fact_period == 'Q4 FY2025' and q.period_granularity == 'QUARTERLY'

def test_movement_is_not_balance():
    q=parse_query('What cash-flow adjustment for accounts receivable did NVIDIA report in FY2022?')
    assert q.target_metric == 'CASH_FLOW_CHANGE_ACCOUNTS_RECEIVABLE'
    assert q.measurement_nature == 'PERIOD_MOVEMENT'

def test_disclosed_percentage_does_not_reverse_operands():
    q=parse_query('What percentage of revenue were research and development expenses for NVIDIA fiscal 2022 in its 2023 10-K?')
    assert q.target_metric == 'R_AND_D_RATIO'
    assert q.calculation is None and q.ratio_metrics == []

def test_twd_currency_identity():
    from strategic_graphrag.schema.financial_observation import split_unit
    assert split_unit('TWD thousands') == ('TWD','thousands')

def test_ratio_uses_both_observations_and_refuses_zero():
    import runpy
    from pathlib import Path
    helper=runpy.run_path(str(Path(__file__).with_name('test_public_calculation_contract.py')))
    obs,path,plan,calculate=[helper[x] for x in ('observation','path','plan','calculate')]
    records=path(obs(40,metric='NET_INCOME',fact_year=2024),obs(100,metric='REVENUE',fact_year=2024))
    request=plan(calculation='RATIO',ratio_metrics=['NET_INCOME','REVENUE'])
    result=calculate([records],request)
    assert result['status']=='PASS' and result['value']==.4 and len(result['observations'])==2
    zero=path(obs(40,metric='NET_INCOME',fact_year=2024),obs(0,metric='REVENUE',fact_year=2024))
    assert calculate([zero],request)['status']=='INVALID_INPUT'
