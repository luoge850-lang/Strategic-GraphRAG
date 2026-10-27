from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.pipeline.financial_table_extractor import extract_financial_table_triples


class Page:
    def __init__(self, row):
        self.row = row

    def extract_tables(self):
        return [[self.row]]


def extract(title, row):
    text = title + '\nYear Ended\n2023 2022\n(In millions)\n' + ' '.join(row)
    return extract_financial_table_triples(Page(row), text, 2023, company_id='NVIDIA_CORPORATION')


def test_specific_research_expense_beats_generic_expense():
    plan = parse_query('What r and d expense did NVIDIA report for fiscal year 2022 in its 2023 10-K?')
    assert plan.target_metric == 'R_AND_D_EXPENSE'


def test_net_receivable_balance_is_not_cash_flow_change():
    balance = extract('CONSOLIDATED BALANCE SHEETS', ['Accounts receivable, net', '3,827', '4,650'])[0]
    movement = extract('CONSOLIDATED STATEMENTS OF CASH FLOWS', ['Accounts receivable', '822', '(2,215)'])[0]
    assert balance['target'] == 'ACCOUNTS_RECEIVABLE'
    assert balance['statement_type'] == 'BALANCE_SHEET'
    assert movement['target'] == 'CASH_FLOW_CHANGE_ACCOUNTS_RECEIVABLE'
    assert movement['statement_type'] == 'CASH_FLOW_STATEMENT'


def test_stock_compensation_component_is_not_total_research_expense():
    component = extract('Note 4 - Stock-Based Compensation', ['Research and development', '1,892', '1,298'])[0]
    total = extract('CONSOLIDATED STATEMENTS OF INCOME', ['Research and development', '7,339', '5,268'])[0]
    assert component['target'] == 'STOCK_BASED_R_AND_D_EXPENSE'
    assert total['target'] == 'R_AND_D_EXPENSE'
