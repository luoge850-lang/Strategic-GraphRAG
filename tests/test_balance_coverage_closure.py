from strategic_graphrag.pipeline.financial_table_extractor import _find_metric, extract_financial_table_triples


def test_balance_labels_are_exact_and_do_not_absorb_components():
    for name, metric in [('Intangible assets, net', 'INTANGIBLE_ASSETS'),
                         ('Other assets', 'OTHER_ASSETS'),
                         ('Retained earnings', 'RETAINED_EARNINGS')]:
        assert _find_metric(name + ' 100 200')[1] == metric
        assert _find_metric('Change in ' + name + ' 100 200') is None
        assert _find_metric(name + ' excluding subsidiaries 100 200') is None


def test_balance_labels_are_not_cash_flow_balances():
    class Page:
        def extract_tables(self):
            return [[['Other assets', '100', '200']]]
    assert extract_financial_table_triples(Page(),
        'CONSOLIDATED STATEMENTS OF CASH FLOWS\n2025 2024\n(In millions)\nOther assets 100 200',
        2025, company_id='NVIDIA_CORPORATION', document_id='report') == []
