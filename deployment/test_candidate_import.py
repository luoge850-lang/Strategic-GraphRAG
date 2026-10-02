import pytest
from deployment.import_candidate import prepare, write_empty


def edge():
    return {"build_id": "build_test", "edge_id": "claim:0", "claim_id": "claim",
        "sentence_id": "sentence", "source": "NVIDIA_CORPORATION", "source_category": "Company",
        "target": "REVENUE", "target_category": "FinancialMetric", "relation": "REPORTS_METRIC",
        "source_filing": "2025-10-K.pdf", "document_sha256": "hash", "filing_year": 2025,
        "fact_year": 2025, "page": 80, "evidence": "Total revenue 130,497 USD millions",
        "metric_value": 130497, "metric_unit": "USD millions", "unit": "USD millions",
        "causal_strength": "DISCLOSED_ONLY", "causal_form": "FINANCIAL_RELATION",
        "table_instance_id": "table", "source_row_id": "Total revenue"}


def test_real_import_plan_preserves_period_and_build_identity():
    plan = prepare({"build_id": "build_test", "edges": [edge()]})
    obs = plan["observations"][0]
    assert obs["fiscal_period"] == "FY2025"
    assert obs["currency"] == "USD" and obs["scale"] == "millions"
    assert obs["build_id"] == "build_test"
    assert plan["relations"]["REPORTS_METRIC"][0]["props"]["observation_id"] == obs["id"]


def test_duplicate_and_foreign_edges_rejected():
    for edges in ([edge(), edge()], [{**edge(), "build_id": "foreign"}]):
        with pytest.raises(ValueError):
            prepare({"build_id": "build_test", "edges": edges})


def test_nonempty_database_refused_before_any_write():
    class Result:
        def single(self):
            return {"count": 1}
    class Transaction:
        def __init__(self):
            self.calls = []
        def run(self, query):
            self.calls.append(query)
            return Result()
    tx = Transaction()
    with pytest.raises(ValueError, match="nonempty"):
        write_empty(tx, {})
    assert tx.calls == ["MATCH (n) RETURN count(n) AS count"]
