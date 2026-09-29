import gc
import json
import tempfile
import weakref
from pathlib import Path

from scripts import run_financial_retrieval_matrix as retrieval
from scripts.run_financial_retrieval_matrix import _cleanup_verified_runtime, _graph_expand


def test_graph_expansion_adds_page_key_to_graph_only_page():
    fused = [
        {
            "page_key": "2024-10-K.pdf#page=20",
            "id": "chunk-seed",
            "document": "seed excerpt",
            "metadata": {"source_filing": "2024-10-K.pdf", "page": 20},
            "score": 0.4,
        }
    ]
    edges_by_page = {
        "2024-10-K.pdf#page=20": [
            {"source": "NVIDIA_CORPORATION", "target": "REVENUE"}
        ],
        "2025-10-K.pdf#page=30": [
            {"source": "REVENUE", "target": "GAMING_MARKET"}
        ],
    }
    page_documents = {
        "2024-10-K.pdf#page=20": {
            "id": "chunk-seed",
            "document": "seed excerpt",
            "metadata": {"source_filing": "2024-10-K.pdf", "page": 20},
        },
        "2025-10-K.pdf#page=30": {
            "id": "chunk-expanded",
            "document": "expanded excerpt",
            "metadata": {"source_filing": "2025-10-K.pdf", "page": 30},
        },
    }

    results = _graph_expand(fused, edges_by_page, page_documents)

    expanded = next(item for item in results if item["id"] == "chunk-expanded")
    assert expanded["page_key"] == "2025-10-K.pdf#page=30"
    assert expanded["graph_expansion"] is True


def test_runtime_finalizer_releases_scratch_client_and_rechecks_package(tmp_path):
    build_dir = tmp_path / "verified-build"
    build_dir.mkdir()
    (build_dir / "artifact_ledger.json").write_text(
        json.dumps({"artifacts": []}), encoding="utf-8"
    )

    class Client:
        closed = False

        def close(self):
            self.closed = True

    client = Client()
    workspace = tempfile.TemporaryDirectory(prefix="test-financial-runtime-")
    scratch = workspace.name
    lease = retrieval._CleanupLease()
    finalizer = weakref.finalize(
        lease,
        _cleanup_verified_runtime,
        {"build_dir": build_dir, "workspace": workspace, "client": client},
    )

    del lease
    gc.collect()

    assert not finalizer.alive
    assert client.closed is True
    assert not Path(scratch).exists()


def test_temporal_only_mode_does_not_run_graph_expansion(monkeypatch):
    bm25 = [{"page_key": "2025-10-K.pdf#80", "score": 1.0}]
    dense = [{"page_key": "2025-10-K.pdf#80", "score": 1.0}]

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("temporal-only diagnostic must not call graph expansion")

    monkeypatch.setattr(retrieval, "_graph_expand", fail_if_called)
    expected = ([{"page_key": "temporal-result"}], {"fiscal_years": [2025]}, "OK", {
        "temporal_candidate_pages_before_filter": 1,
        "temporal_candidate_pages_after_filter": 1,
    })
    monkeypatch.setattr(retrieval, "_temporal_filter", lambda *_args: expected)
    hits, plan, status, counts = retrieval._select_mode_hits(
        "fusion_temporal_only", "revenue in fiscal 2025", bm25, dense, {}, {}
    )
    assert (hits, plan, status) == expected[:3]
    assert counts["graph_expansion_candidate_pages"] is None
    assert counts["temporal_candidate_pages_before_filter"] == 1


def test_explicit_single_year_temporal_filter_rejects_wrong_year_and_missing_edges():
    query = "What was NVIDIA revenue in FY2025?"
    page_2024 = {"page_key": "2024-10-K.pdf#80"}
    page_2025 = {"page_key": "2025-10-K.pdf#80"}
    edges = {
        "2024-10-K.pdf#80": [{"fact_year": 2024, "source_filing": "2024-10-K.pdf"}],
        "2025-10-K.pdf#80": [{"fact_year": 2025, "source_filing": "2025-10-K.pdf"}],
    }

    hits, plan, status, counts = retrieval._temporal_filter(query, [page_2024, page_2025], edges)

    assert plan["fiscal_years"] == [2025]
    assert hits == [page_2025]
    assert status == "OK"
    assert counts == {"temporal_candidate_pages_before_filter": 2, "temporal_candidate_pages_after_filter": 1}

    hits, _, status, _ = retrieval._temporal_filter(query, [page_2025], {})
    assert hits == []
    assert status == "NO_HITS_AFTER_TEMPORAL_FILTER"


def test_single_year_filter_skips_pages_without_a_matching_year_edge():
    query = "What was NVIDIA revenue in FY2025?"
    page_without_edge = {"page_key": "2025-10-K.pdf#80"}
    hits, _, status, _ = retrieval._temporal_filter(query, [page_without_edge], {})
    assert hits == []
    assert status == "NO_HITS_AFTER_TEMPORAL_FILTER"
