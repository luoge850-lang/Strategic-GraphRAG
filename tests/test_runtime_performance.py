from __future__ import annotations

from scripts.benchmark_runtime_performance import percentile, summarize_latencies, _phase_summary


def test_percentile_is_deterministic_and_interpolated() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([], 95) is None


def test_summary_reports_errors_cache_and_stages() -> None:
    rows = [
        {
            "wall_latency_ms": 10.0,
            "error": None,
            "status_code": 200,
            "cache_hit": True,
            "stage_latency_ms": {"graph_path_search_ms": 4.0, "total_ms": 8.0},
        },
        {
            "wall_latency_ms": 20.0,
            "error": "timeout",
            "status_code": None,
            "cache_hit": False,
            "stage_latency_ms": {},
        },
    ]
    summary = _phase_summary(rows, concurrency=1)
    assert summary["success_count"] == 1
    assert summary["error_count"] == 1
    assert summary["cache_hit_count"] == 1
    assert summary["stage_latency_ms"]["graph_path_search_ms"]["p50_ms"] == 4.0


def test_latency_summary_has_tail_metrics() -> None:
    summary = summarize_latencies([1, 2, 3])
    assert summary["count"] == 3
    assert summary["p95_ms"] == 2.9
    assert summary["p99_ms"] == 2.98
