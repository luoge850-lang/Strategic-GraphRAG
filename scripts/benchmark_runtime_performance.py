"""Measure API latency, stage timings, errors, and bounded concurrency.

This is a runtime/performance benchmark, not a quality benchmark.  It sends
the same frozen questions to all four retrieval modes and records whether the
request used the API query cache.  ``use_cache=false`` is reported as a cold
cache miss, while a repeated ``use_cache=true`` request is reported as a cache
hit.  The script cannot prove a process-level cold start unless the caller
restarts the service; that limitation is recorded in the report.

The default is retrieval-only (``synthesize=false``) so latency measurements
do not silently mix external LLM generation with retrieval.  Use
``--synthesize`` for a separately labelled end-to-end run.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from strategic_graphrag.response_contract import response_state


DEFAULT_DATASET = ROOT / "evaluation" / "silver_retrieval_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "runtime_performance_2026-09-19.json"
MODES = ("vector", "graph", "hybrid", "hybrid_temporal")
STAGE_KEYS = (
    "query_understanding_ms",
    "anchor_resolution_ms",
    "ppr_ms",
    "vector_retrieval_ms",
    "graph_path_search_ms",
    "scoring_rerank_ms",
    "llm_synthesis_ms",
    "total_ms",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: Iterable[float], percentile_value: float) -> float | None:
    usable = sorted(float(value) for value in values)
    if not usable:
        return None
    if len(usable) == 1:
        return round(usable[0], 2)
    position = (len(usable) - 1) * percentile_value / 100
    lower = int(position)
    upper = min(lower + 1, len(usable) - 1)
    weight = position - lower
    return round(usable[lower] * (1 - weight) + usable[upper] * weight, 2)


def summarize_latencies(values: Iterable[float]) -> dict[str, float | None]:
    usable = [float(value) for value in values]
    return {
        "count": len(usable),
        "mean_ms": round(statistics.mean(usable), 2) if usable else None,
        "p50_ms": percentile(usable, 50),
        "p95_ms": percentile(usable, 95),
        "p99_ms": percentile(usable, 99),
        "min_ms": round(min(usable), 2) if usable else None,
        "max_ms": round(max(usable), 2) if usable else None,
    }


def _execution_status(row: dict[str, Any]) -> str:
    """Return the response-contract status with legacy-row compatibility."""
    explicit = str(row.get("execution_status") or "").upper().strip()
    if explicit:
        return explicit
    if row.get("error"):
        error_text = str(row.get("error")).upper()
        return "TIMEOUT" if "TIMEOUT" in error_text or "TIMED OUT" in error_text else "DEPENDENCY_ERROR"
    status_code = row.get("status_code")
    if status_code is None or int(status_code) < 400:
        # Older runtime fixtures contain a successful HTTP status but no body
        # contract fields. They remain valid latency/cache measurements.
        return "SUCCEEDED"
    return response_state({}, http_status=int(status_code))["execution_status"]


def _request(base_url: str, item: dict[str, Any], mode: str, *, use_cache: bool, synthesize: bool, timeout: float) -> dict[str, Any]:
    payload = {
        "question": item["question"],
        "max_paths": 10,
        "source_filing": item.get("source_filing"),
        "cross_filing": bool(item.get("cross_filing")),
        "retrieval_mode": mode,
        "vector_top_k": 10,
        "use_cache": use_cache,
        "synthesize": synthesize,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/query",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    network_error = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read().decode("utf-8")
            status_code = response.status
        try:
            payload_response = json.loads(response_body)
            error = None
        except ValueError as exc:
            payload_response = {}
            error = f"INVALID_JSON: {exc}"
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
        payload_response = {}
        status_code = getattr(exc, "code", None)
        error = f"{type(exc).__name__}: {exc}"
        network_error = exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    metadata = payload_response.get("metadata") or {}
    latency = metadata.get("latency_ms") or {}
    cache = metadata.get("cache") or {}
    state = response_state(
        payload_response,
        http_status=status_code,
        error=network_error,
    )
    return {
        "question_id": item.get("id"),
        "mode": mode,
        "synthesize": synthesize,
        "use_cache": use_cache,
        "status_code": status_code,
        "error": error,
        "execution_status": state["execution_status"],
        "answer_status": state["answer_status"],
        "grounding_status": state["grounding_status"],
        "outcome": state["outcome"],
        "status_provenance": state["status_provenance"],
        "wall_latency_ms": round(elapsed_ms, 2),
        "engine_latency_ms": latency.get("total_ms"),
        "cache_hit": bool(cache.get("hit")),
        "cache_age_ms": cache.get("age_ms"),
        "stage_latency_ms": {key: latency.get(key) for key in STAGE_KEYS if latency.get(key) is not None},
        "retrieval_status": (metadata.get("retrieval") or {}).get("status"),
        "grounding_status": (metadata.get("grounding") or {}).get("status"),
    }


def _run_requests(base_url: str, rows: list[dict[str, Any]], mode: str, *, use_cache: bool, synthesize: bool, timeout: float, concurrency: int) -> list[dict[str, Any]]:
    phase_started = time.perf_counter()
    if concurrency <= 1:
        results = [_request(base_url, item, mode, use_cache=use_cache, synthesize=synthesize, timeout=timeout) for item in rows]
    else:
        results = []
        with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="runtime-bench") as executor:
            futures = [executor.submit(_request, base_url, item, mode, use_cache=use_cache, synthesize=synthesize, timeout=timeout) for item in rows]
            for future in as_completed(futures):
                results.append(future.result())
    phase_wall_ms = round((time.perf_counter() - phase_started) * 1000, 2)
    for row in results:
        row["phase_wall_ms"] = phase_wall_ms
    return results


def _phase_summary(rows: list[dict[str, Any]], *, concurrency: int) -> dict[str, Any]:
    successful = [row for row in rows if _execution_status(row) == "SUCCEEDED"]
    errors = [row for row in rows if _execution_status(row) != "SUCCEEDED"]
    wall = summarize_latencies(row["wall_latency_ms"] for row in successful)
    stage_values: dict[str, dict[str, float | None]] = {}
    for key in STAGE_KEYS:
        values = [float(row["stage_latency_ms"][key]) for row in successful if key in row.get("stage_latency_ms", {})]
        if values:
            stage_values[key] = summarize_latencies(values)
    elapsed_seconds = (float(rows[0].get("phase_wall_ms", 0)) / 1000) if rows else 0.0
    return {
        "request_count": len(rows),
        "success_count": len(successful),
        "error_count": len(errors),
        "error_rate": round(len(errors) / len(rows), 4) if rows else None,
        "cache_hit_count": sum(bool(row.get("cache_hit")) for row in successful),
        "cache_miss_count": sum(not bool(row.get("cache_hit")) for row in successful),
        "concurrency": concurrency,
        "throughput_requests_per_second_approx": round(len(successful) / elapsed_seconds, 3) if elapsed_seconds > 0 else None,
        "phase_wall_ms": round(elapsed_seconds * 1000, 2) if rows else None,
        "wall_latency_ms": wall,
        "stage_latency_ms": stage_values,
        "errors": errors[:20],
    }


def benchmark(dataset: list[dict[str, Any]], *, base_url: str, limit: int | None, timeout: float, concurrency: int, synthesize: bool, seed: int = 20260920) -> dict[str, Any]:
    rows = list(dataset[:limit] if limit else dataset)
    random.Random(seed).shuffle(rows)
    all_rows: list[dict[str, Any]] = []
    phases: dict[str, Any] = {}
    for mode in MODES:
        cold_rows = _run_requests(base_url, rows, mode, use_cache=False, synthesize=synthesize, timeout=timeout, concurrency=concurrency)
        phases[f"{mode}:cache_miss"] = _phase_summary(cold_rows, concurrency=concurrency)
        all_rows.extend([{**row, "phase": "cache_miss"} for row in cold_rows])

        warm_rows = _run_requests(base_url, rows, mode, use_cache=True, synthesize=synthesize, timeout=timeout, concurrency=concurrency)
        phases[f"{mode}:cache_hit_or_fill"] = _phase_summary(warm_rows, concurrency=concurrency)
        all_rows.extend([{**row, "phase": "cache_hit_or_fill"} for row in warm_rows])

    return {
        "schema": "strategic-graphrag-runtime-performance/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_type": "API_runtime_not_quality_evaluation",
        "base_url": base_url,
        "dataset": {
            "path": str(DEFAULT_DATASET),
            "status": "AUTO_GENERATED_SILVER_NOT_HUMAN_GOLD",
            "question_count": len(rows),
        },
        "run_config": {
            "modes": list(MODES),
            "synthesize": synthesize,
            "concurrency": concurrency,
            "timeout_seconds": timeout,
            "order_seed": seed,
            "python": platform.python_version(),
            "cache_miss_definition": "API request use_cache=false; does not imply process or dependency cold start",
            "cache_hit_definition": "API request use_cache=true and response metadata.cache.hit=true",
        },
        "phases": phases,
        "rows": all_rows,
        "limitations": [
            "The script cannot restart the API or Neo4j; cache_miss is not a process-level cold-start measurement.",
            "The Silver questions are graph-derived and are not independent human QA.",
            "Retrieval-only runs isolate retrieval and orchestration; synthesis runs include external LLM latency and must be reported separately.",
            "Approximate throughput is calculated from aggregate request wall time and should be complemented by a dedicated load test before deployment claims.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GraphRAG API runtime and stage latency")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=4, help="Number of fixed questions; default keeps a smoke benchmark bounded")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--synthesize", action="store_true", help="Include external LLM answer synthesis; report separately from retrieval-only runs")
    parser.add_argument("--seed", type=int, default=20260920, help="Deterministic question-order seed")
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be positive")
    dataset = load_jsonl(args.dataset)
    report = benchmark(dataset, base_url=args.base_url, limit=args.limit, timeout=args.timeout, concurrency=args.concurrency, synthesize=args.synthesize, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "phases": list(report["phases"]), "questions": report["dataset"]["question_count"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
