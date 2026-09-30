"""Record real HTTP integration trials without relabeling partial answers as passes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


# Source fixture: 2025 Form 10-K, physical page 80, Total revenue row.
# This is an AI/PDF integration fixture, not an independently human-reviewed test.
CASES = [
    {"id": "single_year", "question": "What was NVIDIA revenue in fiscal 2025?", "expected": 130497.0,
     "unit": "USD millions", "source_filing": "2025-10-K.pdf"},
    {"id": "disclosure_version", "question": "What was NVIDIA revenue in fiscal 2023 as disclosed in the 2025 10-K?",
     "expected": 26974.0, "unit": "USD millions", "source_filing": "2025-10-K.pdf"},
    {"id": "growth", "question": "What was NVIDIA revenue growth from fiscal 2023 to fiscal 2025?",
     "expected": (130497.0 - 26974.0) / 26974.0 * 100, "unit": "percent", "cross_filing": True},
    {"id": "unit_conversion", "question": "Convert NVIDIA fiscal 2025 revenue from USD millions to USD billions.",
     "expected": 130.497, "unit": "USD billions", "source_filing": "2025-10-K.pdf"},
    {"id": "unanswerable", "question": "What was NVIDIA revenue in fiscal 2026?", "expected": None,
     "cross_filing": True},
]


def http(base: str, path: str, payload=None, timeout=45) -> dict:
    started = time.perf_counter()
    body = json.dumps(payload).encode() if payload is not None else None
    request = Request(base + path, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            raw, status, content_type = response.read(), response.status, response.headers.get("Content-Type", "")
        error = None
    except HTTPError as exc:
        raw, status, content_type = exc.read(), exc.code, exc.headers.get("Content-Type", "")
        error = "HTTPError"
    except (URLError, TimeoutError, OSError) as exc:
        raw, status, content_type, error = b"", None, "", type(exc).__name__
    result = {"http_status": status, "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
              "error_type": error, "content_type": content_type}
    if "json" in content_type:
        try:
            result["response"] = json.loads(raw)
        except ValueError:
            result["error_type"] = "InvalidJSON"
    return result | {"_bytes": raw}


def score(case: dict, response: dict) -> dict:
    calculation = response.get("calculation") or {}
    status = calculation.get("status")
    value = calculation.get("value")
    if case["expected"] is None:
        passed = status == "INSUFFICIENT_EVIDENCE" and not response.get("citations")
        basis = "out-of-corpus year: explicit insufficient evidence and no unrelated citation"
    else:
        passed = status == "PASS" and isinstance(value, (int, float)) and abs(value - case["expected"]) <= 0.0001 \
                 and calculation.get("unit") == case["unit"]
        basis = "deterministic calculation status, exact source-row value/formula and currency-scale unit"
    return {"status": "PASS" if passed else "FAIL", "basis": basis,
            "calculation_status": status, "actual_value": value, "expected_value": case["expected"],
            "actual_unit": calculation.get("unit"), "expected_unit": case.get("unit")}


def run(base: str, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError("use a new output directory")
    output_dir.mkdir(parents=True)
    rows = []
    readiness = http(base, "/health/ready", timeout=25)
    readiness.pop("_bytes")
    for case in CASES:
        payload = {"question": case["question"], "max_paths": 10, "retrieval_mode": "graph",
                   "synthesize": False, "use_cache": False,
                   "cross_filing": case.get("cross_filing", False), "source_filing": case.get("source_filing")}
        measured = http(base, "/query", payload)
        measured.pop("_bytes")
        response = measured.get("response") or {}
        result = {"case_id": case["id"], "input": payload, "reference": {"level": "AI_PDF_INTEGRATION_FIXTURE",
                  "filing": "2025-10-K.pdf", "physical_page": 80}, **measured, "score": score(case, response)}
        rows.append(result)
    citations = [citation for row in rows for citation in (row.get("response") or {}).get("citations", [])]
    pdf_checks = []
    for filing in sorted({item["source_filing"] for item in citations if item.get("source_filing")}):
        # The application only accepts a filing allowlist; never follow arbitrary returned URLs.
        if filing not in {"2023-10-K.pdf", "2024-10-K.pdf", "2025-10-K.pdf"}:
            continue
        result = http(base, "/evaluation/table-quality/source/" + filing)
        raw = result.pop("_bytes")
        result["source_filing"] = filing
        result["sha256"] = hashlib.sha256(raw).hexdigest() if raw else None
        result["bytes"] = len(raw)
        result["page_checks"] = []
        try:
            import pymupdf
            with pymupdf.open(stream=raw, filetype="pdf") as doc:
                result["page_count"] = len(doc)
                for citation in citations:
                    if citation.get("source_filing") != filing:
                        continue
                    page = citation.get("page")
                    exists = isinstance(page, int) and 1 <= page <= len(doc)
                    text = doc[page - 1].get_text() if exists else ""
                    expected_row = filing == "2025-10-K.pdf" and page == 80
                    result["page_checks"].append({"physical_page": page, "page_exists": exists,
                        "revenue_source_tokens_visible": all(token in text for token in ("130,497", "60,922", "26,974"))
                        if expected_row else None})
        except Exception as exc:
            result["pdf_error_type"] = type(exc).__name__
        pdf_checks.append(result)
    latencies = [row["elapsed_ms"] for row in rows if row["http_status"] is not None]
    successful = [row for row in rows if row["http_status"] == 200]
    summary = {"schema": "live-http-acceptance/v1", "recorded_at": datetime.now(timezone.utc).isoformat(),
        "target_scope": "loopback" if urlsplit(base).hostname in ("127.0.0.1", "localhost", "::1") else "remote",
        "hardware": {"os": platform.system(), "python": platform.python_version(), "logical_cpus": os.cpu_count()},
        "setup": {"concurrency": 1, "queries": len(rows), "repetitions": 1, "result_cache": False,
                  "generation_requested": False, "cold_start": "NOT_MEASURED", "filesystem_cache": "UNCONTROLLED"},
        "requests": {"success": len(successful), "total": len(rows)},
        "scenario_acceptance": {"passed": sum(row["score"]["status"] == "PASS" for row in rows), "total": len(rows)},
        "latency_ms": {"p50": statistics.median(latencies) if latencies else None,
                       "max": max(latencies) if latencies else None,
                       "p95": None, "p95_status": "SMOKE_SAMPLE_TOO_SMALL_FOR_BENCHMARK"},
        "readiness": readiness, "pdf_http_checks": pdf_checks,
        "browser_pdf_click": "NOT_VERIFIED_BY_HTTP_RUNNER", "independent_quality": "NOT_MEASURED"}
    raw_path = output_dir / "requests.jsonl"
    with raw_path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary["raw_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    with (output_dir / "summary.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if urlsplit(base).hostname not in ("127.0.0.1", "localhost", "::1"):
        parser.error("this local acceptance runner only accepts loopback targets")
    summary = run(base, args.output_dir)
    print(json.dumps({"requests": summary["requests"], "acceptance": summary["scenario_acceptance"], "raw_sha256": summary["raw_sha256"]}))
    return 0 if summary["scenario_acceptance"]["passed"] == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
