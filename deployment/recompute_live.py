"""Recompute local smoke assertions from saved responses; never call a service."""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live_acceptance import CASES, score


def recompute(raw: Path, expected_hash: str | None = None) -> dict:
    digest = hashlib.sha256(raw.read_bytes()).hexdigest()
    if expected_hash and digest != expected_hash:
        raise ValueError("raw input hash mismatch")
    rows = [json.loads(line) for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()]
    cases = {case["id"]: case for case in CASES}
    ids = [row["case_id"] for row in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(cases):
        raise ValueError("expected each of the five fixed smoke cases exactly once")
    rescored = []
    for row in rows:
        case = cases[row["case_id"]]
        if row["input"]["question"] != case["question"]:
            raise ValueError("question does not match recorded fixture protocol")
        verdict = score(case, row.get("response") or {})
        passed = row.get("http_status") == 200 and verdict["status"] == "PASS"
        rescored.append({"case_id": row["case_id"], "passed": passed, "score": verdict})
    successes = [row for row in rows if row.get("http_status") == 200]
    durations = [row["elapsed_ms"] for row in rows if row.get("elapsed_ms") is not None]
    return {
        "schema": "recomputed-live-smoke/v1",
        "raw_sha256": digest,
        "reference_level": "AI_PDF_INTEGRATION_FIXTURE_NOT_GOLD",
        "http_success": {"numerator": len(successes), "denominator": len(rows)},
        "scenario_acceptance_all_requests": {"numerator": sum(row["passed"] for row in rescored), "denominator": len(rows)},
        "scenario_acceptance_http_success_subset": {"numerator": sum(row["passed"] for row in rescored), "denominator": len(successes)},
        "latency_all_attempts_ms": {"measured": len(durations), "total": len(rows),
            "p50": statistics.median(durations) if durations else None,
            "max": max(durations) if durations else None, "p95": None,
            "limitation": "five-case smoke run, not a latency benchmark"},
        "cases": rescored,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    print(json.dumps(recompute(args.raw, args.expected_sha256), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
