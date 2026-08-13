"""Run the five required end-to-end query classes and audit citations."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = [
    {"kind": "causal_risk", "question": "How can supply-chain disruption affect NVIDIA revenue?", "cross_filing": False, "source_filing": "2025-10-K.pdf"},
    {"kind": "financial_metric", "question": "What sales, general and administrative expense did NVIDIA report for fiscal 2025?", "cross_filing": False, "source_filing": "2025-10-K.pdf"},
    {"kind": "cross_year", "question": "Compare NVIDIA sales, general and administrative expense in fiscal 2023, 2024, and 2025.", "cross_filing": True, "source_filing": None},
    {"kind": "evidence_lookup", "question": "What filing evidence supports the relationship that supply-chain disruption can decrease NVIDIA revenue in fiscal 2025?", "cross_filing": False, "source_filing": "2025-10-K.pdf"},
    {"kind": "unsupported_abstention", "question": "What exact NVIDIA quantum-computing revenue was reported for fiscal 2025?", "cross_filing": False, "source_filing": "2025-10-K.pdf"},
]


def _claim_index(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            return {
                record["id"]: record.data()
                for record in session.run(
                    "MATCH (c:EvidenceClaim) WHERE c.id IN $ids "
                    "RETURN c.id AS id,c.doc_id AS doc_id,c.page AS page,c.fiscal_year AS fiscal_year,"
                    "c.text AS text,c.verification_status AS verification_status",
                    ids=ids,
                )
            }
    finally:
        driver.close()


def run(base_url: str) -> dict:
    load_dotenv(ROOT / ".env", override=True)
    rows = []
    for item in QUESTIONS:
        started = time.perf_counter()
        row = dict(item)
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/query",
                json={
                    "question": item["question"],
                    "max_paths": 8,
                    "retrieval_mode": "hybrid",
                    "vector_top_k": 8,
                    "source_filing": item["source_filing"],
                    "cross_filing": item["cross_filing"],
                },
                timeout=240,
            )
            payload = response.json()
            grounding = (payload.get("metadata") or {}).get("grounding") or {}
            ids = list(dict.fromkeys(grounding.get("cited_evidence_ids") or []))
            claims = _claim_index(ids)
            path_pairs = {
                (str(evidence_id), int(page), int(year))
                for path in payload.get("paths", [])
                for evidence_id, page, year in zip(
                    path.get("evidence_ids") or [], path.get("pages") or [], path.get("years") or []
                )
                if evidence_id and str(page).isdigit() and str(year).isdigit()
            }
            citation_contract = all(
                claim_id in claims
                and claims[claim_id].get("verification_status") == "VERBATIM"
                and (claim_id, int(claims[claim_id]["page"]), int(claims[claim_id]["fiscal_year"])) in path_pairs
                for claim_id in ids
            )
            answer = str(payload.get("answer") or "")
            abstained = answer.startswith(("[INSUFFICIENT", "[GROUNDING FAILURE]"))
            years_returned = sorted({
                int(year)
                for path in payload.get("paths", [])
                for year in path.get("years", []) or []
                if str(year).isdigit()
            })
            semantic_pass = {
                "causal_risk": not abstained and bool(ids),
                "financial_metric": not abstained and bool(ids) and "3,491" in answer,
                "cross_year": (
                    not abstained
                    and bool(ids)
                    and years_returned == [2023, 2024, 2025]
                    and all(value in answer for value in ("2,440", "2,654", "3,491"))
                ),
                "evidence_lookup": not abstained and bool(ids),
                "unsupported_abstention": abstained,
            }[item["kind"]]
            row.update({
                "http_status": response.status_code,
                "answer": answer,
                "grounding_status": grounding.get("status"),
                "cited_evidence_ids": ids,
                "citation_contract_pass": citation_contract if ids else abstained,
                "semantic_pass": semantic_pass,
                "abstained": abstained,
                "paths": len(payload.get("paths") or []),
                "years_returned": years_returned,
                "error": None,
            })
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        rows.append(row)
    latencies = [row["latency_ms"] for row in rows]
    return {
        "suite": "five_required_query_classes",
        "provider": os.getenv("LLM_PROVIDER"),
        "model": os.getenv("LLM_MODEL"),
        "summary": {
            "evaluated": len(rows),
            "http_200": sum(row.get("http_status") == 200 for row in rows),
            "errors": sum(bool(row.get("error")) for row in rows),
            "citation_contract_pass": sum(bool(row.get("citation_contract_pass")) for row in rows),
            "semantic_pass": sum(bool(row.get("semantic_pass")) for row in rows),
            "unsupported_abstention_pass": any(row["kind"] == "unsupported_abstention" and row.get("abstained") for row in rows),
            "latency_ms": {"min": min(latencies), "max": max(latencies), "mean": round(statistics.mean(latencies), 2)},
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.base_url)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
