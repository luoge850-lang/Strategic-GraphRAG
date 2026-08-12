"""Run standard single-filing queries and audit every cited EvidenceClaim.

This script is intentionally explicit: it records HTTP/engine latency, error
types, returned claims and performs a second read-only Neo4j + PDF quote check.
It sends retrieved 2025-10-K evidence to the configured provider only when the
caller has explicitly authorized that action.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from neo4j import GraphDatabase
import pdfplumber


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = [
    "How do export controls impact NVIDIA revenue in China?",
    "What supply-chain risks could affect NVIDIA financial performance?",
    "What products and markets are connected to NVIDIA revenue risk?",
    "What mitigation strategies does NVIDIA disclose for identified risks?",
    "How did NVIDIA revenue risk change from FY2023 to FY2025 according to the filing?",
]


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 2)


def _load_pdf_pages(pdf_path: Path) -> dict[int, str]:
    with pdfplumber.open(pdf_path) as document:
        return {index: (page.extract_text() or "") for index, page in enumerate(document.pages, 1)}


def _fetch_claims(claim_ids: list[str], filename: str) -> dict[str, dict[str, Any]]:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            rows = session.run(
                """
                MATCH (c:EvidenceClaim)
                WHERE c.id IN $ids
                  AND c.doc_id = replace($filename, '.pdf', '')
                RETURN c.id AS id, c.text AS text,
                       c.page AS page, c.year AS year,
                       c.evidence_char_start AS char_start,
                       c.evidence_char_end AS char_end,
                       c.verification_status AS verification_status
                """,
                ids=claim_ids,
                filename=filename,
            )
            return {str(row["id"]): dict(row) for row in rows}
    finally:
        driver.close()


def _quote_check(claim: dict[str, Any], pages: dict[int, str]) -> dict[str, Any]:
    page_number = int(claim["page"])
    quote = str(claim.get("text") or "").strip()
    page_text = pages.get(page_number, "")
    normalized_quote = re.sub(r"\s+", " ", quote)
    normalized_page = re.sub(r"\s+", " ", page_text)
    exact = normalized_quote in normalized_page if normalized_quote else False
    start = claim.get("char_start")
    end = claim.get("char_end")
    span_valid = isinstance(start, int) and isinstance(end, int) and 0 <= start < end
    return {
        "claim_id": claim["id"],
        "page": page_number,
        "quote_present": bool(quote),
        "quote_matches_pdf_page": exact,
        "char_span_present": span_valid,
        "verification_status": claim.get("verification_status"),
    }


def run(base_url: str, pdf_path: Path, filename: str, output: Path) -> dict[str, Any]:
    load_dotenv(ROOT / ".env", override=True)
    pages = _load_pdf_pages(pdf_path)
    questions = DEFAULT_QUESTIONS
    rows: list[dict[str, Any]] = []

    for question in questions:
        started = time.perf_counter()
        row: dict[str, Any] = {"question": question}
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/query",
                json={
                    "question": question,
                    "max_paths": 5,
                    "retrieval_mode": "hybrid",
                    "vector_top_k": 5,
                    "source_filing": filename,
                },
                timeout=240,
            )
            row["http_status"] = response.status_code
            payload = response.json()
            row["answer"] = payload.get("answer", "")
            row["metadata"] = payload.get("metadata", {})
            row["paths"] = payload.get("paths", [])
            row["evidence_sentences"] = payload.get("evidence_sentences", [])
            grounding = (payload.get("metadata") or {}).get("grounding") or {}
            row["grounding"] = grounding
            cited_ids = list(dict.fromkeys(grounding.get("cited_evidence_ids", [])))
            row["cited_evidence_ids"] = cited_ids
            claims = _fetch_claims(cited_ids, filename) if cited_ids else {}
            row["claim_lookup"] = {
                "requested": cited_ids,
                "found": sorted(claims),
                "missing": sorted(set(cited_ids) - set(claims)),
            }
            row["pdf_quote_audit"] = [_quote_check(claims[claim_id], pages) for claim_id in cited_ids if claim_id in claims]
            answer_text = str(row.get("answer") or "")
            is_explicit_abstention = answer_text.startswith((
                "[INSUFFICIENT EVIDENCE]",
                "[INSUFFICIENT TEMPORAL EVIDENCE]",
                "[GROUNDING FAILURE]",
            ))
            row["response_class"] = "abstention" if is_explicit_abstention else "grounded_answer"
            row["citation_audit_pass"] = (
                not row["claim_lookup"]["missing"]
                and all(item["quote_matches_pdf_page"] and item["char_span_present"] for item in row["pdf_quote_audit"])
            ) if cited_ids else grounding.get("status") in {"NOT_APPLICABLE", "INSUFFICIENT"}
            if not cited_ids and is_explicit_abstention:
                row["citation_audit_pass"] = True
            row["error_type"] = None
        except Exception as exc:  # pragma: no cover - runtime/network diagnostic path
            row["http_status"] = None
            row["answer"] = ""
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)
        row["wall_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        rows.append(row)

    latencies = [float(row["wall_latency_ms"]) for row in rows]
    verified = [row for row in rows if row.get("grounding", {}).get("status") == "VERIFIED"]
    report = {
        "dataset": "standard_single_pdf_queries",
        "source_filing": filename,
        "pdf_pages": len(pages),
        "provider": os.getenv("LLM_PROVIDER"),
        "model": os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL"),
        "evaluated": len(rows),
        "summary": {
            "http_200": sum(row.get("http_status") == 200 for row in rows),
            "verified_grounding": len(verified),
            "citation_audit_pass": sum(bool(row.get("citation_audit_pass")) for row in rows),
            "errors": [row.get("error_type") for row in rows if row.get("error_type")],
            "latency_ms": {
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "mean": round(statistics.mean(latencies), 2) if latencies else None,
            },
        },
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--pdf", type=Path, default=ROOT / "data" / "pdfs" / "2025-10-K.pdf")
    parser.add_argument("--filename", default="2025-10-K.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "standard_query_audit.json")
    args = parser.parse_args()
    report = run(args.base_url, args.pdf, args.filename, args.output)
    print(json.dumps({"summary": report["summary"], "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
