"""Audit the 2026-09-19 table candidates against their cited source PDF pages.

This is a machine-assisted source audit, not a human cell-level annotation.
It deliberately leaves uncertain field association and joint correctness
unresolved for independent reviewers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "evaluation/annotation/table_quality_candidate_2026-09-19.jsonl"
PDFS = {
    "2023-10-K.pdf": ROOT / "data/pdfs_other/2023-10-K.pdf",
    "2024-10-K.pdf": ROOT / "data/pdfs_other/2024-10-K.pdf",
    "2025-10-K.pdf": ROOT / "data/pdfs/2025-10-K.pdf",
}
METRIC_ALIASES = {
    "operating_cost": ("operating expenses", "operating expense"),
    "revenue": ("revenue", "revenues"),
    "gross_profit": ("gross profit",),
    "operating_income": ("operating income",),
    "net_income": ("net income",),
    "research_and_development": ("research and development", "R&D"),
    "research_development": ("research and development", "R&D"),
    "cash_flow": ("cash flow",),
    "inventory": ("inventory",),
    "total_assets": ("total assets",),
    "capital_expenditures": ("capital expenditures", "property and equipment"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%.$()\-]+", " ", text.casefold())).strip()


def value_forms(raw: Any, value: Any) -> set[str]:
    forms = {str(raw or "").strip(), str(value if value is not None else "").strip()}
    result = set()
    for form in forms:
        if not form:
            continue
        compact = re.sub(r"[$,\s]", "", form).strip()
        result.add(compact.casefold())
        try:
            number = float(compact.replace("(", "-").replace(")", ""))
            result.add(str(number).removesuffix(".0"))
        except ValueError:
            pass
    return {form for form in result if form}


def excerpt(text: str, needles: list[str], limit: int = 420) -> str:
    lowered = text.casefold()
    starts = [lowered.find(needle.casefold()) for needle in needles if needle and lowered.find(needle.casefold()) >= 0]
    if not starts:
        return re.sub(r"\s+", " ", text)[:limit]
    start = max(0, min(starts) - 150)
    return re.sub(r"\s+", " ", text[start:start + limit])


def _rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in INPUT.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit() -> list[dict[str, Any]]:
    rows = _rows()
    pdf_text: dict[str, list[str]] = {}
    for name, path in PDFS.items():
        with pdfplumber.open(path) as pdf:
            pdf_text[name] = [page.extract_text() or "" for page in pdf.pages]
    output = []
    for row in rows:
        filing = str(row.get("source_filing") or "")
        try:
            page_num = int(row.get("page"))
        except (ValueError, TypeError):
            page_num = 0
        text_pages = pdf_text.get(filing) or []
        page_exists = 1 <= page_num <= len(text_pages)
        text = text_pages[page_num - 1] if page_exists else ""
        compact_text = re.sub(r"\s+", " ", text).casefold()
        sentence = str(row.get("evidence_sentence") or "").strip()
        sentence_norm = norm(sentence)
        page_norm = norm(text)
        sentence_supported = bool(sentence_norm and sentence_norm in page_norm)

        forms = value_forms(row.get("raw_value"), row.get("value"))
        page_compact = re.sub(r"[$,\s]", "", compact_text)
        numeric_present = any(form in page_compact for form in forms)
        # Page presence only establishes that a number occurs on the page;
        # this cannot establish row/column cell association.
        value_status = "PRESENT_ON_CITED_PAGE_ONLY" if numeric_present else "NOT_FOUND_ON_CITED_PAGE"
        metric_id = re.sub(r"[^a-z0-9]+", "_", str(row.get("metric_id") or "").casefold()).strip("_")
        aliases = METRIC_ALIASES.get(metric_id, (metric_id.replace("_", " "),) if metric_id else ())
        metric_seen = any(norm(alias) in page_norm for alias in aliases if alias)
        row_label = str(row.get("row_label") or "")
        metric_status = "LABEL_CANDIDATE_ON_PAGE" if metric_seen else "NOT_LOCATED_OR_ALIAS_UNMAPPED"

        year = str(row.get("fiscal_year") or "")
        year_seen = bool(year and re.search(rf"(?<!\d){re.escape(year)}(?!\d)", text))
        unit_text = str(row.get("unit") or "").casefold()
        scale_seen = (
            ("million" in unit_text and bool(re.search(r"\bin millions\b|\$ in millions|\bmillions\b", text, re.I)))
            or ("thousand" in unit_text and bool(re.search(r"\bin thousands\b|\bthousands\b", text, re.I)))
            or ("billion" in unit_text and bool(re.search(r"\bin billions\b|\bbillions\b", text, re.I)))
        )
        currency_seen = (
            ("usd" in unit_text and bool(re.search(r"\$|\bUSD\b|U\.S\. dollars?", text, re.I)))
            or ("eur" in unit_text and bool(re.search(r"€|\bEUR\b|euros?", text, re.I)))
        )

        output.append({
            "id": row.get("id"),
            "queue_id": row.get("queue_id"),
            "company_id_candidate": row.get("company_id"),
            "metric_id_candidate": row.get("metric_id"),
            "fiscal_year_candidate": row.get("fiscal_year"),
            "value_candidate": row.get("value"),
            "raw_value_candidate": row.get("raw_value"),
            "unit_candidate": row.get("unit"),
            "row_label_candidate": row_label,
            "column_label_candidate": row.get("column_label"),
            "source_filing": filing,
            "page_candidate": page_num,
            "source_pdf_sha256": sha256(PDFS[filing]) if filing in PDFS else "",
            "locator_check": "PASS_PAGE_EXISTS" if page_exists else "FAIL_PAGE_OUT_OF_RANGE_OR_FILING_MISSING",
            "evidence_sentence_exact_after_whitespace_normalization": sentence_supported,
            "number_check": value_status,
            "sign_check": "REQUIRES_CELL_ALIGNMENT_REVIEW",
            "unit_check": "CURRENCY_CONTEXT_SEEN" if currency_seen else "CURRENCY_CONTEXT_NOT_DETERMINED",
            "scale_check": "SCALE_CONTEXT_SEEN" if scale_seen else "SCALE_CONTEXT_NOT_DETERMINED",
            "year_check": "YEAR_TEXT_PRESENT_NOT_COLUMN_VERIFIED" if year_seen else "YEAR_NOT_LOCATED",
            "metric_check": metric_status,
            "row_column_cell_association": "REQUIRES_HUMAN_TABLE_REVIEW",
            "joint_correctness": "NOT_DETERMINED",
            "review_status": "MACHINE_SOURCE_AUDIT_COMPLETE_PENDING_PRIMARY_HUMAN_REVIEW",
            "reviewer": "",
            "second_reviewer": "",
            "adjudicator": "",
            "source_excerpt": excerpt(text, [sentence, row_label, *forms]),
        })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New JSONL path; existing files are not overwritten")
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {target}")
    rows = audit()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics = {
        "rows": len(rows),
        "page_locator_pass": sum(row["locator_check"] == "PASS_PAGE_EXISTS" for row in rows),
        "evidence_sentence_present": sum(row["evidence_sentence_exact_after_whitespace_normalization"] for row in rows),
        "number_present_on_page_only": sum(row["number_check"] == "PRESENT_ON_CITED_PAGE_ONLY" for row in rows),
        "year_text_present": sum(row["year_check"].startswith("YEAR_TEXT_PRESENT") for row in rows),
        "metric_label_candidate_present": sum(row["metric_check"] == "LABEL_CANDIDATE_ON_PAGE" for row in rows),
        "joint_correctness": "NOT_DETERMINED_PENDING_HUMAN_CELL_REVIEW",
    }
    print(json.dumps({"output": str(target), "summary": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()
