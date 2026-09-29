"""Freeze source-located QA candidates for independent PDF review.

This script does not promote system-generated answers/evidence to truth. It
only builds a family-split reviewer packet with empty reference labels.
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
DEFAULT_SOURCES = [
    "data/evaluation/golden_qa_v2.jsonl",
    "data/evaluation/golden_qa_v3_current.jsonl",
    "data/evaluation/golden_dataset.json",
    "evaluation/golden_qa_human_v2.jsonl",
    "evaluation/silver_retrieval_v1.jsonl",
    "evaluation/annotation/table_quality_candidate_2026-09-19.jsonl",
]
PDF_PATHS = {
    "2023-10-K.pdf": ROOT / "data/pdfs_other/2023-10-K.pdf",
    "2024-10-K.pdf": ROOT / "data/pdfs_other/2024-10-K.pdf",
    "2025-10-K.pdf": ROOT / "data/pdfs/2025-10-K.pdf",
}
FINANCIAL_METRICS = (
    "revenue", "operating income", "gross profit", "net income",
    "operating expenses", "operating cost", "research and development",
    "cash flow", "assets", "liabilities", "inventory", "capital expenditures",
)


def _question_type(row: dict[str, Any], question: str, filing: str) -> str:
    text = question.casefold()
    original = str(row.get("question_type") or row.get("candidate_question_type") or "").casefold()
    if str(row.get("id") or "").startswith("FO_"):
        try:
            fact_year = int(row.get("fiscal_year"))
            filing_year = int(filing[:4])
            if fact_year < filing_year:
                return "fact_year_in_later_disclosure"
        except (ValueError, TypeError):
            pass
        return "single_year_fact"
    if row.get("candidate_answerable") is False or "unsupported" in original or "unanswerable" in original:
        return "unanswerable_or_ambiguous"
    if row.get("answerable") is False and row.get("review_status") == "HUMAN_REVIEWED":
        return "unanswerable_or_ambiguous"
    if any(term in text for term in ("compare", "versus", "year-over-year", "across fiscal", "across fy")) or "temporal_metric" in original:
        return "cross_year_comparison"
    if re.search(r"\b(convert|conversion|percentage change|growth|ratio|calculate the difference)\b", text):
        return "unit_conversion_or_calculation"
    years = sorted({int(year) for year in re.findall(r"\b20\d{2}\b", question)})
    filing_year = int(filing[:4]) if filing else None
    if any(metric in text for metric in FINANCIAL_METRICS) and filing_year and any(year < filing_year for year in years):
        return "fact_year_in_later_disclosure"
    if "multi_hop" in original or any(term in text for term in ("risk", "conditional", "exposed", "mitigat", "cause", "through")):
        return "relation_or_conditional_risk"
    if original in {"single_hop", "direct_relation"}:
        return "relation"
    if any(metric in text for metric in FINANCIAL_METRICS) and years:
        return "single_year_fact"
    return "relation"


def _read(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return value
    return value.get("questions", value.get("data", []))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _question(row: dict[str, Any]) -> str:
    question = str(row.get("question") or row.get("candidate_question") or "").strip()
    if question:
        return question
    if str(row.get("id") or "").startswith("FO_"):
        metric = str(row.get("metric_id") or "financial metric").replace("_", " ")
        year = row.get("fiscal_year")
        filing = _filing(row)
        if year and filing:
            return f"What {metric} did NVIDIA report for fiscal year {year} in its {filing[:4]} 10-K?"
    return ""


def _filing(row: dict[str, Any]) -> str:
    value = str(row.get("source_filing") or row.get("candidate_source_filing") or "").strip()
    match = re.search(r"20\d{2}-10-K\.pdf", value, flags=re.I)
    return match.group(0) if match else ""


def _pages(row: dict[str, Any]) -> list[int]:
    raw = row.get("pages") or row.get("expected_pages") or row.get("candidate_pages") or row.get("gold_pages") or []
    if not raw and row.get("page") not in (None, ""):
        raw = [row.get("page")]
    result: list[int] = []
    for value in raw:
        try:
            page = int(value)
        except (ValueError, TypeError):
            continue
        if page > 0 and page not in result:
            result.append(page)
    return result


def _triples(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    values = row.get("supporting_triples") or row.get("candidate_supporting_triples") or []
    return [
        (
            re.sub(r"\W+", "_", str(item.get("source") or "").casefold()).strip("_"),
            str(item.get("relation") or "").strip().upper(),
            re.sub(r"\W+", "_", str(item.get("target") or "").casefold()).strip("_"),
        )
        for item in values
        if isinstance(item, dict)
    ]


def _family_key(row: dict[str, Any]) -> str:
    question = _question(row).casefold()
    question = re.sub(r"\bfy\s*20\d{2}\b|\b20\d{2}\b", " YEAR ", question)
    question = re.sub(r"\s+", " ", question).strip(" ?.!\t")
    metric_id = str(row.get("metric_id") or "").strip().casefold().replace(" ", "_")
    fact_years = []
    raw_years = row.get("years") or []
    if not isinstance(raw_years, (list, tuple)):
        raw_years = [raw_years]
    for value in raw_years:
        match = re.search(r"20\d{2}", str(value or ""))
        if match:
            fact_years.append(int(match.group(0)))
    if row.get("fiscal_year"):
        try:
            fact_years.append(int(row["fiscal_year"]))
        except (ValueError, TypeError):
            pass
    if not fact_years:
        fact_years.extend(int(value) for value in re.findall(r"\b(?:fy\s*)?(20\d{2})\b", _question(row), flags=re.I))
    fact_years = sorted(set(fact_years))
    triples = _triples(row)
    original_type = str(row.get("question_type") or row.get("candidate_question_type") or "").casefold()
    is_unanswerable_candidate = (
        row.get("candidate_answerable") is False
        or (row.get("answerable") is False and row.get("review_status") == "HUMAN_REVIEWED")
        or "unsupported" in original_type
        or "unanswerable" in original_type
    )
    if is_unanswerable_candidate:
        return "unanswerable:" + question
    metric_triples = [triple for triple in triples if triple[1] == "REPORTS_METRIC"]
    if metric_triples and not metric_id:
        metric_id = metric_triples[0][2]
    if metric_id:
        # Group every period/version of a metric together. This conservative
        # split keeps single-year, comparative, and calculated variants of
        # the same metric from leaking across development and test.
        return f"financial:{metric_id}"
    if triples:
        # Filing version is intentionally excluded so alternate filings or
        # paraphrases of one claimed relation cannot cross the split.
        return "relation:" + json.dumps(triples, separators=(",", ":"))
    question_type = str(row.get("question_type") or row.get("candidate_question_type") or "").casefold()
    if any(token in question_type for token in ("temporal", "metric", "calculation")) or any(m in question for m in FINANCIAL_METRICS):
        metrics = [metric for metric in FINANCIAL_METRICS if metric in question]
        if metrics:
            # Keep the fact family together across alternate wording/year
            # variants; a reviewer can still evaluate each requested period.
            return "financial:" + metrics[0].replace(" ", "_")
    return "question:" + question


def _load_pdf_pages() -> dict[str, int]:
    page_counts = {}
    for name, path in PDF_PATHS.items():
        with pdfplumber.open(path) as pdf:
            page_counts[name] = len(pdf.pages)
    return page_counts


def prepare(seed: int = 20260924) -> list[dict[str, Any]]:
    page_counts = _load_pdf_pages()
    candidates: dict[str, dict[str, Any]] = {}
    for source_name in DEFAULT_SOURCES:
        path = ROOT / source_name
        if not path.is_file():
            continue
        for row in _read(path):
            question = _question(row)
            filing = _filing(row)
            pages = [page for page in _pages(row) if page <= page_counts.get(filing, 0)]
            label_candidate = (
                row.get("candidate_answerable") is False
                or (row.get("answerable") is False and row.get("review_status") == "HUMAN_REVIEWED")
                or "unsupported" in str(row.get("question_type") or row.get("candidate_question_type") or "").casefold()
                or "unanswerable" in str(row.get("question_type") or row.get("candidate_question_type") or "").casefold()
            )
            if not question or not filing or (not pages and not label_candidate):
                continue
            family_key = _family_key(row)
            family_id = hashlib.sha256(family_key.encode("utf-8")).hexdigest()[:20]
            quality = (
                bool(_triples(row)),
                bool(row.get("evidence_claim_ids") or row.get("expected_evidence_ids") or row.get("gold_evidence_ids")),
                len(pages),
                len(question),
            )
            current = candidates.get(family_id)
            source_entry = {
                "file": source_name,
                "record_id": str(row.get("id") or row.get("candidate_id") or ""),
                "prior_review_status": str(row.get("review_status") or row.get("benchmark_status") or "UNREVIEWED"),
                "prior_reviewer_present": bool(row.get("reviewer")),
            }
            if current is None:
                candidates[family_id] = {
                    "family_id": family_id,
                    "family_key": family_key,
                    "question": question,
                    "question_type": _question_type(row, question, filing),
                    "candidate_sources": [source_entry],
                    "candidate_filing": filing,
                    "candidate_pages": pages,
                    "candidate_page_scope": "WHOLE_THREE_FILING_CORPUS_NEGATIVE_SEARCH" if not pages else "POSITIVE_PAGE_HINT_ONLY",
                    "candidate_triples": _triples(row),
                    "candidate_evidence_ids": row.get("evidence_claim_ids") or row.get("expected_evidence_ids") or row.get("gold_evidence_ids") or [],
                    "candidate_answerable": row.get("answerable", row.get("candidate_answerable")),
                    "candidate_answer_sha256": _sha256_text(str(row.get("expected_answer") or row.get("reference_answer") or row.get("candidate_expected_answer") or "")),
                    "candidate_variants": [],
                    "_quality": quality,
                }
            else:
                if source_entry not in current["candidate_sources"]:
                    current["candidate_sources"].append(source_entry)
                if quality > current["_quality"]:
                    current.update({
                        "question": question,
                        "question_type": _question_type(row, question, filing),
                        "candidate_filing": filing,
                        "candidate_pages": pages,
                        "candidate_page_scope": "WHOLE_THREE_FILING_CORPUS_NEGATIVE_SEARCH" if not pages else "POSITIVE_PAGE_HINT_ONLY",
                        "candidate_triples": _triples(row),
                        "candidate_evidence_ids": row.get("evidence_claim_ids") or row.get("expected_evidence_ids") or row.get("gold_evidence_ids") or [],
                        "candidate_answerable": row.get("answerable", row.get("candidate_answerable")),
                        "candidate_answer_sha256": _sha256_text(str(row.get("expected_answer") or row.get("reference_answer") or row.get("candidate_expected_answer") or "")),
                        "_quality": quality,
                    })

    values = list(candidates.values())

    # Candidate-only multi-year and calculation variants are constructed only
    # where the existing PDF-located table annotations contain the requisite
    # fiscal-year observations. No result/answer is generated here.
    table_rows = _read(ROOT / "evaluation/annotation/table_quality_candidate_2026-09-19.jsonl")
    grouped_table: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in table_rows:
        metric = str(row.get("metric_id") or "").casefold().replace(" ", "_")
        filing = _filing(row)
        if metric and filing:
            grouped_table.setdefault((metric, filing), []).append(row)
    metric_labels = {
        "operating_cost": "operating expenses", "cost_of_revenue": "cost of revenue",
        "pretax_income": "pretax income", "r_and_d_expense": "research and development expense",
        "sg_and_a_expense": "sales, general and administrative expense", "cash_and_cash_equivalents": "cash and cash equivalents",
        "marketable_securities": "marketable securities", "net_income": "net income",
        "gross_profit": "gross profit", "revenue": "revenue", "inventories": "inventories",
        "total_liabilities": "total liabilities", "total_assets": "total assets",
        "total_current_assets": "total current assets", "total_current_liabilities": "total current liabilities",
        "accounts_receivable": "accounts receivable",
    }
    for (metric, filing), observations in grouped_table.items():
        years = sorted({int(row["fiscal_year"]) for row in observations if str(row.get("fiscal_year", "")).isdigit()})
        if len(years) < 2:
            continue
        family_key = f"financial:{metric}"
        family_id = hashlib.sha256(family_key.encode("utf-8")).hexdigest()[:20]
        family = candidates.get(family_id)
        if family is None:
            continue
        label = metric_labels.get(metric, metric.replace("_", " "))
        pages = sorted({int(row["page"]) for row in observations if str(row.get("page", "")).isdigit()})
        year_phrase = ", ".join(f"FY{year}" for year in years[:-1]) + f" and FY{years[-1]}"
        filing_year = filing[:4]
        variants = [
            (f"Compare NVIDIA {label} for {year_phrase} as disclosed in its {filing_year} 10-K.", "cross_year_comparison")
        ]
        if metric == "revenue" and len(years) >= 2:
            first_year, second_year = years[-2:]
            variants.append((
                f"Calculate the percentage change in NVIDIA revenue from FY{first_year} to FY{second_year}, using the {filing_year} 10-K.",
                "unit_conversion_or_calculation",
            ))
            if any("million" in str(row.get("unit") or "").casefold() for row in observations):
                variants.append((
                    f"Convert NVIDIA FY{second_year} revenue from USD millions to USD billions using the {filing_year} 10-K.",
                    "unit_conversion_or_calculation",
                ))
        source_ids = sorted({str(row.get("id") or "") for row in observations if row.get("id")})
        for question, question_type in variants:
            if any(item.get("question") == question for item in family["candidate_variants"]):
                continue
            variant_key = f"{family_id}:{question}"
            family["candidate_variants"].append({
                "item_id": "FQAV-" + hashlib.sha256(variant_key.encode("utf-8")).hexdigest()[:16],
                "family_id": family_id,
                "question": question,
                "question_type": question_type,
                "candidate_filing": filing,
                "candidate_pages": pages,
                "candidate_page_scope": "POSITIVE_PAGE_HINT_ONLY",
                "candidate_source_ids": source_ids,
                "reference_answer": "",
                "gold_evidence_ids": [],
                "gold_pages": [],
                "relevance_grades": {},
                "primary_reviewer": "",
                "secondary_reviewer": "",
                "adjudicator": "",
                "review_notes": "",
                "label_status": "PENDING_PRIMARY_PDF_REVIEW",
                "gold_status": "NOT_GOLD",
            })

    ordered = sorted(values, key=lambda item: hashlib.sha256(f"{seed}:{item['family_id']}".encode()).hexdigest())
    dev_count = int(round(len(ordered) * 0.20))
    dev_families = {item["family_id"] for item in ordered[:dev_count]}
    output: list[dict[str, Any]] = []
    for item in sorted(values, key=lambda value: (value["candidate_filing"], min(value["candidate_pages"] or [0]), value["family_id"])):
        item.pop("_quality", None)
        source_path = PDF_PATHS[item["candidate_filing"]]
        item.update({
            "item_id": "FQA-" + item["family_id"],
            "partition": "development" if item["family_id"] in dev_families else "test",
            "source_pdf_sha256": _sha256(source_path),
            "review_scope_pdf_sha256": {name: _sha256(path) for name, path in PDF_PATHS.items()} if not item["candidate_pages"] else {},
            "reference_answer": "",
            "gold_evidence_ids": [],
            "gold_pages": [],
            "relevance_grades": {},
            "primary_reviewer": "",
            "primary_reviewed_at": "",
            "secondary_reviewer": "",
            "secondary_reviewed_at": "",
            "adjudicator": "",
            "review_notes": "",
            "label_status": "PENDING_PRIMARY_PDF_REVIEW",
            "gold_status": "NOT_GOLD",
        })
        for variant in item.get("candidate_variants", []):
            variant["partition"] = item["partition"]
            variant["source_pdf_sha256"] = _sha256(PDF_PATHS[variant["candidate_filing"]])
            variant["gold_status"] = "NOT_GOLD"
        output.append(item)
    return output


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="New JSONL path; existing files are not overwritten")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite reviewer packet: {target}")
    rows = prepare(args.seed)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(target),
        "candidate_families": len(rows),
        "candidate_questions": sum(1 + len(row.get("candidate_variants") or []) for row in rows),
        "development": sum(row["partition"] == "development" for row in rows),
        "test": sum(row["partition"] == "test" for row in rows),
        "review_status": "PENDING_PRIMARY_PDF_REVIEW",
        "gold_status": "NOT_GOLD",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
