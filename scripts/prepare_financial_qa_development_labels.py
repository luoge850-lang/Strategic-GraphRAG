"""Create a small, PDF-checked development label set (never human Gold).

Labels are source-derived by the assistant from the filing PDFs. The script
checks the cited passages and hashes before emitting an immutable JSONL packet.
It deliberately keeps unjudged pages unknown and excludes no-positive items
from retrieval metrics rather than treating them as irrelevant.
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
PACKET = ROOT / "reports/evaluation/financial_qa_review_packet_2026-09-24_v7.jsonl"
PDFS = {
    "2023-10-K.pdf": (ROOT / "data/pdfs_other/2023-10-K.pdf", "89981bbfcd91e20498c1060d7efb39022ac8f85e639f2841091695885aa9f8a8"),
    "2024-10-K.pdf": (ROOT / "data/pdfs_other/2024-10-K.pdf", "536f66d7f1c3413abbf643e0a02bd0aab65639116fe630225f3f93529244658b"),
    "2025-10-K.pdf": (ROOT / "data/pdfs/2025-10-K.pdf", "b67bd67a64488a54886de001c788bc5059a965fc6ef5b4d8b71624951e13df8e"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).casefold()


def _stable_id(value: str, prefix: str) -> str:
    return prefix + hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def _add_label(
    row: dict[str, Any], *, answer: str, answerability: str,
    grades: dict[str, int], notes: str, retrieval_status: str = "SCORED_KNOWN_SUPPORT_PAGES",
) -> dict[str, Any]:
    row = dict(row)
    row["reference_answer"] = answer
    row["answerability_label"] = answerability
    row["gold_page_grades"] = grades
    row["gold_pages"] = [int(key.rsplit("#", 1)[1]) for key, grade in grades.items() if grade > 0]
    row["relevance_grades"] = grades
    row["retrieval_scoring_status"] = retrieval_status
    row["review_notes"] = notes
    row["label_status"] = "AI_SOURCE_REVIEW_NOT_HUMAN"
    row["label_tier"] = "AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW"
    row["source_review_role"] = "AI assistant checked primary filing PDF text and rendered page; not a human reviewer"
    row["human_primary_review"] = "NOT_RUN"
    row["human_secondary_review"] = "NOT_RUN"
    row["adjudication_status"] = "NOT_RUN"
    row["gold_status"] = "NOT_GOLD"
    row["whole_corpus_negative_search"] = False
    return row


def _new_family(
    *, key: str, question: str, question_type: str, filing: str, pages: list[int],
    answer: str, grades: dict[str, int], notes: str,
    variants: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    family_id = _stable_id(key, "DEV-")
    row = {
        "item_id": "FQA-" + family_id,
        "family_id": family_id,
        "family_key": key,
        "partition": "development",
        "question": question,
        "question_type": question_type,
        "candidate_filing": filing,
        "candidate_pages": pages,
        "candidate_page_scope": "PDF_PAGE_CHECKED_NOT_EXHAUSTIVE_RELEVANCE_JUDGMENT",
        "candidate_variants": variants or [],
        "candidate_sources": [{"file": "primary filing PDF", "record_id": key, "prior_review_status": "NOT_FROM_GRAPH_ANSWER"}],
        "source_pdf_sha256": PDFS[filing][1],
    }
    for variant in row["candidate_variants"]:
        variant["family_id"] = family_id
        variant["partition"] = "development"
        variant["candidate_filing"] = filing
        variant["source_pdf_sha256"] = PDFS[filing][1]
        variant["gold_status"] = "NOT_GOLD"
        variant["label_status"] = "AI_SOURCE_REVIEW_NOT_HUMAN"
        variant["label_tier"] = "AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW"
        variant["human_primary_review"] = "NOT_RUN"
        variant["human_secondary_review"] = "NOT_RUN"
        variant["adjudication_status"] = "NOT_RUN"
        variant["source_review_role"] = "AI assistant checked primary filing PDF text; not a human reviewer"
        variant["whole_corpus_negative_search"] = False
    return _add_label(row, answer=answer, answerability="ANSWERABLE", grades=grades, notes=notes)


def prepare() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pdf_pages: dict[str, dict[int, str]] = {}
    checks = {
        ("2023-10-K.pdf", 12): ["could increase our costs"],
        ("2023-10-K.pdf", 43): ["Research and development expenses $ 7,339 $ 5,268"],
        ("2023-10-K.pdf", 44): ["Marketable securities 9,907 19,218"],
        ("2023-10-K.pdf", 56): ["Accounts receivable, net 3,827 4,650", "Total current assets 23,073 28,829"],
        ("2024-10-K.pdf", 36): ["Operating expenses $ 11,329"],
        ("2024-10-K.pdf", 50): ["Revenue $ 60,922 $ 26,974 $ 26,914"],
        ("2024-10-K.pdf", 52): ["Total current assets 44,345 23,073"],
        ("2025-10-K.pdf", 3): ["known and unknown risks"],
        ("2025-10-K.pdf", 5): ["Professional artists, architects and designers", "GeForce GPUs"],
        ("2025-10-K.pdf", 7): ["DRIVE Hyperion"],
        ("2025-10-K.pdf", 10): ["Over the past three years", "export control restrictions"],
        ("2025-10-K.pdf", 14): ["Risks Related to Our Industry and Markets"],
        ("2025-10-K.pdf", 17): ["Extended lead times may occur", "pandemics"],
        ("2025-10-K.pdf", 18): ["We do not assemble, test, or package", "gross margin, revenue"],
        ("2025-10-K.pdf", 22): ["our costs could increase"],
        ("2025-10-K.pdf", 52): ["Revenue $ 130,497 $ 60,922 $ 26,974"],
        ("2023-10-K.pdf", 54): ["Revenue $ 26,974 $ 26,914 $ 16,675"],
    }
    pages_by_filing: dict[str, set[int]] = {}
    for filing, page in checks:
        pages_by_filing.setdefault(filing, set()).add(page)
    for name, (path, expected_hash) in PDFS.items():
        actual_hash = _sha256(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Source PDF hash changed for {name}: {actual_hash}")
        with pdfplumber.open(path) as pdf:
            if any(page > len(pdf.pages) for page in pages_by_filing[name]):
                raise RuntimeError(f"Source page out of range in {name}")
            pdf_pages[name] = {
                page_number: pdf.pages[page_number - 1].extract_text() or ""
                for page_number in sorted(pages_by_filing[name])
            }
    for (filing, page), needles in checks.items():
        text = _norm(pdf_pages[filing][page])
        for needle in needles:
            if _norm(needle) not in text:
                raise RuntimeError(f"Expected source text missing: {filing}#{page}: {needle}")

    original = [json.loads(line) for line in PACKET.read_text(encoding="utf-8").splitlines() if line.strip()]
    development = [row for row in original if row.get("partition") == "development"]
    if len(development) != 16:
        raise RuntimeError(f"Expected the established 16 development families, found {len(development)}")

    existing: dict[str, tuple[str, str, dict[str, int], str, str]] = {
        "FQA-3fbfaf549f970fd5d280": (
            "The FY2023 10-K says climate-related laws could increase costs and may adversely affect later-period results; it describes a possible risk, not a realized cost increase caused by climate change.",
            "ANSWERABLE_WITH_CONDITIONAL_RISK_CAVEAT", {"2023-10-K.pdf#12": 3},
            "Page 12 explicitly uses conditional language ('could'); do not report an observed causal effect.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-459504a0c8aab868b348": (
            "NVIDIA reported research and development expense of $5,268 million for FY2022 in its FY2023 10-K.",
            "ANSWERABLE", {"2023-10-K.pdf#43": 3},
            "Page 43 reports USD millions and columns FY2023/FY2022; candidate hint page 41 is a percentage-of-revenue table, not the requested dollar expense.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-841546b4d92198e1bb46": (
            "NVIDIA reported $19,218 million of marketable securities for FY2022 in the FY2023 10-K.",
            "ANSWERABLE", {"2023-10-K.pdf#44": 3, "2023-10-K.pdf#56": 3},
            "Both the liquidity discussion and balance sheet show the FY2022 amount; repeated facts are multiple sources, not duplicate records to delete.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-9ebad1d7a48a48c4c012": (
            "NVIDIA reported total current assets of $28,829 million as of January 30, 2022 in the FY2023 10-K.",
            "ANSWERABLE", {"2023-10-K.pdf#56": 3},
            "Page 56 is the consolidated balance sheet with date columns and the total-current-assets row.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-e4a8908ea7e94a8710a1": (
            "NVIDIA reported accounts receivable, net, of $4,650 million as of January 30, 2022 in the FY2023 10-K.",
            "ANSWERABLE", {"2023-10-K.pdf#56": 3},
            "Page 56 directly aligns the accounts-receivable row with the FY2022 column; candidate hint page 58 is a cash-flow statement and does not give this balance-sheet stock value.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-34545952401c5dd5a682": (
            "The FY2024 10-K reports total operating expenses of $11,329 million for FY2024; this is a reported metric, not a causal relation.",
            "ANSWERABLE", {"2024-10-K.pdf#36": 3, "2024-10-K.pdf#50": 3},
            "The FY2024 summary and consolidated income statement report the value.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-33764b55e0611be080cc": (
            "The FY2025 10-K reports revenue of $130,497 million for FY2025 versus $60,922 million for FY2024, an increase of $69,575 million (about 114.2%). The filing does not establish that a revenue risk caused this increase.",
            "ANSWERABLE_WITH_SCOPE_CAVEAT", {"2025-10-K.pdf#52": 3},
            "The income statement contains three years of values; the word 'risk' is not evidence of attribution.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-987035dd8ccaceaef0fc": (
            "The FY2025 filing describes shifting and expanding export-control restrictions over the preceding three years, including dated U.S. actions; it does not quantify a year-by-year causal change in NVIDIA revenue.",
            "ANSWERABLE_WITH_SCOPE_CAVEAT", {"2025-10-K.pdf#10": 3},
            "Page 10 describes policy chronology and impact on ability to serve customers; do not infer revenue amounts or causal effect beyond the filing.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-dd161d65dc8249591b85": (
            "The reviewed filing passages do not identify a risk factor that increased NVIDIA revenue in both FY2023 and FY2025. Risk disclosures describe possible adverse effects; no positive causal attribution is established.",
            "ANSWERABLE_AS_EVIDENCE_LIMIT", {"2025-10-K.pdf#14": 1, "2025-10-K.pdf#17": 1},
            "The cited pages are contextual risk disclosures only. The search was not an exhaustive page-by-page negative search, so the no-positive finding is not a completeness claim.", "SCORED_KNOWN_CONTEXT_PAGES_ONLY",
        ),
        "FQA-c2180a4fa48947a326c9": (
            "Page 3 states that known and unknown risks may cause actual results, performance, time frames, or achievements to differ materially from forward-looking statements. This is conditional risk language, not proof that a risk event occurred.",
            "ANSWERABLE_WITH_CONDITIONAL_RISK_CAVEAT", {"2025-10-K.pdf#3": 3},
            "The forward-looking-statements section uses 'may'; preserve its conditional scope.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-72398095458d46295ba0": (
            "The filing associates GeForce GPUs with NVIDIA's Graphics business, but says manufacturing is performed by foundries and subcontractors and that NVIDIA does not assemble, test, or package products. It does not support an unqualified claim that NVIDIA itself manufactures GeForce GPUs.",
            "ANSWERABLE_WITH_PREDICATE_CORRECTION", {"2025-10-K.pdf#5": 2, "2025-10-K.pdf#18": 3},
            "Product/business association is not equivalent to a manufacturer relation; page 18 specifies third-party manufacturing and assembly.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-a22d89953a5d9c3be464": (
            "The filing describes professional visualization as a market NVIDIA serves and says professional artists, architects, and designers use NVIDIA-partner products accelerated by its GPUs and software.",
            "ANSWERABLE_WITH_SCOPE_CAVEAT", {"2025-10-K.pdf#5": 3},
            "Page 5 supports market participation; it does not imply that every referenced product is manufactured by NVIDIA.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-5afddf4bb22cb702148e": (
            "The filing says NVIDIA delivers an end-to-end automated-driving solution under the DRIVE Hyperion brand and describes the platform; it also says products are manufactured/assembled by third parties. 'Produces' is stronger than the cited disclosure supports.",
            "ANSWERABLE_WITH_PREDICATE_CORRECTION", {"2025-10-K.pdf#7": 2, "2025-10-K.pdf#18": 3},
            "Page 7 supports the DRIVE Hyperion solution/platform association, not a claim of in-house physical manufacture.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-02e0949518b4d6c7875b": (
            "The FY2025 filing says pandemics may cause supply constraints and extended lead times, and separately says supplier/manufacturing risks could adversely affect gross margin. It does not state that COVID-19 caused an observed gross-margin change.",
            "ANSWERABLE_WITH_CONDITIONAL_RISK_CAVEAT", {"2025-10-K.pdf#17": 2, "2025-10-K.pdf#18": 3},
            "The pages describe hypothetical/ongoing risk mechanisms; COVID-19-specific realized causality is not established.", "SCORED_KNOWN_SUPPORT_PAGES",
        ),
        "FQA-3bcb4dd7d3f1f7a7ae8f": (
            "The reviewed passages describe possible pandemic/supply-chain disruptions and possible operating costs, but do not establish a COVID-19-to-capital-expenditure causal chain.",
            "ANSWERABLE_AS_EVIDENCE_LIMIT", {"2025-10-K.pdf#17": 1, "2025-10-K.pdf#22": 1},
            "Low-grade pages provide risk context only. No full-filing negative search was performed.", "SCORED_KNOWN_CONTEXT_PAGES_ONLY",
        ),
        "FQA-789e153b6944977a435d": (
            "The reviewed passages do not establish a COVID-19-to-litigation-risk causal chain through supply disruption. The filing's pandemic/supply disclosures are conditional and do not report that causal sequence.",
            "UNANSWERABLE_IN_REVIEWED_SCOPE", {},
            "No exhaustive whole-filing relevance judgment was performed; this item is excluded from retrieval quality denominators, not treated as a negative page label.", "NOT_SCORED_NO_POSITIVE_PAGE_JUDGMENT",
        ),
    }
    variants: dict[str, tuple[str, dict[str, int], str]] = {
        "FQAV-c282e2ba770b72ca": (
            "Research and development expense increased from $5,268 million in FY2022 to $7,339 million in FY2023, a $2,071 million increase; the filing reports a 39% increase.",
            {"2023-10-K.pdf#43": 3}, "The statement and operating-expense table directly align both years, amounts, and change.",
        ),
        "FQAV-fec24883393037be": (
            "Marketable securities decreased from $19,218 million in FY2022 to $9,907 million in FY2023, a decrease of $9,311 million.",
            {"2023-10-K.pdf#44": 3, "2023-10-K.pdf#56": 3}, "Both source pages report the dated balance-sheet values.",
        ),
        "FQAV-ef6bf1e8bc6a41c5": (
            "Total current assets decreased from $28,829 million in FY2022 to $23,073 million in FY2023, a decrease of $5,756 million.",
            {"2023-10-K.pdf#56": 3}, "The consolidated balance sheet directly reports both date columns.",
        ),
        "FQAV-0ea7b124454381d8": (
            "Accounts receivable, net, decreased from $4,650 million in FY2022 to $3,827 million in FY2023, a decrease of $823 million.",
            {"2023-10-K.pdf#56": 3}, "The consolidated balance sheet directly reports both date columns.",
        ),
        "FQAV-ea958195867b7fb5": (
            "Total operating expenses increased from $7,434 million in FY2022 to $11,132 million in FY2023, a $3,698 million increase; the FY2023 filing reports 50% growth.",
            {"2023-10-K.pdf#43": 3, "2023-10-K.pdf#54": 3}, "The source statement and operating-expense table report both periods and the change.",
        ),
    }
    output = []
    for row in development:
        label = existing.get(row.get("item_id"))
        if label is None:
            raise RuntimeError(f"No PDF source review registered for {row.get('item_id')}")
        answer, answerability, grades, notes, retrieval_status = label
        filing = str(row["candidate_filing"])
        if _sha256(PDFS[filing][0]) != row.get("source_pdf_sha256"):
            raise RuntimeError(f"Candidate source hash mismatch for {row['item_id']}")
        labelled = _add_label(
            row, answer=answer, answerability=answerability, grades=grades,
            notes=notes, retrieval_status=retrieval_status,
        )
        for variant in labelled.get("candidate_variants", []):
            if variant.get("item_id") in variants:
                variant_answer, variant_grades, variant_notes = variants[variant["item_id"]]
                variant.update(_add_label(
                    variant, answer=variant_answer, answerability="ANSWERABLE",
                    grades=variant_grades, notes=variant_notes,
                ))
            elif variant.get("label_status") != "AI_SOURCE_REVIEW_NOT_HUMAN":
                raise RuntimeError(f"No independent PDF label is registered for variant {variant.get('item_id')}")
        output.append(labelled)

    output.extend([
        _new_family(
            key="financial:revenue:FY2024:2024-disclosure",
            question="What revenue did NVIDIA report for fiscal year 2024 in its 2024 10-K?",
            question_type="single_year_fact", filing="2024-10-K.pdf", pages=[50],
            answer="NVIDIA reported revenue of $60,922 million (USD) for FY2024 in the FY2024 10-K.",
            grades={"2024-10-K.pdf#50": 3},
            notes="Page 50 consolidated income statement: explicit USD millions heading and FY2024 date column.",
        ),
        _new_family(
            key="financial:total_current_assets:FY2024:2024-disclosure",
            question="What were NVIDIA's total current assets at fiscal year-end 2024, as reported in the 2024 10-K?",
            question_type="single_year_fact", filing="2024-10-K.pdf", pages=[52],
            answer="NVIDIA reported total current assets of $44,345 million as of January 28, 2024.",
            grades={"2024-10-K.pdf#52": 3},
            notes="Page 52 consolidated balance sheet gives the date, row, and USD millions heading.",
        ),
        _new_family(
            key="financial:revenue:FY2025:2025-disclosure",
            question="What revenue did NVIDIA report for fiscal year 2025 in its 2025 10-K?",
            question_type="single_year_fact", filing="2025-10-K.pdf", pages=[52],
            answer="NVIDIA reported revenue of $130,497 million (USD) for FY2025 in the FY2025 10-K.",
            grades={"2025-10-K.pdf#52": 3},
            notes="Page 52 consolidated income statement explicitly states '(In millions)' and lists FY2025.",
            variants=[{
                "item_id": "FQAV-DEV-USD-BILLIONS-2025-REVENUE",
                "question": "Convert NVIDIA FY2025 revenue of $130,497 million from its 2025 10-K into USD billions.",
                "question_type": "unit_conversion_or_calculation",
                "candidate_pages": [52],
                "reference_answer": "$130.497 billion USD (130,497 million ÷ 1,000).",
                "gold_pages": [52],
                "gold_page_grades": {"2025-10-K.pdf#52": 3},
                "relevance_grades": {"2025-10-K.pdf#52": 3},
                "answerability_label": "ANSWERABLE",
                "retrieval_scoring_status": "SCORED_KNOWN_SUPPORT_PAGES",
                "review_notes": "Source page states USD millions; unit conversion is deterministic and preserves USD.",
            }],
        ),
        _new_family(
            key="financial:revenue:FY2023-vs-FY2022:2023-disclosure",
            question="Using NVIDIA's 2023 10-K, what was the change in revenue from FY2022 to FY2023 in USD millions and percent?",
            question_type="cross_year_comparison", filing="2023-10-K.pdf", pages=[54],
            answer="Revenue increased by $60 million, from $26,914 million in FY2022 to $26,974 million in FY2023; the increase is 60/26,914 = 0.2229%, or 0.2% rounded to one decimal place.",
            grades={"2023-10-K.pdf#54": 3},
            notes="Page 54 consolidated income statement reports both values in USD millions; percentage is independently calculated from those source values.",
        ),
    ])

    if len(output) != 20 or len({row["family_id"] for row in output}) != 20:
        raise RuntimeError("Expected 20 distinct development semantic families")
    hashes = {name: value[1] for name, value in PDFS.items()}
    return output, {
        "schema": "financial-qa-source-reviewed-development/v1",
        "label_tier": "AI_PDF_SOURCE_DIAGNOSIS_NOT_HUMAN_REVIEW",
        "gold_status": "NOT_GOLD",
        "human_primary_review": "NOT_RUN",
        "human_secondary_review": "NOT_RUN",
        "adjudication": "NOT_RUN",
        "source_pdf_sha256": hashes,
        "source_page_checks": len(checks),
        "label_generator_sha256": _sha256(Path(__file__).resolve()),
        "input_packet_sha256": _sha256(PACKET),
        "family_count": len(output),
        "question_count_including_variants": sum(1 + len(row.get("candidate_variants") or []) for row in output),
        "known_positive_page_families": sum(bool(row["gold_page_grades"]) for row in output),
        "no_positive_page_judgment_families": sum(not row["gold_page_grades"] for row in output),
        "limitation": "Only explicitly listed source pages are judged relevant. Other pages/hits are unjudged, not irrelevant; this is a development diagnostic, not an independent test or human Gold set.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    output_path = Path(args.output).resolve()
    manifest_path = Path(args.manifest).resolve()
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite a label dataset or its manifest")
    rows, manifest = prepare()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    manifest["dataset_sha256"] = _sha256(output_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": str(output_path), "manifest": str(manifest_path), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
