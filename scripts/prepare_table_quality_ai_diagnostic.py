"""Create a reproducible, explicitly non-human visual diagnostic for table candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_PDF = "2023-10-K.pdf"
EXPECTED_PDF_SHA256 = "89981bbfcd91e20498c1060d7efb39022ac8f85e639f2841091695885aa9f8a8"
RENDERED_PAGES = {37, 41, 43, 44, 54, 55, 56, 58, 65}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_key(row: dict[str, Any]) -> tuple[str, int, str, str]:
    try:
        value = f"{float(row['value']):.12g}"
        year = int(row["fiscal_year"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"candidate row lacks a normalized year/value: {row.get('queue_id')}") from exc
    return (str(row.get("metric_id") or ""), year, value, str(row.get("unit") or ""))


def _diagnose(row: dict[str, Any], duplicate_count: int) -> dict[str, Any]:
    page = int(row["page"])
    if page == 65:
        period_status = "FAIL_PERIOD_NOT_STATED_IN_TABLE"
        metric_status = "FAIL_MELLANOX_PURCHASE_PRICE_ALLOCATION_NOT_NVIDIA_FY_METRIC"
        joint_status = "FLAG_PERIOD_AND_METRIC_MISATTRIBUTION"
        rationale = (
            "物理页 65 的 115 和 699 位于 Mellanox 收购购买价分摊表；该表没有 FY2023 列，"
            "不能标成 NVIDIA FY2023 合并口径的现金及现金等价物或有价证券。"
        )
    elif page == 41:
        period_status = "VISUAL_MATCH"
        metric_status = "NEEDS_PERCENT_OF_REVENUE_BASIS_IN_TYPED_METRIC"
        joint_status = "FLAG_PERCENTAGE_BASIS_CONTEXT"
        rationale = (
            "物理页 41 明示本表将损益项目表示为 revenue 的百分比；候选 unit=percent，"
            "但结构化事实应显式保存分母/比率口径，不能与同名美元金额混用。"
        )
    elif page == 58 and str(row.get("metric_id")) in {"accounts_receivable", "inventories"}:
        period_status = "VISUAL_MATCH"
        metric_status = "NEEDS_CASH_FLOW_ADJUSTMENT_QUALIFIER"
        joint_status = "FLAG_STATEMENT_TYPE_CONTEXT"
        rationale = (
            "物理页 58 的应收账款与存货位于经营活动现金流调节项；括号数为负值。"
            "数值、符号、年份和 USD millions 与单元格相符，但应把指标类型标成现金流变动，"
            "不能与资产负债表余额混为一谈。"
        )
    else:
        period_status = "VISUAL_MATCH"
        metric_status = "VISUAL_ROW_LABEL_OR_NORMALIZED_ALIAS_MATCH"
        joint_status = "AI_VISUAL_NO_OBVIOUS_MISMATCH"
        rationale = "渲染原文中的表头、行名、候选数值与候选年度列视觉对应；仍需人工确认规范化指标。"

    return {
        "queue_id": row["queue_id"],
        "candidate_id": row["id"],
        "review_tier": "AI_PDF_VISUAL_DIAGNOSTIC_NOT_HUMAN_REVIEW",
        "source_filing": row["source_filing"],
        "source_pdf_sha256": EXPECTED_PDF_SHA256,
        "physical_pdf_page": page,
        "rendered_page_visually_inspected": page in RENDERED_PAGES,
        "candidate_metric_id": row["metric_id"],
        "candidate_fiscal_year": int(row["fiscal_year"]),
        "candidate_value": row["value"],
        "candidate_raw_value": row["raw_value"],
        "candidate_unit": row["unit"],
        "candidate_row_label": row["row_label"],
        "candidate_column_label": row["column_label"],
        "candidate_evidence_sentence": row["evidence_sentence"],
        "numeric_value_cell_visual_status": "VISUAL_MATCH",
        "sign_visual_status": "VISUAL_MATCH_ACCOUNTING_PARENTHESES_PRESERVED",
        "unit_scale_visual_status": "VISUAL_MATCH",
        "source_page_visual_status": "VISUAL_MATCH",
        "fiscal_year_column_visual_status": period_status,
        "metric_semantics_visual_status": metric_status,
        "ai_joint_diagnostic_status": joint_status,
        "exact_candidate_fact_duplicate_group_size": duplicate_count,
        "rationale_zh": rationale,
        "primary_human_reviewer": "",
        "secondary_human_reviewer": "",
        "adjudication_status": "PENDING",
    }


def prepare(candidate_path: Path, pdf_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_pdf_hash = _sha256(pdf_path)
    if actual_pdf_hash != EXPECTED_PDF_SHA256:
        raise ValueError(f"source PDF hash mismatch: {actual_pdf_hash}")
    rows = _read_jsonl(candidate_path)
    if len(rows) != 60:
        raise ValueError(f"expected 60 existing table candidates; found {len(rows)}")
    pages = {int(row["page"]) for row in rows}
    if pages != RENDERED_PAGES:
        raise ValueError(f"rendered-page coverage mismatch: {sorted(pages)}")
    if any(row.get("source_filing") != EXPECTED_PDF for row in rows):
        raise ValueError("this visual diagnostic is scoped to the audited 2023 filing only")

    frequencies = Counter(_candidate_key(row) for row in rows)
    output = [
        _diagnose(row, frequencies[_candidate_key(row)])
        for row in sorted(rows, key=lambda item: (int(item["page"]), str(item["queue_id"])))
    ]
    status_counts = Counter(row["ai_joint_diagnostic_status"] for row in output)
    report = {
        "schema": "table-quality-ai-visual-diagnostic/v1",
        "review_tier": "AI_PDF_VISUAL_DIAGNOSTIC_NOT_HUMAN_REVIEW",
        "source_candidate_file": candidate_path.relative_to(ROOT).as_posix(),
        "source_candidate_sha256": _sha256(candidate_path),
        "source_filing": EXPECTED_PDF,
        "source_pdf_sha256": actual_pdf_hash,
        "candidate_count": len(rows),
        "unique_candidate_metric_year_value_unit_keys": len(frequencies),
        "duplicate_candidate_key_groups": sum(count > 1 for count in frequencies.values()),
        "duplicate_excess_rows": sum(count - 1 for count in frequencies.values() if count > 1),
        "visually_inspected_physical_pages": sorted(pages),
        "numeric_value_cell_visual_match": len(output),
        "sign_visual_match": len(output),
        "unit_scale_visual_match": len(output),
        "source_page_visual_match": len(output),
        "fiscal_year_column_visual_match": sum(row["fiscal_year_column_visual_status"] == "VISUAL_MATCH" for row in output),
        "joint_ai_diagnostic_status_counts": dict(sorted(status_counts.items())),
        "not_an_accuracy_estimate": True,
        "human_review_status": "NOT_RUN",
        "limitations": [
            "This is a one-agent visual diagnostic, not a human annotation or independent accuracy estimate.",
            "The sample consists of all 60 existing candidates from one filing and is not a recall sample.",
            "Repeated metric/year/value/unit candidates include duplicate evidence pages and are not independent facts.",
        ],
    }
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=ROOT / "evaluation/annotation/table_quality_candidate_2026-09-19.jsonl",
    )
    parser.add_argument("--pdf", type=Path, default=ROOT / "data/pdfs_other/2023-10-K.pdf")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/evaluation/table_quality_ai_visual_diagnostic_2026-09-24_v2.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=ROOT / "reports/evaluation/table_quality_ai_visual_diagnostic_summary_2026-09-24_v2.json",
    )
    args = parser.parse_args()
    output, summary = args.output.resolve(), args.summary.resolve()
    if output.exists() or summary.exists():
        raise FileExistsError("Refusing to overwrite an existing visual diagnostic artifact")
    rows, report = prepare(args.candidates.resolve(), args.pdf.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(output), "summary": str(summary)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
