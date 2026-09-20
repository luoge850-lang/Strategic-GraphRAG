"""Evaluate structured financial-table extraction against an independent Gold file.

This evaluator deliberately separates row retrieval from field correctness. A
record that finds the right metric but the wrong year, unit, sign, or source
page is not a correct table cell. The input files are JSONL and are never
treated as human Gold unless the Gold file declares ``review_status`` as
``HUMAN_REVIEWED`` for every row. The table annotation queue format is also
accepted directly: its nested ``gold`` object is flattened for evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


ROOT = Path(__file__).resolve().parent.parent
FIELDS = (
    "company_id", "fiscal_year", "metric_id", "value", "unit", "source_filing",
    "page", "row_label", "column_label", "table_name",
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(value)
    return rows


def _text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _number(value: Any) -> Optional[Decimal]:
    if value in (None, ""):
        return None
    raw = str(value).strip().replace(",", "").replace("$", "").replace("%", "")
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("() ")
    try:
        number = Decimal(raw)
    except (InvalidOperation, ValueError):
        return None
    return -abs(number) if negative else number


def _equal(field: str, expected: Any, actual: Any) -> bool:
    if field in {"value"}:
        left, right = _number(expected), _number(actual)
        return left is not None and right is not None and left == right
    if field in {"fiscal_year", "page"}:
        try:
            return int(expected) == int(actual)
        except (TypeError, ValueError):
            return False
    return _text(expected) == _text(actual)


def _row_id(row: Dict[str, Any]) -> str:
    return str(row.get("id") or row.get("claim_id") or "").strip()


def _evidence_text(row: Dict[str, Any]) -> str:
    return str(
        row.get("evidence_text")
        or row.get("evidence_sentence")
        or row.get("row_evidence")
        or ""
    ).strip()


def _f1(precision: float, recall: float) -> float:
    return round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0


def normalize_gold_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten the Demo queue's nested Gold fields without losing review metadata."""
    normalized: List[Dict[str, Any]] = []
    for row in rows:
        nested = row.get("gold")
        if not isinstance(nested, dict):
            normalized.append(dict(row))
            continue
        gold = dict(nested)
        gold["id"] = row.get("id") or row.get("queue_id") or row.get("claim_id")
        gold["review_status"] = row.get("review_status")
        gold["reviewer"] = row.get("reviewer")
        gold["review_notes"] = row.get("review_notes")
        normalized.append(gold)
    return normalized


def _field_metrics(gold: List[Dict[str, Any]], predicted: Dict[str, Dict[str, Any]], field: str) -> Dict[str, Any]:
    expected_rows = [_row_id(row) for row in gold if _row_id(row)]
    present = [row_id for row_id in expected_rows if row_id in predicted]
    correct = sum(_equal(field, next(row for row in gold if _row_id(row) == row_id).get(field), predicted[row_id].get(field)) for row_id in present)
    denominator = len(expected_rows)
    recall = correct / denominator if denominator else 0.0
    precision = correct / len(predicted) if predicted else 0.0
    return {
        "correct": correct,
        "gold_rows": denominator,
        "predicted_rows": len(predicted),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": _f1(precision, recall),
    }


def evaluate(gold: Iterable[Dict[str, Any]], predicted: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    gold_rows = list(gold)
    pred_rows = list(predicted)
    has_support_labels = any("cell_supported" in row for row in gold_rows)
    supported_gold_rows = [
        row for row in gold_rows
        if not has_support_labels or row.get("cell_supported") is True
    ]
    gold_by_id = {_row_id(row): row for row in supported_gold_rows if _row_id(row)}
    pred_by_id = {_row_id(row): row for row in pred_rows if _row_id(row)}
    matched_ids = sorted(set(gold_by_id) & set(pred_by_id))
    row_precision = len(matched_ids) / len(pred_by_id) if pred_by_id else 0.0
    row_recall = len(matched_ids) / len(gold_by_id) if gold_by_id else 0.0

    field_results = {
        field: _field_metrics(supported_gold_rows, pred_by_id, field)
        for field in FIELDS
    }
    complete_cell = sum(
        all(_equal(field, gold_by_id[row_id].get(field), pred_by_id[row_id].get(field))
            for field in ("company_id", "fiscal_year", "metric_id", "value", "unit", "source_filing", "page"))
        for row_id in matched_ids
    )
    evidence_aligned = sum(
        bool(_text(gold_by_id[row_id].get("evidence_text")) and
             _text(gold_by_id[row_id].get("evidence_text")) in _text(_evidence_text(pred_by_id[row_id])))
        for row_id in matched_ids
    )
    reviewer_statuses = Counter(str(row.get("review_status") or "UNSPECIFIED") for row in gold_rows)
    return {
        "schema": "strategic-graphrag-table-quality/v1",
        "gold_rows": len(gold_rows),
        "predicted_rows": len(pred_rows),
        "matched_row_ids": len(matched_ids),
        "row_retrieval": {
            "precision": round(row_precision, 4),
            "recall": round(row_recall, 4),
            "f1": _f1(row_precision, row_recall),
        },
        "table_cell_accuracy": round(complete_cell / len(supported_gold_rows), 4) if supported_gold_rows else 0.0,
        "numeric_exact_match": field_results["value"],
        "unit_accuracy": field_results["unit"],
        "evidence_alignment": {
            "aligned": evidence_aligned,
            "matched_rows": len(matched_ids),
            "rate": round(evidence_aligned / len(matched_ids), 4) if matched_ids else 0.0,
        },
        "field_metrics": field_results,
        "gold_review_status_counts": dict(reviewer_statuses),
        "gold_is_independent_human": bool(gold_rows) and reviewer_statuses == Counter({"HUMAN_REVIEWED": len(gold_rows)}),
        "candidate_gold_labels": {
            "total_rows": len(gold_rows),
            "supported_rows_evaluated": len(supported_gold_rows),
            "unsupported_rows_excluded_from_gold_set": sum(
                row.get("cell_supported") is False for row in gold_rows
            ),
            "uncertain_rows": sum(
                row.get("cell_supported") == "uncertain" for row in gold_rows
            ),
        },
        "limitations": [
            "Metrics are only publication-grade when Gold rows are independently reviewed.",
            "A candidate extraction file is not a Gold file and cannot establish recall.",
            "Numeric equality is exact after sign and decimal normalization; unit scale equivalence must be explicitly annotated.",
            "When cell_supported labels are present, row metrics are candidate-conditioned and do not measure recall over unobserved table cells.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predicted", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_gold = load_jsonl(args.gold)
    queued_gold = [
        row for row in raw_gold
        if isinstance(row.get("gold"), dict)
        and row.get("review_status") != "HUMAN_REVIEWED"
    ]
    if queued_gold:
        raise SystemExit(
            f"Gold queue contains {len(queued_gold)} rows not marked HUMAN_REVIEWED; "
            "finish or exclude them before evaluation."
        )
    report = evaluate(normalize_gold_rows(raw_gold), load_jsonl(args.predicted))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
