"""Audit the completed extraction annotation sample.

This report treats ``uncertain`` as incorrect, as required by the current
annotation protocol.  The sample contains extracted claims, not a complete
gold relation inventory, so the reported relation score is a precision-like
sample estimate.  Recall remains explicitly unavailable until each sampled
evidence unit has a complete gold relation set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE = ROOT / "evaluation" / "annotation" / "extraction_sample_v1.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "extraction_annotation_audit_v1.json"

LABEL_FIELDS = (
    "source_entity_correct",
    "target_entity_correct",
    "relation_correct",
    "evidence_supports_relation",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _label_value(row: dict[str, Any], field: str) -> Any:
    return (row.get("labels") or {}).get(field)


def _is_correct(value: Any) -> bool:
    # ``uncertain`` is deliberately counted as incorrect in this protocol.
    return value is True


def _normalized_evidence(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def _triple(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_id") or ""),
        str(row.get("relation_type") or ""),
        str(row.get("target_id") or ""),
    )


def _triple_display(triple: tuple[str, str, str]) -> str:
    return f"{triple[0]} --[{triple[1]}]--> {triple[2]}"


def _label_vector(row: dict[str, Any]) -> tuple[bool, bool, bool, bool]:
    return tuple(_is_correct(_label_value(row, field)) for field in LABEL_FIELDS)


def analyze(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    status_counts = Counter(str(row.get("annotation_status") or "MISSING") for row in rows)
    label_counts = {
        field: Counter(
            "true" if _label_value(row, field) is True
            else "uncertain" if _label_value(row, field) == "uncertain"
            else "false_or_missing"
            for row in rows
        )
        for field in LABEL_FIELDS
    }

    correct_counts = {
        field: sum(_is_correct(_label_value(row, field)) for row in rows)
        for field in LABEL_FIELDS
    }
    exact_triple_count = sum(
        all(_is_correct(_label_value(row, field)) for field in LABEL_FIELDS[:3])
        for row in rows
    )
    evidence_supported_count = sum(
        all(_is_correct(_label_value(row, field)) for field in LABEL_FIELDS)
        for row in rows
    )

    groups: defaultdict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[_triple(row)].append(row)
    duplicate_groups = {triple: group for triple, group in groups.items() if len(group) > 1}

    duplicate_pair_count = 0
    duplicate_agreeing_pair_count = 0
    mixed_label_groups = 0
    duplicate_group_rows = []
    for triple, group in sorted(duplicate_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        vectors = [_label_vector(row) for row in group]
        vector_set = set(vectors)
        mixed = len(vector_set) > 1
        mixed_label_groups += int(mixed)
        pairs = list(combinations(vectors, 2))
        duplicate_pair_count += len(pairs)
        duplicate_agreeing_pair_count += sum(left == right for left, right in pairs)
        duplicate_group_rows.append(
            {
                "triple": _triple_display(triple),
                "source_id": triple[0],
                "relation_type": triple[1],
                "target_id": triple[2],
                "count": len(group),
                "doc_ids": sorted({str(row.get("doc_id") or "") for row in group}),
                "pages": sorted({int(row["page"]) for row in group if str(row.get("page", "")).isdigit()}),
                "different_evidence_count": len({_normalized_evidence(row.get("evidence")) for row in group}),
                "mixed_normalized_labels": mixed,
                "normalized_label_vectors": [list(vector) for vector in vectors],
                "claim_ids": [row.get("claim_id") for row in group],
            }
        )

    exact_row_keys = Counter(
        (
            row.get("doc_id"),
            row.get("page"),
            *_triple(row),
            _normalized_evidence(row.get("evidence")),
        )
        for row in rows
    )
    exact_duplicate_excess = sum(count - 1 for count in exact_row_keys.values() if count > 1)

    def rate(count: int) -> float | None:
        return round(count / total, 4) if total else None

    return {
        "schema": "strategic-graphrag-extraction-annotation-audit/v1",
        "annotation_protocol": {
            "uncertain_is_incorrect": True,
            "sample_unit": "extracted_claim",
            "recall_status": "NOT_MEASURED",
            "recall_reason": "The sample has predicted claims but no complete gold relation inventory per evidence unit.",
        },
        "sample": {
            "rows": total,
            "status_counts": dict(status_counts),
            "by_doc_id": dict(Counter(str(row.get("doc_id") or "MISSING") for row in rows)),
            "by_extraction_method": dict(Counter(str(row.get("extraction_method") or "MISSING") for row in rows)),
        },
        "label_counts": {field: dict(counts) for field, counts in label_counts.items()},
        "precision_like_sample_estimates": {
            "source_entity_correct_rate": rate(correct_counts["source_entity_correct"]),
            "target_entity_correct_rate": rate(correct_counts["target_entity_correct"]),
            "relation_correct_rate": rate(correct_counts["relation_correct"]),
            "evidence_support_rate": rate(correct_counts["evidence_supports_relation"]),
            "exact_triple_correct_rate": rate(exact_triple_count),
            "evidence_supported_exact_triple_rate": rate(evidence_supported_count),
            "f1": None,
            "f1_reason": "F1 is not defined from this positive-prediction-only sample without a complete gold relation set and recall.",
        },
        "duplicate_audit": {
            "logical_triple_groups": len(groups),
            "duplicate_logical_triple_groups": len(duplicate_groups),
            "rows_in_duplicate_groups": sum(len(group) for group in duplicate_groups.values()),
            "duplicate_excess_rows": sum(len(group) - 1 for group in duplicate_groups.values()),
            "duplicate_groups_with_different_evidence": sum(
                len({_normalized_evidence(row.get("evidence")) for row in group}) > 1
                for group in duplicate_groups.values()
            ),
            "duplicate_groups_with_same_evidence": sum(
                len({_normalized_evidence(row.get("evidence")) for row in group}) == 1
                for group in duplicate_groups.values()
            ),
            "exact_same_doc_page_triple_evidence_excess_rows": exact_duplicate_excess,
            "mixed_label_groups": mixed_label_groups,
            "duplicate_pair_label_agreement": (
                round(duplicate_agreeing_pair_count / duplicate_pair_count, 4)
                if duplicate_pair_count else None
            ),
            "agreement_note": "This is duplicate-triple label agreement, not inter-annotator agreement or Cohen/Fleiss kappa.",
            "groups": duplicate_group_rows,
        },
        "annotation_metadata": {
            "rows_without_annotator": sum(not row.get("annotator") for row in rows),
            "rows_with_notes": sum(bool(row.get("notes")) for row in rows),
            "rows_with_missing_gold_relations": sum(bool((row.get("labels") or {}).get("missing_gold_relations")) for row in rows),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit extraction annotation labels without changing them")
    parser.add_argument("--sample", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = analyze(load_jsonl(args.sample))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
