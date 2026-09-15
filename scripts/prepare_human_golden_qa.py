"""Prepare a blank human Golden QA review set from generated candidates.

The generated candidate set is copied into ``candidate_*`` reference fields;
human-owned fields remain blank until a reviewer fills them.  This script is
deliberately local-only: it does not call an LLM, Neo4j, or any external
service, and it never overwrites the candidate input file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
CURRENT_CANDIDATE = ROOT / "data" / "evaluation" / "golden_qa_v3_current.jsonl"
LEGACY_CANDIDATE = ROOT / "data" / "evaluation" / "golden_qa_v2.jsonl"
DEFAULT_INPUT = CURRENT_CANDIDATE if CURRENT_CANDIDATE.exists() else LEGACY_CANDIDATE
DEFAULT_OUTPUT = ROOT / "evaluation" / "golden_qa_human_v2.jsonl"

HUMAN_FIELDS = (
    "reference_answer",
    "gold_evidence_ids",
    "gold_pages",
    "relevant_evidence_grades",
    "answerable",
    "requires_abstention",
    "reviewer",
    "review_notes",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            rows.append(value)
    return rows


def _blank_human_row(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return one review row without copying candidate labels into gold fields."""

    candidate_fields = {f"candidate_{key}": value for key, value in candidate.items()}
    return {
        "id": candidate.get("id"),
        "question": candidate.get("question", ""),
        **candidate_fields,
        "reference_answer": "",
        "gold_evidence_ids": [],
        "gold_pages": [],
        "relevant_evidence_grades": {},
        "answerable": None,
        "requires_abstention": None,
        "reviewer": "",
        "review_notes": "",
        "review_status": "HUMAN_REVIEW_PENDING",
    }


def prepare_human_golden_qa(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    overwrite: bool = False,
) -> int:
    """Write a blank human-review JSONL and return its row count.

    ``overwrite`` applies only to the independent output file.  The candidate
    input is always read-only and cannot be selected as the output path.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("The human-review output must not overwrite the candidate input")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing human-review file: {output_path}; use --force to replace it"
        )

    candidates = _load_jsonl(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for candidate in candidates:
            handle.write(json.dumps(_blank_human_row(candidate), ensure_ascii=False) + "\n")
    return len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a blank human Golden QA review set from generated candidates"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing independent human-review output file",
    )
    args = parser.parse_args()
    count = prepare_human_golden_qa(args.input, args.output, overwrite=args.force)
    print(f"Wrote {count} HUMAN_REVIEW_PENDING rows to {args.output}")


if __name__ == "__main__":
    main()
