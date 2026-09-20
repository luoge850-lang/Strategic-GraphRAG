"""Export a non-Gold table-cell annotation queue from the current graph.

The output is intentionally labelled as a candidate queue. It contains the
system prediction and an empty human Gold section; it must not be used as
recall ground truth until an independent reviewer fills the fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strategic_graphrag.api.server import get_schema_manager


def export(limit: int) -> List[Dict[str, Any]]:
    rows = get_schema_manager()._read(
        """
        MATCH (observation:FinancialObservation)-[:SUPPORTED_BY_CLAIM]->(claim:EvidenceClaim)
        RETURN observation.id AS id,
               observation.company_id AS company_id,
               observation.metric_id AS metric_id,
               observation.fiscal_year AS fiscal_year,
               observation.value AS value,
               observation.raw_value AS raw_value,
               observation.unit AS unit,
               observation.source_filing AS source_filing,
               observation.page AS page,
               observation.row_label AS row_label,
               observation.column_label AS column_label,
               observation.table_name AS table_name,
               observation.statement_type AS statement_type,
               claim.id AS claim_id,
               claim.text AS evidence_sentence,
               claim.table_context AS table_context
        ORDER BY observation.source_filing, observation.page, observation.metric_id,
                 observation.fiscal_year, observation.id
        LIMIT $limit
        """,
        limit=max(1, min(int(limit), 500)),
    )
    queue = []
    for row in rows:
        prediction = dict(row)
        queue_identity = "|".join([
            str(row.get("source_filing") or ""),
            str(row.get("page") or ""),
            str(row.get("metric_id") or ""),
            str(row.get("fiscal_year") or ""),
            str(row.get("row_label") or ""),
            str(row.get("column_label") or ""),
            str(row.get("claim_id") or row.get("id") or ""),
        ])
        queue_id = "TABLE-Q-" + hashlib.sha256(
            queue_identity.encode("utf-8")
        ).hexdigest()[:20]
        prediction.update({
            "queue_id": queue_id,
            "candidate_id": queue_id,
            "candidate_schema_version": "table-candidate/v1",
            "review_status": "UNLABELED_CANDIDATE",
            "reviewer": "",
            "review_notes": "",
            "gold": {
                "company_id": "",
                "fiscal_year": "",
                "metric_id": "",
                "value": "",
                "unit": "",
                "source_filing": "",
                "page": "",
                "row_label": "",
                "column_label": "",
                "table_name": "",
                "evidence_text": "",
                "cell_supported": "",
            },
            "annotation_instruction": (
                "Compare the displayed row/header/context with the original PDF. "
                "Fill every gold field, preserve accounting parentheses as negative, "
                "and mark cell_supported=false when the row or value is not actually "
                "supported by the cited page."
            ),
        })
        queue.append(prediction)
    return queue


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evaluation" / "annotation" / "table_quality_candidate_2026-09-19.jsonl",
    )
    args = parser.parse_args()
    rows = export(args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "rows": len(rows),
        "status": "UNLABELED_CANDIDATE_NOT_GOLD",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
