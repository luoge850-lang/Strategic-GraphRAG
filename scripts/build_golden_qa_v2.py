"""Build a reviewable, evidence-linked QA set for the active single filing.

This is intentionally a draft builder. It never writes to Neo4j and does not
call an LLM. Every item must be human-reviewed before it is treated as a final
golden test case.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent


def _display(record: dict, key: str) -> str:
    return str(record.get(f"{key}_name") or record.get(key) or "UNKNOWN")


def build_dataset(doc_id: str, filename: str, output: Path, single_count: int = 25, multi_count: int = 10) -> int:
    load_dotenv(ROOT / ".env")
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    with driver.session() as session:
        single_rows = [
            record.data()
            for record in session.run(
                """
                MATCH (claim:EvidenceClaim {doc_id:$doc_id})-[:ABOUT_SOURCE]->(src)
                MATCH (claim)-[:ABOUT_TARGET]->(tgt)
                WHERE claim.verification_status = 'VERBATIM'
                RETURN claim.id AS claim_id, claim.text AS claim_text,
                       claim.page AS page, claim.year AS year,
                       src.id AS source, coalesce(src.name, src.id) AS source_name,
                       tgt.id AS target, coalesce(tgt.name, tgt.id) AS target_name,
                       claim.relation_type AS relation
                ORDER BY claim.page, claim.id
                LIMIT $limit
                """,
                doc_id=doc_id,
                limit=single_count,
            )
        ]
        multi_rows = [
            record.data()
            for record in session.run(
                """
                MATCH (src)-[r1]->(mid)-[r2]->(tgt)
                WHERE coalesce(r1.source_filing, r1.filing, '') = $filename
                  AND coalesce(r2.source_filing, r2.filing, '') = $filename
                  AND r1.evidence_id IS NOT NULL AND r2.evidence_id IS NOT NULL
                  AND EXISTS {
                      MATCH (c1:EvidenceClaim {id:r1.evidence_id})
                      WHERE c1.verification_status = 'VERBATIM'
                  }
                  AND EXISTS {
                      MATCH (c2:EvidenceClaim {id:r2.evidence_id})
                      WHERE c2.verification_status = 'VERBATIM'
                  }
                MATCH (c1:EvidenceClaim {id:r1.evidence_id})
                MATCH (c2:EvidenceClaim {id:r2.evidence_id})
                RETURN src.id AS source, coalesce(src.name, src.id) AS source_name,
                       mid.id AS middle, coalesce(mid.name, mid.id) AS middle_name,
                       tgt.id AS target, coalesce(tgt.name, tgt.id) AS target_name,
                       type(r1) AS relation1, type(r2) AS relation2,
                       c1.id AS claim1_id, c1.text AS claim1_text,
                       c1.page AS page1, c1.year AS year1,
                       c2.id AS claim2_id, c2.text AS claim2_text,
                       c2.page AS page2, c2.year AS year2
                ORDER BY page1, page2, source, middle, target
                LIMIT $limit
                """,
                filename=filename,
                limit=multi_count,
            )
        ]
    driver.close()

    items = []
    for index, row in enumerate(single_rows, start=1):
        source = _display(row, "source")
        target = _display(row, "target")
        items.append(
            {
                "id": f"GQ-{index:03d}",
                "question_type": "single_hop",
                "difficulty": "basic",
                "question": f"What evidence-backed {row.get('relation', 'relationship')} relationship connects {source} and {target} in the NVIDIA 2025 10-K?",
                "expected_answer": row.get("claim_text", ""),
                "atomic_facts": [row.get("claim_text", "")],
                "answerable": True,
                "source_filing": filename,
                "evidence_claim_ids": [row["claim_id"]],
                "supporting_triples": [
                    {
                        "source": row["source"],
                        "relation": row["relation"],
                        "target": row["target"],
                    }
                ],
                "pages": [row["page"]],
                "years": [row["year"]],
                "review_status": "AUTO_GENERATED_REGRESSION_CANDIDATE",
            }
        )

    offset = len(items)
    for index, row in enumerate(multi_rows, start=1):
        source = _display(row, "source")
        middle = _display(row, "middle")
        target = _display(row, "target")
        items.append(
            {
                "id": f"GQ-{offset + index:03d}",
                "question_type": "multi_hop",
                "difficulty": "multi_hop",
                "question": f"What evidence-backed two-step relationship connects {source} to {target} through {middle} in the NVIDIA 2025 10-K?",
                "expected_answer": f"{row.get('claim1_text', '')} {row.get('claim2_text', '')}".strip(),
                "atomic_facts": [row.get("claim1_text", ""), row.get("claim2_text", "")],
                "answerable": True,
                "source_filing": filename,
                "evidence_claim_ids": [row["claim1_id"], row["claim2_id"]],
                "supporting_triples": [
                    {"source": row["source"], "relation": row["relation1"], "target": row["middle"]},
                    {"source": row["middle"], "relation": row["relation2"], "target": row["target"]},
                ],
                "pages": [row["page1"], row["page2"]],
                "years": [row["year1"], row["year2"]],
                "review_status": "AUTO_GENERATED_REGRESSION_CANDIDATE",
            }
        )

    offset = len(items)
    temporal_questions = [
        "How did the relationship between export controls and NVIDIA revenue change from FY2023 to FY2025 according to the filing?",
        "Compare NVIDIA's supply-chain risk evidence across FY2023 and FY2025 in this filing.",
        "What year-over-year change does this single 2025 filing prove for NVIDIA's revenue risk?",
        "Which risk factor increased NVIDIA revenue in both FY2023 and FY2025?",
        "What causal trend is established across multiple NVIDIA 10-K filings?",
    ]
    for index, question in enumerate(temporal_questions, start=1):
        items.append(
            {
                "id": f"GQ-{offset + index:03d}",
                "question_type": "unanswerable_temporal",
                "difficulty": "abstention",
                "question": question,
                "expected_answer": "INSUFFICIENT_TEMPORAL_EVIDENCE",
                "atomic_facts": [],
                "answerable": False,
                "source_filing": filename,
                "evidence_claim_ids": [],
                "supporting_triples": [],
                "pages": [],
                "years": [2025],
                "review_status": "AUTO_GENERATED_REGRESSION_CANDIDATE",
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence-linked Golden QA draft")
    parser.add_argument("--doc_id", default="2025-10-K")
    parser.add_argument("--filename", default="2025-10-K.pdf")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation" / "golden_qa_v2.jsonl")
    parser.add_argument("--single-count", type=int, default=25)
    parser.add_argument("--multi-count", type=int, default=10)
    args = parser.parse_args()
    print(build_dataset(args.doc_id, args.filename, args.output, args.single_count, args.multi_count))


if __name__ == "__main__":
    main()
