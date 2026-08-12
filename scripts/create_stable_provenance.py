"""Create a reproducibility manifest for the single-PDF stable candidate."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def neo4j_snapshot(filename: str) -> dict:
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            counts = session.run(
                """
                MATCH (c:EvidenceClaim {doc_id: replace($filename, '.pdf', '')})
                WITH count(c) AS claims
                MATCH (n)-[r]->(m)
                WHERE coalesce(r.source_filing, r.filing, '') = $filename
                RETURN claims, count(r) AS filing_relationships,
                       count(DISTINCT n) + count(DISTINCT m) AS storage_nodes
                """,
                filename=filename,
            ).single()
            indexes = [
                dict(row)
                for row in session.run(
                    "SHOW FULLTEXT INDEXES YIELD name, state, populationPercent "
                    "RETURN name, state, populationPercent ORDER BY name"
                )
            ]
            return {
                "claims": counts["claims"] if counts else None,
                "filing_relationships": counts["filing_relationships"] if counts else None,
                "storage_nodes": counts["storage_nodes"] if counts else None,
                "fulltext_indexes": indexes,
            }
    finally:
        driver.close()


def main() -> None:
    load_dotenv(ROOT / ".env", override=True)
    filename = "2025-10-K.pdf"
    tracked = {
        "pdf": ROOT / "data" / "pdfs" / filename,
        "kg_audit": ROOT / "reports" / "final_kg_audit.json",
        "page_coverage": ROOT / "reports" / "final_page_coverage.json",
        "pipeline_stats": ROOT / "reports" / "post_optimization_pipeline_stats.json",
        "standard_query_audit": ROOT / "reports" / "standard_query_audit.json",
        "golden_qa": ROOT / "data" / "evaluation" / "golden_qa_v2.jsonl",
        "golden_results": ROOT / "reports" / "golden_qa_v2_results.json",
    }
    manifest = {
        "status": "SINGLE_PDF_STABLE_CANDIDATE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_filing": filename,
        "llm_provider": os.getenv("LLM_PROVIDER"),
        "llm_model": os.getenv("LLM_MODEL") or os.getenv("DEEPSEEK_MODEL"),
        "dataset_status": "AUTO_GENERATED_REGRESSION_CANDIDATE_NOT_HUMAN_GOLD",
        "evaluation_protocol": "hybrid retrieval, structural metrics only; LLM judge disabled",
        "files_sha256": {name: sha256(path) for name, path in tracked.items()},
        "neo4j_snapshot": neo4j_snapshot(filename),
        "limitations": [
            "This is one NVIDIA 2025 10-K; it cannot establish cross-year trends.",
            "Golden QA is automatically generated and must be human-reviewed before academic claims.",
            "Faithfulness and Answer Relevance LLM-judge scores were not run in this candidate.",
            "This candidate is not a production deployment or a final paper result.",
        ],
    }
    output = ROOT / "reports" / "stable_single_pdf_provenance.json"
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
