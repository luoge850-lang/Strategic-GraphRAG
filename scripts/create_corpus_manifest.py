"""Create a reproducibility manifest for the active three-filing corpus."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_PDFS = [
    ROOT / "data" / "pdfs_other" / "2023-10-K.pdf",
    ROOT / "data" / "pdfs_other" / "2024-10-K.pdf",
    ROOT / "data" / "pdfs" / "2025-10-K.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, capture_output=True, check=False
    )
    return result.stdout.strip()


def main() -> None:
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "corpus_manifest.json"
    load_dotenv(ROOT / ".env", override=True)
    driver = GraphDatabase.driver(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"]),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE") or None) as session:
            rows = [dict(row) for row in session.run(
                """
                MATCH (c:EvidenceClaim)
                WHERE c.doc_id + '.pdf' IN $filings AND c.verification_status='VERBATIM'
                RETURN c.doc_id + '.pdf' AS filing, count(c) AS claims,
                       count(DISTINCT c.page) AS evidence_pages,
                       count(DISTINCT c.chunk_id) AS evidence_chunks
                ORDER BY filing
                """,
                filings=[path.name for path in ACTIVE_PDFS],
            )]
    finally:
        driver.close()
    manifest = {
        "schema": "strategic-graphrag-corpus-manifest/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_id": "nvidia-10k-2023-2025-v3",
        "active_filings": [
            {"filename": path.name, "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in ACTIVE_PDFS
        ],
        "graph_inventory": rows,
        "claim_id_version": "v2",
        "ontology_sha256": sha256(ROOT / "strategic_graphrag" / "ontology" / "financial_ontology.json"),
        "provider": os.getenv("LLM_PROVIDER"),
        "extraction_model": os.getenv("LLM_EXTRACTION_MODEL") or os.getenv("LLM_MODEL"),
        "query_model": os.getenv("LLM_QUERY_MODEL") or os.getenv("LLM_MODEL"),
        "report_model": os.getenv("LLM_REPORT_MODEL") or os.getenv("LLM_MODEL"),
        "embedding_model": os.getenv("GRAPH_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "vector_collection": os.getenv("GRAPH_VECTOR_COLLECTION", "nvidia_sec_filings_active"),
        "git": {
            "branch": git_value("branch", "--show-current"),
            "head": git_value("rev-parse", "HEAD"),
            "dirty": bool(git_value("status", "--porcelain")),
        },
        "evaluation_status": "NO_CURRENT_HUMAN_GOLDEN_QA",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
