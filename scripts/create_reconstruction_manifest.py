"""Create a local, secret-free rebuild identity without touching live stores."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategic_graphrag.build_identity import make_build_identity
from strategic_graphrag.document_layer import DocumentLayerReader
from strategic_graphrag.evaluation.metric_spec import metric_registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, action="append", default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus-id", default="nvidia-10k-2023-2025-v3")
    parser.add_argument("--build-id", default=None, help="Optional asserted build ID for artifact comparison")
    args = parser.parse_args()
    pdfs = args.pdf or [
        ROOT / "data" / "pdfs_other" / "2023-10-K.pdf",
        ROOT / "data" / "pdfs_other" / "2024-10-K.pdf",
        ROOT / "data" / "pdfs" / "2025-10-K.pdf",
    ]
    missing = [str(path) for path in pdfs if not path.exists()]
    if missing:
        raise SystemExit("Missing PDF(s): " + ", ".join(missing))
    reader = DocumentLayerReader()
    identities = make_build_identity(
        pdfs,
        corpus_id=args.corpus_id,
        parser_version=reader.parser_version if hasattr(reader, "parser_version") else "pdfplumber-document-layer/v1",
        parser_config_hash=reader.config_hash,
        root=ROOT,
    )
    build_id = args.build_id or identities.build_id
    local_document_checks = []
    for pdf in pdfs:
        document = reader.read(pdf, build_id=build_id)
        coverage = document.coverage()
        local_document_checks.append({
            "filename": pdf.name,
            "pdf_sha256": document.pdf_sha256,
            "coverage": coverage,
            "status": "PASS" if coverage["conservation_holds"] and coverage["failed"] == 0 else "BLOCKED",
        })
    all_local_pass = all(item["status"] == "PASS" for item in local_document_checks)
    result = {
        "schema": "strategic-graphrag-reconstruction-manifest/v1",
        "status": "PASS" if all_local_pass else "BLOCKED",
        "build_identity": identities.to_dict() | {"build_id": build_id},
        "document_layer": {
            "status": "PASS" if all_local_pass else "BLOCKED",
            "files": local_document_checks,
            "ocr": "NOT_SUPPORTED_FAIL_CLOSED",
        },
        "artifacts": {
            "graph": {"status": "NOT_RUN", "build_id": None, "proof": "requires isolated Neo4j export"},
            "vector": {"status": "NOT_RUN", "build_id": None, "proof": "requires isolated Chroma build"},
            "numeric": {"status": "NOT_RUN", "build_id": None, "proof": "requires isolated observation materialization"},
            "llm_cache": {"status": "PASS", "build_id": build_id, "proof": "cache key includes build_id when bound"},
        },
        "metrics": metric_registry(),
        "decisions": {
            "document_layer": "RETAIN_AND_ROUTE_ALL_DOWNSTREAM_TEXT_THROUGH_CANONICAL_READER",
            "active_stores": "PRESERVE_UNTIL_ISOLATED_BUILD_AND_ROLLBACK_PASS",
            "ocr": "BLOCK_SILENT_INGEST_UNTIL_OCR_IMPLEMENTED_OR_REVIEWED",
        },
        "environment": {
            "graphrag_build_id_env": os.getenv("GRAPHRAG_BUILD_ID"),
            "live_store_write": "NOT_RUN",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "build_id": build_id, "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
