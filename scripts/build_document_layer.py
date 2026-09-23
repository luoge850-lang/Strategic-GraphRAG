"""Build a read-only canonical document-layer artifact for one or more PDFs.

The command never contacts Neo4j, Chroma, an OCR service, or an LLM.  Its
``PASS`` result means only that every physical PDF page has a retained parse
record and the page ledger conserves pages.  ``OCR_REQUIRED`` pages remain
pending and are not silently admitted as parsed text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategic_graphrag.build_identity import build_id_from_env
from strategic_graphrag.document_layer import DocumentLayerReader, DOCUMENT_LAYER_SCHEMA


def build(pdf: Path, *, build_id: str | None = None) -> dict:
    document = DocumentLayerReader().read(pdf, build_id=build_id)
    coverage = document.coverage()
    status = "PASS" if coverage["conservation_holds"] and coverage["failed"] == 0 else "BLOCKED"
    return {
        "schema": DOCUMENT_LAYER_SCHEMA,
        "status": status,
        "document": document.to_dict(),
        "limitations": [
            "OCR is not configured; OCR_REQUIRED pages are pending and not parsed.",
            "Table coordinates are not claimed unless the PDF backend exposes them.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-id", default=build_id_from_env())
    args = parser.parse_args()
    records = []
    for pdf in args.pdf:
        if not pdf.exists():
            raise SystemExit(f"PDF not found: {pdf}")
        records.append(build(pdf, build_id=args.build_id))
    status = "PASS" if all(item["status"] == "PASS" for item in records) else "BLOCKED"
    result = {
        "schema": "document-layer-audit/v1",
        "status": status,
        "build_id": args.build_id,
        "files": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": status, "files": len(records), "output": str(args.output)}, ensure_ascii=False))
    if status == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
