"""Run a bounded parser pilot without changing the active stores.

The pilot freezes a deterministic set of potentially difficult pages from the
active filings, records the current pdfplumber document-layer baseline, and
probes Docling only when it is already installed in the project environment.
It never installs dependencies, writes indexes, or treats parser coverage as
semantic extraction accuracy.  With no Docling package the result is
intentionally ``NOT_RUN`` rather than a fabricated comparison.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategic_graphrag.build_identity import build_id_from_env
from strategic_graphrag.document_layer import DocumentLayerReader


DEFAULT_PDFS = (
    ROOT / "data" / "pdfs_other" / "2023-10-K.pdf",
    ROOT / "data" / "pdfs_other" / "2024-10-K.pdf",
    ROOT / "data" / "pdfs" / "2025-10-K.pdf",
)


def _candidate_score(page: Any) -> float:
    """Rank layout-risk proxies; this is triage, not a quality label."""
    text_length = len(page.normalized_text or "")
    footnote_like = len(re.findall(r"\b(?:note|notes|source|thereof)\b", page.normalized_text or "", re.I))
    table_cells = sum(len(table.cells) for table in page.tables)
    return round(
        len(page.tables) * 10.0
        + min(table_cells / 20.0, 10.0)
        + min(footnote_like / 2.0, 5.0)
        + (3.0 if text_length < 200 else 0.0),
        3,
    )


def _page_record(document: Any, page: Any) -> dict[str, Any]:
    return {
        "filename": document.filename,
        "document_id": document.document_id,
        "pdf_sha256": document.pdf_sha256,
        "physical_page_number": page.physical_page_number,
        "parse_status": page.parse_status,
        "text_parse_status": page.text_parse_status,
        "table_parse_status": page.table_parse_status,
        "ocr_status": page.ocr_status,
        "layout_status": page.layout_status,
        "text_chars": len(page.normalized_text or ""),
        "text_block_count": len(page.text_blocks),
        "table_count": len(page.tables),
        "table_cell_count": sum(len(table.cells) for table in page.tables),
        "table_error": page.table_error,
        "triage_score": _candidate_score(page),
    }


def _docling_probe(pdf_paths: list[Path]) -> dict[str, Any]:
    if importlib.util.find_spec("docling") is None:
        return {
            "status": "NOT_RUN",
            "available": False,
            "reason": "docling is not installed in the project environment; no dependency was installed by this audit",
            "files": [],
        }
    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        converter = DocumentConverter()
        files: list[dict[str, Any]] = []
        for pdf in pdf_paths:
            result = converter.convert(str(pdf))
            exported = result.document.export_to_dict()
            tables = exported.get("tables") if isinstance(exported, dict) else None
            files.append({
                "filename": pdf.name,
                "status": "PASS",
                "export_keys": sorted(exported) if isinstance(exported, dict) else [],
                "document_table_count": len(tables) if isinstance(tables, list) else None,
            })
        return {
            "status": "PASS",
            "available": True,
            "reason": "Docling conversion completed; page-level equivalence still requires manual review",
            "files": files,
        }
    except Exception as exc:  # keep a failed optional pilot explicit and recoverable
        return {
            "status": "BLOCKED",
            "available": True,
            "reason": f"Docling pilot failed: {type(exc).__name__}: {exc}",
            "files": [],
        }


def run(pdf_paths: list[Path], *, candidate_count: int, build_id: str | None) -> dict[str, Any]:
    reader = DocumentLayerReader()
    documents = [reader.read(pdf, build_id=build_id) for pdf in pdf_paths]
    pages = [_page_record(document, page) for document in documents for page in document.pages]
    candidates = sorted(
        pages,
        key=lambda item: (-float(item["triage_score"]), item["filename"], int(item["physical_page_number"])),
    )[:candidate_count]
    for rank, item in enumerate(candidates, start=1):
        item["candidate_rank"] = rank
        item["selection_reason"] = "deterministic layout-risk triage; requires parser equivalence review"
    coverage = {
        "total_pages": len(pages),
        "parsed_pages": sum(item["parse_status"] == "PARSED" for item in pages),
        "pending_pages": sum(item["parse_status"] == "OCR_REQUIRED" for item in pages),
        "failed_pages": sum(item["parse_status"] == "FAILED" for item in pages),
        "table_pages": sum(item["table_count"] > 0 for item in pages),
    }
    docling = _docling_probe(pdf_paths)
    overall_status = docling["status"] if docling["status"] != "PASS" else "PASS"
    return {
        "schema": "parser-pilot/v1",
        "status": overall_status,
        "build_id": build_id,
        "parser_baseline": {
            "parser": "pdfplumber-document-layer/v1",
            "status": "PASS" if coverage["failed_pages"] == 0 else "BLOCKED",
            "coverage": coverage,
            "candidate_count": len(candidates),
        },
        "docling": docling,
        "candidates": candidates,
        "limitations": [
            "Triage score selects review pages; it is not a parser quality metric.",
            "A successful conversion does not establish page/table/footnote equivalence.",
            "No OCR, graph, vector, numeric materialization, or active-store write is performed.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", nargs="*", type=Path, default=list(DEFAULT_PDFS))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-count", type=int, default=12)
    parser.add_argument("--build-id", default=build_id_from_env())
    args = parser.parse_args()
    if args.candidate_count < 1:
        raise SystemExit("--candidate-count must be positive")
    missing = [str(pdf) for pdf in args.pdf if not pdf.exists()]
    if missing:
        raise SystemExit(f"PDF not found: {missing}")
    result = run(args.pdf, candidate_count=args.candidate_count, build_id=args.build_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "build_id": result["build_id"],
        "candidate_count": len(result["candidates"]),
        "docling_status": result["docling"]["status"],
        "output": str(args.output),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
