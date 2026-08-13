"""Read-only PDF coverage audit for the GraphRAG ingestion contract.

This script deliberately does not call an LLM and does not connect to Neo4j.
It answers the first coverage question before an expensive rebuild:

    How many pages contain text, which SEC section owns each page, and which
    pages/chunks would be sent to the extractor under the current config?

The output is a deterministic planning ledger.  It must not be interpreted as
evidence that a page contains a valid relationship; only the ingestion pipeline
can produce strict EvidenceClaims after extraction and verbatim validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategic_graphrag.pipeline.section_detector import SectionDetector
from strategic_graphrag.pipeline.text_splitter import RecursiveTextSplitter


def audit_pdf(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> dict:
    splitter = RecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    detector = SectionDetector()

    with pdfplumber.open(str(pdf_path)) as pdf:
        page_texts = [page.extract_text() or "" for page in pdf.pages]
        detector.scan(pdf)
        target_indices = set(detector.get_target_pages(pdf))

    pages = []
    for index, text in enumerate(page_texts):
        page_number = index + 1
        stripped = text.strip()
        section_id = detector.get_section_for_page(page_number)
        selected = index in target_indices
        extraction_text = detector.get_page_extraction_text(page_number, text)
        extraction_chars = len(extraction_text.strip())
        chunks = (
            splitter.split_text(extraction_text)
            if selected and extraction_chars >= 200
            else []
        )
        pages.append(
            {
                "page": page_number,
                "section_id": section_id,
                "section": detector.get_page_section_label(page_number),
                "text_chars": len(stripped),
                "extraction_text_chars": extraction_chars,
                "has_text": bool(stripped),
                "meets_min_content": extraction_chars >= 200,
                "selected_for_extraction": selected,
                "section_heading_line": detector.get_section_start_line(page_number),
                "chunk_count": len(chunks),
                "exclusion_reason": (
                    None
                    if selected and extraction_chars >= 200
                    else (
                        "section_not_in_target_scope"
                        if not selected
                        else "below_min_content_threshold"
                    )
                ),
            }
        )

    section_counts = Counter(page["section_id"] or "UNCLASSIFIED" for page in pages)
    selected = [page for page in pages if page["selected_for_extraction"]]
    processable = [
        page
        for page in selected
        if page["meets_min_content"]
    ]
    return {
        "status": "PASS",
        "pdf": str(pdf_path),
        "chunk_config": {
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        },
        "target_sections": sorted(detector.target_sections),
        "total_pages": len(pages),
        "pages_with_text": sum(page["has_text"] for page in pages),
        "pages_meeting_min_content": sum(page["meets_min_content"] for page in pages),
        "selected_pages": len(selected),
        "processable_selected_pages": len(processable),
        "excluded_pages": len(pages) - len(processable),
        "excluded_page_reasons": dict(
            Counter(
                page["exclusion_reason"]
                for page in pages
                if page["exclusion_reason"]
            )
        ),
        "estimated_chunks": sum(page["chunk_count"] for page in processable),
        "section_page_counts": dict(section_counts),
        "detected_sections": [
            {
                "section_id": span.section_id,
                "start_page": span.start_page,
                "end_page": span.end_page,
                "confidence": span.confidence,
            }
            for span in detector.sections
        ],
        "pages": pages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only PDF coverage audit")
    parser.add_argument("pdf", nargs="+", type=Path, help="PDF path(s) to inspect")
    parser.add_argument("--chunk-size", type=int, default=2400)
    parser.add_argument("--chunk-overlap", type=int, default=300)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    reports = []
    for pdf_path in args.pdf:
        if not pdf_path.exists():
            raise SystemExit(f"PDF not found: {pdf_path}")
        reports.append(audit_pdf(pdf_path, args.chunk_size, args.chunk_overlap))

    result = {"contract": "read_only_pdf_section_and_chunk_coverage_v1", "files": reports}
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
