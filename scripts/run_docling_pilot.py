"""Compare the canonical pdfplumber layer with a pinned Docling pilot.

The script is run from an isolated Docling environment.  Automatic checks are
lexical/layout diagnostics only; the emitted review queue is the handoff for
human semantic judgement and is never called an independent Gold set.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    import psutil
except ImportError:  # pragma: no cover - optional measurement dependency
    psutil = None


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def numbers(value: Any) -> List[str]:
    return sorted(set(re.findall(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?", str(value or ""))))


def units(value: Any) -> List[str]:
    text = str(value or "").lower()
    return sorted(set(re.findall(r"\b(?:usd|dollars?|million(?:s)?|billion(?:s)?|thousand(?:s)?|percent|%)\b", text)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_rows(path: Path) -> List[Dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return list(value.get("candidates") or [])


def baseline_snapshot_rows(snapshot_dir: Path, candidates: List[Dict[str, Any]]) -> Dict[tuple[str, int], Dict[str, Any]]:
    wanted = {(row["filename"], int(row["physical_page_number"])) for row in candidates}
    rows: Dict[tuple[str, int], Dict[str, Any]] = {}
    for snapshot in snapshot_dir.glob("*.pdf.json"):
        document = json.loads(snapshot.read_text(encoding="utf-8"))
        for page in document.get("pages", []):
            key = (document["filename"], int(page["physical_page_number"]))
            if key not in wanted:
                continue
            table_text = "\n".join(
                " | ".join(cell.get("normalized_value", "") for cell in table.get("cells", []))
                for table in page.get("tables", [])
            )
            rows[key] = {
                "text": page.get("normalized_text", ""),
                "text_chars": len(page.get("normalized_text", "")),
                "table_count": len(page.get("tables", [])),
                "table_cell_count": sum(len(table.get("cells", [])) for table in page.get("tables", [])),
                "numbers": numbers(page.get("normalized_text", "") + "\n" + table_text),
                "units": units(page.get("normalized_text", "") + "\n" + table_text),
                "footnote_like": [
                    line for line in str(page.get("normalized_text", "")).splitlines()
                    if re.search(r"\b(?:note|notes|source|thereof)\b", line, re.I)
                ],
                "coordinates": sum(
                    1 for block in page.get("text_blocks", []) if block.get("bbox")
                ) + sum(
                    1 for table in page.get("tables", [])
                    for cell in table.get("cells", []) if cell.get("bbox")
                ),
            }
    return rows


def prov_page(item: Dict[str, Any]) -> int | None:
    for provenance in item.get("prov", []) or []:
        value = provenance.get("page_no")
        if isinstance(value, int):
            return value
    return None


def extract_docling_pages(document: Any) -> Dict[int, Dict[str, Any]]:
    exported = document.export_to_dict()
    pages: Dict[int, Dict[str, Any]] = {}
    for text_item in exported.get("texts", []) or []:
        page = prov_page(text_item)
        if page is None:
            continue
        row = pages.setdefault(page, {"texts": [], "tables": [], "coordinates": 0, "footnotes": []})
        text = normalize(text_item.get("text") or text_item.get("orig") or "")
        if text:
            row["texts"].append(text)
        row["coordinates"] += sum(1 for prov in text_item.get("prov", []) or [] if prov.get("bbox"))
        if re.search(r"\b(?:note|notes|source|thereof)\b", text, re.I):
            row["footnotes"].append(text)
    for table_item in exported.get("tables", []) or []:
        page_numbers = sorted({prov_page(table_item)} - {None})
        data = table_item.get("data") or {}
        cells = data.get("table_cells") or data.get("cells") or []
        for page in page_numbers:
            row = pages.setdefault(page, {"texts": [], "tables": [], "coordinates": 0, "footnotes": []})
            row["tables"].append({
                "cell_count": len(cells),
                "row_count": data.get("num_rows") or data.get("row_count"),
                "column_count": data.get("num_cols") or data.get("column_count"),
                "text": normalize(" ".join(
                    str(cell.get("text") or cell.get("content") or "")
                    if isinstance(cell, dict) else str(cell)
                    for cell in cells
                )),
            })
            row["coordinates"] += sum(1 for prov in table_item.get("prov", []) or [] if prov.get("bbox"))
    for row in pages.values():
        row["text"] = "\n".join(row["texts"])
        table_text = "\n".join(table["text"] for table in row["tables"])
        row["text_chars"] = len(row["text"])
        row["table_count"] = len(row["tables"])
        row["table_cell_count"] = sum(table["cell_count"] for table in row["tables"])
        row["numbers"] = numbers(row["text"] + "\n" + table_text)
        row["units"] = units(row["text"] + "\n" + table_text)
    return pages


def compare_page(baseline: Dict[str, Any], docling: Dict[str, Any] | None) -> Dict[str, Any]:
    if docling is None:
        return {"status": "FAIL", "reason": "Docling did not emit this page"}
    number_overlap = len(set(baseline["numbers"]) & set(docling["numbers"])) / max(len(set(baseline["numbers"])), 1)
    unit_overlap = len(set(baseline["units"]) & set(docling["units"])) / max(len(set(baseline["units"])), 1)
    return {
        "status": "REVIEW_REQUIRED",
        "automatic_diagnostics": {
            "text_length_ratio": round(docling["text_chars"] / max(baseline["text_chars"], 1), 4),
            "table_count_delta": docling["table_count"] - baseline["table_count"],
            "table_cell_count_delta": docling["table_cell_count"] - baseline["table_cell_count"],
            "numeric_token_recall": round(number_overlap, 4),
            "unit_token_recall": round(unit_overlap, 4),
            "baseline_coordinate_count": baseline["coordinates"],
            "docling_coordinate_count": docling["coordinates"],
            "baseline_footnote_candidates": len(baseline["footnote_like"]),
            "docling_footnote_candidates": len(docling["footnotes"]),
        },
        "manual_review": {
            "reading_order": "",
            "table_rows_columns_headers": "",
            "numeric_values_units_periods": "",
            "footnotes": "",
            "source_coordinates": "",
            "semantic_decision": "",
            "reviewer": "",
            "review_status": "UNLABELED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument("--pdf", type=Path, action="append", default=None)
    parser.add_argument("--build-id", default=None, help="Immutable build identity for the baseline snapshot used by this pilot")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from docling.document_converter import DocumentConverter  # type: ignore
    from pypdf import PdfReader, PdfWriter  # type: ignore

    candidates = candidate_rows(args.candidate_report)
    baseline = baseline_snapshot_rows(args.snapshot_dir, candidates)
    converter = DocumentConverter()
    all_docling: Dict[str, Dict[int, Dict[str, Any]]] = {}
    file_runs = []
    process = psutil.Process() if psutil else None
    pdfs = [path.resolve() for path in args.pdf] if args.pdf else sorted(args.pdf_dir.glob("*.pdf")) if args.pdf_dir else []
    if len(pdfs) != 3:
        raise SystemExit(f"expected exactly three --pdf inputs or a directory containing three PDFs, got {len(pdfs)}")
    for pdf in pdfs:
        selected_pages = sorted({
            int(row["physical_page_number"])
            for row in candidates
            if row["filename"] == pdf.name
        })
        if not selected_pages:
            continue
        # Docling is evaluated on the fixed 12-page pilot, not on an
        # unbounded 395-page conversion. The page map keeps citations tied to
        # original physical pages and the original PDF hash remains recorded.
        input_dir = args.output.parent / "docling_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        pilot_pdf = input_dir / pdf.name
        writer = PdfWriter()
        reader = PdfReader(str(pdf))
        for page_number in selected_pages:
            writer.add_page(reader.pages[page_number - 1])
        with pilot_pdf.open("wb") as handle:
            writer.write(handle)
        started = time.perf_counter()
        rss_before = process.memory_info().rss if process else None
        result = converter.convert(str(pilot_pdf))
        local_pages = extract_docling_pages(result.document)
        pages = {
            original_page: local_pages.get(local_page, {})
            for local_page, original_page in enumerate(selected_pages, start=1)
        }
        rss_after = process.memory_info().rss if process else None
        all_docling[pdf.name] = pages
        file_runs.append({
            "filename": pdf.name,
            "pdf_sha256": sha256(pdf),
            "pilot_pdf_sha256": sha256(pilot_pdf),
            "pilot_pages": selected_pages,
            "pilot_input": str(pilot_pdf),
            "status": "PASS",
            "docling_version": "2.129.0",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "docling_pages_observed": len(local_pages),
        })

    comparisons = []
    for candidate in candidates:
        key = (candidate["filename"], int(candidate["physical_page_number"]))
        comparisons.append({
            "filename": key[0],
            "physical_page_number": key[1],
            "pdf_sha256": candidate["pdf_sha256"],
            "baseline": baseline.get(key, {}),
            "docling": all_docling.get(key[0], {}).get(key[1]),
            "comparison": compare_page(baseline.get(key, {"text_chars": 0, "numbers": [], "units": [], "table_count": 0, "table_cell_count": 0, "coordinates": 0, "footnote_like": []}), all_docling.get(key[0], {}).get(key[1])),
        })
    output = {
        "schema": "parser-pilot-docling/v1",
        "build_id": args.build_id,
        "status": "PASS" if all(item["comparison"]["status"] == "REVIEW_REQUIRED" for item in comparisons) else "FAIL",
        "docling": {
            "version": importlib.metadata.version("docling"),
            "docling_core": importlib.metadata.version("docling-core"),
            "license": "MIT (package metadata; verify for redistribution)",
            "environment": sys.executable,
            "pilot_input_mode": "cropped_candidate_pages_from_original_pdfs",
            "model_runtime": "CPU RapidOCR/layout pipeline; downloaded model files are environment assets",
        },
        "baseline_report": str(args.candidate_report),
        "files": file_runs,
        "comparisons": comparisons,
        "manual_review_queue": [
            {"filename": item["filename"], "physical_page_number": item["physical_page_number"], "review_status": "UNLABELED", "semantic_gold": None}
            for item in comparisons
        ],
        "limitations": [
            "Automatic token/shape overlap is not semantic correctness or independent human Gold.",
            "The 12 pages are a deterministic pilot sample, not a 395-page accuracy estimate.",
            "Human review must fill the manual_review fields before parser replacement decisions.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"status": output["status"], "candidate_count": len(comparisons), "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
