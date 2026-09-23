"""Canonical PDF document representation used by downstream pipeline stages.

The project previously let the graph pipeline, vector indexer, and coverage
auditor extract PDF text independently.  That makes it possible for a graph
claim and a vector chunk to refer to different page text without noticing.
This module is deliberately dependency-light and read-only: it creates a
stable document/page/table representation and an explicit page coverage
ledger.  It does not write Neo4j, Chroma, or an OCR result.

OCR is intentionally fail-closed.  A page with no extractable text is marked
``OCR_REQUIRED`` rather than being treated as successfully parsed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pdfplumber


DOCUMENT_LAYER_SCHEMA = "document-layer/v1"
PARSER_VERSION = "pdfplumber-document-layer/v1"
PAGE_STATUSES = {"PARSED", "OCR_REQUIRED", "FAILED"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    """Normalize presentation whitespace without changing the raw evidence."""
    value = unicodedata.normalize("NFKC", str(value or "")).replace("\u00a0", " ")
    lines = []
    for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _stable_id(prefix: str, *parts: Any) -> str:
    material = "|".join(str(part or "") for part in parts)
    return f"{prefix}_{hashlib.sha256(material.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True)
class TextBlock:
    block_id: str
    text: str
    normalized_text: str
    reading_order: int
    bbox: Dict[str, float]
    block_type: str = "TEXT"


@dataclass(frozen=True)
class Footnote:
    footnote_id: str
    text: str
    normalized_text: str
    marker: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None


@dataclass(frozen=True)
class Cell:
    cell_id: str
    row_index: int
    column_index: int
    raw_value: str
    normalized_value: str
    bbox: Optional[Dict[str, float]] = None
    header_level: int = 0
    header_path: Tuple[str, ...] = ()
    is_header: bool = False
    footnote_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class Table:
    table_id: str
    table_index: int
    cells: Tuple[Cell, ...]
    row_count: int
    column_count: int
    header_rows: Tuple[int, ...] = ()
    footnote_ids: Tuple[str, ...] = ()
    raw_matrix: Tuple[Tuple[str, ...], ...] = ()


@dataclass
class Page:
    physical_page_number: int
    printed_page_number: Optional[str]
    width: Optional[float]
    height: Optional[float]
    raw_text: str
    normalized_text: str
    text_blocks: List[TextBlock] = field(default_factory=list)
    tables: List[Table] = field(default_factory=list)
    footnotes: List[Footnote] = field(default_factory=list)
    reading_order: List[str] = field(default_factory=list)
    section: Optional[str] = None
    parser_version: str = PARSER_VERSION
    config_hash: str = ""
    parse_status: str = "PARSED"
    error: Optional[str] = None
    normalization_notes: List[str] = field(default_factory=list)
    text_parse_status: str = "PARSED"
    table_parse_status: str = "NOT_PRESENT"
    table_error: Optional[str] = None
    ocr_status: str = "NOT_REQUIRED"
    layout_status: str = "OBSERVED"


@dataclass
class Document:
    document_id: str
    filename: str
    pdf_sha256: str
    total_pages: int
    pages: List[Page]
    parser_version: str = PARSER_VERSION
    config_hash: str = ""
    build_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def coverage(self) -> Dict[str, Any]:
        counts = {status: 0 for status in PAGE_STATUSES}
        for page in self.pages:
            counts[page.parse_status] = counts.get(page.parse_status, 0) + 1
        accounted = sum(counts.values())
        result = {
            "total_pages": self.total_pages,
            "successful_parse": counts.get("PARSED", 0),
            "excluded": 0,
            "pending": counts.get("OCR_REQUIRED", 0),
            "failed": counts.get("FAILED", 0),
            "status_counts": counts,
            "conservation_holds": accounted == self.total_pages and len(self.pages) == self.total_pages,
            "ocr_supported": False,
        }
        return result

    def assert_coverage(self) -> None:
        coverage = self.coverage()
        if not coverage["conservation_holds"]:
            raise ValueError(f"page coverage does not conserve pages: {coverage}")

    def to_dict(self) -> Dict[str, Any]:
        self.assert_coverage()
        value = asdict(self)
        value["schema"] = DOCUMENT_LAYER_SCHEMA
        value["coverage"] = self.coverage()
        return value

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _config_hash(config: Dict[str, Any]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _printed_page_number(raw_text: str) -> Optional[str]:
    lines = [line.strip() for line in str(raw_text or "").splitlines() if line.strip()]
    candidates = lines[:5] + lines[-5:]
    for line in candidates:
        match = re.fullmatch(r"(?:page\s+)?([ivxlcdm]+|\d{1,4})", line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _word_bbox(word: Dict[str, Any]) -> Dict[str, float]:
    return {
        key: number
        for key in ("x0", "top", "x1", "bottom")
        if (number := _finite_number(word.get(key))) is not None
    }


def _text_blocks(page: Any, page_number: int, document_identity: str = "") -> List[TextBlock]:
    try:
        words = page.extract_words(use_text_flow=True, keep_blank_chars=False) or []
    except Exception:
        words = []
    rows: List[Dict[str, Any]] = []
    for word in sorted(words, key=lambda item: (float(item.get("top", 0)), float(item.get("x0", 0)))):
        top = float(word.get("top", 0))
        row = next((candidate for candidate in reversed(rows) if abs(candidate["top"] - top) <= 2.0), None)
        if row is None:
            row = {"top": top, "words": []}
            rows.append(row)
        row["words"].append(word)
    blocks: List[TextBlock] = []
    for order, row in enumerate(rows):
        row_words = sorted(row["words"], key=lambda item: float(item.get("x0", 0)))
        text = " ".join(str(word.get("text") or "") for word in row_words).strip()
        if not text:
            continue
        values = [_word_bbox(word) for word in row_words]
        numeric = [value for value in values if value]
        bbox = {
            "x0": min(value.get("x0", 0.0) for value in numeric),
            "top": min(value.get("top", 0.0) for value in numeric),
            "x1": max(value.get("x1", 0.0) for value in numeric),
            "bottom": max(value.get("bottom", 0.0) for value in numeric),
        } if numeric else {}
        blocks.append(TextBlock(
            block_id=_stable_id("tb", document_identity, page_number, order, text),
            text=text,
            normalized_text=normalize_text(text),
            reading_order=order,
            bbox=bbox,
        ))
    return blocks


def _tables(
    page: Any,
    page_number: int,
    document_identity: str = "",
) -> Tuple[List[Table], Optional[str]]:
    try:
        matrices = page.extract_tables() or []
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    tables: List[Table] = []
    try:
        for table_index, matrix in enumerate(matrices):
            if not matrix:
                continue
            rows: List[Tuple[str, ...]] = []
            cells: List[Cell] = []
            max_columns = max(len(row or []) for row in matrix)
            for row_index, row in enumerate(matrix):
                normalized_row = tuple(normalize_text(value or "") for value in (row or []))
                normalized_row += ("",) * (max_columns - len(normalized_row))
                rows.append(normalized_row)
                raw_row = list(row or [])
                for column_index, value in enumerate(normalized_row):
                    cells.append(Cell(
                        cell_id=_stable_id(
                            "cell", document_identity, page_number,
                            table_index, row_index, column_index,
                        ),
                        row_index=row_index,
                        column_index=column_index,
                        raw_value=str(raw_row[column_index] if column_index < len(raw_row) else ""),
                        normalized_value=value,
                        header_level=1 if row_index == 0 else 0,
                        header_path=(value,) if row_index == 0 and value else (),
                        is_header=row_index == 0,
                    ))
            tables.append(Table(
                table_id=_stable_id(
                    "table", document_identity, page_number, table_index,
                    json.dumps(rows, ensure_ascii=False),
                ),
                table_index=table_index,
                cells=tuple(cells),
                row_count=len(rows),
                column_count=max_columns,
                header_rows=(0,) if rows else (),
                raw_matrix=tuple(rows),
            ))
    except Exception as exc:
        return [], f"{type(exc).__name__}: {exc}"
    return tables, None


class DocumentLayerReader:
    """Read one PDF into the canonical document layer without side effects."""

    def __init__(self, *, parser_config: Optional[Dict[str, Any]] = None):
        self.parser_version = PARSER_VERSION
        self.parser_config = dict(parser_config or {
            "text_flow": True,
            "word_line_tolerance": 2.0,
            "ocr": "unsupported_fail_closed",
        })
        self.config_hash = _config_hash(self.parser_config)

    def read(
        self,
        pdf_path: str | Path,
        *,
        section_resolver: Optional[Callable[[int], Optional[str]]] = None,
        build_id: Optional[str] = None,
    ) -> Document:
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(path)
        document = Document(
            document_id=path.stem,
            filename=path.name,
            pdf_sha256=sha256_file(path),
            total_pages=0,
            pages=[],
            config_hash=self.config_hash,
            build_id=build_id,
            metadata={"ocr": "NOT_SUPPORTED", "source_path": str(path)},
        )
        with pdfplumber.open(str(path)) as pdf:
            document.total_pages = len(pdf.pages)
            for index, source_page in enumerate(pdf.pages, start=1):
                try:
                    raw_text = source_page.extract_text() or ""
                    normalized = normalize_text(raw_text)
                    blocks = _text_blocks(source_page, index, document.pdf_sha256)
                    tables, table_error = _tables(source_page, index, document.pdf_sha256)
                    text_status = "PARSED" if normalized else "EMPTY"
                    table_status = (
                        "FAILED" if table_error
                        else "PARSED" if tables
                        else "NOT_PRESENT"
                    )
                    status = (
                        "FAILED" if table_error
                        else "PARSED" if normalized or tables
                        else "OCR_REQUIRED"
                    )
                    notes = []
                    if tables and not normalized:
                        notes.append("table_present_without_text_layer")
                    if status == "OCR_REQUIRED":
                        notes.append("no_extractable_text_and_ocr_not_configured")
                    if table_error:
                        notes.append("table_parse_failed")
                    document.pages.append(Page(
                        physical_page_number=index,
                        printed_page_number=_printed_page_number(raw_text),
                        width=_finite_number(getattr(source_page, "width", None)),
                        height=_finite_number(getattr(source_page, "height", None)),
                        raw_text=raw_text,
                        normalized_text=normalized,
                        text_blocks=blocks,
                        tables=tables,
                        reading_order=[block.block_id for block in blocks],
                        section=section_resolver(index) if section_resolver else None,
                        config_hash=self.config_hash,
                        parse_status=status,
                        text_parse_status=text_status,
                        table_parse_status=table_status,
                        table_error=table_error,
                        ocr_status="REQUIRED" if status == "OCR_REQUIRED" else "NOT_REQUIRED",
                        layout_status="OBSERVED" if blocks else "UNKNOWN",
                        normalization_notes=notes,
                    ))
                except Exception as exc:  # page-level failure is retained, never dropped
                    document.pages.append(Page(
                        physical_page_number=index,
                        printed_page_number=None,
                        width=None,
                        height=None,
                        raw_text="",
                        normalized_text="",
                        section=section_resolver(index) if section_resolver else None,
                        config_hash=self.config_hash,
                        parse_status="FAILED",
                        error=f"{type(exc).__name__}: {exc}",
                        text_parse_status="FAILED",
                        table_parse_status="FAILED",
                        table_error=f"{type(exc).__name__}: {exc}",
                        ocr_status="UNKNOWN",
                        layout_status="UNKNOWN",
                    ))
        document.assert_coverage()
        return document


__all__ = [
    "DOCUMENT_LAYER_SCHEMA", "PARSER_VERSION", "Document", "Page", "TextBlock",
    "Table", "Cell", "Footnote", "DocumentLayerReader", "normalize_text", "sha256_file",
]
