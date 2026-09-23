# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: PDF → Knowledge Graph Pipeline Orchestrator
===============================================================
End-to-end data engineering pipeline:

  PDF Files → Text Extraction → Section Detection → Chunking
  → LLM + Rule Triple Extraction → Filtering → Neo4j Ingestion
  → Vector Baseline Construction (ChromaDB)

Usage:
    python -m strategic_graphrag.pipeline.pipeline --pdf_dir data/pdfs
    python -m strategic_graphrag.pipeline.pipeline --pdf_dir data/pdfs --year 2025
"""

import os
import re
import sys
import json
import glob
import hashlib
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

# PDF extraction
import pdfplumber
from .text_splitter import RecursiveTextSplitter

# Internal modules
from .section_detector import SectionDetector
from .extractor import TripleExtractor
from .ingestor import GraphIngestor
from .financial_table_extractor import extract_financial_table_triples
from ..document_layer import DocumentLayerReader, DOCUMENT_LAYER_SCHEMA
from ..llm_response_cache import LLMResponseCacheError
from ..ontology.entity_registry import resolve_entity

logger = logging.getLogger("Pipeline")


def _get_filter_rejection_counts(extractor) -> Dict[str, int]:
    """Snapshot the latest filter diagnostics without changing old callers."""
    getter = getattr(extractor, "get_filter_rejection_counts", None)
    if not callable(getter):
        return {}
    counts = getter()
    return dict(counts or {})


# =============================================================================
# Pipeline Configuration
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the PDF → KG pipeline."""
    pdf_dir: str = "data/pdfs"
    chunk_size: int = 2400
    chunk_overlap: int = 300
    use_llm: bool = True
    use_rules: bool = True
    # Verify candidates on every target-section page by default.  The rule
    # engine still runs first and supplies deterministic candidates; restricting
    # LLM calls to Risk Factors made MD&A and Business systematically sparse.
    llm_on_all_target_pages: bool = True
    # Keep the extraction scope aligned with the detector.  Item 7A and the
    # financial statements are needed for revenue/cost/liquidity questions;
    # the detector stops the scope before exhibits and signatures.
    target_sections: Tuple[str, ...] = (
        "RISK_FACTORS",
        "MD_AND_A",
        "BUSINESS",
        "QUANTITATIVE_MARKET_RISK",
        "FINANCIAL_STATEMENTS",
    )
    min_content_chars: int = 200
    llm_provider: Optional[str] = None
    model_name: Optional[str] = None
    # Do not infer the disclosure company from the filename.  The caller must
    # supply a verified document-registration identity; otherwise table facts
    # remain explicitly pending review.
    company_id: Optional[str] = None
    llm_cache_mode: Optional[str] = None
    llm_cache_path: Optional[str] = None
    require_llm: bool = False
    allow_multiple_pdfs: bool = False
    replace_existing_filing: bool = False
    year_override: Optional[int] = None
    # Extraction-only mode for repeatability and regression checks.  It must
    # not connect to Neo4j or run post-processing, so a second extraction can
    # be compared without changing the frozen graph.
    dry_run: bool = False
    # Optional isolated-build export.  Normal runs keep the historical compact
    # statistics shape; staging builds can retain the accepted triples for a
    # local graph index without writing Neo4j.
    capture_triples: bool = False
    # Optional explicit destination for unresolved financial-table candidates.
    # No queue file is written unless the caller opts in.
    pending_table_queue_path: Optional[str] = None
    # Optional JSON/JSONL document-registration registry. When supplied, a
    # company identity is accepted only for a filename+SHA256 match marked
    # VERIFIED/HUMAN_REVIEWED.
    document_registry_path: Optional[str] = None
    build_id: Optional[str] = None


# =============================================================================
# Pipeline Orchestrator
# =============================================================================

class KnowledgeGraphPipeline:
    """
    Orchestrates the full PDF → Knowledge Graph data engineering pipeline.

    Architecture:
      1. Load PDFs from directory
      2. For each PDF:
         a. Detect SEC sections (Item 1A, Item 7, etc.)
         b. Chunk relevant pages
         c. Extract triples (LLM + Rule dual-engine)
         d. Filter and canonicalize
         e. Ingest into Neo4j (6-layer schema)
      3. Run post-processing (dedup, hubness pruning)
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()

        # Initialize components
        self.section_detector = SectionDetector(
            target_sections=set(self.config.target_sections)
        )
        self.extractor = TripleExtractor(
            model_name=self.config.model_name,
            provider=self.config.llm_provider,
            cache_mode=self.config.llm_cache_mode,
            cache_path=self.config.llm_cache_path,
        )
        self.ingestor = GraphIngestor()
        # Persist the provider-safe values that the runtime actually resolved,
        # not potentially stale raw environment variables.
        self.ingestor.llm_provider = getattr(self.extractor.llm, "provider", "unknown")
        self.ingestor.llm_model = getattr(
            self.extractor.llm, "default_model", "unknown"
        )
        if self.config.build_id:
            self.ingestor.build_id = self.config.build_id

        # Text splitter
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.document_reader = DocumentLayerReader()

        # Statistics
        self.stats: Dict = {}

    def _ensure_commit_connection(self) -> None:
        """Fail before any filing write when the pre-commit reconnect fails."""
        if not self.ingestor.ensure_connection():
            raise RuntimeError(
                "Neo4j connection unavailable before filing commit; "
                "existing filing was preserved."
            )

    @staticmethod
    def _evidence_span(text: str, evidence: str) -> Tuple[Optional[int], Optional[int]]:
        """Locate an evidence quote in page text, tolerating PDF line breaks."""
        raw_text = str(text or "")
        raw_evidence = str(evidence or "").strip()
        if not raw_text or not raw_evidence:
            return None, None
        direct = raw_text.find(raw_evidence)
        if direct >= 0:
            return direct, direct + len(raw_evidence)

        collapsed_chars: List[str] = []
        original_offsets: List[int] = []
        pending_space = False
        for index, char in enumerate(raw_text):
            if char.isspace():
                pending_space = bool(collapsed_chars)
                continue
            if pending_space and collapsed_chars and collapsed_chars[-1] != " ":
                collapsed_chars.append(" ")
                original_offsets.append(index)
            collapsed_chars.append(char)
            original_offsets.append(index)
            pending_space = False

        collapsed_text = "".join(collapsed_chars)
        collapsed_evidence = re.sub(r"\s+", " ", raw_evidence).strip()
        start = collapsed_text.find(collapsed_evidence)
        if start < 0 or start >= len(original_offsets):
            return None, None
        end_index = min(start + len(collapsed_evidence) - 1, len(original_offsets) - 1)
        return original_offsets[start], original_offsets[end_index] + 1

    @staticmethod
    def _annotate_table_candidate(
        triple: Dict,
        *,
        filename: str,
        page_num: int,
        document_sha256: Optional[str] = None,
        company_resolution_source: str = "pending_review",
    ) -> Dict:
        """Attach deterministic review/round-trip identity to a table row."""
        source_table_id = str(
            triple.get("table_id") or triple.get("table_name") or "UNKNOWN_TABLE"
        )
        source_row_id = str(
            triple.get("row_id") or triple.get("target") or "UNKNOWN_ROW"
        )
        periods = []
        try:
            periods = [
                str(item.get("period"))
                for item in json.loads(triple.get("metric_values_json") or "[]")
                if isinstance(item, dict) and item.get("period") is not None
            ]
        except (TypeError, ValueError, json.JSONDecodeError):
            periods = []
        identity_payload = "|".join([
            str(document_sha256 or "UNKNOWN"),
            filename,
            str(page_num),
            source_table_id,
            source_row_id,
            ",".join(periods),
            str(triple.get("row_evidence") or triple.get("evidence_sentence") or ""),
        ])
        candidate_id = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()[:20]
        table_instance_id = (
            f"{str(document_sha256 or 'UNKNOWN')[:16]}:{filename}:{page_num}:table_"
            f"{hashlib.sha256(source_table_id.encode('utf-8')).hexdigest()[:12]}"
        )
        triple.update({
            "candidate_id": candidate_id,
            "queue_id": candidate_id,
            "table_instance_id": table_instance_id,
            "source_table_id": source_table_id,
            "source_row_id": source_row_id,
            "row_id": f"{table_instance_id}:row_{source_row_id}",
            "column_ids": periods or [str(triple.get("column_id") or "")],
            "source_span_type": "text",
            "bbox": None,
            "raw_values": triple.get("metric_values_json"),
            "unit_source": "table_header_or_row",
            "source_filing": filename,
            "source_page": page_num,
            "document_sha256": document_sha256,
            "company_resolution_source": company_resolution_source,
        })
        return triple

    @staticmethod
    def _load_document_registry(path_value: Optional[str]) -> List[Dict]:
        if not path_value:
            return []
        path = Path(path_value)
        if not path.exists():
            raise FileNotFoundError(f"Document registry not found: {path}")
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                return [
                    item for line in handle if line.strip()
                    for item in [json.loads(line)]
                    if isinstance(item, dict)
                ]
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value.get("documents", value.get("rows", []))
        return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []

    def _resolve_registered_company(
        self,
        *,
        filename: str,
        document_sha256: str,
    ) -> Tuple[Optional[str], str]:
        """Resolve only a canonical Company identity bound to this document."""
        configured = str(self.config.company_id or "").strip()
        registry = self._load_document_registry(self.config.document_registry_path)
        if registry:
            matches = [
                row for row in registry
                if str(row.get("filename") or row.get("source_filing") or "") == filename
                and str(row.get("document_sha256") or row.get("sha256") or "") == document_sha256
                and str(row.get("review_status") or row.get("status") or "").upper()
                in {"VERIFIED", "HUMAN_REVIEWED"}
            ]
            if not matches:
                return None, "pending_registry_review"
            raw_company = str(matches[0].get("company_id") or "").strip()
            if not raw_company:
                return None, "pending_registry_review"
            canonical, label = resolve_entity(raw_company, "Company")
            if label != "Company":
                return None, "pending_registry_review"
            if configured:
                configured_canonical, configured_label = resolve_entity(configured, "Company")
                if configured_label != "Company" or configured_canonical != canonical:
                    raise ValueError(
                        f"company_id conflicts with the verified document registry for {filename}"
                    )
            return canonical, "document_registry"
        if not configured:
            return None, "pending_review"
        canonical, label = resolve_entity(configured, "Company")
        if label != "Company":
            raise ValueError("company_id must resolve to a canonical Company entity")
        return canonical, "verified_config"

    @staticmethod
    def _table_review_record(
        triple: Dict,
        *,
        review_status: str,
        reason: str,
    ) -> Dict:
        """Return a JSON-safe, auditable copy for the annotation queue."""
        record = dict(triple)
        record["review_status"] = review_status
        record["review_reason"] = reason
        record["candidate_schema_version"] = "table-candidate/v1"
        record.setdefault("reviewer", "")
        record.setdefault("review_notes", "")
        record.setdefault("gold", {
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
        })
        return record

    def _write_pending_table_queue(self, candidates: List[Dict]) -> int:
        """Append new pending table candidates when an explicit path is set."""
        path_value = self.config.pending_table_queue_path
        if not path_value or not candidates:
            return 0
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing_ids = set()
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(item, dict) and item.get("candidate_id"):
                        existing_ids.add(str(item["candidate_id"]))
        written = 0
        with path.open("a", encoding="utf-8") as handle:
            for candidate in candidates:
                candidate_id = str(candidate.get("candidate_id") or "")
                if not candidate_id or candidate_id in existing_ids:
                    continue
                handle.write(json.dumps(candidate, ensure_ascii=False, default=str) + "\n")
                existing_ids.add(candidate_id)
                written += 1
        return written

    # ── PDF Text Extraction ──

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text from a PDF file, page by page.

        Returns:
            (full_text, page_metadata_list)
        """
        pages_meta = []
        full_text_parts = []
        document = self.document_reader.read(pdf_path, build_id=self.config.build_id)
        logger.info(f"  Extracting text from {document.total_pages} pages...")

        for page in document.pages:
            text = page.normalized_text
            if len(text.strip()) >= self.config.min_content_chars:
                full_text_parts.append(text)
                pages_meta.append({
                    "page": page.physical_page_number,
                    "char_count": len(text),
                    "document_id": document.document_id,
                    "pdf_sha256": document.pdf_sha256,
                    "parser_version": document.parser_version,
                    "config_hash": document.config_hash,
                })

        full_text = "\n\n".join(full_text_parts)
        logger.info(f"  Extracted {len(full_text):,} chars from {len(pages_meta)} content pages")
        return full_text, pages_meta

    # ── Process a Single PDF ──

    def process_pdf(self, pdf_path: str) -> Dict:
        """
        Process a single PDF: extract → detect sections → chunk → extract triples → ingest.

        Returns processing statistics dict.
        """
        filename = os.path.basename(pdf_path)
        with open(pdf_path, "rb") as pdf_file:
            document_sha256 = hashlib.sha256(pdf_file.read()).hexdigest()
        registered_company_id, company_resolution_source = self._resolve_registered_company(
            filename=filename,
            document_sha256=document_sha256,
        )
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING: {filename}")
        logger.info(f"{'='*60}")

        # Extract year from filename
        year_match = re.search(r"(20\d{2})", pdf_path)
        year = self.config.year_override or (int(year_match.group(1)) if year_match else 2024)

        # Build the standard document layer once.  The downstream extraction
        # loop consumes these page texts; it does not silently create a second
        # independent text representation.
        canonical_document = self.document_reader.read(
            pdf_path, build_id=self.config.build_id
        )
        canonical_document.assert_coverage()

        # Step 1: Open PDF and detect sections
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = canonical_document.total_pages
            logger.info(f"  Total pages: {total_pages}")

            # ISSUE-FIX #2: Detect document type from first pages.
            # Non-SEC files (Seeking Alpha transcripts, GPU whitepapers, blog posts)
            # have different writing styles and produce low-quality triples.
            # Cache page text once for the rest of this run.  Apart from
            # avoiding repeated PDF decoding, this gives us a deterministic
            # page-level coverage ledger: every page is accounted for, even
            # when it is outside the extraction scope.
            page_texts = [page.normalized_text for page in canonical_document.pages]
            first_pages_text = "\n".join(page_texts[:3])
            doc_type = SectionDetector.detect_document_type(first_pages_text)
            logger.info(f"  Document type: {doc_type}")

            if doc_type == "NON_SEC":
                logger.warning(
                    f"  SKIPPING non-SEC document: {filename}. "
                    f"This file does not contain SEC filing markers and would "
                    f"produce low-quality extraction."
                )
                return {
                    "filename": filename,
                    "status": "skipped_non_sec",
                    "doc_type": doc_type,
                    "triples": 0,
                    "build_id": self.config.build_id,
                    "document_layer": {
                        "schema": DOCUMENT_LAYER_SCHEMA,
                        "coverage": canonical_document.coverage(),
                    },
                }
            if doc_type not in ("SEC_10K", "SEC_10Q"):
                logger.warning(
                    f"  SKIPPING non-primary SEC document ({doc_type}): {filename}. "
                    f"Only 10-K and 10-Q filings are processed for reliable extraction."
                )
                return {
                    "filename": filename,
                    "status": "skipped_sec_other",
                    "doc_type": doc_type,
                    "triples": 0,
                    "build_id": self.config.build_id,
                    "document_layer": {
                        "schema": DOCUMENT_LAYER_SCHEMA,
                        "coverage": canonical_document.coverage(),
                    },
                }

            # Detect sections
            self.section_detector.scan(pdf)
            for canonical_page in canonical_document.pages:
                canonical_page.section = self.section_detector.get_page_section_label(
                    canonical_page.physical_page_number
                )
            logger.info(f"  Sections detected:\n{self.section_detector.describe()}")

            # Get target pages (0-indexed)
            target_indices = self.section_detector.get_target_pages(pdf)
            logger.info(f"  Target pages: {len(target_indices)} pages in scope")

            if not target_indices:
                logger.warning(f"  No target pages found. Skipping {filename}")
                return {
                    "filename": filename,
                    "status": "skipped",
                    "triples": 0,
                    "build_id": self.config.build_id,
                    "document_layer": {
                        "schema": DOCUMENT_LAYER_SCHEMA,
                        "coverage": canonical_document.coverage(),
                    },
                }

            # Step 2: Extract text from target pages
            all_triples = []
            total_ingested = 0
            pending_batches = []
            pending_table_candidates: List[Dict] = []
            rejected_table_candidates: List[Dict] = []
            page_stats = []
            target_index_set = set(target_indices)
            coverage_pages = []
            coverage_by_page = {}
            for page_index, page_text in enumerate(page_texts):
                page_num = page_index + 1
                text_chars = len(page_text.strip())
                section_id = self.section_detector.get_section_for_page(page_num)
                selected = page_index in target_index_set
                record = {
                    "page": page_num,
                    "section_id": section_id,
                    "section": self.section_detector.get_page_section_label(page_num),
                    "selected_for_extraction": selected,
                    "section_heading_line": self.section_detector.get_section_start_line(page_num),
                    "text_chars": text_chars,
                    "extraction_text_chars": text_chars,
                    "meets_min_content": text_chars >= self.config.min_content_chars,
                    "parse_status": "text_extracted" if text_chars else "empty_text",
                    "exclusion_reason": (
                        None if selected else "section_not_in_target_scope"
                    ),
                    "chunk_count": 0,
                    "raw_candidate_triples": 0,
                    "filtered_candidate_triples": 0,
                    "deduplicated_triples": 0,
                    "strict_triples": 0,
                    "evidence_spans": 0,
                    "llm_enabled": False,
                    "llm_calls": 0,
                    "llm_accepted_triples": 0,
                    "table_candidates": 0,
                    "table_pending_candidates": 0,
                    "table_rejected_candidates": 0,
                    "table_accepted_candidates": 0,
                    "table_strict_triples": 0,
                    "filter_rejection_counts": {},
                }
                if selected and text_chars < self.config.min_content_chars:
                    record["parse_status"] = "below_min_content"
                    record["exclusion_reason"] = "below_min_content_threshold"
                coverage_pages.append(record)
                coverage_by_page[page_num] = record
        extraction_method_counts = {
            "LLM_EXTRACTION": 0,
            "RULE_EXTRACTION": 0,
            "TABLE_EXTRACTION": 0,
        }

        self.ingestor.reset_batch_state()

        # The discovery pass above intentionally closes its PDF handle before
        # remote extraction begins. Reopen it for table geometry; retaining a
        # pdfplumber Page from the closed handle makes extract_tables() fail
        # silently and would drop all deterministic metric disclosures.
        extraction_pdf = pdfplumber.open(pdf_path)
        for idx in target_indices:
                page = extraction_pdf.pages[idx]
                page_num = idx + 1
                section_id = self.section_detector.get_section_for_page(page_num)
                section_label = self.section_detector.get_page_section_label(page_num)

                raw_text = page_texts[idx]
                # On a page where a new section heading appears after a long
                # carry-over block, keep only the text from that heading
                # onward. This preserves the page while preventing the
                # previous section's table/narrative from entering the chunk.
                text = self.section_detector.get_page_extraction_text(
                    page_num, raw_text
                )
                coverage_record = coverage_by_page[page_num]
                coverage_record["extraction_text_chars"] = len(text.strip())
                if len(text.strip()) < self.config.min_content_chars:
                    continue

                # Step 3: Extract triples — dual strategy:
                #   (a) Rule engine: run on FULL page text for cross-paragraph context
                #   (b) LLM engine: run per chunk (token limits), larger chunks for semantics
                page_triples = []

                # Quantitative table rows need a dedicated path. They are
                # disclosures, not causal prose, so they become
                # REPORTS_METRIC facts with value/unit/period metadata.
                table_triples = []
                if section_id in {"MD_AND_A", "FINANCIAL_STATEMENTS"}:
                    table_triples = extract_financial_table_triples(
                        page,
                        text,
                        year,
                        company_id=registered_company_id,
                        document_id=os.path.basename(pdf_path),
                        report_period=f"FY{year}",
                    )
                    table_triples = [
                        self._annotate_table_candidate(
                            triple,
                            filename=filename,
                            page_num=page_num,
                            document_sha256=document_sha256,
                            company_resolution_source=company_resolution_source,
                        )
                        for triple in table_triples
                    ]
                    raw_table_candidates = list(table_triples)
                    verified_table_triples = []
                    for triple in raw_table_candidates:
                        if triple.get("subject_resolution_status") == "PENDING_COMPANY_REVIEW":
                            pending_table_candidates.append(
                                self._table_review_record(
                                    triple,
                                    review_status="UNLABELED_CANDIDATE",
                                    reason="MISSING_VERIFIED_COMPANY_ID",
                                )
                            )
                        else:
                            verified_table_triples.append(triple)
                    # Pending identity candidates remain visible in the
                    # annotation queue but never enter filter_triples or the
                    # active graph as company facts.
                    table_triples = verified_table_triples
                    for triple in table_triples:
                        triple["_source"] = "table"
                        triple["statement_type"] = section_id
                    page_triples.extend(table_triples)
                else:
                    raw_table_candidates = []

                # ── Rule extraction on FULL page text ──
                # P0-FIX: Rules need the full page to find entities that co-occur
                # across paragraph boundaries within the same page. Chunking was
                # breaking causal chains (e.g., risk mentioned in ¶1, metric in ¶3).
                if self.config.use_rules:
                    rule_triples = self.extractor.rule_extract(text)
                    for t in rule_triples:
                        t["_source"] = "rule"
                    page_triples.extend(rule_triples)

                # ── LLM extraction: ONLY for RISK_FACTORS pages ──
                # Cloud LLM (Groq) has 100K TPD limit. Using it only on
                # risk pages (~5/filing) conserves quota while still capturing
                # the highest-density causal language.
                # LLM extraction runs on every selected target-section page
                # by default; --llm_risk_only preserves the quota-saving mode.
                is_risk_page = ("Item 1A" in section_label or
                                "Risk Factor" in section_label)
                llm_enabled_for_page = (
                    self.config.use_llm
                    and (self.config.llm_on_all_target_pages or is_risk_page)
                )
                chunks = self.splitter.split_text(text) if llm_enabled_for_page else []
                llm_calls_before = self.extractor.llm_calls
                llm_accepted_before = self.extractor.llm_accepted_triples
                if llm_enabled_for_page:
                    for chunk_index, chunk in enumerate(chunks):
                        llm_triples = self.extractor.llm_extract(chunk)
                        for t in llm_triples:
                            if isinstance(t, dict):
                                t["_source"] = "llm"
                                t["_chunk_index"] = chunk_index
                                page_triples.append(t)

                raw_candidate_count = len(page_triples)

                # Filter and canonicalize
                page_triples = self.extractor.filter_triples(page_triples, text)
                filtered_candidate_count = len(page_triples)
                filter_rejection_counts = _get_filter_rejection_counts(self.extractor)

                accepted_table_candidate_ids = {
                    str(triple.get("candidate_id"))
                    for triple in page_triples
                    if triple.get("_source") == "table" and triple.get("candidate_id")
                }
                rejected_on_filter = [
                    triple for triple in table_triples
                    if str(triple.get("candidate_id")) not in accepted_table_candidate_ids
                ]
                for triple in rejected_on_filter:
                    rejected_table_candidates.append(
                        self._table_review_record(
                            triple,
                            review_status="REJECTED_FILTER",
                            reason="TABLE_CANDIDATE_FILTER_REJECTED",
                        )
                    )

                for triple in page_triples:
                    evidence_start, evidence_end = self._evidence_span(
                        text, triple.get("evidence_sentence", "")
                    )
                    if evidence_start is not None:
                        triple["evidence_char_start"] = evidence_start
                        triple["evidence_char_end"] = evidence_end
                    if triple.get("_chunk_index") is None and chunks:
                        evidence = triple.get("evidence_sentence", "")
                        triple["_chunk_index"] = next(
                            (
                                chunk_index
                                for chunk_index, chunk in enumerate(chunks)
                                if self._evidence_span(chunk, evidence)[0] is not None
                            ),
                            0,
                        )

                # Deduplicate within page — LLM first (higher semantic quality),
                # then rules fill gaps (P1-FIX #3)
                seen_keys = set()
                unique_triples = []
                llm_triples = [t for t in page_triples if t.get("_source") == "llm"]
                rule_triples = [t for t in page_triples if t.get("_source") == "rule"]
                table_triples = [t for t in page_triples if t.get("_source") == "table"]
                for t in table_triples + llm_triples + rule_triples:
                    key = (
                        str(t.get("source", "")),
                        str(t.get("relation", "")),
                        str(t.get("target", "")),
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        source_marker = t.pop("_source", None)
                        t["extraction_method"] = {
                            "llm": "LLM_EXTRACTION",
                            "table": "TABLE_EXTRACTION",
                        }.get(source_marker, "RULE_EXTRACTION")
                        extraction_method_counts[t["extraction_method"]] += 1
                        unique_triples.append(t)

                # Stable TextUnit identity follows the same contract used by
                # the vector index: filing + page + chunk.  The claim keeps
                # this metadata so a later audit can join KG evidence back to
                # the exact extraction unit without storing raw chunk text in
                # Neo4j.
                chunk_ids = [
                    f"{filename}:{page_num}:{chunk_index}"
                    for chunk_index in range(len(chunks))
                ]
                for triple in unique_triples:
                    evidence_start = triple.get("evidence_char_start")
                    evidence_chunk = int(triple.get("_chunk_index", 0) or 0)
                    if evidence_start is not None and chunks and "_chunk_index" not in triple:
                        running = 0
                        for chunk_index, chunk in enumerate(chunks):
                            if running <= evidence_start <= running + len(chunk):
                                evidence_chunk = chunk_index
                                break
                            running += max(len(chunk) - self.splitter.chunk_overlap, 1)
                    triple["chunk_id"] = (
                        chunk_ids[evidence_chunk] if chunk_ids else f"{filename}:{page_num}:0"
                    )
                    # Keep source identity on the captured candidate itself.
                    # The Neo4j writer already receives page/year as parallel
                    # batch arguments, but an isolated build must be able to
                    # materialize the same evidence without reconstructing
                    # that positional relationship from logs.
                    triple["source_filing"] = filename
                    triple["source_page"] = page_num
                    triple["filing_year"] = year
                    triple["document_sha256"] = document_sha256
                    triple.pop("_chunk_index", None)

                coverage_record.update({
                    "parse_status": "processed",
                    "chunk_count": len(chunks),
                    "raw_candidate_triples": raw_candidate_count,
                    "filtered_candidate_triples": filtered_candidate_count,
                    "deduplicated_triples": len(unique_triples),
                    "chunk_ids": chunk_ids,
                    "strict_triples": len(unique_triples),
                    "evidence_spans": sum(
                        1 for triple in unique_triples
                        if triple.get("evidence_char_start") is not None
                    ),
                    "llm_enabled": llm_enabled_for_page,
                    "llm_calls": self.extractor.llm_calls - llm_calls_before,
                    "llm_accepted_triples": (
                        self.extractor.llm_accepted_triples - llm_accepted_before
                    ),
                    "table_candidates": len(raw_table_candidates),
                    "table_pending_candidates": sum(
                        1 for candidate in pending_table_candidates
                        if candidate.get("source_page") == page_num
                    ),
                    "table_rejected_candidates": len(rejected_on_filter),
                    "table_accepted_candidates": len(accepted_table_candidate_ids),
                    "table_strict_triples": sum(
                        1 for triple in unique_triples
                        if triple.get("extraction_method") == "TABLE_EXTRACTION"
                    ),
                    "filter_rejection_counts": dict(filter_rejection_counts),
                    "exclusion_reason": None,
                })

                # Stage the batch in memory.  Replacement is deliberately
                # deferred until the entire filing has passed extraction and
                # the optional LLM quality gate, preventing data loss when a
                # provider times out midway through a rebuild.
                if unique_triples:
                    pending_batches.append({
                        "triples": unique_triples,
                        "page": page_num,
                        "year": year,
                        "section": section_label,
                    })
                    logger.info(
                        f"  Page {page_num} [{section_label}]: "
                        f"{len(unique_triples)} triples staged"
                    )

                page_stats.append({
                    "page": page_num,
                    "section": section_label,
                    "text_chars": len(text),
                    "rule_candidates": len(rule_triples) if self.config.use_rules else 0,
                    "llm_enabled": llm_enabled_for_page,
                    "llm_calls": self.extractor.llm_calls - llm_calls_before,
                    "llm_accepted_triples": (
                        self.extractor.llm_accepted_triples - llm_accepted_before
                    ),
                    "strict_triples": len(unique_triples),
                    "evidence_spans": sum(
                        1 for triple in unique_triples
                        if triple.get("evidence_char_start") is not None
                    ),
                    "chunk_count": len(chunks),
                    "raw_candidate_triples": raw_candidate_count,
                    "filtered_candidate_triples": filtered_candidate_count,
                    "deduplicated_triples": len(unique_triples),
                    "table_candidates": len(raw_table_candidates),
                    "table_pending_candidates": sum(
                        1 for candidate in pending_table_candidates
                        if candidate.get("source_page") == page_num
                    ),
                    "table_rejected_candidates": len(rejected_on_filter),
                    "table_accepted_candidates": len(accepted_table_candidate_ids),
                    "table_strict_triples": sum(
                        1 for triple in unique_triples
                        if triple.get("extraction_method") == "TABLE_EXTRACTION"
                    ),
                    "filter_rejection_counts": dict(filter_rejection_counts),
                })

                all_triples.extend(unique_triples)
        extraction_pdf.close()

        extraction_stats = self.extractor.get_llm_stats()
        logger.info("  LLM extraction stats: %s", extraction_stats)
        if self.config.require_llm:
            if not self.config.use_llm or not self.extractor.llm_available:
                raise RuntimeError(
                    "LLM is required for this rebuild, but no configured "
                    "provider is available. Existing filing was preserved."
                )
            if extraction_stats["calls"] == 0:
                raise RuntimeError(
                    "LLM is required but no LLM extraction calls were made. "
                    "Existing filing was preserved."
                )
            if extraction_stats["failures"] > 0:
                raise RuntimeError(
                    "LLM extraction had failed calls; refusing to replace "
                    f"the filing: {extraction_stats}. Existing filing was preserved."
                )
            if extraction_stats["accepted_triples"] == 0:
                raise RuntimeError(
                    "LLM returned no evidence-grounded triples; refusing to "
                    "replace the filing. Existing filing was preserved."
                )

        if self.config.dry_run:
            logger.info(
                "  DRY RUN: extraction completed; skipping Neo4j replacement, "
                "ingestion, and post-processing."
            )
        else:
            # Commit phase: only after the whole PDF is staged and, when
            # enabled, the LLM quality gate has passed. Re-check the driver
            # because a long extraction run can outlive Neo4j's routing
            # connection. A failed reconnect must occur before any filing
            # replacement or write.
            self._ensure_commit_connection()

            # Ingestion must also work in the deterministic rules/tables-only mode.
            if self.config.replace_existing_filing:
                self.ingestor.replace_filing(filename)

            self.ingestor.create_document_node(
                filename=filename,
                doc_type="10-K" if "10-K" in filename else "10-Q",
                fiscal_year=year,
                total_pages=total_pages,
                document_sha256=document_sha256,
            )
            for batch in pending_batches:
                ingested = self.ingestor.ingest_batch(
                    triples=batch["triples"],
                    filename=filename,
                    pages=[batch["page"]] * len(batch["triples"]),
                    year=batch["year"],
                    sections=[batch["section"]] * len(batch["triples"]),
                    document_sha256=document_sha256,
                )
                total_ingested += ingested
                if ingested > 0:
                    logger.info(
                        f"  Page {batch['page']} [{batch['section']}]: "
                        f"{len(batch['triples'])} triples, {ingested} ingested"
                    )

        # Step 6: Log filing statistics
        batch_stats = self.ingestor.get_stats()
        logger.info(f"\n  ── Filing Summary: {filename} ──")
        logger.info(f"  Total triples extracted: {len(all_triples)}")
        logger.info(f"  Total ingested: {total_ingested}")
        for rel, count in sorted(batch_stats["relations"].items(), key=lambda x: -x[1]):
            logger.info(f"    {rel}: {count}")

        selected_pages = [
            record for record in coverage_pages
            if record["selected_for_extraction"]
        ]
        section_page_counts = {}
        for record in coverage_pages:
            section_id = record.get("section_id") or "UNCLASSIFIED"
            section_page_counts[section_id] = section_page_counts.get(section_id, 0) + 1
        coverage_ledger = {
            "contract": (
                "All pages are parsed and accounted for; only pages in target_sections "
                "are sent to extraction. Strict triples require verbatim evidence."
            ),
            "target_sections": list(self.config.target_sections),
            "total_pages": total_pages,
            "pages_with_text": sum(r["text_chars"] > 0 for r in coverage_pages),
            "pages_meeting_min_content": sum(
                r["meets_min_content"] for r in coverage_pages
            ),
            "selected_pages": len(selected_pages),
            "selected_content_pages": sum(
                r["selected_for_extraction"]
                and r["meets_min_content"]
                for r in coverage_pages
            ),
            "pages_with_strict_triples": sum(
                r["strict_triples"] > 0 for r in coverage_pages
            ),
            "strict_triples": sum(r["strict_triples"] for r in coverage_pages),
            "raw_candidate_triples": sum(
                r["raw_candidate_triples"] for r in coverage_pages
            ),
            "filtered_candidate_triples": sum(
                r["filtered_candidate_triples"] for r in coverage_pages
            ),
            "evidence_spans": sum(r["evidence_spans"] for r in coverage_pages),
            "section_page_counts": section_page_counts,
            "pages": coverage_pages,
        }

        table_quality = {
            "schema_version": "table-quality/v1",
            "candidate_count": sum(
                page.get("table_candidates", 0) for page in coverage_pages
            ),
            "accepted_count": sum(
                page.get("table_accepted_candidates", 0) for page in coverage_pages
            ),
            "pending_count": len(pending_table_candidates),
            "rejected_count": len(rejected_table_candidates),
            "conservation_holds": (
                sum(page.get("table_candidates", 0) for page in coverage_pages)
                == sum(page.get("table_accepted_candidates", 0) for page in coverage_pages)
                + len(pending_table_candidates)
                + len(rejected_table_candidates)
            ),
            "pending_queue_path": self.config.pending_table_queue_path,
            "pending_candidates": pending_table_candidates,
            "rejected_candidates": rejected_table_candidates,
        }

        return {
            "filename": filename,
            "document_sha256": document_sha256,
            "build_id": self.config.build_id,
            "document_layer": {
                "schema": DOCUMENT_LAYER_SCHEMA,
                "document_id": canonical_document.document_id,
                "parser_version": canonical_document.parser_version,
                "config_hash": canonical_document.config_hash,
                "coverage": canonical_document.coverage(),
            },
            "year": year,
            "total_pages": total_pages,
            "target_pages": len(target_indices),
            "triples_extracted": len(all_triples),
            "triples_ingested": total_ingested,
            "llm": self.extractor.get_llm_stats(),
            "llm_cache": self.extractor.response_cache.stats(),
            "llm_provider": getattr(self.extractor.llm, "provider", None),
            "llm_model": getattr(self.extractor.llm, "default_model", None),
            "prompt_version": getattr(self.ingestor, "prompt_version", None),
            "extraction_run_id": getattr(self.ingestor, "run_id", None),
            "extraction_method_counts": extraction_method_counts,
            "table_quality": table_quality,
            "pending_table_candidates": pending_table_candidates,
            "accepted_triples": all_triples if self.config.capture_triples else None,
            "page_stats": page_stats,
            "coverage_ledger": coverage_ledger,
            "pages_with_strict_triples": sum(
                1 for page in page_stats if page["strict_triples"] > 0
            ),
            "relations": dict(batch_stats["relations"]),
            "entities": dict(batch_stats["entities"]),
            "dry_run": self.config.dry_run,
            "status": "completed",
        }

    # ── Batch Processing ──

    def process_batch(
        self,
        pdf_dir: str = None,
        pdf_paths: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Process all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files

        Returns:
            List of per-file processing statistics
        """
        if pdf_paths:
            pdf_files = [os.path.abspath(path) for path in pdf_paths]
            missing = [path for path in pdf_files if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(
                    "Explicit PDF path(s) not found: " + ", ".join(missing)
                )
            logger.info("Explicit PDF files: %s", pdf_files)
        else:
            pdf_dir = pdf_dir or self.config.pdf_dir
            pdf_dir = os.path.abspath(pdf_dir)
            logger.info(f"PDF directory: {pdf_dir}")

            if not os.path.isdir(pdf_dir):
                logger.error(f"Directory not found: {pdf_dir}")
                return []

            pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
        if not pdf_files:
            logger.error(f"No PDF files found in {pdf_dir}")
            return []

        if len(pdf_files) > 1 and not self.config.allow_multiple_pdfs:
            raise RuntimeError(
                "Single-PDF stabilization mode is active: found "
                f"{len(pdf_files)} PDFs in {pdf_dir}. Keep exactly one PDF "
                "until the ontology and evidence contract are validated, or "
                "rerun with --allow_multiple_pdfs."
            )

        logger.info(f"Found {len(pdf_files)} PDF file(s)")
        results = []

        # Extraction-only validation intentionally has no external graph side
        # effect.  In normal mode connect before processing and keep the
        # existing post-processing behavior unchanged.
        if not self.config.dry_run and not self.ingestor.connect():
            logger.error("Failed to connect to Neo4j. Aborting.")
            return []

        try:
            for i, pdf_path in enumerate(pdf_files, 1):
                logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
                try:
                    result = self.process_pdf(pdf_path)
                    result["pending_table_queue_written"] = self._write_pending_table_queue(
                        result.get("pending_table_candidates", [])
                    )
                    results.append(result)
                except Exception as e:
                    logger.error(f"ERROR processing {pdf_path}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    if isinstance(e, LLMResponseCacheError):
                        raise
                    results.append({
                        "filename": os.path.basename(pdf_path),
                        "status": "error",
                        "error": str(e),
                    })

            if not self.config.dry_run:
                # Post-processing
                logger.info("\n" + "=" * 60)
                logger.info("POST-PROCESSING")
                logger.info("=" * 60)
                self.ingestor.deduplicate_relations()
                self.ingestor.enforce_hubness(max_out_edges=30)

                # Final stats
                final_stats = self.ingestor.get_stats()
                logger.info(f"\nFINAL STATISTICS:")
                logger.info(f"  Total unique triples: {final_stats['total_relations']}")
                logger.info(f"  Total entities: {final_stats['total_entities']}")
                for rel, count in sorted(final_stats["relations"].items(), key=lambda x: -x[1]):
                    logger.info(f"    {rel}: {count}")

        finally:
            self.ingestor.close()

        self.stats = {"files": results}
        return results

    # ── Export ──

    def save_stats(self, output_path: str = "pipeline_stats.json"):
        """Save pipeline statistics to JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.stats, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Pipeline stats saved to {output_path}")


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Strategic-GraphRAG: PDF → Knowledge Graph Pipeline"
    )
    parser.add_argument(
        "--pdf_dir", type=str, default="data/pdfs",
        help="Directory containing SEC filing PDFs"
    )
    parser.add_argument(
        "--pdf", dest="pdf_paths", action="append", default=None,
        help="Process an explicit PDF path; repeat for multiple selected filings"
    )
    parser.add_argument(
        "--no_llm", action="store_true",
        help="Disable LLM extraction (rule-based only)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=2400,
        help="Text chunk size for LLM extraction"
    )
    parser.add_argument(
        "--llm_provider", type=str, default=None,
        choices=("gemini", "groq", "deepseek", "ollama", "local"),
        help="Override the provider from LLM_PROVIDER"
    )
    parser.add_argument(
        "--model_name", type=str, default=None,
        help="Optional provider-specific model override; otherwise use provider default"
    )
    parser.add_argument(
        "--company_id", type=str, default=None,
        help=(
            "Verified disclosure company ID from document registration only; "
            "omit to keep table facts pending human review"
        )
    )
    parser.add_argument(
        "--pending_table_queue", type=str, default=None,
        help=(
            "Optional JSONL path for unresolved table candidates; no queue is "
            "written unless this flag is supplied"
        )
    )
    parser.add_argument(
        "--document_registry", type=str, default=None,
        help=(
            "Optional JSON/JSONL filename+SHA256 registry; only verified "
            "company identities from a matching row are accepted"
        )
    )
    parser.add_argument(
        "--build_id", type=str, default=None,
        help="Shared build identity from a reconstruction manifest; omit only for unbound dry-runs",
    )
    parser.add_argument(
        "--require_llm", action="store_true",
        help="Abort before replacement if any LLM extraction call fails or yields no accepted triples"
    )
    parser.add_argument(
        "--llm_risk_only", action="store_true",
        help="Use LLM verification only on Risk Factors pages (legacy quota-saving mode)"
    )
    parser.add_argument(
        "--year", type=int, default=None,
        help="Override fiscal year for all PDFs"
    )
    parser.add_argument(
        "--allow_multiple_pdfs", action="store_true",
        help="Explicitly opt into multi-PDF ingestion after single-PDF validation"
    )
    parser.add_argument(
        "--replace_existing_filing", action="store_true",
        help="Delete only the same filing's evidence and edges before re-ingestion"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Extract and validate without connecting to Neo4j or changing the graph"
    )
    parser.add_argument(
        "--capture_triples", action="store_true",
        help="Retain accepted triples in the local stats artifact for an isolated staging build",
    )
    parser.add_argument(
        "--output_stats", type=str, default="pipeline_stats.json",
        help="Path to save processing statistics JSON"
    )

    args = parser.parse_args()

    config = PipelineConfig(
        pdf_dir=args.pdf_dir,
        chunk_size=args.chunk_size,
        use_llm=not args.no_llm,
        llm_provider=args.llm_provider,
        model_name=args.model_name,
        company_id=args.company_id,
        llm_on_all_target_pages=not args.llm_risk_only,
        require_llm=args.require_llm,
        allow_multiple_pdfs=args.allow_multiple_pdfs,
        replace_existing_filing=args.replace_existing_filing,
        year_override=args.year,
        dry_run=args.dry_run,
        capture_triples=args.capture_triples,
        pending_table_queue_path=args.pending_table_queue,
        document_registry_path=args.document_registry,
        build_id=args.build_id,
    )

    pipeline = KnowledgeGraphPipeline(config)
    results = pipeline.process_batch(args.pdf_dir, pdf_paths=args.pdf_paths)
    pipeline.save_stats(args.output_stats)

    # Print summary
    completed = [r for r in results if r.get("status") == "completed"]
    total_triples = sum(
        r.get("triples_extracted", 0) if args.dry_run
        else r.get("triples_ingested", 0)
        for r in completed
    )
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Files processed: {len(results)}")
    print(f"  Completed: {len(completed)}")
    label = "extracted (dry run)" if args.dry_run else "ingested"
    print(f"  Total triples {label}: {total_triples}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
