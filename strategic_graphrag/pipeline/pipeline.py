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

logger = logging.getLogger("Pipeline")


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
    require_llm: bool = False
    allow_multiple_pdfs: bool = False
    replace_existing_filing: bool = False
    year_override: Optional[int] = None


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
        )
        self.ingestor = GraphIngestor()
        # Persist the provider-safe values that the runtime actually resolved,
        # not potentially stale raw environment variables.
        self.ingestor.llm_provider = getattr(self.extractor.llm, "provider", "unknown")
        self.ingestor.llm_model = getattr(
            self.extractor.llm, "default_model", "unknown"
        )

        # Text splitter
        self.splitter = RecursiveTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Statistics
        self.stats: Dict = {}

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

    # ── PDF Text Extraction ──

    def extract_text_from_pdf(self, pdf_path: str) -> Tuple[str, List[Dict]]:
        """
        Extract text from a PDF file, page by page.

        Returns:
            (full_text, page_metadata_list)
        """
        pages_meta = []
        full_text_parts = []

        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
            logger.info(f"  Extracting text from {total} pages...")

            for i, page in enumerate(pdf.pages):
                page_num = i + 1
                text = page.extract_text() or ""
                if len(text.strip()) >= self.config.min_content_chars:
                    full_text_parts.append(text)
                    pages_meta.append({
                        "page": page_num,
                        "char_count": len(text),
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
        logger.info(f"\n{'='*60}")
        logger.info(f"PROCESSING: {filename}")
        logger.info(f"{'='*60}")

        # Extract year from filename
        year_match = re.search(r"(20\d{2})", pdf_path)
        year = self.config.year_override or (int(year_match.group(1)) if year_match else 2024)

        # Step 1: Open PDF and detect sections
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"  Total pages: {total_pages}")

            # ISSUE-FIX #2: Detect document type from first pages.
            # Non-SEC files (Seeking Alpha transcripts, GPU whitepapers, blog posts)
            # have different writing styles and produce low-quality triples.
            # Cache page text once for the rest of this run.  Apart from
            # avoiding repeated PDF decoding, this gives us a deterministic
            # page-level coverage ledger: every page is accounted for, even
            # when it is outside the extraction scope.
            page_texts = [page.extract_text() or "" for page in pdf.pages]
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
                }

            # Detect sections
            self.section_detector.scan(pdf)
            logger.info(f"  Sections detected:\n{self.section_detector.describe()}")

            # Get target pages (0-indexed)
            target_indices = self.section_detector.get_target_pages(pdf)
            logger.info(f"  Target pages: {len(target_indices)} pages in scope")

            if not target_indices:
                logger.warning(f"  No target pages found. Skipping {filename}")
                return {"filename": filename, "status": "skipped", "triples": 0}

            # Step 2: Extract text from target pages
            all_triples = []
            total_ingested = 0
            pending_batches = []
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
                    "table_strict_triples": 0,
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
                        page, text, year
                    )
                    for triple in table_triples:
                        triple["_source"] = "table"
                        triple["statement_type"] = section_id
                    page_triples.extend(table_triples)

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
                    "table_candidates": len(table_triples),
                    "table_strict_triples": sum(
                        1 for triple in unique_triples
                        if triple.get("extraction_method") == "TABLE_EXTRACTION"
                    ),
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
                    "table_candidates": len(table_triples),
                    "table_strict_triples": sum(
                        1 for triple in unique_triples
                        if triple.get("extraction_method") == "TABLE_EXTRACTION"
                    ),
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

        # Commit phase: only after the whole PDF is staged and, when enabled,
        # the LLM quality gate has passed. Ingestion must also work in the
        # deterministic rules/tables-only mode.
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

        return {
            "filename": filename,
            "document_sha256": document_sha256,
            "year": year,
            "total_pages": total_pages,
            "target_pages": len(target_indices),
            "triples_extracted": len(all_triples),
            "triples_ingested": total_ingested,
            "llm": self.extractor.get_llm_stats(),
            "llm_provider": getattr(self.extractor.llm, "provider", None),
            "llm_model": getattr(self.extractor.llm, "default_model", None),
            "prompt_version": getattr(self.ingestor, "prompt_version", None),
            "extraction_run_id": getattr(self.ingestor, "run_id", None),
            "extraction_method_counts": extraction_method_counts,
            "page_stats": page_stats,
            "coverage_ledger": coverage_ledger,
            "pages_with_strict_triples": sum(
                1 for page in page_stats if page["strict_triples"] > 0
            ),
            "relations": dict(batch_stats["relations"]),
            "entities": dict(batch_stats["entities"]),
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

        # Connect to Neo4j
        if not self.ingestor.connect():
            logger.error("Failed to connect to Neo4j. Aborting.")
            return []

        try:
            for i, pdf_path in enumerate(pdf_files, 1):
                logger.info(f"\n[{i}/{len(pdf_files)}] Processing: {os.path.basename(pdf_path)}")
                try:
                    result = self.process_pdf(pdf_path)
                    results.append(result)
                except Exception as e:
                    logger.error(f"ERROR processing {pdf_path}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    results.append({
                        "filename": os.path.basename(pdf_path),
                        "status": "error",
                        "error": str(e),
                    })

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
        llm_on_all_target_pages=not args.llm_risk_only,
        require_llm=args.require_llm,
        allow_multiple_pdfs=args.allow_multiple_pdfs,
        replace_existing_filing=args.replace_existing_filing,
        year_override=args.year,
    )

    pipeline = KnowledgeGraphPipeline(config)
    results = pipeline.process_batch(args.pdf_dir, pdf_paths=args.pdf_paths)
    pipeline.save_stats(args.output_stats)

    # Print summary
    completed = [r for r in results if r.get("status") == "completed"]
    total_triples = sum(r.get("triples_ingested", 0) for r in completed)
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE")
    print(f"{'='*60}")
    print(f"  Files processed: {len(results)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Total triples ingested: {total_triples}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
