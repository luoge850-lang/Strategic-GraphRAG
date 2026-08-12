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
    target_sections: Tuple[str, ...] = ("RISK_FACTORS", "MD_AND_A", "BUSINESS")
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
            first_pages_text = ""
            for i in range(min(3, total_pages)):
                first_pages_text += (pdf.pages[i].extract_text() or "") + "\n"
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
            extraction_method_counts = {
                "LLM_EXTRACTION": 0,
                "RULE_EXTRACTION": 0,
            }

            self.ingestor.reset_batch_state()

            for idx in target_indices:
                page = pdf.pages[idx]
                page_num = idx + 1
                section_label = self.section_detector.get_page_section_label(page_num)

                text = page.extract_text() or ""
                if len(text.strip()) < self.config.min_content_chars:
                    continue

                # Step 3: Extract triples — dual strategy:
                #   (a) Rule engine: run on FULL page text for cross-paragraph context
                #   (b) LLM engine: run per chunk (token limits), larger chunks for semantics
                page_triples = []

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
                is_risk_page = ("Item 1A" in section_label or
                                "Risk Factor" in section_label)
                llm_enabled_for_page = (
                    self.config.use_llm
                    and (self.config.llm_on_all_target_pages or is_risk_page)
                )
                llm_calls_before = self.extractor.llm_calls
                llm_accepted_before = self.extractor.llm_accepted_triples
                if llm_enabled_for_page:
                    chunks = self.splitter.split_text(text)
                    for chunk in chunks:
                        llm_triples = self.extractor.llm_extract(chunk)
                        for t in llm_triples:
                            if isinstance(t, dict):
                                t["_source"] = "llm"
                                page_triples.append(t)

                # Filter and canonicalize
                page_triples = self.extractor.filter_triples(page_triples, text)

                for triple in page_triples:
                    evidence_start, evidence_end = self._evidence_span(
                        text, triple.get("evidence_sentence", "")
                    )
                    if evidence_start is not None:
                        triple["evidence_char_start"] = evidence_start
                        triple["evidence_char_end"] = evidence_end

                # Deduplicate within page — LLM first (higher semantic quality),
                # then rules fill gaps (P1-FIX #3)
                seen_keys = set()
                unique_triples = []
                llm_triples = [t for t in page_triples if t.get("_source") == "llm"]
                rule_triples = [t for t in page_triples if t.get("_source") == "rule"]
                for t in llm_triples + rule_triples:
                    key = (
                        str(t.get("source", "")),
                        str(t.get("relation", "")),
                        str(t.get("target", "")),
                    )
                    if key not in seen_keys:
                        seen_keys.add(key)
                        source_marker = t.pop("_source", None)
                        t["extraction_method"] = (
                            "LLM_EXTRACTION" if source_marker == "llm"
                            else "RULE_EXTRACTION"
                        )
                        extraction_method_counts[t["extraction_method"]] += 1
                        unique_triples.append(t)

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
                })

                all_triples.extend(unique_triples)

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

            # Commit phase: only after the whole PDF is staged and validated.
            if self.config.replace_existing_filing:
                self.ingestor.replace_filing(filename)

            self.ingestor.create_document_node(
                filename=filename,
                doc_type="10-K" if "10-K" in filename else "10-Q",
                fiscal_year=year,
                total_pages=total_pages,
            )
            for batch in pending_batches:
                ingested = self.ingestor.ingest_batch(
                    triples=batch["triples"],
                    filename=filename,
                    pages=[batch["page"]] * len(batch["triples"]),
                    year=batch["year"],
                    sections=[batch["section"]] * len(batch["triples"]),
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

        return {
            "filename": filename,
            "year": year,
            "total_pages": total_pages,
            "target_pages": len(target_indices),
            "triples_extracted": len(all_triples),
            "triples_ingested": total_ingested,
            "llm": self.extractor.get_llm_stats(),
            "extraction_method_counts": extraction_method_counts,
            "page_stats": page_stats,
            "pages_with_strict_triples": sum(
                1 for page in page_stats if page["strict_triples"] > 0
            ),
            "relations": dict(batch_stats["relations"]),
            "entities": dict(batch_stats["entities"]),
            "status": "completed",
        }

    # ── Batch Processing ──

    def process_batch(self, pdf_dir: str = None) -> List[Dict]:
        """
        Process all PDFs in a directory.

        Args:
            pdf_dir: Directory containing PDF files

        Returns:
            List of per-file processing statistics
        """
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
    results = pipeline.process_batch(args.pdf_dir)
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
