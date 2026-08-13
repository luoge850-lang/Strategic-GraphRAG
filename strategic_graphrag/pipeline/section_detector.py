# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: SEC 10-K Section Detector
=============================================
Identifies and maps SEC filing sections to focus extraction on
relevant content (Item 1A Risk Factors, Item 7 MD&A, Item 8 Financials).

Supports 10-K, 10-Q, and Annual Report document structures.
"""

import re
import logging
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger("SectionDetector")

# =============================================================================
# SEC 10-K Standard Section Structure
# =============================================================================

SECTION_PATTERNS: Dict[str, List[str]] = {
    # NOTE: RISK_FACTORS must come BEFORE BUSINESS so "item 1a" matches before "item 1"
    "RISK_FACTORS": [
        "item 1a.", "item 1a ", "risk factors",
        "item 1a\n",
    ],
    "BUSINESS": [
        # Do not use the bare word "business": it occurs throughout the
        # filing and in the running "Table of Contents" header.
        "item 1.", "item 1 ", "item 1\n",
    ],
    "UNRESOLVED_STAFF_COMMENTS": [
        "item 1b.", "item 1b ", "unresolved staff comments",
    ],
    "PROPERTIES": [
        "item 2.", "item 2 ", "properties",
    ],
    "LEGAL_PROCEEDINGS": [
        "item 3.", "item 3 ", "legal proceedings",
    ],
    "MINE_SAFETY": [
        "item 4.", "item 4 ", "mine safety disclosures",
    ],
    "MARKET_FOR_REGISTRANTS_COMMON_EQUITY": [
        "item 5.", "item 5 ", "market for registrant",
    ],
    "SELECTED_FINANCIAL_DATA": [
        "item 6.", "item 6 ", "selected financial data", "[reserved]",
    ],
    "MD_AND_A": [
        "item 7.", "item 7 ",
        "item 7\n",
    ],
    "QUANTITATIVE_MARKET_RISK": [
        "item 7a.", "item 7a ", "quantitative and qualitative disclosures about market risk",
    ],
    "FINANCIAL_STATEMENTS": [
        "item 8.", "item 8 ", "item 8\n",
        # NVIDIA's 10-K explicitly places the detailed statements and notes
        # under Part IV, Item 15.  Treat that continuation as part of the
        # financial-statements extraction scope as well.
        "item 15.", "item 15 ", "item 15\n",
    ],
    "CHANGES_IN_ACCOUNTANTS": [
        "item 9.", "item 9 ", "changes in and disagreements with accountants",
    ],
    "CONTROLS_AND_PROCEDURES": [
        "item 9a.", "item 9a ", "controls and procedures",
    ],
    "OTHER_INFORMATION": [
        "item 9b.", "item 9b ", "other information",
    ],
    "DIRECTORS_AND_OFFICERS": [
        "item 10.", "item 10 ", "directors, executive officers",
    ],
    "EXECUTIVE_COMPENSATION": [
        "item 11.", "item 11 ", "executive compensation",
    ],
    "SECURITY_OWNERSHIP": [
        "item 12.", "item 12 ", "security ownership",
    ],
    "RELATED_TRANSACTIONS": [
        "item 13.", "item 13 ", "certain relationships and related transactions",
    ],
    "PRINCIPAL_ACCOUNTANT_FEES": [
        "item 14.", "item 14 ", "principal accountant fees",
    ],
    "EXHIBITS": [
        "exhibit index", "exhibit ",
    ],
    "SIGNATURES": [
        "signatures",
    ],
}

# Sections we care about for financial risk analysis
TARGET_SECTIONS = {
    "RISK_FACTORS",
    "MD_AND_A",
    "QUANTITATIVE_MARKET_RISK",
    "FINANCIAL_STATEMENTS",
    "BUSINESS",
}

# Section descriptions for display
SECTION_DISPLAY_NAMES = {
    "RISK_FACTORS": "Item 1A - Risk Factors",
    "MD_AND_A": "Item 7 - Management's Discussion and Analysis",
    "QUANTITATIVE_MARKET_RISK": "Item 7A - Quantitative and Qualitative Market Risk",
    "FINANCIAL_STATEMENTS": "Item 8/15 - Financial Statements and Schedules",
    "BUSINESS": "Item 1 - Business",
    "MARKET_FOR_REGISTRANTS_COMMON_EQUITY": "Item 5 - Market Information",
    "SELECTED_FINANCIAL_DATA": "Item 6 - Selected Financial Data",
    # Keep the display contract ASCII-only so reports render consistently on
    # Windows terminals and in the frontend.
    "EXHIBITS": "Exhibits",
    "SIGNATURES": "Signatures",
    "RISK_FACTORS": "Item 1A — Risk Factors",
    "MD_AND_A": "Item 7 — Management's Discussion and Analysis",
    "QUANTITATIVE_MARKET_RISK": "Item 7A — Quantitative & Qualitative Market Risk",
    "FINANCIAL_STATEMENTS": "Item 8 — Financial Statements",
    "BUSINESS": "Item 1 — Business",
    "MARKET_FOR_REGISTRANTS_COMMON_EQUITY": "Item 5 — Market Information",
    "SELECTED_FINANCIAL_DATA": "Item 6 — Selected Financial Data",
}

# Minimum characters for a page to be considered "content-bearing"
MIN_CONTENT_CHARS = 200

# Re-assert the display labels after the legacy mapping above.  The source
# file historically contained mojibake in these labels; keeping the runtime
# contract ASCII-only avoids corrupt section names in JSON and the frontend.
SECTION_DISPLAY_NAMES.update({
    "RISK_FACTORS": "Item 1A - Risk Factors",
    "MD_AND_A": "Item 7 - Management's Discussion and Analysis",
    "QUANTITATIVE_MARKET_RISK": "Item 7A - Quantitative and Qualitative Market Risk",
    "FINANCIAL_STATEMENTS": "Item 8/15 - Financial Statements and Schedules",
    "BUSINESS": "Item 1 - Business",
    "MARKET_FOR_REGISTRANTS_COMMON_EQUITY": "Item 5 - Market Information",
    "SELECTED_FINANCIAL_DATA": "Item 6 - Selected Financial Data",
    "EXHIBITS": "Exhibits",
    "SIGNATURES": "Signatures",
})

# Item numbers are much more reliable than searching for words such as
# "business" or "risk factors": those words appear in ordinary prose and in
# the repeated running header of NVIDIA's PDF export.
ITEM_TO_SECTION = {
    "1": "BUSINESS",
    "1a": "RISK_FACTORS",
    "1b": "UNRESOLVED_STAFF_COMMENTS",
    "2": "PROPERTIES",
    "3": "LEGAL_PROCEEDINGS",
    "4": "MINE_SAFETY",
    "5": "MARKET_FOR_REGISTRANTS_COMMON_EQUITY",
    "6": "SELECTED_FINANCIAL_DATA",
    "7": "MD_AND_A",
    "7a": "QUANTITATIVE_MARKET_RISK",
    "8": "FINANCIAL_STATEMENTS",
    "9": "CHANGES_IN_ACCOUNTANTS",
    "9a": "CONTROLS_AND_PROCEDURES",
    "9b": "OTHER_INFORMATION",
    "10": "DIRECTORS_AND_OFFICERS",
    "11": "EXECUTIVE_COMPENSATION",
    "12": "SECURITY_OWNERSHIP",
    "13": "RELATED_TRANSACTIONS",
    "14": "PRINCIPAL_ACCOUNTANT_FEES",
    # Part IV contains the detailed statements and notes referenced by Item 8.
    "15": "FINANCIAL_STATEMENTS",
    "16": "OTHER_INFORMATION",
}


def _heading_section_details(text: str) -> List[Tuple[str, int]]:
    """Return SEC Item headings that occur as standalone page lines.

    NVIDIA's filings repeat a Table of Contents header on every page.  A
    substring scan therefore turns ordinary references such as ``Item 1A``
    into false section boundaries.  Only a line beginning with an Item number
    is accepted here; the TOC is filtered separately by ``scan``.
    """
    sections: List[Tuple[str, int]] = []
    # Inspect the first text block of the page.  A real section heading can
    # appear after a short carry-over block (for example, the executive
    # officers table immediately before Item 1A), so a strict line-number
    # cutoff would silently lose valid section starts.
    for line_index, raw_line in enumerate((text or "").splitlines()[:80]):
        line = " ".join(raw_line.replace("\u00a0", " ").split())
        normalized = line.lower().rstrip(" .:")
        if normalized == "exhibit index" or re.match(r"^exhibit\s+\d", normalized):
            return [("EXHIBITS", line_index)]
        if normalized == "signatures":
            return [("SIGNATURES", line_index)]
        match = re.match(
            r"^(?:part\s+(?:i|ii|iii|iv)\s+)?item\s+([0-9]{1,2}[a-z]?)\s*[.:]?\s+",
            line,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        section_id = ITEM_TO_SECTION.get(match.group(1).lower())
        if section_id and section_id not in {item[0] for item in sections}:
            # A page such as the start of Item 8 may also list Item 9 and
            # Item 9A immediately below it.  Only the first heading starts a
            # span; later headings belong to the next page boundary.
            return [(section_id, line_index)]
    return sections


def _heading_section_ids(text: str) -> List[str]:
    """Backward-compatible section-only view used by lightweight callers."""
    return [section_id for section_id, _line_index in _heading_section_details(text)]


@dataclass
class SectionSpan:
    """A detected section spanning a range of pages."""
    section_id: str
    display_name: str
    start_page: int
    end_page: int
    confidence: float = 1.0


class SectionDetector:
    """
    Detects SEC 10-K/10-Q section boundaries using pattern matching
    on page text to restrict extraction to relevant financial sections.
    """

    def __init__(self, target_sections: Set[str] = None):
        self.target_sections = target_sections or TARGET_SECTIONS
        self.sections: List[SectionSpan] = []
        self._page_section_map: Dict[int, str] = {}
        self._page_section_start_line: Dict[int, int] = {}

    def scan(self, pdf) -> List[SectionSpan]:
        """
        Scan all pages of a PDF and identify section boundaries.

        Args:
            pdf: pdfplumber.PDF object

        Returns:
            List of detected SectionSpan objects
        """
        total_pages = len(pdf.pages)
        logger.info(f"Scanning {total_pages} pages for SEC section boundaries...")
        self.sections = []
        self._page_section_map = {}
        self._page_section_start_line = {}

        # First pass: detect section start pages
        detection_candidates: List[Tuple[int, str, float, int]] = []  # (page, section, confidence, pattern_len)
        start_line_candidates: Dict[int, int] = {}
        for i, page in enumerate(pdf.pages):
            page_num = i + 1  # 1-indexed
            text = (page.extract_text() or "").lower()
            if len(text.strip()) < 50:
                continue

            # Check first 500 chars for section headers (they appear at top)
            header_zone = text[:500]
            # NVIDIA's PDFs repeat a "Table of Contents" running header on
            # nearly every page.  The actual TOC page contains many Item
            # markers, so do not let it become a section boundary.  A normal
            # content page may mention one or two other items in prose, hence
            # the deliberately conservative threshold.
            toc_item_count = len(re.findall(r"\bitem\s+\d+[a-z]?\s*[.]?", header_zone))
            if "table of contents" in header_zone and toc_item_count >= 4:
                continue
            for section_id, heading_line in _heading_section_details(text):
                effective_page = page_num
                if effective_page <= total_pages:
                    detection_candidates.append(
                        (effective_page, section_id, 1.0, len(section_id))
                    )
                    start_line_candidates.setdefault(effective_page, heading_line)

        # Deduplicate and sort by page number
        # Prefer more specific (longer pattern) matches when multiple candidates
        seen_pages: Dict[int, Tuple[str, float, int]] = {}  # (section_id, confidence, pattern_len)
        for page_num, section_id, confidence, pattern_len in sorted(detection_candidates):
            if page_num not in seen_pages:
                seen_pages[page_num] = (section_id, confidence, pattern_len)
            else:
                old_conf = seen_pages[page_num][1]
                old_len = seen_pages[page_num][2]
                # Keep higher confidence; tie-break on longer pattern (more specific match)
                if confidence > old_conf or (confidence == old_conf and pattern_len > old_len):
                    seen_pages[page_num] = (section_id, confidence, pattern_len)

        # Repeated "Exhibit X" headings occur on almost every exhibit page and
        # "Signatures" may occur in both the filing body and an exhibit.  A
        # boundary is valid only when it transitions from a different section;
        # duplicate starts are discarded.
        compact_pages: List[int] = []
        for candidate_page in sorted(seen_pages.keys()):
            candidate_section = seen_pages[candidate_page][0]
            if compact_pages:
                previous_page = compact_pages[-1]
                previous_section = seen_pages[previous_page][0]
                if candidate_section == previous_section:
                    continue
            compact_pages.append(candidate_page)

        # Build section spans
        sorted_pages = compact_pages
        for i, start_page in enumerate(sorted_pages):
            section_id, confidence, _ = seen_pages[start_page]
            end_page = (sorted_pages[i + 1] - 1) if i + 1 < len(sorted_pages) else total_pages
            display_name = SECTION_DISPLAY_NAMES.get(section_id, section_id.replace("_", " ").title())

            span = SectionSpan(
                section_id=section_id,
                display_name=display_name,
                start_page=start_page,
                end_page=end_page,
                confidence=confidence,
            )
            self.sections.append(span)
            self._page_section_start_line[start_page] = start_line_candidates.get(
                start_page, 0
            )

            # Map each page in the span to this section
            for p in range(start_page, end_page + 1):
                if p not in self._page_section_map:
                    self._page_section_map[p] = section_id

        # Log results
        target_found = [s for s in self.sections if s.section_id in self.target_sections]
        logger.info(f"Detected {len(self.sections)} sections, {len(target_found)} are target sections:")
        for s in target_found:
            logger.info(f"  {s.display_name}: Pages {s.start_page}–{s.end_page} (confidence: {s.confidence:.0%})")

        return self.sections

    def get_section_for_page(self, page_num: int) -> Optional[str]:
        """Get the section ID for a given page number."""
        return self._page_section_map.get(page_num)

    def is_target_page(self, page_num: int) -> bool:
        """Check if a page belongs to a target section."""
        section = self.get_section_for_page(page_num)
        return section in self.target_sections if section else False

    def get_section_start_line(self, page_num: int) -> int:
        """Return the line containing the section heading on a boundary page."""
        return self._page_section_start_line.get(page_num, 0)

    def is_mixed_boundary_page(self, page_num: int) -> bool:
        """Whether a section heading appears after substantial carry-over text."""
        return self.get_section_start_line(page_num) >= 30

    def get_page_extraction_text(self, page_num: int, text: str) -> str:
        """Drop page content before a late section heading."""
        start_line = self.get_section_start_line(page_num)
        lines = str(text or "").splitlines()
        if start_line < 30 or start_line >= len(lines):
            return str(text or "")
        return "\n".join(lines[start_line:]).strip()

    def get_target_pages(self, pdf) -> List[int]:
        """
        Get list of page indices (0-indexed) that belong to target sections.
        If no sections were detected, returns a broad range around detected content.
        """
        if not self._page_section_map:
            return list(range(len(pdf.pages)))

        # Primary: pages in target sections
        target_pages = [
            p - 1 for p, sec in self._page_section_map.items()
            if sec in self.target_sections
        ]

        if target_pages:
            return sorted(set(target_pages))

        # Fallback: pages 1-80 (typical risk + MD&A range in 10-K)
        fallback_end = min(80, len(pdf.pages))
        logger.warning(f"No target sections detected. Falling back to pages 1-{fallback_end}.")
        return list(range(fallback_end))

    def get_page_section_label(self, page_num: int) -> str:
        """Get human-readable section label for a page."""
        section_id = self.get_section_for_page(page_num)
        if section_id:
            return SECTION_DISPLAY_NAMES.get(section_id, section_id)
        return "Other"

    def describe(self) -> str:
        """Return a human-readable summary of detected sections."""
        if not self.sections:
            return "No sections detected."
        lines = []
        for s in self.sections:
            marker = "★" if s.section_id in self.target_sections else " "
            lines.append(f"  {marker} {s.display_name}: Pages {s.start_page}–{s.end_page}")
        return "\n".join(lines)

    @property
    def target_section_count(self) -> int:
        """Number of detected target sections."""
        return sum(1 for s in self.sections if s.section_id in self.target_sections)

    @property
    def total_pages_in_scope(self) -> int:
        """Total pages covered by target sections."""
        return sum(
            s.end_page - s.start_page + 1
            for s in self.sections
            if s.section_id in self.target_sections
        )

    # ═══════════════════════════════════════════════════════════
    # Document Type Detection (Issue-FIX #2)
    # ═══════════════════════════════════════════════════════════

    # SEC filings have distinctive markers in the first few pages
    SEC_MARKERS = [
        "united states securities and exchange commission",
        "securities and exchange commission",
        "washingtond.c.20549", "washington, d.c. 20549",
    ]

    FORM_MARKERS = [
        "form 10-k", "form 10-q", "form 8-k", "form s-1",
        "annual report pursuant to section 13",
        "quarterly report pursuant to section 13",
        "transition report pursuant to section 13",
    ]

    NON_SEC_MARKERS = [
        "seeking alpha", "earnings call transcript", "earnings conference call",
        "gpu architecture", "blog post", "white paper", "whitepaper",
        "press release", "news release",
    ]

    @staticmethod
    def detect_document_type(first_pages_text: str) -> str:
        """
        Classify document type from first few pages of extracted text.

        Returns:
            'SEC_10K', 'SEC_10Q', 'SEC_OTHER', or 'NON_SEC'
        """
        tl = first_pages_text.lower()

        # Check SEC markers FIRST — these are authoritative.
        # SEC filings may mention "press release" or "news" in their
        # corporate communications disclosure, which would be a false
        # positive if we checked NON_SEC markers first.
        has_sec_header = any(m in tl for m in SectionDetector.SEC_MARKERS)
        has_form = any(m in tl for m in SectionDetector.FORM_MARKERS)

        if has_sec_header or has_form:
            if "form 10-k" in tl or "annual report pursuant to section 13" in tl:
                return "SEC_10K"
            if "form 10-q" in tl or "quarterly report pursuant to section 13" in tl:
                return "SEC_10Q"
            return "SEC_OTHER"

        # No SEC markers — check if it's a known non-SEC document type
        for marker in SectionDetector.NON_SEC_MARKERS:
            if marker in tl:
                return "NON_SEC"

        # No SEC markers, no known non-SEC markers → unclear, default NON_SEC
        return "NON_SEC"

    @staticmethod
    def is_sec_filing(first_pages_text: str) -> bool:
        """Quick check: is this an SEC filing?"""
        return SectionDetector.detect_document_type(first_pages_text) != "NON_SEC"
