# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG v2.0: Graph-Constrained Query Understanding Module
====================================================================
Parses natural language financial questions into structured graph queries.

Implements: Source entity → Target entity → Relation constraint → Time window
Reference: Microsoft GraphRAG (Edge et al., 2024), Neuro-Symbolic AI patterns
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class StructuredQuery:
    """Parsed representation of a financial analysis question."""
    raw_question: str

    # Target entities to anchor the graph search
    source_entities: List[str] = field(default_factory=list)  # e.g., ["US_EXPORT_CONTROL"]
    target_entity: str = ""  # e.g., "REVENUE", "NET_INCOME"
    target_metric: str = ""  # e.g., "REVENUE"

    # Relation constraint
    analysis_type: str = ""  # IMPACT_ANALYSIS | MITIGATION_ANALYSIS | RISK_EXPOSURE | TEMPORAL_TREND
    causal_direction: str = ""  # FORWARD (source→target) | BACKWARD (target→source) | BIDIRECTIONAL

    # Time window
    fiscal_year_start: Optional[int] = None
    fiscal_year_end: Optional[int] = None
    temporal_required: bool = False

    # Graph search parameters
    max_hops: int = 4
    min_confidence: float = 0.3
    relation_types: List[str] = field(default_factory=list)
    exclude_relations: List[str] = field(default_factory=list)

    # Evidence requirements
    require_explicit_causal: bool = True  # Only CONFIRMED_CAUSAL + STRONG_ASSOCIATION
    require_multi_year: bool = False  # Need cross-year evidence

    def to_search_params(self) -> Dict:
        """Convert to dictionary for the path finder."""
        return {
            "anchor_entities": self.source_entities,
            "max_hops": self.max_hops,
            "relation_preference": self.relation_types,
            "year_constraint": self.fiscal_year_start,
            "temporal_required": self.temporal_required,
        }


# ═══════════════════════════════════════════════════════════════
# Intent → Structured Query Parser
# ═══════════════════════════════════════════════════════════════

# Financial metric keywords for target detection
FINANCIAL_METRICS_MAP = {
    "revenue": "REVENUE",
    "sales": "REVENUE",
    "income": "NET_INCOME",
    "net income": "NET_INCOME",
    "profit": "NET_INCOME",
    "margin": "GROSS_MARGIN",
    "gross margin": "GROSS_MARGIN",
    "operating margin": "OPERATING_MARGIN",
    "eps": "EARNINGS_PER_SHARE",
    "earnings per share": "EARNINGS_PER_SHARE",
    "cash flow": "CASH_FLOW",
    "free cash flow": "FREE_CASH_FLOW",
    "cost": "OPERATING_COST",
    "operating cost": "OPERATING_COST",
    "expense": "OPERATING_COST",
    "sales, general and administrative": "SG_AND_A_EXPENSE",
    "sales general and administrative": "SG_AND_A_EXPENSE",
    "sg&a": "SG_AND_A_EXPENSE",
    "sg and a": "SG_AND_A_EXPENSE",
    "market value": "MARKET_VALUE",
    "stock price": "MARKET_VALUE",
    "market cap": "MARKET_VALUE",
}

# Analysis type detection
ANALYSIS_PATTERNS = {
    "IMPACT_ANALYSIS": [
        r"how (do|does|did|will|would).+?(impact|affect|influence|change)",
        r"what (is|was|are|were) the (impact|effect|consequence)",
        r"(impact|effect) of .+? on",
        r"how (much|significantly).+?(affect|decrease|increase|reduce)",
    ],
    "MITIGATION_ANALYSIS": [
        r"how (do|does|did).+?(mitigate|address|manage|handle|deal with|counter|hedge)",
        r"what (strategies|measures|actions).+?(mitigate|address)",
        r"risk (mitigation|management|reduction)",
    ],
    "RISK_EXPOSURE": [
        r"what (are|were|is) the (risks?|threats?|vulnerabilities)",
        r"how (exposed|vulnerable|susceptible) is",
        r"(risk|threat) (assessment|profile|landscape)",
        r"what (could|might|may) (threaten|endanger|jeopardize)",
    ],
    "TEMPORAL_TREND": [
        r"how (has|have|did).+?(evolve|change|trend|grow|decline) over time",
        r"(over the years|across fiscal years|since 20\d{2}|between 20\d{2})",
        r"(historical|temporal|year.over.year|quarter.over.quarter)",
        r"(trajectory|evolution|progression|trend)",
    ],
}


def parse_query(question: str) -> StructuredQuery:
    """
    Parse a natural language financial question into a StructuredQuery.

    Returns a StructuredQuery with resolved entities, analysis type,
    and graph search constraints.
    """
    q = StructuredQuery(raw_question=question)
    q_lower = question.lower()

    # Step 1: Detect analysis type
    for atype, patterns in ANALYSIS_PATTERNS.items():
        if any(re.search(p, q_lower) for p in patterns):
            q.analysis_type = atype
            break
    if not q.analysis_type:
        q.analysis_type = "IMPACT_ANALYSIS"  # default

    # Step 2: Extract target financial metric
    for keyword, metric_id in sorted(FINANCIAL_METRICS_MAP.items(),
                                     key=lambda x: -len(x[0])):  # longest match first
        if keyword in q_lower:
            q.target_metric = metric_id
            q.target_entity = metric_id
            break

    # Step 3: Extract temporal constraints
    year_match = re.findall(r"(20\d{2})", question)
    if year_match:
        years = [int(y) for y in year_match]
        q.fiscal_year_start = min(years)
        q.fiscal_year_end = max(years)
        if len(years) >= 2:
            q.temporal_required = True
            q.require_multi_year = True

    # Step 4: Detect temporal trend intent
    if q.analysis_type == "TEMPORAL_TREND":
        q.temporal_required = True
        q.require_multi_year = True
        q.max_hops = 5  # Temporal paths need more hops

    # Step 5: Set relation types based on analysis type
    if q.analysis_type == "IMPACT_ANALYSIS":
        q.relation_types = ["CAUSES", "DECREASES", "INCREASES", "CONSTRAINS",
                            "AFFECTS_SEGMENT", "CONSTRAINS_MARKET", "IMPACTS",
                            "EXPOSED_THROUGH", "TRIGGERS"]
        q.causal_direction = "FORWARD"
    elif q.analysis_type == "MITIGATION_ANALYSIS":
        q.relation_types = ["MITIGATES", "IMPLEMENTS", "ADDRESSES", "EXECUTES",
                            "DECREASES", "CAUSES"]
    elif q.analysis_type == "RISK_EXPOSURE":
        q.relation_types = ["EXPOSED_TO", "EXPOSED_THROUGH", "CAUSES",
                            "CONSTRAINS", "AFFECTS_SEGMENT"]
    elif q.analysis_type == "TEMPORAL_TREND":
        q.relation_types = ["CAUSES", "DECREASES", "INCREASES", "PRECEDES",
                            "OCCURS_DURING", "REPORTED_IN", "TRIGGERS"]

    # Step 6: Exclude downgrade-only relations from causal search
    q.exclude_relations = ["DISCLOSES", "MENTIONS", "POSSIBLE_RELATION"]

    return q


def format_query_context(sq: StructuredQuery, intent_display: str) -> str:
    """Generate a human-readable summary of how the query was understood."""
    parts = [f"Analysis Type: {sq.analysis_type}"]
    if sq.target_metric:
        parts.append(f"Target Metric: {sq.target_metric}")
    if sq.fiscal_year_start:
        yr = f"FY{sq.fiscal_year_start}"
        if sq.fiscal_year_end and sq.fiscal_year_end != sq.fiscal_year_start:
            yr += f"-{sq.fiscal_year_end}"
        parts.append(f"Time Window: {yr}")
    parts.append(f"Max Hops: {sq.max_hops}")
    parts.append(f"Causal Only: {sq.require_explicit_causal}")
    return " | ".join(parts)
