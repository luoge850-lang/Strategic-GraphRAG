# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Relationship Inference Rules
================================================
Deterministic rules for inferring the correct financial relationship
type between two entities based on their categories and context text.

No more generic RELATED_TO or IMPACTS — every relationship is typed
with strict financial semantics.
"""

import re
from typing import Dict, Optional, Set, Tuple

from .entity_registry import norm_id

# =============================================================================
# Metric directionality
# =============================================================================

POSITIVE_METRICS = {
    "REVENUE", "GROSS_MARGIN", "OPERATING_MARGIN", "NET_INCOME",
    "GROSS_PROFIT", "OPERATING_INCOME", "TOTAL_ASSETS",
    "TOTAL_CURRENT_ASSETS", "TOTAL_SHAREHOLDERS_EQUITY",
    "CASH_FLOW", "FREE_CASH_FLOW", "EARNINGS_PER_SHARE",
    "RETURN_ON_EQUITY", "MARKET_VALUE", "EBITDA", "CURRENT_RATIO",
}

NEGATIVE_METRICS = {
    "OPERATING_COST", "COST_OF_REVENUE", "R_AND_D_EXPENSE",
    "SG_AND_A_EXPENSE", "CAPEX", "DEBT_TO_EQUITY",
}

# =============================================================================
# v2.0 Strict Causal Verb Detection (Judea Pearl SCM-inspired)
# =============================================================================

# Only these explicit causal verbs can produce CONFIRMED_CAUSAL
CAUSAL_VERBS_EXPLICIT = [
    "cause", "causes", "caused", "causing",
    "lead to", "leads to", "led to", "leading to",
    "result in", "results in", "resulted in", "resulting in",
    "due to", "because of", "as a result of",
    "driven by", "drives", "trigger", "triggers", "triggered",
    "mitigate", "mitigates", "mitigated", "mitigating",
    "reduce", "reduces", "reduced", "reducing",
    "increase", "increases", "increased", "increasing",
    "decrease", "decreases", "decreased", "decreasing",
]

# These imply causation but do not state it explicitly → STRONG_ASSOCIATION
CAUSAL_VERBS_IMPLIED = [
    "affect", "affects", "affected", "impact", "impacts", "impacted",
    "influence", "influences", "influenced",
    "linked to", "associated with", "related to", "correlated with",
    "contribute to", "contributes to", "stem from", "stems from",
]

# These indicate uncertainty → WEAK_ASSOCIATION at best
SPECULATIVE_MARKERS = [
    "may", "might", "could", "potentially", "possibly",
    "subject to", "uncertain", "depending on", "if",
    "risk of", "exposed to", "vulnerable to", "susceptible to",
]


def detect_causal_strength(text: str) -> str:
    """
    v2.0: Detect causal strength using explicit verb detection.
    Only returns CONFIRMED_CAUSAL when an explicit causal verb is present.
    No causal language → DISCLOSED_ONLY (cannot assert causation).
    """
    if not text:
        return "DISCLOSED_ONLY"

    tl = text.lower()

    # Tier 1: Explicit causal verb REQUIRED for CONFIRMED_CAUSAL
    if any(verb in tl for verb in CAUSAL_VERBS_EXPLICIT):
        return "CONFIRMED_CAUSAL"

    # Tier 2: Implied causal language
    if any(verb in tl for verb in CAUSAL_VERBS_IMPLIED):
        return "STRONG_ASSOCIATION"

    # Tier 3: Speculative markers only
    if any(marker in tl for marker in SPECULATIVE_MARKERS):
        return "WEAK_ASSOCIATION"

    # Tier 4: Nothing — text does not assert any causal relationship
    return "DISCLOSED_ONLY"


# =============================================================================
# Core relationship inference engine
# =============================================================================

def infer_relation(
    source_category: str,
    target_category: str,
    context_text: str = "",
    source_name: str = "",
    target_name: str = "",
) -> str:
    """
    Infer the correct financial relationship type between two entity categories.

    This is the central reasoning function — it replaces generic relations
    with strict financial semantics.

    Args:
        source_category: Neo4j label of source node (e.g., 'RiskFactor')
        target_category: Neo4j label of target node (e.g., 'FinancialMetric')
        context_text: Evidence sentence for linguistic analysis
        source_name: Display name of source entity
        target_name: Display name of target entity

    Returns:
        A valid relationship type string (e.g., 'DECREASES', 'MITIGATES')
    """
    ctx = context_text.lower()
    sn = source_name.upper()
    tn = target_name.upper()

    # ── Self-loops: same entity ──
    if norm_id(source_name) == norm_id(target_name):
        return "CAUSES"

    # ── Company → X ──
    if source_category == "Company" and target_category == "RiskFactor":
        return "EXPOSED_TO"
    if source_category == "Company" and target_category == "Strategy":
        return "IMPLEMENTS"
    if source_category == "Company" and target_category == "FinancialMetric":
        return "REPORTED_IN"
    if source_category == "Company" and target_category == "Region":
        return "OPERATES_IN"
    if source_category == "Company" and target_category == "Market":
        return "OPERATES_IN"
    if source_category == "Company" and target_category == "Product":
        return "PRODUCES"
    if source_category == "Company" and target_category == "Company":
        return "COMPETES_WITH"

    # ── Strategy → X ──
    if source_category == "Strategy" and target_category == "RiskFactor":
        return "MITIGATES"
    if source_category == "Strategy" and target_category == "FinancialMetric":
        if _signal_reduce(ctx):
            return "DECREASES"
        if _signal_increase(ctx):
            return "INCREASES"
        # Strategy usually aims to increase revenue / decrease costs
        if tn in POSITIVE_METRICS:
            return "INCREASES"
        if tn in NEGATIVE_METRICS:
            return "DECREASES"
        return "CAUSES"

    # ── RiskFactor → X ──
    if source_category == "RiskFactor" and target_category == "FinancialMetric":
        # Risk usually harms positive metrics and increases negative ones
        if tn in POSITIVE_METRICS:
            if _signal_increase(ctx) and not _signal_reduce(ctx):
                return "INCREASES"
            return "DECREASES"
        if tn in NEGATIVE_METRICS:
            if _signal_reduce(ctx) and not _signal_increase(ctx):
                return "DECREASES"
            return "INCREASES"
        # Context-based disambiguation
        if _signal_reduce(ctx):
            return "DECREASES"
        if _signal_increase(ctx):
            return "INCREASES"
        return "CAUSES"

    if source_category == "RiskFactor" and target_category == "RiskFactor":
        if re.search(r'\b(amplify|exacerbate|worsen|intensify|magnify|compound)\b', ctx):
            return "AMPLIFIES"
        return "CAUSES"

    if source_category == "RiskFactor" and target_category == "Mechanism":
        return "TRIGGERS"

    if source_category == "RiskFactor" and target_category == "Event":
        return "CAUSES"

    # ── Mechanism → X ──
    if source_category == "Mechanism" and target_category == "FinancialMetric":
        if tn in POSITIVE_METRICS:
            return "DECREASES"
        if tn in NEGATIVE_METRICS:
            return "INCREASES"
        return "CAUSES"
    if source_category == "Mechanism" and target_category == "RiskFactor":
        return "AGGRAVATES"

    # ── Regulation → X ──
    if source_category == "Regulation" and target_category == "Market":
        return "CONSTRAINS"
    if source_category == "Regulation" and target_category == "Company":
        return "CONSTRAINS"
    if source_category == "Regulation" and target_category == "Product":
        return "CONSTRAINS"
    if source_category == "Regulation" and target_category == "RiskFactor":
        return "CAUSES"

    # ── Market → X ──
    if source_category == "Market" and target_category == "RiskFactor":
        return "CAUSES"
    if source_category == "Market" and target_category == "FinancialMetric":
        if _signal_decline(ctx):
            return "DECREASES"
        return "INCREASES"

    # ── Region → X ──
    if source_category == "Region" and target_category == "RiskFactor":
        return "EXPOSED_TO"

    # ── Product → X ──
    if source_category == "Product" and target_category == "Market":
        return "DEPENDS_ON"
    if source_category == "Product" and target_category == "FinancialMetric":
        return "INCREASES"

    # ── Event → X ──
    if source_category == "Event" and target_category == "RiskFactor":
        return "CAUSES"
    if source_category == "Event" and target_category == "Mechanism":
        return "TRIGGERS"
    if source_category == "Event" and target_category == "FinancialMetric":
        if _signal_decline(ctx):
            return "DECREASES"
        return "INCREASES"

    # ── Metric → Metric ──
    if source_category == "FinancialMetric" and target_category == "FinancialMetric":
        if sn == "OPERATING_COST" and tn in POSITIVE_METRICS:
            return "DECREASES"
        if sn == "REVENUE" and tn in POSITIVE_METRICS:
            return "INCREASES"
        if sn in NEGATIVE_METRICS and tn in POSITIVE_METRICS:
            return "DECREASES"
        if sn in POSITIVE_METRICS and tn in POSITIVE_METRICS:
            return "INCREASES"
        return "CAUSES"

    # ── Fallback: default to CAUSES ──
    return "CAUSES"


# =============================================================================
# Context signal detectors (helper functions)
# =============================================================================

# ═══════════════════════════════════════════════════════════════
# P1-FIX #4: Concession & Negation Detection
# ═══════════════════════════════════════════════════════════════

# Concession markers — the sentence acknowledges the entity but subverts the
# expected causal direction.  "Despite X, Y increased" → X did NOT cause Y.
CONCESSION_MARKERS = [
    r'\bdespite\b', r'\balthough\b', r'\beven though\b',
    r'\bnotwithstanding\b', r'\bregardless of\b',
    r'\bwhile\b.*\b(?:remain|continu|persist)\b',
]

# Negation markers — the sentence explicitly denies causation or impact.
NEGATION_MARKERS = [
    r'\b(?:did|does|do|has|have|had|is|are|was|were|will|would|can|could|may|might)\s+not\b',
    r'\bno\s+(?:significant|material|meaningful|measurable|direct|adverse)\b',
    r'\bnot\s+(?:material|significant|expected|anticipated|likely)\b',
    r'\b(?:failed|unable)\s+to\b',
    r"\b(?:doesn't|don't|didn't|hasn't|haven't|isn't|aren't|wasn't|weren't|won't|wouldn't|couldn't|can't)\b",
    r'\bwithout\s+(?:a\s+)?(?:material|significant)\s+(?:impact|effect)\b',
]

# Counterfactual markers — the sentence describes what WOULD have happened,
# not what DID happen.
COUNTERFACTUAL_MARKERS = [
    r'\bwould\s+(?:have|be)\b',
    r'\bcould\s+(?:have|be)\b',
    r'\bif\b.*\bwould\b',
    r'\bhad\b.*\bwould\b',
]

# But/however after a positive signal → likely a concession pivot
PIVOT_MARKERS = [
    r',?\s+(?:but|however|yet|nevertheless|nonetheless)\s+',
]


def _is_concession(text: str) -> bool:
    """Check if text contains concession language that subverts the direction."""
    tl = text.lower()
    return any(re.search(p, tl) for p in CONCESSION_MARKERS)


def _is_negated(text: str) -> bool:
    """Check if causal/impact language is negated in the text."""
    tl = text.lower()
    return any(re.search(p, tl) for p in NEGATION_MARKERS)


def _is_counterfactual(text: str) -> bool:
    """Check if text describes hypothetical, not actual events."""
    tl = text.lower()
    return any(re.search(p, tl) for p in COUNTERFACTUAL_MARKERS)


def _has_concession_pivot(text: str) -> bool:
    """Check if text has a but/however pivot that may reverse the initial claim."""
    tl = text.lower()
    return any(re.search(p, tl) for p in PIVOT_MARKERS)


def _clean_signal(text: str) -> str:
    """
    Split a sentence at concession pivots and return only the main clause.
    If the sentence structure is 'A but B', the causal signal is in B,
    not A. Returns the portion after the LAST pivot.
    """
    tl = text.lower()
    pivot_positions = []
    for p in PIVOT_MARKERS:
        for m in re.finditer(p, tl):
            pivot_positions.append(m.start())
    if pivot_positions:
        # Return text after the last pivot
        last_pivot = max(pivot_positions)
        return text[last_pivot:]
    return text


# ═══════════════════════════════════════════════════════════════

def _signal_reduce(ctx: str) -> bool:
    """Detect reduction/decrease language in context.
    P1-FIX: Excludes negated and concessive matches."""
    if _is_negated(ctx) or _is_counterfactual(ctx):
        return False
    # Check the main clause (after any concession pivot)
    main = _clean_signal(ctx) if _has_concession_pivot(ctx) else ctx
    return bool(re.search(
        r'\b(reduce|decrease|lower|minimize|cut|decline|drop|'
        r'fell|shrink|contract|diminish|erode|impair|hurt|'
        r'damage|harm|adversely|negatively|loss)\b',
        main
    ))

def _signal_increase(ctx: str) -> bool:
    """Detect increase/growth language in context.
    P1-FIX: Excludes negated and concessive matches."""
    if _is_negated(ctx) or _is_counterfactual(ctx):
        return False
    main = _clean_signal(ctx) if _has_concession_pivot(ctx) else ctx
    return bool(re.search(
        r'\b(increase|growth|grow|rise|raise|improve|enhance|'
        r'boost|expand|gain|climb|surge|accelerate|strengthen|'
        r'positively|favorably)\b',
        main
    ))

def _signal_decline(ctx: str) -> bool:
    """Detect market/financial decline language.
    P1-FIX: Excludes negated and concessive matches."""
    if _is_negated(ctx) or _is_counterfactual(ctx):
        return False
    main = _clean_signal(ctx) if _has_concession_pivot(ctx) else ctx
    return bool(re.search(
        r'\b(decline|slowdown|contraction|downturn|recession|'
        r'depression|crisis|crash|bear|weak|soft)\b',
        main
    ))


# =============================================================================
# Relationship validation
# =============================================================================

# v2.0: Evidence-Grounded Causal Ontology
# Downgrade-only: weak signals → POSSIBLE_RELATION, DISCLOSES, MENTIONS
# Causal edges (CAUSES, MITIGATES, etc.) require EXPLICIT causal verb evidence

def classify_causal_form(source_category: str, target_category: str) -> str:
    """Describe whether a relation is direct, mediated, or structural.

    This label prevents a direct risk-to-metric disclosure from being
    presented as if a mechanism node had already been extracted.
    """
    if source_category in {"Regulation", "RegulationChange"}:
        return "REGULATORY_CONSTRAINT_OR_DRIVER"
    if source_category == "RiskFactor" and target_category == "FinancialMetric":
        return "DIRECT_DISCLOSED_IMPACT"
    if source_category == "Mechanism" or target_category == "Mechanism":
        return "MECHANISM_MEDIATED"
    if source_category in {"Event", "RiskEvent"}:
        return "EVENT_LINK"
    if source_category == "FinancialMetric" or target_category == "FinancialMetric":
        return "FINANCIAL_RELATION"
    return "STRUCTURAL_OR_EXPOSURE"


VALID_RELATIONS = {
    # ── Causal (require explicit evidence) ──
    "CAUSES", "TRIGGERS", "AMPLIFIES",
    "INCREASES", "DECREASES",
    "MITIGATES", "IMPLEMENTS",
    "CONSTRAINS", "EXPOSED_TO",
    # ── Multi-hop intermediate (v2.0) ──
    "AFFECTS_SEGMENT", "CONSTRAINS_MARKET", "EXPOSED_THROUGH", "IMPACTS",
    "EXECUTES", "ADDRESSES",
    # ── Structural ──
    "OPERATES_IN", "PRODUCES", "COMPETES_WITH", "DEPENDS_ON",
    "REGULATED_BY", "SUPPLIES_TO",
    # ── Temporal ──
    "OCCURS_DURING", "PRECEDES", "REPORTED_IN", "REPORTS_METRIC",
    # ── Evidence ──
    "HAS_EVIDENCE", "BELONGS_TO", "SUPPORTS",
    # ── Downgrade relations (weak/uncertain signals) ──
    "DISCLOSES", "MENTIONS", "POSSIBLE_RELATION",
}

# v2.0 Causal Strength Tiers (Judea Pearl SCM-inspired)
CAUSAL_STRENGTHS = {
    # Tier 1: Explicit causal language in evidence
    "CONFIRMED_CAUSAL",
    # Tier 2: Strong implication (e.g., "as a result of", "due to")
    "STRONG_ASSOCIATION",
    # Tier 3: Co-occurrence in risk disclosure section
    "WEAK_ASSOCIATION",
    # Tier 4: Entity disclosed but no causal language
    "DISCLOSED_ONLY",
    # Tier 5: System-inferred, lowest confidence
    "INFERRED",
    # Legacy compatibility
    "DIRECT_CAUSALITY", "INDIRECT_CAUSALITY",
    "RISK_ASSOCIATION", "SPECULATIVE_RELATION", "DISCLOSED_EXPOSURE",
}

# v2.0 Entity Categories — intermediate semantic layers
ENTITY_CATEGORIES = {
    # Layer 1: Entity
    "Company", "Product", "Market", "Region",
    # Layer 2: External Driver
    "Regulation", "RegulationChange", "MacroEvent", "GeopoliticalEvent",
    # Layer 3: Risk Driver
    "RiskFactor", "RiskDriver", "RiskEvent",
    # Layer 4: Transmission
    "Mechanism", "BusinessSegment",
    # Layer 5: Impact
    "FinancialMetric", "FinancialImpact",
    # Layer 6: Mitigation
    "Strategy", "MitigationAction",
    # Layer 7: Temporal
    "Year", "Quarter", "Event",
    # Layer 8: Evidence
    "Document", "Sentence", "EvidenceClaim",
}

# Domain/range constraints for the relations that participate in causal
# retrieval.  The old validator checked that both labels existed, but that
# still allowed accidental edges such as Product -> CAUSES -> Region.  These
# rules intentionally remain conservative while preserving the combinations
# already used by the single-filing baseline.
RELATION_CATEGORY_RULES: Dict[str, Set[Tuple[str, str]]] = {
    "CAUSES": {
        ("Regulation", "RiskFactor"),
        ("RegulationChange", "RiskFactor"),
        ("MacroEvent", "RiskFactor"),
        ("GeopoliticalEvent", "RiskFactor"),
        ("Market", "RiskFactor"),
        ("Event", "RiskFactor"),
        ("RiskFactor", "RiskFactor"),
        ("RiskFactor", "FinancialMetric"),
        ("Mechanism", "RiskFactor"),
        ("Mechanism", "FinancialMetric"),
    },
    "TRIGGERS": {
        ("Regulation", "RiskFactor"), ("RegulationChange", "RiskFactor"),
        ("MacroEvent", "RiskFactor"), ("GeopoliticalEvent", "RiskFactor"),
        ("Event", "RiskFactor"), ("RiskFactor", "RiskFactor"),
    },
    "AMPLIFIES": {
        ("RiskFactor", "RiskFactor"), ("RiskFactor", "FinancialMetric"),
        ("Mechanism", "RiskFactor"), ("Mechanism", "FinancialMetric"),
    },
    "INCREASES": {
        ("RiskFactor", "RiskFactor"), ("RiskFactor", "FinancialMetric"),
        ("Mechanism", "RiskFactor"), ("Mechanism", "FinancialMetric"),
        ("Market", "RiskFactor"), ("Market", "FinancialMetric"),
        ("Event", "RiskFactor"), ("Event", "FinancialMetric"),
        ("Regulation", "RiskFactor"), ("Regulation", "FinancialMetric"),
    },
    "DECREASES": {
        ("RiskFactor", "RiskFactor"), ("RiskFactor", "FinancialMetric"),
        ("Mechanism", "RiskFactor"), ("Mechanism", "FinancialMetric"),
        ("Market", "RiskFactor"), ("Market", "FinancialMetric"),
        ("Event", "RiskFactor"), ("Event", "FinancialMetric"),
        ("Regulation", "RiskFactor"), ("Regulation", "FinancialMetric"),
    },
    "MITIGATES": {
        ("Strategy", "RiskFactor"), ("Strategy", "FinancialMetric"),
        ("MitigationAction", "RiskFactor"), ("MitigationAction", "FinancialMetric"),
    },
    "IMPLEMENTS": {
        ("Company", "Strategy"), ("Company", "MitigationAction"),
    },
    "CONSTRAINS": {
        ("Regulation", "Market"), ("Regulation", "RiskFactor"),
        ("Regulation", "FinancialMetric"), ("RiskFactor", "Market"),
        ("RiskFactor", "FinancialMetric"),
    },
    "EXPOSED_TO": {
        ("Company", "RiskFactor"), ("BusinessSegment", "RiskFactor"),
        ("Market", "RiskFactor"), ("Region", "RiskFactor"),
    },
    "AFFECTS_SEGMENT": {
        ("RiskFactor", "BusinessSegment"), ("Event", "BusinessSegment"),
        ("Market", "BusinessSegment"),
    },
    "CONSTRAINS_MARKET": {
        ("Regulation", "Market"), ("RegulationChange", "Market"),
        ("RiskFactor", "Market"),
    },
    "EXPOSED_THROUGH": {
        ("Company", "Market"), ("Company", "Region"),
        ("RiskFactor", "Mechanism"),
    },
    "IMPACTS": {
        ("RiskFactor", "FinancialMetric"), ("Mechanism", "FinancialMetric"),
        ("Event", "FinancialMetric"), ("Market", "FinancialMetric"),
    },
    "EXECUTES": {
        ("Company", "Strategy"), ("Company", "MitigationAction"),
    },
    "ADDRESSES": {
        ("Strategy", "RiskFactor"), ("MitigationAction", "RiskFactor"),
    },
    "OPERATES_IN": {
        ("Company", "Market"), ("Company", "Region"),
        ("BusinessSegment", "Market"), ("BusinessSegment", "Region"),
    },
    "PRODUCES": {
        ("Company", "Product"), ("Company", "BusinessSegment"),
        ("BusinessSegment", "Product"),
    },
    "COMPETES_WITH": {
        ("Company", "Company"), ("Company", "Product"),
        ("Product", "Company"), ("Product", "Product"),
    },
    "DEPENDS_ON": {
        ("Company", "Company"), ("Company", "Product"),
        ("Company", "Market"), ("Product", "Company"),
        ("Product", "Product"), ("BusinessSegment", "Product"),
        ("BusinessSegment", "Market"),
    },
    "REGULATED_BY": {
        ("Company", "Regulation"), ("Product", "Regulation"),
        ("Market", "Regulation"), ("BusinessSegment", "Regulation"),
    },
    "SUPPLIES_TO": {
        ("Company", "Company"), ("Product", "Company"),
        ("Company", "Market"), ("Product", "Market"),
    },
    "OCCURS_DURING": {
        ("Event", "Year"), ("Event", "Quarter"),
        ("RiskEvent", "Year"), ("RiskEvent", "Quarter"),
    },
    "PRECEDES": {
        ("Event", "Event"), ("RiskEvent", "RiskEvent"),
        ("Event", "RiskEvent"), ("RiskEvent", "Event"),
    },
    "REPORTED_IN": {
        ("EvidenceClaim", "Document"), ("Document", "Year"),
    },
    "REPORTS_METRIC": {
        ("Company", "FinancialMetric"),
    },
    "DISCLOSES": {
        ("Document", "EvidenceClaim"),
    },
    "MENTIONS": {
        ("EvidenceClaim", "Company"), ("EvidenceClaim", "Product"),
        ("EvidenceClaim", "Market"), ("EvidenceClaim", "Region"),
        ("EvidenceClaim", "RiskFactor"), ("EvidenceClaim", "FinancialMetric"),
        ("EvidenceClaim", "Strategy"), ("EvidenceClaim", "Event"),
    },
    "BELONGS_TO": {
        ("Sentence", "Document"), ("EvidenceClaim", "Document"),
        ("Event", "Year"), ("RiskEvent", "Year"),
    },
    "SUPPORTS": {
        ("Sentence", "EvidenceClaim"), ("Sentence", "RiskFactor"),
        ("Sentence", "FinancialMetric"),
    },
}


def validate_triple(
    source_category: str,
    target_category: str,
    relation: str,
    source_name: str = "",
    target_name: str = "",
) -> Tuple[bool, str]:
    """
    Validate a triple against the ontology constraints.

    Returns:
        (is_valid, reason)
    """
    # Check entity categories
    if source_category not in ENTITY_CATEGORIES:
        return False, f"Invalid source category: {source_category}"
    if target_category not in ENTITY_CATEGORIES:
        return False, f"Invalid target category: {target_category}"

    # Check relationship type
    if relation not in VALID_RELATIONS:
        return False, f"Invalid relation: {relation}"

    # Prevent self-referencing
    if norm_id(source_name) == norm_id(target_name):
        return False, "Self-referencing relation not allowed"

    # Prevent MITIGATES from non-Strategy sources
    if relation == "MITIGATES" and source_category != "Strategy":
        return False, f"MITIGATES requires Strategy source, got {source_category}"

    # Prevent IMPLEMENTS from non-Company sources
    if relation == "IMPLEMENTS" and source_category != "Company":
        return False, f"IMPLEMENTS requires Company source, got {source_category}"

    # Prevent HAS_EVIDENCE from non-relationship sources
    if relation == "HAS_EVIDENCE":
        return False, "HAS_EVIDENCE must be attached to a relationship, not a node"

    allowed_pairs = RELATION_CATEGORY_RULES.get(relation)
    if allowed_pairs is not None and (source_category, target_category) not in allowed_pairs:
        return False, (
            f"Invalid category pair for {relation}: "
            f"{source_category} -> {target_category}"
        )

    return True, "OK"
