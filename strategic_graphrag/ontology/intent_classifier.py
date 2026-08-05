# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Query Intent Classifier
===========================================
Classifies user queries into financial analysis intents to
guide GraphRAG retrieval strategy and relationship filtering.
"""

import re
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


# =============================================================================
# Intent taxonomy with keyword signatures
# =============================================================================

@dataclass
class IntentSignature:
    """Keyword signature for intent classification."""
    intent_id: str
    display_name: str
    keywords: List[str]
    entity_preference: List[str]  # Preferred entity types for retrieval
    relation_preference: List[str]  # Preferred relationship types
    temporal_required: bool = False


INTENT_REGISTRY: Dict[str, IntentSignature] = {
    "CAUSAL_CHAIN": IntentSignature(
        intent_id="CAUSAL_CHAIN",
        display_name="Causal Chain Analysis",
        keywords=[
            "how does", "how do", "impact", "affect", "lead to",
            "cause", "effect", "consequence", "result in",
            "ripple effect", "chain", "transmission", "propagate",
            "cascade", "spillover", "through what mechanism",
        ],
        entity_preference=["RiskFactor", "Mechanism", "FinancialMetric", "Regulation"],
        relation_preference=["CAUSES", "TRIGGERS", "AMPLIFIES", "AGGRAVATES", "DECREASES", "INCREASES", "MITIGATES"],
        temporal_required=False,
    ),
    "RISK_EXPOSURE": IntentSignature(
        intent_id="RISK_EXPOSURE",
        display_name="Risk Exposure Assessment",
        keywords=[
            "what risk", "risk factor", "exposed to", "exposure",
            "vulnerable", "threat", "risk assessment", "risk profile",
            "what are the risks", "key risks", "primary risks",
        ],
        entity_preference=["Company", "RiskFactor", "Market", "Region"],
        relation_preference=["EXPOSED_TO", "AMPLIFIES"],
        temporal_required=False,
    ),
    "MITIGATION_STRATEGY": IntentSignature(
        intent_id="MITIGATION_STRATEGY",
        display_name="Risk Mitigation Strategy",
        keywords=[
            "mitigate", "mitigation", "mitigating", "hedge", "countermeasure",
            "address risk", "manage risk", "reduce risk",
            "strategy against", "protect against", "diversif",
            "how does nvidia address", "how does nvidia manage",
            "how does nvidia mitigate", "how does nvidia handle",
            "what strategy", "what strategies", "what measures",
            "risk management", "risk mitigation",
            "managing risk", "managing supply chain",
            "how do they mitigate", "how do they manage",
            "how do they address", "how do they handle",
        ],
        entity_preference=["Strategy", "RiskFactor", "Company"],
        relation_preference=["MITIGATES", "IMPLEMENTS"],
        temporal_required=False,
    ),
    "FINANCIAL_IMPACT": IntentSignature(
        intent_id="FINANCIAL_IMPACT",
        display_name="Financial Impact Quantification",
        keywords=[
            "revenue", "income", "margin", "profit", "cost",
            "earnings", "eps", "cash flow", "capex", "expense",
            "financial impact", "bottom line", "quarterly",
            "fiscal", "operating income", "gross margin",
            "how much", "what percentage", "what is the revenue",
        ],
        entity_preference=["FinancialMetric", "RiskFactor", "Strategy", "Company"],
        relation_preference=["INCREASES", "DECREASES", "CAUSES", "REPORTED_IN"],
        temporal_required=True,
    ),
    "REGULATORY_ANALYSIS": IntentSignature(
        intent_id="REGULATORY_ANALYSIS",
        display_name="Regulatory Impact Analysis",
        keywords=[
            "regulation", "regulatory", "export control", "tariff",
            "trade restriction", "compliance", "antitrust",
            "ftc", "eu", "bis", "entity list", "sanction",
            "chips act", "ai act", "government",
        ],
        entity_preference=["Regulation", "Company", "Market", "RiskFactor"],
        relation_preference=["CONSTRAINS", "REGULATED_BY", "CAUSES", "DECREASES"],
        temporal_required=False,
    ),
    "COMPETITIVE_LANDSCAPE": IntentSignature(
        intent_id="COMPETITIVE_LANDSCAPE",
        display_name="Competitive Landscape Analysis",
        keywords=[
            "competitor", "competitive", "competition", "amd",
            "intel", "rival", "market share", "compete",
            "alternative", "threat from", "versus", "vs",
        ],
        entity_preference=["Company", "Product", "Market"],
        relation_preference=["COMPETES_WITH", "PRODUCES", "DEPENDS_ON"],
        temporal_required=False,
    ),
    "SUPPLY_CHAIN": IntentSignature(
        intent_id="SUPPLY_CHAIN",
        display_name="Supply Chain Analysis",
        keywords=[
            "supply chain", "supplier", "manufacturing", "foundry",
            "tsmc", "samsung", "fab", "wafer", "packaging",
            "cowos", "hbm", "procurement", "logistics",
            "inventory", "shortage", "bottleneck",
        ],
        entity_preference=["Company", "Product", "RiskFactor", "Region", "Mechanism"],
        relation_preference=["DEPENDS_ON", "SUPPLIES_TO", "EXPOSED_TO", "TRIGGERS"],
        temporal_required=False,
    ),
    "TEMPORAL_TREND": IntentSignature(
        intent_id="TEMPORAL_TREND",
        display_name="Temporal Trend Analysis",
        keywords=[
            "trend", "over time", "over the years", "historical",
            "year over year", "yoy", "quarter over quarter",
            "qoq", "growth rate", "trajectory", "evolution",
            "since", "from 20", "between 20", "during 20",
        ],
        entity_preference=["FinancialMetric", "Year", "Event"],
        relation_preference=["REPORTED_IN", "PRECEDES", "OCCURS_DURING"],
        temporal_required=True,
    ),
    "GEOPOLITICAL": IntentSignature(
        intent_id="GEOPOLITICAL",
        display_name="Geopolitical Risk Analysis",
        keywords=[
            "china", "taiwan", "geopolitical", "trade war",
            "sanction", "export ban", "national security",
            "cfius", "decouple", "reshore", "nearshoring",
            "sovereign", "conflict", "tension",
        ],
        entity_preference=["Region", "Regulation", "RiskFactor", "Company"],
        relation_preference=["CONSTRAINS", "EXPOSED_TO", "CAUSES"],
        temporal_required=False,
    ),
}


def classify_intent(query: str) -> Tuple[str, IntentSignature]:
    """
    Classify a user query into a financial analysis intent.

    Returns:
        (intent_id, IntentSignature) with the best matching intent.
    """
    query_lower = query.lower()

    # ── Compound intent detection: regulation/geopolitical + financial impact ──
    # These queries ask "how does regulation X impact financial metric Y"
    # They need REGULATORY_ANALYSIS with FINANCIAL_IMPACT awareness
    REGULATORY_SIGNALS = [
        "export control", "tariff", "trade restriction", "sanction",
        "regulation", "regulatory", "entity list", "bis", "commerce department",
        "chips act", "national security", "trade war",
    ]
    FINANCIAL_SIGNALS = [
        "revenue", "income", "margin", "profit", "earnings",
        "financial impact", "bottom line", "market cap",
    ]

    has_regulatory = any(s in query_lower for s in REGULATORY_SIGNALS)
    has_financial = any(s in query_lower for s in FINANCIAL_SIGNALS)
    has_causal = any(kw in query_lower for kw in ["how does", "how do", "impact", "affect", "lead to", "effect on"])

    # Compound: regulation + financial → REGULATORY_ANALYSIS with causal chain
    if has_regulatory and (has_financial or has_causal):
        sig = INTENT_REGISTRY["REGULATORY_ANALYSIS"]
        return "REGULATORY_ANALYSIS", sig

    # Geopolitical + financial → GEOPOLITICAL
    GEOPOLITICAL_SIGNALS = ["china", "taiwan", "geopolitical", "decouple", "reshore"]
    has_geopolitical = any(s in query_lower for s in GEOPOLITICAL_SIGNALS)
    if has_geopolitical and (has_financial or has_causal):
        sig = INTENT_REGISTRY["GEOPOLITICAL"]
        return "GEOPOLITICAL", sig

    # ── Standard keyword scoring ──
    best_score = 0
    best_intent_id = "CAUSAL_CHAIN"
    best_signature = INTENT_REGISTRY["CAUSAL_CHAIN"]

    # High-priority mitigation keywords
    MITIGATION_STRONG_SIGNALS = [
        "mitigate", "mitigation", "mitigating", "how does nvidia address",
        "how does nvidia manage", "how does nvidia mitigate",
        "risk management", "risk mitigation",
    ]

    mitigation_signal_count = sum(1 for kw in MITIGATION_STRONG_SIGNALS if kw in query_lower)
    if mitigation_signal_count > 0:
        sig = INTENT_REGISTRY["MITIGATION_STRATEGY"]
        score = sum(1 for kw in sig.keywords if kw in query_lower)
        density = score / max(len(sig.keywords), 1)
        weighted_score = (score + density * 2) * 2.0
        if weighted_score > best_score:
            best_score = weighted_score
            best_intent_id = "MITIGATION_STRATEGY"
            best_signature = sig

    for intent_id, sig in INTENT_REGISTRY.items():
        score = sum(1 for kw in sig.keywords if kw in query_lower)
        density = score / max(len(sig.keywords), 1)
        weighted_score = score + density * 2
        if mitigation_signal_count > 0 and intent_id != "MITIGATION_STRATEGY":
            weighted_score *= 0.5
        if weighted_score > best_score:
            best_score = weighted_score
            best_intent_id = intent_id
            best_signature = sig

    return best_intent_id, best_signature


def get_retrieval_strategy(query: str) -> Dict:
    """
    Generate a retrieval strategy based on query intent classification.
    Returns a dict with entity and relationship preferences for Cypher generation.
    """
    intent_id, sig = classify_intent(query)
    return {
        "intent": intent_id,
        "display_name": sig.display_name,
        "entity_preference": sig.entity_preference,
        "relation_preference": sig.relation_preference,
        "temporal_required": sig.temporal_required,
        "max_hops": 4 if intent_id in ("CAUSAL_CHAIN", "TEMPORAL_TREND", "GEOPOLITICAL") else 3,
    }


def extract_financial_entities_from_query(query: str) -> List[str]:
    """
    Extract financial entity keywords from a user query.
    Used for anchor node identification in Cypher generation.
    """
    entities = []

    # Company names
    company_patterns = [
        r'\b(nvidia|amd|intel|tsmc|samsung|broadcom|qualcomm|micron|microsoft|google|amazon|apple|meta|huawei)\b',
    ]
    for pat in company_patterns:
        matches = re.findall(pat, query, re.IGNORECASE)
        entities.extend(m.upper() for m in matches)

    # Financial metrics
    metric_patterns = [
        r'\b(revenue|income|margin|profit|earnings|eps|cash flow|'
        r'capex|operating cost|gross margin|net income|free cash flow|'
        r'return on equity|ebitda)\b',
    ]
    for pat in metric_patterns:
        matches = re.findall(pat, query, re.IGNORECASE)
        entities.extend(m.upper().replace(" ", "_") for m in matches)

    # Risk types (use flexible matching for plural forms)
    risk_patterns = [
        r'\b(supply\s*chains?|export\s*controls?|regulation|competition|'
        r'cyber|geopolitical|inflation|tariffs?|sanctions?|'
        r'intellectual\s*property|talent)\b',
    ]
    for pat in risk_patterns:
        matches = re.findall(pat, query, re.IGNORECASE)
        entities.extend(m.upper().replace(" ", "_") for m in matches)

    # Region names
    region_patterns = [
        r'\b(china|taiwan|united states|europe|japan|korea|'
        r'singapore|israel|india|asia)\b',
    ]
    for pat in region_patterns:
        matches = re.findall(pat, query, re.IGNORECASE)
        entities.extend(m.upper().replace(" ", "_") for m in matches)

    # Deduplicate and map to canonical knowledge graph entity IDs
    entities = list(dict.fromkeys(entities))

    # Map extracted query terms to canonical knowledge graph entity IDs
    from .entity_registry import CANONICAL_MAP, norm_id
    canonical_entities = []
    for e in entities:
        e_norm = norm_id(e)  # lowercase, underscores
        # Direct lookup by normalized ID
        if e_norm in CANONICAL_MAP:
            canonical_entities.append(CANONICAL_MAP[e_norm][0])
            continue
        # Try stripping trailing 's'
        e_stripped = e_norm.rstrip('s')
        if len(e_stripped) > 3 and e_stripped in CANONICAL_MAP:
            canonical_entities.append(CANONICAL_MAP[e_stripped][0])
            continue
        # Partial match: entity registry key contains query term or vice versa
        for key, (name, cat) in CANONICAL_MAP.items():
            if len(key) > 4 and (e_norm in key or key in e_norm):
                canonical_entities.append(name)
                break
        else:
            canonical_entities.append(e)

    return list(dict.fromkeys(entities + canonical_entities))
