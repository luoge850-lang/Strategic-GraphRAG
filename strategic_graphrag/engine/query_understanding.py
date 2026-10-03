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
from ..schema.financial_observation import measurement_identity


@dataclass
class StructuredQuery:
    """Structured QueryPlan kept under the historical class name.

    An unknown question is never silently promoted to a causal analysis.  The
    explicit plan fields are serialized into API/evaluation traces while the
    historical attributes remain available to the graph engine.
    """
    raw_question: str

    # QueryPlan contract
    task_type: str = "UNCLASSIFIED"
    company: Optional[str] = None
    fact_period: Optional[str] = None
    disclosure_as_of: Optional[str] = None
    document_scope: Optional[str] = None
    comparison: Optional[str] = None
    calculation: Optional[str] = None
    calculation_target_currency: Optional[str] = None
    calculation_target_scale: Optional[str] = None
    evidence_budget: int = 10
    ambiguity: List[str] = field(default_factory=list)
    business_scope: str = "CONSOLIDATED"
    measurement_nature: str = "AMOUNT"
    period_granularity: str = "ANNUAL"
    value_kind: str = "AMOUNT"

    # Target entities to anchor the graph search
    source_entities: List[str] = field(default_factory=list)  # e.g., ["US_EXPORT_CONTROL"]
    target_entity: str = ""  # e.g., "REVENUE", "NET_INCOME"
    target_metric: str = ""  # e.g., "REVENUE"

    # Relation constraint
    analysis_type: str = ""  # FACT | UNCLASSIFIED | IMPACT_ANALYSIS | ...
    causal_direction: str = ""  # FORWARD (source→target) | BACKWARD (target→source) | BIDIRECTIONAL

    # Time window
    fiscal_year_start: Optional[int] = None
    fiscal_year_end: Optional[int] = None
    fiscal_years: List[int] = field(default_factory=list)
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
            "task_type": self.task_type,
            "fact_period": self.fact_period,
            "disclosure_as_of": self.disclosure_as_of,
            "document_scope": self.document_scope,
            "comparison": self.comparison,
            "calculation": self.calculation,
            "calculation_target_currency": self.calculation_target_currency,
            "calculation_target_scale": self.calculation_target_scale,
            "evidence_budget": self.evidence_budget,
            "ambiguity": list(self.ambiguity),
        }

    def to_dict(self) -> Dict:
        """Return a JSON-safe, versioned QueryPlan for audit traces."""
        return {
            "schema": "query-plan/v1",
            "business_scope": self.business_scope,
            "measurement_nature": self.measurement_nature,
            "period_granularity": self.period_granularity,
            "value_kind": self.value_kind,
            "raw_question": self.raw_question,
            "task_type": self.task_type,
            "company": self.company,
            "target_metric": self.target_metric,
            "fact_period": self.fact_period,
            "disclosure_as_of": self.disclosure_as_of,
            "document_scope": self.document_scope,
            "comparison": self.comparison,
            "calculation": self.calculation,
            "calculation_target_currency": self.calculation_target_currency,
            "calculation_target_scale": self.calculation_target_scale,
            "evidence_budget": self.evidence_budget,
            "ambiguity": list(self.ambiguity),
            "analysis_type": self.analysis_type,
            "causal_direction": self.causal_direction,
            "fiscal_year_start": self.fiscal_year_start,
            "fiscal_year_end": self.fiscal_year_end,
            "fiscal_years": list(self.fiscal_years),
            "temporal_required": self.temporal_required,
            "require_multi_year": self.require_multi_year,
            "max_hops": self.max_hops,
            "relation_types": list(self.relation_types),
            "exclude_relations": list(self.exclude_relations),
            "require_explicit_causal": self.require_explicit_causal,
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
    "research and development": "R_AND_D_EXPENSE",
    "r and d expense": "R_AND_D_EXPENSE",
    "r&d": "R_AND_D_EXPENSE",
    "r_and_d_expense": "R_AND_D_EXPENSE",
    "sales, general and administrative": "SG_AND_A_EXPENSE",
    "sales general and administrative": "SG_AND_A_EXPENSE",
    "sg&a": "SG_AND_A_EXPENSE",
    "sg and a": "SG_AND_A_EXPENSE",
    "market value": "MARKET_VALUE",
    "stock price": "MARKET_VALUE",
    "market cap": "MARKET_VALUE",
    # Filing-table metrics are part of the active FinancialMetric registry and
    # must be addressable by natural-language and canonical-token queries.
    "cash and cash equivalents": "CASH_AND_CASH_EQUIVALENTS",
    "cash_and_cash_equivalents": "CASH_AND_CASH_EQUIVALENTS",
    "accounts receivable": "ACCOUNTS_RECEIVABLE",
    "accounts_receivable": "ACCOUNTS_RECEIVABLE",
    "accounts payable": "ACCOUNTS_PAYABLE",
    "accounts_payable": "ACCOUNTS_PAYABLE",
    "cost of revenue": "COST_OF_REVENUE",
    "marketable securities": "MARKETABLE_SECURITIES",
    "marketable_securities": "MARKETABLE_SECURITIES",
    "inventories": "INVENTORIES",
    "total current assets": "TOTAL_CURRENT_ASSETS",
    "total_current_assets": "TOTAL_CURRENT_ASSETS",
    "total assets": "TOTAL_ASSETS",
    "total_assets": "TOTAL_ASSETS",
    "total current liabilities": "TOTAL_CURRENT_LIABILITIES",
    "total_current_liabilities": "TOTAL_CURRENT_LIABILITIES",
    "total liabilities": "TOTAL_LIABILITIES",
    "total_liabilities": "TOTAL_LIABILITIES",
    "total shareholders equity": "TOTAL_SHAREHOLDERS_EQUITY",
    "total_shareholders_equity": "TOTAL_SHAREHOLDERS_EQUITY",
    "gross profit": "GROSS_PROFIT",
    "gross_profit": "GROSS_PROFIT",
    "operating income": "OPERATING_INCOME",
    "operating_income": "OPERATING_INCOME",
    "pretax income": "PRETAX_INCOME",
    "pretax_income": "PRETAX_INCOME",
    "income tax expense": "INCOME_TAX_EXPENSE",
    "income_tax_expense": "INCOME_TAX_EXPENSE",
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


def _explicit_fiscal_period(question: str) -> str | None:
    """Preserve explicit subannual scope instead of collapsing it to its year."""
    text = str(question or "")
    patterns = (
        (r"\bQ([1-4])\s*(?:FY|fiscal(?:\s+year)?)\s*[-:]?\s*(20\d{2})\b", "quarter_first"),
        (r"\bQ([1-4])\s*(20\d{2})\b", "quarter_first_bare"),
        (r"\b(?:FY|fiscal(?:\s+year)?)\s*[-:]?\s*(20\d{2})\s*Q([1-4])\b", "quarter_last"),
        (r"\b(20\d{2})\s*Q([1-4])\b", "year_quarter"),
    )
    for pattern, order in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        if order in {"quarter_first", "quarter_first_bare"}:
            quarter, year = match.group(1), match.group(2)
        elif order == "quarter_last":
            year, quarter = match.group(1), match.group(2)
        else:
            year, quarter = match.group(1), match.group(2)
        return f"Q{quarter} FY{year}"

    ordinal = re.search(r"\b(first|second|third|fourth) quarter(?: of)?(?: the)?\s+(?:FY|fiscal(?: year)?)\s*(20\d{2})", text, re.I)
    if ordinal:
        return f"Q{['first','second','third','fourth'].index(ordinal.group(1).lower())+1} FY{ordinal.group(2)}"

    half = re.search(
        r"\bH([1-2])\s*(?:FY|fiscal(?:\s+year)?)\s*[-:]?\s*(20\d{2})\b|"
        r"\b(?:FY|fiscal(?:\s+year)?)\s*[-:]?\s*(20\d{2})\s*H([1-2])\b",
        text,
        re.IGNORECASE,
    )
    if half:
        if half.group(1):
            return f"H{half.group(1)} FY{half.group(2)}"
        return f"H{half.group(4)} FY{half.group(3)}"
    annual = re.search(
        r"\b(?:FY|fiscal(?:\s+year)?|financial\s+year)\s*[-:]?\s*(20\d{2})\b",
        text,
        re.IGNORECASE,
    )
    if annual:
        return f"FY{annual.group(1)}"
    return None


def _year_from_period_label(period: str) -> int | None:
    match = re.search(r"(?<!\d)(20\d{2})(?!\d)", str(period or ""))
    return int(match.group(1)) if match else None


def parse_query(question: str) -> StructuredQuery:
    """
    Parse a natural language financial question into a StructuredQuery.

    Returns a StructuredQuery with resolved entities, analysis type,
    and graph search constraints.
    """
    q = StructuredQuery(raw_question=question)
    q_lower = question.lower()
    explicit_period = _explicit_fiscal_period(question)

    # Keep disclosure period and fact period separate.  For example, a 2025
    # 10-K can disclose FY2024, and treating the two as the same year leaks
    # document scope into fact scope.
    fiscal_year_token = r"(?:fy|fiscal(?:\s+year)?|financial\s+year)\s*[-:]?\s*(20\d{2})"
    document_year_group = None
    as_of_match = re.search(
        rf"\b(20\d{{2}})\s*(?:annual\s+)?report\b.*?\b{fiscal_year_token}",
        q_lower,
    )
    if as_of_match:
        document_year, fact_year = as_of_match.group(1), as_of_match.group(2)
        document_year_group = 1
    else:
        # Accept the common reversed wording used in filings and user
        # questions: "FY2024 revenue in the 2025 filing" or
        # "the 2025 filing discloses FY2024 revenue".  The filing year is
        # document scope; the FY year is the fact period.
        fact_first = re.search(
            rf"\b{fiscal_year_token}\b.*?\b(20\d{{2}})\s*(?:(?:annual\s+)?report|(?:form\s+)?10-k(?:\s+filing)?|filing)\b",
            q_lower,
        )
        filing_first = re.search(
            rf"\b(20\d{{2}})\s*(?:(?:annual\s+)?report|(?:form\s+)?10-k(?:\s+filing)?|filing)\b.*?\b{fiscal_year_token}\b",
            q_lower,
        )
        if fact_first:
            fact_year, document_year = fact_first.group(1), fact_first.group(2)
            as_of_match = fact_first
            document_year_group = 2
        elif filing_first:
            document_year, fact_year = filing_first.group(1), filing_first.group(2)
            as_of_match = filing_first
            document_year_group = 1
        else:
            document_year = fact_year = None
    if as_of_match and document_year and fact_year:
        q.disclosure_as_of = f"FY{document_year}"
        q.fact_period = (
            explicit_period
            if explicit_period and _year_from_period_label(explicit_period) == int(fact_year)
            else f"FY{fact_year}"
        )
        q.document_scope = f"{document_year}-10-K.pdf"

    if re.search(r"\b(compare|versus|vs\.?|between)\b", q_lower):
        q.comparison = "POINT_TO_POINT"
    elif re.search(r"\b(from|over|through|across)\b", q_lower) and len(re.findall(r"20\d{2}", question)) >= 2:
        q.comparison = "DELTA_OVER_TIME"
    has_financial_metric = any(
        keyword in q_lower for keyword in FINANCIAL_METRICS_MAP
    )
    if re.search(r"\b(convert|conversion)\b", q_lower):
        q.calculation = "UNIT_CONVERSION"
        if re.search(r"\bbillion", q_lower):
            q.calculation_target_scale = "billions"
        elif re.search(r"\bmillion", q_lower):
            q.calculation_target_scale = "millions"
        elif re.search(r"\bthousand", q_lower):
            q.calculation_target_scale = "thousands"
        if re.search(r"\b(?:usd|us dollars?|dollars?)\b", q_lower):
            q.calculation_target_currency = "USD"
        elif re.search(r"\b(?:eur|euros?)\b", q_lower):
            q.calculation_target_currency = "EUR"
        elif re.search(r"\b(?:gbp|pounds? sterling)\b", q_lower):
            q.calculation_target_currency = "GBP"
    elif has_financial_metric and re.search(r"\b(ratio|percentage of|percent of)\b", q_lower):
        q.calculation = "RATIO"
    elif has_financial_metric and re.search(
        r"\b(growth(?: rate| percentage)?|percentage change|percent change|year[- ]over[- ]year|yoy)\b",
        q_lower,
    ):
        q.calculation = "PERCENT_CHANGE"
    elif has_financial_metric and re.search(
        r"\b(absolute difference|difference|delta|calculate the change|change from .* to)\b",
        q_lower,
    ):
        q.calculation = "ABSOLUTE_CHANGE"

    # Step 1: Detect analysis type
    for atype, patterns in ANALYSIS_PATTERNS.items():
        if any(re.search(p, q_lower) for p in patterns):
            q.analysis_type = atype
            break
    if not q.analysis_type:
        q.analysis_type = "FACT" if q.target_metric or re.search(
            r"\b(what|which|how much|reported|value|amount|page|cite)\b", q_lower
        ) else "UNCLASSIFIED"

    # Step 2: Extract target financial metric
    for keyword, metric_id in sorted(FINANCIAL_METRICS_MAP.items(),
                                     key=lambda x: -len(x[0])):  # longest match first
        if keyword in q_lower:
            q.target_metric = metric_id
            q.target_entity = metric_id
            break

    if re.search(r"\bnvidia(?:'s| corporation)?\b", q_lower):
        q.company = "NVIDIA_CORPORATION"
    if q.analysis_type == "UNCLASSIFIED" and q.target_metric:
        q.analysis_type = "FACT"

    # Step 3: Extract temporal constraints
    if explicit_period and explicit_period.startswith('Q'):
        q.period_granularity = "QUARTERLY"
    elif re.search(r"\b(quarter|q[1-4]|half.year|monthly|month)\b",q_lower):
        q.ambiguity.append("subannual_scope_not_reliably_resolved")
    scopes = {'data center':'DATA_CENTER', 'data centre':'DATA_CENTER', 'gaming':'GAMING',
              'automotive':'AUTOMOTIVE','professional visualization':'PROFESSIONAL_VISUALIZATION',
              'compute & networking':'COMPUTE_NETWORKING','graphics segment':'GRAPHICS'}
    matched_scopes = {value for name,value in scopes.items() if name in q_lower}
    if len(matched_scopes)>1:
        q.ambiguity.append("multiple_business_scopes_require_explicit_comparison")
    elif matched_scopes:
        q.business_scope=next(iter(matched_scopes))
        if q.target_metric=='REVENUE':
            q.target_metric=q.target_entity=q.business_scope+'_REVENUE'
        else:q.ambiguity.append("unsupported_business_scope_for_metric")
    elif re.search(r"\b(segment|division|business unit|regional|geographic)\b",q_lower):
        q.ambiguity.append("business_scope_not_resolved")
    flow_change = bool(re.search(r"cash[- ]?flow|cash flows|working capital|operating assets",q_lower)) and bool(re.search(r"adjustment|change|movement",q_lower))
    if flow_change and q.target_metric in {'ACCOUNTS_RECEIVABLE','INVENTORIES','ACCOUNTS_PAYABLE'}:
        q.target_metric=q.target_entity='CASH_FLOW_CHANGE_'+q.target_metric
        q.measurement_nature='PERIOD_MOVEMENT'
        q.calculation=None
    elif re.search(r"\b(balance|at year.end|at fiscal year.end)\b",q_lower):
        q.measurement_nature='BALANCE'
    percentages={'R_AND_D_EXPENSE':'R_AND_D_RATIO','SG_AND_A_EXPENSE':'SG_AND_A_RATIO',
                 'OPERATING_COST':'OPERATING_EXPENSE_RATIO','COST_OF_REVENUE':'COST_OF_REVENUE_RATIO',
                 'GROSS_PROFIT':'GROSS_MARGIN','NET_INCOME':'NET_MARGIN','OPERATING_INCOME':'OPERATING_MARGIN'}
    if q.calculation=='RATIO' and not re.search(r"\b(calculate|compute|divide|divided)\b",q_lower):
        if q.target_metric in percentages:
            q.target_metric=q.target_entity=percentages[q.target_metric]
            q.measurement_nature='DISCLOSED_RATIO'
            q.calculation=None
    if re.search(r"percentage[- ]point|percentage points",q_lower):
        q.calculation='PERCENTAGE_POINT_CHANGE'
    if re.search(r"\b(adjustment|movement)\b",q_lower) and q.measurement_nature!='PERIOD_MOVEMENT':
        q.ambiguity.append("movement_qualifier_not_resolved")

    year_matches = list(re.finditer(r"(20\d{2})", question))
    year_match = [match.group(1) for match in year_matches]
    fact_years = list(year_match)
    if as_of_match:
        # The filing/report year identifies the document, not the fact period.
        # Remove the exact matched document-year occurrence. Do not remove the
        # first equal year string: the fact endpoint may share the same year
        # as the filing (for example FY2023 disclosed in the 2023 10-K).
        document_span = as_of_match.span(document_year_group or 1)
        document_occurrence = next(
            (index for index, match in enumerate(year_matches)
             if (match.start(1), match.end(1)) == document_span),
            None,
        )
        document_year = document_year or as_of_match.group(document_year_group or 1)
        fact_years = [
            year for index, year in enumerate(year_match)
            if index != document_occurrence
        ]
        if not fact_years:
            fact_years = [fact_year or as_of_match.group(2)]
    if year_match:
        years = [int(y) for y in fact_years]
        q.fiscal_years = sorted(set(years))
        q.fiscal_year_start = min(years)
        q.fiscal_year_end = max(years)
        if len(set(years)) >= 2:
            q.temporal_required = True
            q.require_multi_year = True

    if q.fact_period is None and explicit_period:
        q.fact_period = explicit_period
    if q.fact_period is None and q.fiscal_year_start is not None and q.fiscal_year_start == q.fiscal_year_end:
        q.fact_period = f"FY{q.fiscal_year_start}"
    if q.document_scope is None and len(set(year_match)) == 1 and re.search(r"\b(10-k|filing|report)\b", q_lower):
        q.document_scope = f"{year_match[0]}-10-K.pdf"

    q.task_type = {
        "TEMPORAL_TREND": "COMPARISON",
        "IMPACT_ANALYSIS": "RELATION",
        "MITIGATION_ANALYSIS": "RELATION",
        "RISK_EXPOSURE": "RELATION",
        "FACT": "CALCULATION" if q.calculation else "FACT",
        "UNCLASSIFIED": "UNCLASSIFIED",
    }.get(q.analysis_type, "UNCLASSIFIED")
    if q.analysis_type in {"IMPACT_ANALYSIS", "MITIGATION_ANALYSIS", "RISK_EXPOSURE"}:
        q.calculation = None
        q.calculation_target_currency = None
        q.calculation_target_scale = None
    elif q.calculation:
        q.task_type = "CALCULATION"
    if q.comparison and q.task_type == "FACT":
        q.task_type = "COMPARISON"
    explicit_relation_query = bool(
        re.search(r"\b(?:relationship|relation|connects?|connected|link)\b", q_lower)
        and re.search(r"\b[A-Z][A-Z_]{2,}\b", question)
    )
    if explicit_relation_query and not q.calculation:
        q.task_type = "RELATION"
        relation_match = re.search(r"\b(REPORTS_METRIC)\b", question, re.IGNORECASE)
        if relation_match:
            q.relation_types = [relation_match.group(1).upper()]
    if not q.target_metric and q.task_type in {"FACT", "CALCULATION", "COMPARISON"}:
        q.ambiguity.append("metric_not_resolved")

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
    elif q.analysis_type == "FACT":
        q.relation_types = ["REPORTS_METRIC"] if q.target_metric else []

    # Step 6: Exclude downgrade-only relations from causal search
    q.exclude_relations = ["DISCLOSES", "MENTIONS", "POSSIBLE_RELATION"]

    identity=measurement_identity(q.target_metric,q.fact_period or '', 'percent' if q.measurement_nature=='DISCLOSED_RATIO' else '')
    q.measurement_nature=identity['measurement_nature']
    q.value_kind=identity['value_kind']
    return q


def format_query_context(sq: StructuredQuery, intent_display: str) -> str:
    """Generate a human-readable summary of how the query was understood."""
    parts = [f"Task Type: {sq.task_type}", f"Analysis Type: {sq.analysis_type}"]
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


# Public name for new callers; the alias preserves compatibility with old
# imports while the serialized schema is now explicitly called QueryPlan.
QueryPlan = StructuredQuery
