"""Canonical financial-observation records derived from table evidence.

An EvidenceClaim says that a table row was disclosed.  A
FinancialObservation represents one value in that row for one fiscal period.
Keeping the two concepts separate makes numeric retrieval, temporal comparison,
and provenance validation possible without parsing JSON at query time.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional


_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_NUMBER_TOKEN_RE = re.compile(r"(?<![A-Za-z])\(?-?\$?\d[\d,]*(?:\.\d+)?\)?")
_PERCENT_TOKEN_RE = re.compile(r"\(?-?\d[\d,]*(?:\.\d+)?\)?\s*%")
_PERCENT_METRIC_MAP = {
    "GROSS_PROFIT": "GROSS_MARGIN",
    "OPERATING_INCOME": "OPERATING_MARGIN",
    "NET_INCOME": "NET_MARGIN",
    "COST_OF_REVENUE": "COST_OF_REVENUE_RATIO",
    "OPERATING_COST": "OPERATING_EXPENSE_RATIO",
    "R_AND_D_EXPENSE": "R_AND_D_RATIO",
    "SG_AND_A_EXPENSE": "SG_AND_A_RATIO",
}


@dataclass(frozen=True)
class FinancialObservation:
    id: str
    measurement_key: str
    company_id: str
    metric_id: str
    claim_id: str
    source_filing: str
    page: int
    fiscal_period: str
    fiscal_year: int
    value: float
    raw_value: str
    unit: str
    currency: Optional[str]
    scale: Optional[str]
    statement_type: str
    table_name: str
    row_label: str
    column_label: str
    valid_from: str
    valid_to: str
    comparability_status: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_numeric_value(value: Any) -> Optional[float]:
    """Parse SEC accounting notation without silently changing its sign."""
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    raw = str(value or "").strip()
    negative = raw.startswith("(") and raw.endswith(")")
    cleaned = raw.replace(",", "").replace("$", "").replace("%", "").strip("() ")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -abs(number) if negative else number


def split_unit(unit: str) -> tuple[Optional[str], Optional[str]]:
    normalized = str(unit or "").strip()
    lower = normalized.casefold()
    currency = "USD" if lower.startswith("usd") else None
    if "million" in lower:
        scale = "millions"
    elif "thousand" in lower:
        scale = "thousands"
    elif "billion" in lower:
        scale = "billions"
    else:
        scale = None
    return currency, scale


def fiscal_period_bounds(period: str) -> tuple[str, str, int]:
    """Return conservative period bounds without inventing exact fiscal dates."""
    text = str(period or "").strip()
    match = _YEAR_RE.search(text)
    if not match:
        raise ValueError(f"Financial observation has no fiscal year: {period!r}")
    year = int(match.group(1))
    # The source currently gives a fiscal-year label, not exact start/end
    # dates. Preserve that granularity explicitly instead of pretending these
    # are calendar-year boundaries.
    label = f"FY{year}"
    return label, label, year


def build_financial_observations(
    triple: Dict[str, Any],
    *,
    claim_id: str,
    company_id: str,
    metric_id: str,
    source_filing: str,
    page: int,
    filing_year: int,
    section: str,
) -> List[Dict[str, Any]]:
    """Expand one REPORTS_METRIC claim into period-specific observations."""
    if str(triple.get("relation", "")).upper() != "REPORTS_METRIC":
        return []
    raw_values = triple.get("metric_values_json") or "[]"
    try:
        values = json.loads(raw_values) if isinstance(raw_values, str) else raw_values
    except (TypeError, json.JSONDecodeError):
        values = []
    if not isinstance(values, list) or not values:
        fallback_value = triple.get("metric_value")
        if fallback_value not in (None, ""):
            values = [{"period": triple.get("metric_period") or filing_year, "value": fallback_value}]

    unit = str(triple.get("metric_unit", "reported units")).strip() or "reported units"
    evidence = str(triple.get("evidence_sentence") or "")
    number_tokens = _NUMBER_TOKEN_RE.findall(evidence)
    percent_tokens = _PERCENT_TOKEN_RE.findall(evidence)
    percentage_only_row = bool(number_tokens) and len(percent_tokens) == len(number_tokens)
    if percentage_only_row:
        unit = "percent"
        if metric_id.upper() == "REVENUE":
            return []
        mapped_metric = _PERCENT_METRIC_MAP.get(metric_id.upper())
        if mapped_metric:
            metric_id = mapped_metric.lower()
    currency, scale = split_unit(unit)
    # A percentage-of-revenue table always reports Revenue as 100%. That row
    # is a denominator label, not a revenue amount or growth observation.
    if metric_id.upper() == "REVENUE" and unit.casefold() == "percent":
        return []
    statement_type = str(triple.get("statement_type") or section or "UNKNOWN").strip()
    table_name = str(triple.get("table_name") or "UNKNOWN_TABLE").strip()
    row_label = str(triple.get("row_label") or triple.get("target") or metric_id).strip()
    comparability = str(triple.get("comparability_status") or "UNASSESSED").upper()
    observations: List[Dict[str, Any]] = []

    for item in values:
        if not isinstance(item, dict):
            continue
        raw_value = str(item.get("value", "")).strip()
        numeric_value = parse_numeric_value(raw_value)
        if numeric_value is None:
            continue
        period = str(item.get("period") or filing_year).strip()
        try:
            valid_from, valid_to, fiscal_year = fiscal_period_bounds(period)
        except ValueError:
            continue
        column_label = str(item.get("column_label") or period).strip()
        identity = "|".join(
            [claim_id, metric_id, period, raw_value, unit, source_filing, str(page)]
        )
        observation_id = "FO_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24].upper()
        measurement_identity = "|".join(
            [company_id, metric_id, period, raw_value, unit]
        )
        measurement_key = "FM_" + hashlib.sha256(
            measurement_identity.encode("utf-8")
        ).hexdigest()[:20].upper()
        observations.append(
            FinancialObservation(
                id=observation_id,
                measurement_key=measurement_key,
                company_id=company_id,
                metric_id=metric_id,
                claim_id=claim_id,
                source_filing=source_filing,
                page=int(page),
                fiscal_period=period,
                fiscal_year=fiscal_year,
                value=numeric_value,
                raw_value=raw_value,
                unit=unit,
                currency=currency,
                scale=scale,
                statement_type=statement_type,
                table_name=table_name,
                row_label=row_label,
                column_label=column_label,
                valid_from=valid_from,
                valid_to=valid_to,
                comparability_status=comparability,
            ).to_dict()
        )
    return observations
