"""Deterministic extraction of quantitative facts from SEC filing tables.

Tables are not causal prose.  This module emits evidence-backed
``REPORTS_METRIC`` facts while keeping the original row as the provenance
quote and recording values, units, and disclosed periods as metadata.
"""

from __future__ import annotations

import json
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


TABLE_METRICS: Sequence[Tuple[str, str]] = (
    ("cash and cash equivalents", "CASH_AND_CASH_EQUIVALENTS"),
    ("marketable securities", "MARKETABLE_SECURITIES"),
    ("accounts receivable", "ACCOUNTS_RECEIVABLE"),
    ("inventories", "INVENTORIES"),
    ("total current assets", "TOTAL_CURRENT_ASSETS"),
    ("total assets", "TOTAL_ASSETS"),
    ("total current liabilities", "TOTAL_CURRENT_LIABILITIES"),
    ("total liabilities", "TOTAL_LIABILITIES"),
    ("total shareholders' equity", "TOTAL_SHAREHOLDERS_EQUITY"),
    ("shareholders' equity", "TOTAL_SHAREHOLDERS_EQUITY"),
    ("cost of revenue", "COST_OF_REVENUE"),
    ("total revenue", "REVENUE"),
    ("net income per diluted share", "EARNINGS_PER_SHARE"),
    ("diluted earnings per share", "EARNINGS_PER_SHARE"),
    ("basic earnings per share", "EARNINGS_PER_SHARE"),
    ("research and development expenses", "R_AND_D_EXPENSE"),
    ("gross profit", "GROSS_PROFIT"),
    ("gross margin", "GROSS_MARGIN"),
    ("income before income tax", "PRETAX_INCOME"),
    ("income tax expense", "INCOME_TAX_EXPENSE"),
    ("operating expenses", "OPERATING_COST"),
    ("total operating expenses", "OPERATING_COST"),
    ("research and development", "R_AND_D_EXPENSE"),
    ("sales general and administrative", "SG_AND_A_EXPENSE"),
    ("operating income", "OPERATING_INCOME"),
    ("net income", "NET_INCOME"),
    ("revenue", "REVENUE"),
    ("cash flow from operating activities", "OPERATING_CASH_FLOW"),
    ("capital expenditures", "CAPEX"),
)

_NUMBER_RE = re.compile(r"(?<![A-Za-z])\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")
_YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")


def _clean_cell(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _row_text(row: Iterable[object]) -> str:
    return " ".join(cell for cell in (_clean_cell(v) for v in row) if cell).strip()


def _find_metric(row_text: str) -> Optional[Tuple[str, str]]:
    lower = _normalise(row_text)
    # Match the semantic row label, not an arbitrary substring in a longer
    # disclosure.  This prevents ``% of net revenue`` and ``deferred revenue``
    # from becoming false Revenue facts, and prevents tax-credit rows from
    # becoming R&D facts.
    lower = re.sub(
        r"\bup\s+[-(]?\d[\d,]*(?:\.\d+)?%?\)?(?:\s+pts?)?",
        " ",
        lower,
        flags=re.IGNORECASE,
    )
    label = _NUMBER_RE.sub(" ", lower)
    label = re.sub(r"[$%(),*\u2020\u2021]", " ", label)
    label = re.sub(r"\bchange\b", " ", label)
    label = re.sub(r"\s+", " ", label).strip()
    for alias, metric_id in sorted(TABLE_METRICS, key=lambda item: -len(item[0])):
        if label == alias:
            return alias, metric_id
        if label.startswith(alias + " "):
            suffix = label[len(alias) + 1:]
            if metric_id in {"REVENUE", "GROSS_PROFIT", "NET_INCOME"}:
                continue
            if suffix in {"loss", "expenses", "expense"}:
                return alias, metric_id
        if metric_id == "REVENUE" and label == "total revenue":
            return alias, metric_id
    return None


def _numeric_values(row_text: str) -> List[str]:
    # Change annotations such as "Up 114%" are not reported values.
    row_text = re.sub(
        r"\bup\s+[-(]?\d[\d,]*(?:\.\d+)?%?\)?(?:\s+pts?)?",
        "",
        row_text,
        flags=re.IGNORECASE,
    )
    # Financial tables commonly append a year-over-year change percentage
    # after the reported dollar values (``$ 12,914 ... 49 %``).  It is not a
    # value belonging to the metric row.  Percentage-of-revenue rows do not
    # contain a dollar marker, so they remain unaffected.
    row_text = re.sub(
        r"\s+\(?-?\d[\d,]*(?:\.\d+)?\)?\s*%\s*$",
        "",
        row_text,
    )
    values = []
    for match in _NUMBER_RE.findall(row_text):
        parenthesized = match.startswith("(") and match.endswith(")")
        cleaned = match.replace("$", "").replace(",", "").strip("()")
        # SEC tables use accounting parentheses for negative amounts.  Keep
        # that sign in the structured value instead of silently turning cash
        # outflows and contra-balances into positive figures.
        if parenthesized and cleaned and not cleaned.startswith("-"):
            cleaned = f"-{cleaned}"
        if cleaned and cleaned != "-":
            values.append(cleaned)
    return values


def _find_exact_row(page_text: str, row_text: str, metric_alias: str) -> str:
    target = _normalise(row_text)
    lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
    for line in lines:
        if _normalise(line) == target:
            return line[:500]
    alias = _normalise(metric_alias)
    target_values = _numeric_values(row_text)
    for line in lines:
        if alias in _normalise(line) and _numeric_values(line) == target_values:
            return line[:500]
    for line in lines:
        if alias in _normalise(line) and _numeric_values(line):
            return line[:500]
    return ""


def _periods(page_text: str, filing_year: int) -> List[int]:
    """Infer periods from the table header before falling back to page years.

    MD&A pages often mention an older filing in surrounding prose.  Reading
    every year on the page therefore overstates the periods represented by a
    table (for example, a 2025/2024 percentage table on a page that also
    mentions the 2023 filing).  Prefer the compact ``Year Ended`` header
    window, which is the actual table scope.
    """
    text = str(page_text or "")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header_markers = ("year ended", "years ended")
    for index, line in enumerate(lines):
        lower = line.lower()
        if lower not in header_markers:
            continue
        window = " ".join(lines[index:index + 4])
        years = []
        for value in _YEAR_RE.findall(window):
            year = int(value)
            if year not in years:
                years.append(year)
        if len(years) >= 2:
            return years[:5]

    years = []
    for value in _YEAR_RE.findall(text):
        year = int(value)
        if year not in years:
            years.append(year)
    return years[:5] or [filing_year]


def _periods_for_evidence(page_text: str, evidence: str, filing_year: int) -> List[int]:
    """Use the nearest table header for pages containing multiple tables."""
    lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
    target = _normalise(evidence)
    matches = [i for i, line in enumerate(lines) if _normalise(line) == target]
    if not matches:
        return _periods(page_text, filing_year)
    index = matches[0]
    for cursor in range(index, max(-1, index - 14), -1):
        # Date columns are frequently split across two lines, e.g.
        # ``Jan 26, 2025 Jan 28, 2024 Change`` or ``January 29, January 30,``
        # followed by ``2023 2022 Change``. Read a compact header window rather
        # than assigning periods from the first line containing two years.
        window = " ".join(lines[max(0, cursor - 1):cursor + 2])
        years = []
        for value in _YEAR_RE.findall(window):
            year = int(value)
            if year not in years:
                years.append(year)
        if len(years) >= 2:
            # A comparison table contains two reported-period columns followed
            # by dollar-change and percent-change columns. Only the first two
            # dates are value semantics for each metric row.
            return years[:2]
    return _periods(page_text, filing_year)


def _is_percentage_table(page_text: str) -> bool:
    context = _normalise(page_text)
    return (
        "percentage of revenue" in context
        or "expressed as a percentage" in context
    )


def _unit(page_text: str, row_text: str) -> str:
    context = f"{page_text}\n{row_text}".lower()
    value_text = re.sub(
        r"\bup\s+[-(]?\d[\d,]*(?:\.\d+)?%?\)?(?:\s+pts?)?",
        "",
        row_text,
        flags=re.IGNORECASE,
    )
    # SEC MD&A comparison tables often place a single currency marker in the
    # header/first row, so subsequent rows contain no literal ``$``. A header
    # such as ``$ %`` plus ``($ in millions)`` still makes the row a currency
    # disclosure; the final percent belongs to the change column and is
    # removed by _numeric_values.
    currency_table = bool(
        re.search(r"\(\$\s+in\s+(?:millions|thousands)\)", page_text, re.IGNORECASE)
        or re.search(r"(?m)^\s*\$\s+%\s*$", page_text)
    )
    if "$" in row_text or currency_table:
        if "per diluted share" in row_text.lower() or "per share" in row_text.lower():
            return "USD per share"
        if "in millions" in context:
            return "USD millions"
        if "in thousands" in context:
            return "USD thousands"
        return "USD"
    if "%" in value_text or "margin" in row_text.lower():
        return "percent"
    if "in millions" in context:
        return "USD millions"
    if "in thousands" in context:
        return "USD thousands"
    # pdfplumber can drop the percent glyph from a percentage table row; the
    # surrounding MD&A table still identifies it as a percent-of-revenue view.
    if _is_percentage_table(page_text):
        return "percent"
    return "reported units"


def extract_financial_table_triples(page, page_text: str, filing_year: int) -> List[Dict]:
    """Extract strict numeric disclosure triples from one PDF page."""
    try:
        tables = page.extract_tables() or []
    except Exception:
        return []

    triples: List[Dict] = []
    seen = set()
    for table in tables:
        for row in table or []:
            row_text = _row_text(row)
            metric = _find_metric(row_text)
            if not metric or not _numeric_values(row_text):
                continue
            metric_alias, metric_id = metric
            # MD&A contains a percentage-of-revenue table whose row labels
            # reuse income-statement names.  In that context "Gross profit
            # 75.0" is a margin percentage, not USD gross profit.
            percentage_table = _is_percentage_table(page_text)
            if percentage_table and metric_id == "GROSS_PROFIT":
                metric_id = "GROSS_MARGIN"
            elif percentage_table and metric_id == "OPERATING_INCOME":
                metric_id = "OPERATING_MARGIN"
            evidence = _find_exact_row(page_text, row_text, metric_alias)
            if len(evidence) < 20:
                continue
            periods = _periods_for_evidence(page_text, evidence, filing_year)
            period_text = ",".join(str(year) for year in periods)
            unit = _unit(page_text, evidence)
            # The table may report dollars while nearby prose mentions a
            # percentage view of the same metric.  Use row-level units first.
            if unit == "percent" and "$" in evidence:
                unit = "USD millions" if "in millions" in page_text.lower() else "USD"
            key = (metric_id, evidence)
            if key in seen:
                continue
            seen.add(key)
            # pdfplumber may split currency symbols into separate cells.  The
            # exact page line is the authoritative row for values and units.
            values = _numeric_values(evidence)
            if not values:
                values = _numeric_values(row_text)
            # Values are positionally paired with the reported period columns.
            # Discard trailing dollar-change/percentage-change columns instead
            # of presenting them as additional fiscal-year values.
            if periods and len(values) > len(periods):
                values = values[:len(periods)]
            if not values:
                continue
            period_values = [
                {"period": str(period), "value": value}
                for period, value in zip(periods, values)
            ]
            triples.append({
                "source": "NVIDIA_CORPORATION",
                "source_category": "Company",
                "target": metric_id,
                "target_category": "FinancialMetric",
                "relation": "REPORTS_METRIC",
                "causal_strength": "DISCLOSED_ONLY",
                "relation_polarity": "reported",
                "modality": "observed",
                "temporal_scope": period_text,
                "evidence_sentence": evidence,
                "metric_values_json": json.dumps(period_values, ensure_ascii=False),
                "metric_value": values[0],
                "metric_unit": _unit(page_text, evidence),
                "metric_period": str(periods[0]) if periods else str(filing_year),
            })
    return triples
