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


def _table_name(page_text: str, evidence: str) -> str:
    """Recover the nearest human-readable table heading above a metric row."""
    lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
    target = _normalise(evidence)
    row_index = next(
        (index for index, line in enumerate(lines) if _normalise(line) == target),
        len(lines),
    )
    rejected = re.compile(
        r"^(year|years) ended$|^\$\s*%$|^\(?\$ in |^\(?in\s+(?:millions|thousands)\)?$|^\(continued\)$|^20\d{2}(?:\s+20\d{2})*",
        re.IGNORECASE,
    )

    # Prefer document/table-level headings over the nearest prose paragraph.
    # In SEC text extraction, a row can be separated from its heading by many
    # rows, while an unrelated subsection title may be closer in raw order.
    for line in reversed(lines[max(0, row_index - 30):row_index]):
        lower = line.lower()
        if (
            "consolidated statements" in lower
            or "consolidated balance sheets" in lower
            or lower in {
                "liquidity and capital resources",
                "results of operations",
            }
        ):
            return line

    # Summary/section tables normally put a short title directly above the
    # period header (for example, "Fiscal Year 2023 Summary").
    period_index = next(
        (
            index
            for index in range(row_index - 1, -1, -1)
            if re.search(r"^years? ended$", lines[index], re.IGNORECASE)
        ),
        None,
    )
    if period_index is not None:
        for line in reversed(lines[max(0, period_index - 6):period_index]):
            lower = line.lower()
            if rejected.search(line):
                continue
            if (
                "following table" in lower
                or "summary" in lower
                or (len(line) <= 80 and not _NUMBER_RE.search(line))
            ):
                return line

    for line in reversed(lines[max(0, row_index - 12):row_index]):
        if rejected.search(line) or _NUMBER_RE.search(line):
            continue
        if 4 <= len(line) <= 120:
            return line
    return "UNKNOWN_TABLE"


def _table_context(page_text: str, evidence: str, max_chars: int = 500) -> str:
    """Return a verbatim page excerpt containing the table header and row.

    A metric row by itself is not a self-contained evidence unit: it loses the
    statement title, period columns, and scale.  Keep the row values separate
    for parsing, but persist a compact, page-verbatim context excerpt for
    citation and human review.
    """
    raw_lines = [line.strip() for line in str(page_text or "").splitlines() if line.strip()]
    target = _normalise(evidence)
    row_index = next(
        (index for index, line in enumerate(raw_lines) if _normalise(line) == target),
        None,
    )
    if row_index is None:
        return str(evidence or "")[:max_chars]

    header_index = next(
        (
            index
            for index in range(row_index - 1, -1, -1)
            if re.search(r"^years? ended$", raw_lines[index], re.IGNORECASE)
        ),
        None,
    )
    if header_index is not None:
        # Keep the table description, period header, scale line when present,
        # and the target row. Do not copy every intervening metric row into a
        # single claim; that would blur which values support this metric.
        description_index = next(
            (
                index
                for index in range(header_index - 1, max(-1, header_index - 4), -1)
                if "following table" in raw_lines[index].lower()
                or "consolidated statements" in raw_lines[index].lower()
            ),
            None,
        )
        start = description_index if description_index is not None else max(0, header_index - 1)
        header_end = header_index
        for index in range(header_index + 1, min(row_index, header_index + 5) + 1):
            line = raw_lines[index]
            is_date_header = bool(
                _YEAR_RE.search(line)
                or re.search(
                    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|"
                    r"may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|"
                    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
                    line,
                    re.IGNORECASE,
                )
            )
            is_scale_header = bool(
                re.search(r"\(\$\s+in\s+(?:millions|thousands)\)", line, re.IGNORECASE)
                or re.search(r"^\s*\$\s+%\s*$", line)
            )
            if not (is_date_header or is_scale_header):
                break
            header_end = index
        selected = raw_lines[start:header_end + 1]
        if row_index > header_end:
            selected.append(raw_lines[row_index])
    else:
        table_name = _table_name(page_text, evidence)
        heading_index = next(
            (
                index
                for index in range(row_index - 1, -1, -1)
                if _normalise(raw_lines[index]) == _normalise(table_name)
            ),
            max(0, row_index - 3),
        )
        selected = raw_lines[heading_index:row_index + 1]

    candidate = "\n".join(selected)
    if len(candidate) <= max_chars:
        return candidate
    row_line = raw_lines[row_index]
    header = "\n".join(selected[:-1]) if selected[-1] == row_line else selected[0]
    available = max_chars - len(row_line) - 1
    if available <= 0:
        return row_line[-max_chars:]
    return f"{header[:available]}\n{row_line}"


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
            elif percentage_table and metric_id == "NET_INCOME":
                metric_id = "NET_MARGIN"
            elif percentage_table and metric_id == "COST_OF_REVENUE":
                metric_id = "COST_OF_REVENUE_RATIO"
            elif percentage_table and metric_id == "OPERATING_COST":
                metric_id = "OPERATING_EXPENSE_RATIO"
            elif percentage_table and metric_id == "R_AND_D_EXPENSE":
                metric_id = "R_AND_D_RATIO"
            elif percentage_table and metric_id == "SG_AND_A_EXPENSE":
                metric_id = "SG_AND_A_RATIO"
            elif percentage_table and metric_id == "REVENUE":
                # Revenue=100% is the denominator of this presentation, not
                # an amount or a growth metric.
                continue
            row_evidence = _find_exact_row(page_text, row_text, metric_alias)
            if len(row_evidence) < 20:
                continue
            evidence = _table_context(page_text, row_evidence)
            periods = _periods_for_evidence(page_text, row_evidence, filing_year)
            period_text = ",".join(str(year) for year in periods)
            unit = _unit(page_text, row_evidence)
            # The table may report dollars while nearby prose mentions a
            # percentage view of the same metric.  Use row-level units first.
            if unit == "percent" and "$" in row_evidence:
                unit = "USD millions" if "in millions" in page_text.lower() else "USD"
            key = (metric_id, row_evidence)
            if key in seen:
                continue
            seen.add(key)
            # pdfplumber may split currency symbols into separate cells.  The
            # exact page line is the authoritative row for values and units.
            values = _numeric_values(row_evidence)
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
                # The row is the cross-parser verbatim citation unit.  The
                # surrounding header/period/unit context is persisted
                # separately for human review and table interpretation.
                "evidence_sentence": row_evidence,
                "row_evidence": row_evidence,
                "table_context": evidence,
                "metric_values_json": json.dumps(period_values, ensure_ascii=False),
                "metric_value": values[0],
                "metric_unit": _unit(page_text, row_evidence),
                "metric_period": str(periods[0]) if periods else str(filing_year),
                # Resolve the heading from the exact row, not from the
                # multi-line citation context.  Passing the full context
                # makes the row lookup miss and can select an unrelated
                # heading near the end of the page.
                "table_name": _table_name(page_text, row_evidence),
                "row_label": metric_alias,
                "statement_type": "FINANCIAL_TABLE",
                "comparability_status": "UNASSESSED",
            })
    return triples
