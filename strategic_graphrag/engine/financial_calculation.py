"""Fail-closed deterministic calculations over typed filing observations."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from decimal import Decimal, DecimalException
from typing import Any, Iterable

from ..schema.financial_observation import FinancialObservation, split_unit


_SCALE = {
    "units": Decimal("1"),
    "unit": Decimal("1"),
    "thousands": Decimal("1000"),
    "thousand": Decimal("1000"),
    "millions": Decimal("1000000"),
    "million": Decimal("1000000"),
    "billions": Decimal("1000000000"),
    "billion": Decimal("1000000000"),
}
_CURRENCY_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
_ANNUAL_PERIOD_RE = re.compile(r"^(?:(?:FY|FISCAL\s+YEAR)\s*)?(20\d{2})$", re.IGNORECASE)
_QUARTER_FIRST_RE = re.compile(r"^Q([1-4])\s*(?:FY\s*)?(20\d{2})$", re.IGNORECASE)
_QUARTER_LAST_RE = re.compile(r"^(?:FY\s*)?(20\d{2})\s*Q([1-4])$", re.IGNORECASE)


def _normalized_scale(value: Any) -> str | None:
    scale = str(value or "").strip().casefold()
    aliases = {
        "unit": "units", "thousand": "thousands",
        "million": "millions", "billion": "billions",
    }
    scale = aliases.get(scale, scale)
    return scale or None


def _unit_metadata_conflict(observation: FinancialObservation) -> str | None:
    parsed_currency, parsed_scale = split_unit(observation.unit)
    explicit_currency = str(observation.currency or "").strip().upper() or None
    explicit_scale = _normalized_scale(observation.scale)
    unit = str(observation.unit or "").strip().casefold()
    if parsed_currency and explicit_currency and parsed_currency != explicit_currency:
        return "currency_field_conflicts_with_unit"
    if parsed_scale and explicit_scale and _normalized_scale(parsed_scale) != explicit_scale:
        return "scale_field_conflicts_with_unit"
    if unit in {"percent", "%", "percentage", "percentage points"}:
        if explicit_currency or (explicit_scale and explicit_scale != "percent"):
            return "percentage_unit_conflicts_with_currency_or_scale"
    return None


def _period_descriptor(value: Any) -> tuple[str, int] | None:
    text = re.sub(r"\s+", " ", str(value or "").strip()).upper()
    quarter = _QUARTER_FIRST_RE.fullmatch(text)
    if quarter:
        return f"Q{quarter.group(1)}", int(quarter.group(2))
    quarter = _QUARTER_LAST_RE.fullmatch(text)
    if quarter:
        return f"Q{quarter.group(2)}", int(quarter.group(1))
    annual = _ANNUAL_PERIOD_RE.fullmatch(text)
    if annual:
        return "FY", int(annual.group(1))
    return None


def _period_contract_error(observation: FinancialObservation) -> str | None:
    descriptor = _period_descriptor(observation.fiscal_period)
    if descriptor is None:
        return "unsupported_or_unresolved_observation_period"
    if descriptor[1] != observation.fiscal_year:
        return "fiscal_period_year_conflicts_with_fiscal_year"
    return None


def _line_provenance_signature(observation: FinancialObservation) -> tuple[str, ...]:
    return (
        observation.company_id.strip().upper().replace(" ", "_"),
        observation.metric_id.strip().upper().replace(" ", "_"),
        observation.claim_id.strip(),
        observation.source_filing.strip().casefold(),
        str(observation.page),
        observation.table_name.strip(),
        observation.row_label.strip().casefold(),
        observation.unit.strip().casefold(),
        str(observation.currency or "").strip().upper(),
        _normalized_scale(observation.scale) or "",
        str(observation.build_id or "").strip(),
    )


def _comparability_assessment(
    items: list[FinancialObservation],
) -> tuple[dict[str, Any] | None, str | None]:
    statuses = {str(item.comparability_status or "").strip().upper() for item in items}
    if "NOT_COMPARABLE" in statuses:
        return None, "observations_explicitly_not_comparable"
    if len(statuses) != 1:
        return None, "observation_comparability_conflict"
    if statuses == {"COMPARABLE"}:
        return {"status": "COMPARABLE", "basis": "explicit_observation_status"}, None
    if statuses != {"UNASSESSED"}:
        return None, "observation_comparability_not_confirmed"

    signatures = {_line_provenance_signature(item) for item in items}
    if len(signatures) != 1:
        return None, "observation_comparability_not_confirmed"
    sample = items[0]
    source_fields_present = bool(
        sample.company_id
        and sample.metric_id
        and sample.claim_id
        and sample.source_filing
        and sample.page > 0
        and sample.table_name
        and sample.table_name != "UNKNOWN_TABLE"
        and sample.row_label
        and sample.unit
        and sample.build_id
    )
    if not source_fields_present:
        return None, "observation_comparability_not_confirmed"
    return {
        "status": "CONDITIONALLY_COMPARABLE",
        "basis": "same_disclosure_claim_page_table_row_and_unit",
        "disclosure_version": sample.source_filing,
        "evidence_id": sample.claim_id,
        "page": sample.page,
        "table_name": sample.table_name,
        "row_label": sample.row_label,
    }, None


def _comparison_reason(assessment: dict[str, Any]) -> str:
    if assessment.get("status") == "CONDITIONALLY_COMPARABLE":
        return "same_disclosure_row_conditionally_comparable"
    return "same_metric_comparable_years"


def _public_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _result(status: str, operation: str, reason: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": "deterministic-calculation/v1",
        "status": status,
        "operation": operation,
        "observations": [],
        "value": None,
        "unit": None,
        "display": reason.replace("_", " ").capitalize() + ".",
        "reason_code": reason,
        **extra,
    }


def _record(observation: FinancialObservation) -> dict[str, Any]:
    record = asdict(observation)
    record["fact_period"] = observation.fiscal_period
    record["disclosure_version"] = observation.source_filing
    record["evidence_id"] = observation.claim_id
    record["source"] = f"{observation.source_filing}#page={observation.page}"
    return record


def _year(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        match = _CURRENCY_RE.search(str(value or ""))
        return int(match.group(1)) if match else None


def _unit(observation: FinancialObservation) -> tuple[str | None, str | None, str]:
    parsed_currency, parsed_scale = split_unit(observation.unit)
    currency = str(observation.currency or parsed_currency or "").strip().upper() or None
    scale = _normalized_scale(observation.scale or parsed_scale)
    normalized_unit = str(observation.unit or "").strip()
    if normalized_unit.casefold() in {"percent", "%", "percentage", "percentage points"}:
        scale = "percent"
    return currency, scale, normalized_unit


def _basis_value(
    observation: FinancialObservation,
) -> tuple[tuple[str, str], Decimal] | None:
    currency, scale, unit = _unit(observation)
    try:
        value = Decimal(str(observation.value))
    except (DecimalException, TypeError, ValueError):
        return None
    if not value.is_finite():
        return None
    if currency and scale in _SCALE:
        return ("currency", currency), value * _SCALE[scale]
    if scale == "percent" or unit.casefold() in {"percent", "%", "percentage", "percentage points"}:
        return ("percent", "percent"), value
    if unit:
        return ("unit", unit.casefold()), value
    return None


def _valid_observations(
    items: Iterable[Any],
) -> tuple[list[FinancialObservation], int, list[tuple[FinancialObservation, str]]]:
    valid: list[FinancialObservation] = []
    invalid = 0
    contract_errors: list[tuple[FinancialObservation, str]] = []
    for item in items:
        if not isinstance(item, FinancialObservation):
            invalid += 1
            continue
        try:
            metadata_ok = bool(
                item.company_id
                and item.metric_id
                and item.claim_id
                and item.source_filing
                and item.build_id
                and item.page > 0
                and item.fiscal_year > 0
                and item.fiscal_period
                and item.unit
            )
            if metadata_ok:
                contract_error = _unit_metadata_conflict(item) or _period_contract_error(item)
                if contract_error:
                    contract_errors.append((item, contract_error))
                    continue
            basis = _basis_value(item)
        except (TypeError, ValueError):
            basis = None
            metadata_ok = False
        if not basis or not metadata_ok:
            invalid += 1
            continue
        valid.append(item)
    return valid, invalid, contract_errors


def _requested_years(plan: dict[str, Any], operation: str) -> tuple[list[int], bool]:
    start = _year(plan.get("fiscal_year_start"))
    end = _year(plan.get("fiscal_year_end"))
    explicit_years = sorted({
        parsed for value in (plan.get("fiscal_years") or [])
        if (parsed := _year(value)) is not None
    })
    fact_year = _year(plan.get("fact_period"))
    comparison = str(plan.get("comparison") or "").upper()

    if operation in {"CROSS_YEAR", "ABSOLUTE_CHANGE", "PERCENT_CHANGE"}:
        if len(explicit_years) >= 2:
            if operation == "CROSS_YEAR":
                return explicit_years, False
            return [explicit_years[0], explicit_years[-1]], False
        if start is None or end is None:
            return [], False
        if start == end:
            return [], True
        if operation == "CROSS_YEAR" and comparison == "DELTA_OVER_TIME":
            return list(range(min(start, end), max(start, end) + 1)), False
        return sorted({start, end}), False

    if start is not None and end is not None and start != end:
        return [], True
    one_year = start if start is not None else end
    if fact_year is not None:
        if one_year is not None and one_year != fact_year:
            return [], True
        one_year = fact_year
    return ([one_year] if one_year is not None else []), False


def _version_filter(plan: dict[str, Any], items: list[FinancialObservation]) -> tuple[list[FinancialObservation], str | None]:
    document_scope = str(plan.get("document_scope") or "").strip().casefold()
    disclosure_as_of = str(plan.get("disclosure_as_of") or "").strip()
    version_year = _year(disclosure_as_of)
    if document_scope and version_year:
        scope_year = _year(document_scope)
        if scope_year is not None and scope_year != version_year:
            return [], "conflicting_disclosure_scope"

    selected = items
    if document_scope:
        selected = [
            item for item in selected
            if item.source_filing.strip().casefold() == document_scope
        ]
    if version_year is not None:
        selected = [
            item for item in selected
            if _year(item.source_filing) == version_year
        ]
    return selected, None


def _choose_year(
    year: int,
    items: list[FinancialObservation],
    *,
    require_comparability: bool = False,
) -> tuple[FinancialObservation | None, str | None]:
    candidates = [item for item in items if item.fiscal_year == year]
    if not candidates:
        return None, None
    normalized: list[tuple[FinancialObservation, tuple[str, str], Decimal]] = []
    for item in candidates:
        measured = _basis_value(item)
        if measured is None:
            continue
        basis, value = measured
        normalized.append((item, basis, value))
    if not normalized:
        return None, None
    bases = {basis for _, basis, _ in normalized}
    values = {value for _, _, value in normalized}
    periods = {_period_descriptor(item.fiscal_period) for item, _, _ in normalized}
    companies = {item.company_id.strip().upper().replace(" ", "_") for item, _, _ in normalized}
    if len(companies) != 1:
        return None, "company_scope_required_for_multiple_companies"
    comparability = {
        str(item.comparability_status or "").strip().upper()
        for item, _, _ in normalized
    }
    if require_comparability and len(comparability) != 1:
        return None, "observation_comparability_conflict"
    if require_comparability and comparability == {"UNASSESSED"}:
        if len({_line_provenance_signature(item) for item, _, _ in normalized}) != 1:
            return None, "observation_comparability_provenance_conflict"
    if len(bases) != 1 or len(values) != 1 or len(periods) != 1:
        return None, "conflicting_fact_values"
    normalized.sort(key=lambda row: (row[0].source_filing, row[0].claim_id, row[0].id))
    return normalized[0][0], None


def _compatible_pair(
    first: FinancialObservation,
    second: FinancialObservation,
) -> tuple[Decimal, Decimal, tuple[str, str]] | None:
    first_measured = _basis_value(first)
    second_measured = _basis_value(second)
    if first_measured is None or second_measured is None:
        return None
    if first_measured[0] != second_measured[0]:
        return None
    return first_measured[1], second_measured[1], first_measured[0]


def calculate_public(
    paths: Iterable[Any],
    *,
    query_plan: dict[str, Any],
) -> dict[str, Any]:
    """Calculate only when a typed plan and complete, typed observations agree."""
    plan = query_plan if isinstance(query_plan, dict) else {}
    task = str(plan.get("task_type") or "UNCLASSIFIED").upper()
    if task == "RELATION":
        return _result("NOT_APPLICABLE", "none", "causal_relation_query_is_not_numeric_calculation")
    raw_operation = str(plan.get("calculation") or "").strip().upper()
    aliases = {
        "DETERMINISTIC_FINANCIAL_OPERATION": "UNSPECIFIED",
        "GROWTH_RATE": "PERCENT_CHANGE",
        "PERCENTAGE_CHANGE": "PERCENT_CHANGE",
        "ABSOLUTE_DIFFERENCE": "ABSOLUTE_CHANGE",
        "DIFFERENCE": "ABSOLUTE_CHANGE",
    }
    operation = aliases.get(raw_operation, raw_operation)
    if not operation:
        if task == "FACT":
            operation = "FACT"
        elif task == "COMPARISON":
            operation = "CROSS_YEAR"
        elif task in {"RELATION", "UNCLASSIFIED"}:
            return _result("NOT_APPLICABLE", "none", "not_a_numeric_fact_request")
        else:
            operation = "UNSPECIFIED"

    names = {
        "FACT": "fact",
        "CROSS_YEAR": "cross_year",
        "UNIT_CONVERSION": "unit_conversion",
        "ABSOLUTE_CHANGE": "absolute_change",
        "PERCENT_CHANGE": "percent_change",
    }
    operation_name = names.get(operation, operation.lower() or "none")
    if operation == "UNSPECIFIED" or operation not in names:
        return _result("OPERATION_UNSUPPORTED", operation_name, "operation_not_supported_by_structured_plan")

    ambiguities = plan.get("ambiguity") or []
    metric = str(plan.get("target_metric") or "").strip().upper().replace(" ", "_")
    if ambiguities or not metric:
        return _result("AMBIGUOUS", operation_name, "query_plan_has_unresolved_fields")

    requested_years, year_ambiguous = _requested_years(plan, operation)
    if year_ambiguous:
        return _result("AMBIGUOUS", operation_name, "query_plan_year_scope_is_inconsistent")
    requested_period = plan.get("fact_period") or plan.get("fiscal_period")
    period_constraint = _period_descriptor(requested_period)
    if requested_period and period_constraint is None:
        return _result(
            "AMBIGUOUS", operation_name, "query_plan_period_is_unresolved",
            requested_fact_period=requested_period,
        )

    raw_observations: list[Any] = []
    for path in paths or []:
        raw_observations.extend(getattr(path, "financial_observations", []) or [])
    observations, invalid_count, contract_errors = _valid_observations(raw_observations)
    company = str(plan.get("company") or "").strip().upper().replace(" ", "_")
    expected_build_id = str(plan.get("build_id") or "").strip()
    metric_matches = lambda item: item.metric_id.strip().upper().replace(" ", "_") == metric
    relevant_errors = [
        (item, reason) for item, reason in contract_errors
        if metric_matches(item)
        and (not company or item.company_id.strip().upper().replace(" ", "_") == company)
        and (not expected_build_id or item.build_id == expected_build_id)
    ]
    scoped_error_items, scope_error = _version_filter(
        plan, [item for item, _ in relevant_errors]
    )
    if scope_error:
        return _result("AMBIGUOUS", operation_name, scope_error)
    scoped_error_ids = {id(item) for item in scoped_error_items}
    relevant_errors = [
        (item, reason) for item, reason in relevant_errors
        if id(item) in scoped_error_ids
    ]
    requested_year_set = set(requested_years)
    relevant_errors = [
        (item, reason) for item, reason in relevant_errors
        if not requested_year_set
        or not ({_year(item.fiscal_period), item.fiscal_year} - {None}).isdisjoint(requested_year_set)
        or not _year(item.fiscal_period) or item.fiscal_year <= 0
    ]
    if period_constraint is not None:
        relevant_errors = [
            (item, reason) for item, reason in relevant_errors
            if _period_descriptor(item.fiscal_period) is None
            or _period_descriptor(item.fiscal_period)[0] == period_constraint[0]
        ]
    if relevant_errors:
        first_error = relevant_errors[0][1]
        return _result(
            "INVALID_INPUT", operation_name, first_error,
            invalid_observations=len(relevant_errors),
            observations=[_record(item) for item, _ in relevant_errors],
        )
    observations = [
        item for item in observations
        if metric_matches(item)
        and (not company or item.company_id.strip().upper().replace(" ", "_") == company)
    ]
    observations, scope_error = _version_filter(plan, observations)
    if scope_error:
        return _result("AMBIGUOUS", operation_name, scope_error)
    if not observations:
        return _result(
            "INSUFFICIENT_EVIDENCE", operation_name,
            "no_complete_typed_observation_for_requested_company_metric_and_disclosure",
            invalid_observations=invalid_count,
        )
    if not expected_build_id:
        return _result(
            "INSUFFICIENT_EVIDENCE", operation_name,
            "candidate_build_identity_missing",
            observations=[_record(item) for item in observations],
        )
    observed_build_ids = {str(item.build_id or "").strip() for item in observations}
    if observed_build_ids != {expected_build_id}:
        return _result(
            "AMBIGUOUS", operation_name, "observation_build_identity_mismatch",
            expected_build_id=expected_build_id,
            observed_build_ids=sorted(observed_build_ids),
            observations=[_record(item) for item in observations],
        )

    if period_constraint is not None:
        observations = [
            item for item in observations
            if _period_descriptor(item.fiscal_period)
            and _period_descriptor(item.fiscal_period)[0] == period_constraint[0]
        ]
        if not observations:
            return _result(
                "INSUFFICIENT_EVIDENCE", operation_name,
                "requested_fact_period_granularity_missing",
                requested_fact_period=requested_period,
                invalid_observations=invalid_count,
            )

    years_available = sorted({item.fiscal_year for item in observations})
    if not requested_years:
        if operation in {"CROSS_YEAR", "ABSOLUTE_CHANGE", "PERCENT_CHANGE"}:
            return _result(
                "AMBIGUOUS", operation_name, "explicit_comparison_years_required",
                observations=[_record(item) for item in observations],
                available_fact_years=years_available,
            )
        if len(years_available) != 1:
            return _result("AMBIGUOUS", operation_name, "fact_year_not_resolved_and_multiple_years_available")
        requested_years = years_available

    if operation in {"CROSS_YEAR", "ABSOLUTE_CHANGE", "PERCENT_CHANGE"} and period_constraint is None:
        return _result(
            "AMBIGUOUS", operation_name, "comparison_period_granularity_required",
            observations=[_record(item) for item in observations],
            requested_years=requested_years,
        )
    if operation in {"FACT", "UNIT_CONVERSION"} and period_constraint is None and requested_years:
        return _result(
            "AMBIGUOUS", operation_name, "fact_period_granularity_required",
            observations=[_record(item) for item in observations],
            requested_years=requested_years,
        )

    selected: dict[int, FinancialObservation] = {}
    for year in requested_years:
        item, conflict = _choose_year(
            year,
            observations,
            require_comparability=operation in {"CROSS_YEAR", "ABSOLUTE_CHANGE", "PERCENT_CHANGE"},
        )
        if conflict:
            return _result(
                "AMBIGUOUS", operation_name, conflict,
                fact_year=year,
                conflicting_observations=[
                    _record(candidate) for candidate in observations
                    if candidate.fiscal_year == year
                ],
            )
        if item is not None:
            selected[year] = item
    missing_years = [year for year in requested_years if year not in selected]
    if missing_years:
        return _result(
            "INSUFFICIENT_EVIDENCE", operation_name, "requested_fact_years_missing",
            missing_years=missing_years,
            observations=[_record(item) for item in observations],
        )

    selected_items = [selected[year] for year in requested_years]
    selected_companies = {
        item.company_id.strip().upper().replace(" ", "_") for item in selected_items
    }
    if not company and len(selected_companies) != 1:
        return _result(
            "AMBIGUOUS", operation_name, "company_scope_required_for_multiple_companies",
            available_companies=sorted(selected_companies),
            observations=[_record(item) for item in selected_items],
        )
    detailed_records = [
        _record(item) for item in observations
        if item.fiscal_year in requested_years
    ]
    if operation in {"CROSS_YEAR", "ABSOLUTE_CHANGE", "PERCENT_CHANGE"}:
        period_families = {
            _period_descriptor(item.fiscal_period)[0] for item in selected_items
        }
        if len(period_families) != 1:
            return _result(
                "AMBIGUOUS", operation_name, "comparison_period_granularity_mismatch",
                observations=detailed_records,
            )
        comparability_assessment, comparability_error = _comparability_assessment(selected_items)
        if comparability_error:
            status = (
                "AMBIGUOUS"
                if comparability_error in {
                    "observations_explicitly_not_comparable",
                    "observation_comparability_conflict",
                    "observation_comparability_provenance_conflict",
                }
                else "INSUFFICIENT_EVIDENCE"
            )
            return _result(
                status, operation_name, comparability_error,
                observations=detailed_records,
            )
    if operation == "FACT":
        if len(selected_items) != 1:
            return _result("AMBIGUOUS", operation_name, "single_fact_year_required")
        item = selected_items[0]
        numeric = _public_float(item.value)
        if numeric is None:
            return _result("INVALID_INPUT", operation_name, "non_finite_public_result")
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "value": numeric,
            "unit": item.unit,
            "display": f"{numeric:,.3f} {item.unit}",
            "reason_code": "typed_fact_observation",
        }

    if operation == "UNIT_CONVERSION":
        if len(selected_items) != 1:
            return _result("AMBIGUOUS", operation_name, "single_source_fact_required_for_conversion")
        item = selected_items[0]
        source_currency, source_scale, _ = _unit(item)
        target_scale = str(plan.get("calculation_target_scale") or "").strip().lower()
        target_currency = str(plan.get("calculation_target_currency") or source_currency or "").strip().upper()
        if target_scale not in _SCALE or source_scale not in _SCALE or not source_currency:
            return _result(
                "OPERATION_UNSUPPORTED", operation_name, "currency_and_supported_source_target_scales_required",
                observations=detailed_records,
            )
        if target_currency != source_currency:
            return _result(
                "OPERATION_UNSUPPORTED", operation_name, "currency_exchange_rate_not_in_protocol",
                observations=detailed_records,
            )
        try:
            base_value = Decimal(str(item.value)) * _SCALE[source_scale]
            value = base_value / _SCALE[target_scale]
        except DecimalException:
            return _result("INVALID_INPUT", operation_name, "non_finite_conversion_result")
        if not value.is_finite():
            return _result("INVALID_INPUT", operation_name, "non_finite_conversion_result")
        result_value = _public_float(value)
        if result_value is None:
            return _result("INVALID_INPUT", operation_name, "non_finite_public_result")
        unit = f"{target_currency} {target_scale}"
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "value": result_value,
            "unit": unit,
            "display": f"{result_value:,.3f} {unit}",
            "reason_code": "exact_scale_conversion_no_fx",
        }

    if operation == "CROSS_YEAR":
        # A comparison across three or more requested periods is a series,
        # not a two-point arithmetic operation. Preserve each observation's
        # native unit instead of silently coercing unlike scales.
        bases = {_basis_value(item)[0] for item in selected_items}
        if len(bases) != 1:
            return _result(
                "AMBIGUOUS", operation_name,
                "mixed_or_unrecognized_measurement_units",
                observations=detailed_records,
            )
        public_values = [_public_float(item.value) for item in selected_items]
        if any(value is None for value in public_values):
            return _result("INVALID_INPUT", operation_name, "non_finite_public_result")
        series = [
            {
                "fiscal_year": item.fiscal_year,
                "fiscal_period": item.fiscal_period,
                "value": public_value,
                "unit": item.unit,
            }
            for item, public_value in zip(selected_items, public_values)
        ]
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "comparability_assessment": comparability_assessment,
            "value": series,
            "unit": "per-observation",
            "display": "; ".join(
                f"{item['fiscal_period']}: {item['value']:,.3f} {item['unit']}"
                for item in series
            ),
            "reason_code": "complete_typed_multi_year_series",
        }

    if len(selected_items) != 2:
        return _result("AMBIGUOUS", operation_name, "exactly_two_fact_years_required")
    first, second = selected_items
    compatible = _compatible_pair(first, second)
    if compatible is None:
        return _result(
            "AMBIGUOUS", operation_name, "mixed_or_unrecognized_measurement_units",
            observations=detailed_records,
        )
    first_base, second_base, basis = compatible
    try:
        change = second_base - first_base
    except DecimalException:
        return _result("INVALID_INPUT", operation_name, "non_finite_difference_result")
    if operation == "ABSOLUTE_CHANGE":
        scale = _unit(second)[1]
        currency = basis[1] if basis[0] == "currency" else None
        if basis[0] == "currency" and scale not in _SCALE:
            return _result("OPERATION_UNSUPPORTED", operation_name, "unsupported_output_scale")
        try:
            value = change / _SCALE[scale] if currency and scale else change
        except DecimalException:
            return _result("INVALID_INPUT", operation_name, "non_finite_difference_result")
        if not value.is_finite():
            return _result("INVALID_INPUT", operation_name, "non_finite_difference_result")
        unit = second.unit
        numeric = _public_float(value)
        if numeric is None:
            return _result("INVALID_INPUT", operation_name, "non_finite_public_result")
        label = "percentage points" if basis[0] == "percent" else unit
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "comparability_assessment": comparability_assessment,
            "value": numeric,
            "unit": label,
            "display": f"FY{second.fiscal_year} - FY{first.fiscal_year} = {numeric:,.3f} {label}",
            "reason_code": _comparison_reason(comparability_assessment),
        }

    if operation == "PERCENT_CHANGE":
        if basis[0] != "currency":
            return _result(
                "OPERATION_UNSUPPORTED", operation_name,
                "percent_change_requires_currency_observations",
                observations=detailed_records,
            )
        if first_base == 0:
            return _result("INVALID_INPUT", operation_name, "zero_baseline_for_percent_change", observations=detailed_records)
        try:
            percent = change / abs(first_base) * Decimal("100")
        except DecimalException:
            return _result("INVALID_INPUT", operation_name, "non_finite_percent_change_result")
        if not percent.is_finite():
            return _result("INVALID_INPUT", operation_name, "non_finite_percent_change_result")
        numeric = _public_float(percent)
        if numeric is None:
            return _result("INVALID_INPUT", operation_name, "non_finite_public_result")
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "comparability_assessment": comparability_assessment,
            "value": numeric,
            "unit": "percent",
            "display": f"FY{first.fiscal_year} to FY{second.fiscal_year}: {numeric:,.3f}% change",
            "reason_code": _comparison_reason(comparability_assessment),
        }

    return _result("OPERATION_UNSUPPORTED", operation_name, "operation_not_supported")


def dependency_failure(dependency: str) -> dict[str, Any]:
    return _result(
        "DEPENDENCY_ERROR", "none", "required_dependency_unavailable",
        dependency=dependency,
    )
