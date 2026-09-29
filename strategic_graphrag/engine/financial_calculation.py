"""Fail-closed deterministic calculations over typed filing observations."""

from __future__ import annotations

import math
import re
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
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
    scale = str(observation.scale or parsed_scale or "").strip().lower() or None
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
    except (InvalidOperation, TypeError, ValueError):
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


def _valid_observations(items: Iterable[Any]) -> tuple[list[FinancialObservation], int]:
    valid: list[FinancialObservation] = []
    invalid = 0
    for item in items:
        if not isinstance(item, FinancialObservation):
            invalid += 1
            continue
        try:
            basis = _basis_value(item)
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
        except (TypeError, ValueError):
            basis = None
            metadata_ok = False
        if not basis or not metadata_ok:
            invalid += 1
            continue
        valid.append(item)
    return valid, invalid


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
    if len(bases) != 1 or len(values) != 1:
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

    raw_observations: list[Any] = []
    for path in paths or []:
        raw_observations.extend(getattr(path, "financial_observations", []) or [])
    observations, invalid_count = _valid_observations(raw_observations)
    company = str(plan.get("company") or "").strip().upper().replace(" ", "_")
    observations = [
        item for item in observations
        if item.metric_id.strip().upper().replace(" ", "_") == metric
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
    expected_build_id = str(plan.get("build_id") or "").strip()
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

    requested_years, year_ambiguous = _requested_years(plan, operation)
    years_available = sorted({item.fiscal_year for item in observations})
    if year_ambiguous:
        return _result("AMBIGUOUS", operation_name, "query_plan_year_scope_is_inconsistent")
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

    selected: dict[int, FinancialObservation] = {}
    for year in requested_years:
        item, conflict = _choose_year(year, observations)
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
    detailed_records = [
        _record(item) for item in observations
        if item.fiscal_year in requested_years
    ]
    if operation == "FACT":
        if len(selected_items) != 1:
            return _result("AMBIGUOUS", operation_name, "single_fact_year_required")
        item = selected_items[0]
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "value": float(item.value),
            "unit": item.unit,
            "display": f"{float(item.value):,.3f} {item.unit}",
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
        base_value = Decimal(str(item.value)) * _SCALE[source_scale]
        value = base_value / _SCALE[target_scale]
        if not value.is_finite():
            return _result("INVALID_INPUT", operation_name, "non_finite_conversion_result")
        result_value = float(value)
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
        series = [
            {
                "fiscal_year": item.fiscal_year,
                "value": float(item.value),
                "unit": item.unit,
            }
            for item in selected_items
        ]
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "value": series,
            "unit": "per-observation",
            "display": "; ".join(
                f"FY{item['fiscal_year']}: {item['value']:,.3f} {item['unit']}"
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
    change = second_base - first_base
    if operation == "ABSOLUTE_CHANGE":
        scale = _unit(second)[1]
        currency = basis[1] if basis[0] == "currency" else None
        if basis[0] == "currency" and scale not in _SCALE:
            return _result("OPERATION_UNSUPPORTED", operation_name, "unsupported_output_scale")
        value = change / _SCALE[scale] if currency and scale else change
        if not value.is_finite():
            return _result("INVALID_INPUT", operation_name, "non_finite_difference_result")
        unit = second.unit
        numeric = float(value)
        label = "percentage points" if basis[0] == "percent" else unit
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "value": numeric,
            "unit": label,
            "display": f"FY{second.fiscal_year} - FY{first.fiscal_year} = {numeric:,.3f} {label}",
            "reason_code": "same_metric_comparable_years",
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
        percent = change / abs(first_base) * Decimal("100")
        if not percent.is_finite():
            return _result("INVALID_INPUT", operation_name, "non_finite_percent_change_result")
        numeric = float(percent)
        return {
            "schema": "deterministic-calculation/v1",
            "status": "PASS",
            "operation": operation_name,
            "observations": detailed_records,
            "value": numeric,
            "unit": "percent",
            "display": f"FY{first.fiscal_year} to FY{second.fiscal_year}: {numeric:,.3f}% change",
            "reason_code": "same_metric_comparable_years",
        }

    return _result("OPERATION_UNSUPPORTED", operation_name, "operation_not_supported")


def dependency_failure(dependency: str) -> dict[str, Any]:
    return _result(
        "DEPENDENCY_ERROR", "none", "required_dependency_unavailable",
        dependency=dependency,
    )
