import json

from scripts.build_temporal_change_model import (
    build_fact_versions,
    classify,
    metric_for_year,
    parse_number,
)


def test_parse_accounting_numbers():
    assert parse_number("$1,250.5") == 1250.5
    assert parse_number("(42)") == -42.0


def test_metric_year_selection_and_increase():
    row = {
        "earlier_year": 2023,
        "later_year": 2024,
        "relation_type": "REPORTS_METRIC",
        "earlier_metric_values_json": json.dumps([{"period": "2023", "value": "10"}]),
        "later_metric_values_json": json.dumps([{"period": "2024", "value": "15"}]),
        "earlier_unit": "USD millions", "later_unit": "USD millions",
        "earlier_text": "$10", "later_text": "$15",
    }
    result = classify(row)
    assert metric_for_year(row["earlier_metric_values_json"], 2023) == 10.0
    assert result["change_type"] == "METRIC_INCREASED"
    assert result["absolute_delta"] == 5.0
    assert result["percent_delta"] == 50.0


def test_narrative_repetition_has_no_direction():
    result = classify({"earlier_year": 2023, "later_year": 2024, "relation_type": "EXPOSED_TO"})
    assert result["change_type"] == "CONTINUED_DISCLOSURE"
    assert result["quantitative"] is False


def test_metric_measurement_signature_mismatch_is_not_compared():
    result = classify({
        "earlier_year": 2023, "later_year": 2024, "relation_type": "REPORTS_METRIC",
        "earlier_unit": "USD millions", "later_unit": "USD millions",
        "earlier_text": "Cost of revenue $ 138", "later_text": "Cost of revenue 27.3",
    })
    assert result["change_type"] == "METRIC_NOT_COMPARABLE"


def test_observation_nodes_are_preferred_over_legacy_json():
    row = {
        "earlier_year": 2023,
        "later_year": 2024,
        "relation_type": "REPORTS_METRIC",
        "earlier_unit": "USD millions",
        "later_unit": "USD millions",
        "earlier_text": "$10",
        "later_text": "$15",
        "earlier_observations": [{"fiscal_period": "FY2023", "value": 100.0}],
        "later_observations": [{"fiscal_period": "FY2024", "value": 160.0}],
        "earlier_metric_values_json": json.dumps([{"period": "2023", "value": "10"}]),
        "later_metric_values_json": json.dumps([{"period": "2024", "value": "15"}]),
    }
    result = classify(row)
    assert result["from_value"] == 100.0
    assert result["to_value"] == 160.0


def test_bitemporal_versions_close_record_state_without_claiming_falsehood():
    claims = [
        {
            "claim_id": "claim_2024",
            "source_id": "NVIDIA_CORPORATION",
            "relation_type": "EXPOSED_TO",
            "target_id": "EXPORT_CONTROL_RISK",
            "doc_id": "2024-10-K",
            "page": 20,
            "filing_fiscal_year": 2024,
            "evidence_referenced_period": "FY2024",
        },
        {
            "claim_id": "claim_2025",
            "source_id": "NVIDIA_CORPORATION",
            "relation_type": "EXPOSED_TO",
            "target_id": "EXPORT_CONTROL_RISK",
            "doc_id": "2025-10-K",
            "page": 21,
            "filing_fiscal_year": 2025,
            "evidence_referenced_period": "FY2025",
        },
    ]
    facts = build_fact_versions(claims, migration_recorded_at="2026-08-17T00:00:00+00:00")
    assert len(facts) == 2
    assert facts[0]["invalidation_status"] == "SUPERSEDED_DISCLOSURE"
    assert facts[0]["truth_status"] == "DISCLOSED_FACT"
    assert facts[0]["recorded_time_precision"] == "MIGRATION_TIME"
    assert facts[1]["is_current_record"] is True
    assert facts[0]["fact_key"] == facts[1]["fact_key"]
