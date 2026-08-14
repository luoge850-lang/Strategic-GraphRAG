import json

from scripts.build_temporal_change_model import classify, metric_for_year, parse_number


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
