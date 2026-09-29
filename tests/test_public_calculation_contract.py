from __future__ import annotations

from dataclasses import replace

from strategic_graphrag.engine.graph_rag_engine import CausalPath, GraphRAGEngine
from strategic_graphrag.schema.financial_observation import (
    FinancialObservation,
    split_unit,
)
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.engine.graph_rag_engine import CausalPathFinder
from strategic_graphrag.staging_index import StagingGraphIndex


def observation(
    value: float,
    *,
    fact_year: int = 2024,
    disclosure_year: int = 2025,
    unit: str = "USD millions",
    company: str = "NVIDIA_CORPORATION",
    metric: str = "REVENUE",
    evidence_id: str | None = None,
) -> FinancialObservation:
    currency, scale = split_unit(unit)
    period = f"FY{fact_year}"
    claim_id = evidence_id or f"claim_{fact_year}_{disclosure_year}_{value}"
    return FinancialObservation(
        id=f"obs_{claim_id}",
        measurement_key=f"measurement_{company}_{metric}_{period}",
        company_id=company,
        metric_id=metric,
        claim_id=claim_id,
        source_filing=f"{disclosure_year}-10-K.pdf",
        page=50,
        fiscal_period=period,
        fiscal_year=fact_year,
        value=value,
        raw_value=str(value),
        unit=unit,
        currency=currency,
        scale=scale,
        statement_type="INCOME_STATEMENT",
        table_name="CONSOLIDATED_STATEMENTS_OF_INCOME",
        row_label="Revenue",
        column_label=period,
        valid_from=period,
        valid_to=period,
        comparability_status="COMPARABLE",
        build_id="build_test",
    )


def path(*observations: FinancialObservation) -> CausalPath:
    return CausalPath(
        path_id="path_0",
        nodes=["NVIDIA_CORPORATION", "REVENUE"],
        node_labels=["Company", "FinancialMetric"],
        relationships=["REPORTS_METRIC"],
        causal_strengths=["DISCLOSED_ONLY"],
        evidence=["Total revenue from the consolidated statement."],
        pages=[item.page for item in observations] or [50],
        years=[item.fiscal_year for item in observations],
        evidence_ids=[item.claim_id for item in observations],
        filings=[item.source_filing for item in observations],
        metric_values=[item.value for item in observations],
        metric_units=[item.unit for item in observations],
        financial_observations=list(observations),
        total_hops=1,
    )


def plan(**overrides):
    result = {
        "task_type": "FACT",
        "build_id": "build_test",
        "company": "NVIDIA_CORPORATION",
        "target_metric": "REVENUE",
        "fact_period": "FY2024",
        "fiscal_year_start": 2024,
        "fiscal_year_end": 2024,
        "document_scope": None,
        "disclosure_as_of": None,
        "comparison": None,
        "calculation": None,
        "calculation_target_currency": None,
        "calculation_target_scale": None,
        "ambiguity": [],
    }
    result.update(overrides)
    return result


def calculate(paths, query_plan):
    return GraphRAGEngine._public_calculation(
        paths, query_plan=query_plan, question="deliberately not used by calculation"
    )


def test_eur_millions_convert_to_eur_billions_without_usd_relabeling():
    result = calculate(
        [path(observation(1234.5, unit="EUR millions"))],
        plan(
            calculation="UNIT_CONVERSION",
            calculation_target_currency="EUR",
            calculation_target_scale="billions",
        ),
    )
    assert result["status"] == "PASS"
    assert result["value"] == 1.2345
    assert result["unit"] == "EUR billions"


def test_conversion_does_not_apply_an_unrequested_currency_exchange():
    result = calculate(
        [path(observation(1234.5, unit="EUR millions"))],
        plan(
            calculation="UNIT_CONVERSION",
            calculation_target_currency="USD",
            calculation_target_scale="billions",
        ),
    )
    assert result["status"] == "OPERATION_UNSUPPORTED"
    assert result["value"] is None


def test_comparison_without_explicit_years_is_not_an_empty_pass():
    result = calculate(
        [path(observation(10, fact_year=2023), observation(15, fact_year=2024))],
        plan(
            task_type="COMPARISON",
            fact_period=None,
            fiscal_year_start=None,
            fiscal_year_end=None,
            comparison="POINT_TO_POINT",
        ),
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["observations"]


def test_conflicting_fact_year_across_disclosures_is_ambiguous():
    result = calculate(
        [
            path(observation(100, fact_year=2024, disclosure_year=2024)),
            path(observation(105, fact_year=2024, disclosure_year=2025)),
        ],
        plan(),
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["value"] is None


def test_explicit_disclosure_version_selects_only_that_observation():
    result = calculate(
        [
            path(observation(100, fact_year=2024, disclosure_year=2024)),
            path(observation(105, fact_year=2024, disclosure_year=2025)),
        ],
        plan(document_scope="2025-10-K.pdf", disclosure_as_of="FY2025"),
    )
    assert result["status"] == "PASS"
    assert result["value"] == 105
    assert {item["source_filing"] for item in result["observations"]} == {"2025-10-K.pdf"}


def test_conflicting_values_within_same_disclosure_are_not_overwritten():
    result = calculate(
        [
            path(observation(100, fact_year=2024, evidence_id="claim_a")),
            path(observation(101, fact_year=2024, evidence_id="claim_b")),
        ],
        plan(document_scope="2025-10-K.pdf", disclosure_as_of="FY2025"),
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["value"] is None


def test_unimplemented_ratio_never_falls_back_to_first_fact():
    result = calculate(
        [path(observation(100, fact_year=2023), observation(125, fact_year=2024))],
        plan(calculation="RATIO"),
    )
    assert result["status"] == "OPERATION_UNSUPPORTED"
    assert result["value"] is None


def test_negative_values_keep_their_sign_during_scale_conversion():
    result = calculate(
        [path(observation(-1250, unit="USD millions"))],
        plan(
            calculation="UNIT_CONVERSION",
            calculation_target_currency="USD",
            calculation_target_scale="billions",
        ),
    )
    assert result["status"] == "PASS"
    assert result["value"] == -1.25


def test_percentage_change_with_zero_baseline_is_invalid_not_pass():
    result = calculate(
        [path(observation(0, fact_year=2023), observation(10, fact_year=2024))],
        plan(
            task_type="CALCULATION",
            calculation="PERCENT_CHANGE",
            fiscal_year_start=2023,
            fiscal_year_end=2024,
            fact_period=None,
        ),
    )
    assert result["status"] == "INVALID_INPUT"
    assert result["value"] is None


def test_mixed_currency_comparison_is_not_silently_normalized():
    result = calculate(
        [
            path(observation(100, fact_year=2023, unit="USD millions")),
            path(observation(100, fact_year=2024, unit="EUR millions")),
        ],
        plan(
            task_type="COMPARISON",
            calculation=None,
            fact_period=None,
            comparison="POINT_TO_POINT",
            fiscal_year_start=2023,
            fiscal_year_end=2024,
        ),
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["value"] is None


def test_requested_missing_year_is_insufficient_evidence():
    result = calculate(
        [path(observation(100, fact_year=2023))],
        plan(
            task_type="COMPARISON",
            fact_period=None,
            comparison="POINT_TO_POINT",
            fiscal_year_start=2023,
            fiscal_year_end=2024,
        ),
    )
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["missing_years"] == [2024]


def test_non_finite_observations_cannot_produce_a_pass():
    bad = replace(observation(10), value=float("inf"))
    result = calculate([path(bad)], plan())
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["value"] is None


def test_legacy_parallel_value_arrays_are_not_treated_as_typed_observations():
    legacy_path = path()
    legacy_path.financial_observations = []
    legacy_path.metric_values = [123.0]
    legacy_path.metric_units = ["USD millions"]
    result = calculate([legacy_path], plan())
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["value"] is None


def test_calculation_depends_on_query_plan_not_free_form_question_text():
    fact_path = path(observation(500, unit="EUR millions"))
    result = GraphRAGEngine._public_calculation(
        [fact_path],
        query_plan=plan(
            calculation="UNIT_CONVERSION",
            calculation_target_currency="EUR",
            calculation_target_scale="billions",
        ),
        question="convert to USD 9000 billion growth ratio",
    )
    assert result["status"] == "PASS"
    assert result["unit"] == "EUR billions"
    assert result["value"] == 0.5


def test_parser_serializes_conversion_target_in_structured_plan():
    parsed = parse_query(
        "Convert NVIDIA FY2024 revenue from EUR millions to billions"
    ).to_dict()
    assert parsed["calculation"] == "UNIT_CONVERSION"
    assert parsed["calculation_target_currency"] == "EUR"
    assert parsed["calculation_target_scale"] == "billions"
    assert parsed["target_metric"] == "REVENUE"
    assert parsed["fact_period"] == "FY2024"


def test_parser_separates_fact_year_from_10k_disclosure_year_without_fy_abbreviation():
    parsed = parse_query(
        "What revenue did NVIDIA report for fiscal year 2022 in its 2023 10-K?"
    ).to_dict()
    assert parsed["fact_period"] == "FY2022"
    assert parsed["disclosure_as_of"] == "FY2023"
    assert parsed["document_scope"] == "2023-10-K.pdf"
    assert parsed["fiscal_years"] == [2022]


def test_same_year_fact_and_form_10k_disclosure_is_not_a_multiyear_comparison():
    question = (
        "What was NVIDIA's total revenue in fiscal year 2025 "
        "(reported in the 2025 Form 10-K)?"
    )
    parsed = parse_query(question).to_dict()
    temporal = GraphRAGEngine._build_temporal_context(
        question, year_start=None, year_end=None
    )

    assert parsed["task_type"] == "FACT"
    assert parsed["fact_period"] == "FY2025"
    assert parsed["disclosure_as_of"] == "FY2025"
    assert parsed["document_scope"] == "2025-10-K.pdf"
    assert parsed["fiscal_years"] == [2025]
    assert parsed["temporal_required"] is False
    assert parsed["require_multi_year"] is False
    assert temporal["requested"] is False


def test_parser_retains_fact_endpoint_when_its_year_matches_filing_year():
    parsed = parse_query(
        "Calculate NVIDIA revenue growth from FY2022 to FY2023 using the 2023 10-K"
    ).to_dict()
    assert parsed["calculation"] == "PERCENT_CHANGE"
    assert parsed["document_scope"] == "2023-10-K.pdf"
    assert parsed["fiscal_years"] == [2022, 2023]


def test_parser_marks_ratio_as_an_explicit_unsupported_operation():
    parsed = parse_query("Calculate NVIDIA's revenue to operating cost ratio").to_dict()
    assert parsed["calculation"] == "RATIO"
    assert parsed["task_type"] == "CALCULATION"


def test_parser_classifies_explicit_canonical_relationship_question_as_relation():
    parsed = parse_query(
        "What REPORTS_METRIC relationship connects NVIDIA_CORPORATION to REVENUE?"
    ).to_dict()
    assert parsed["task_type"] == "RELATION"
    assert parsed["relation_types"] == ["REPORTS_METRIC"]
    result = calculate([], parsed | {"build_id": "build_test"})
    assert result["status"] == "NOT_APPLICABLE"


def test_growth_word_is_structured_as_percent_change_not_fact():
    parsed = parse_query(
        "What was NVIDIA revenue growth from FY2023 to FY2024?"
    ).to_dict()
    assert parsed["calculation"] == "PERCENT_CHANGE"
    assert parsed["task_type"] == "CALCULATION"


def test_three_explicit_comparison_years_are_all_preserved_in_the_plan():
    parsed = parse_query(
        "Compare NVIDIA revenue across FY2023, FY2024, and FY2025"
    ).to_dict()
    assert parsed["fiscal_years"] == [2023, 2024, 2025]
    result = calculate(
        [
            path(observation(10, fact_year=2023)),
            path(observation(15, fact_year=2024)),
            path(observation(20, fact_year=2025)),
        ],
        parsed | {"build_id": "build_test"},
    )
    assert result["status"] == "PASS"
    assert [item["fiscal_year"] for item in result["observations"]] == [2023, 2024, 2025]


def test_percentage_change_uses_two_typed_comparable_years():
    result = calculate(
        [path(observation(100, fact_year=2023)), path(observation(120, fact_year=2024))],
        plan(
            task_type="CALCULATION",
            calculation="PERCENT_CHANGE",
            fact_period=None,
            fiscal_year_start=2023,
            fiscal_year_end=2024,
        ),
    )
    assert result["status"] == "PASS"
    assert result["value"] == 20
    assert result["unit"] == "percent"


def test_absolute_change_normalizes_mixed_scales_without_losing_currency():
    result = calculate(
        [
            path(observation(1000, fact_year=2023, unit="USD millions")),
            path(observation(1.2, fact_year=2024, unit="USD billions")),
        ],
        plan(
            task_type="CALCULATION",
            calculation="ABSOLUTE_CHANGE",
            fact_period=None,
            fiscal_year_start=2023,
            fiscal_year_end=2024,
        ),
    )
    assert result["status"] == "PASS"
    assert result["value"] == 0.2
    assert result["unit"] == "USD billions"


def test_build_mismatch_cannot_mix_observations_into_a_pass():
    result = calculate(
        [path(observation(10))],
        plan(build_id="build_other"),
    )
    assert result["status"] == "AMBIGUOUS"
    assert result["reason_code"] == "observation_build_identity_mismatch"


def test_missing_build_identity_blocks_numeric_success():
    result = calculate(
        [path(observation(10))],
        plan(build_id=None),
    )
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["reason_code"] == "candidate_build_identity_missing"


def test_staging_adapter_emits_the_same_typed_observation_contract():
    edge = {
        "claim_id": "claim_staging_1",
        "build_id": "build_test",
        "source": "NVIDIA_CORPORATION",
        "source_category": "Company",
        "target": "REVENUE",
        "target_category": "FinancialMetric",
        "relation": "REPORTS_METRIC",
        "causal_strength": "DISCLOSED_ONLY",
        "causal_form": "FINANCIAL_RELATION",
        "evidence": "Total revenue was EUR 1,250 million.",
        "page": 50,
        "fact_year": 2024,
        "filing_year": 2025,
        "source_filing": "2025-10-K.pdf",
        "metric_value": "1250",
        "metric_unit": "EUR millions",
        "unit": "EUR millions",
        "currency": "EUR",
        "scale": "millions",
    }
    adapter = StagingGraphIndex({"build_id": "build_test", "edges": [edge]})
    [result] = adapter.find_metric_disclosures(
        "REVENUE", year_start=2024, year_end=2024, build_id="build_test"
    )
    [typed] = result.financial_observations
    assert isinstance(typed, FinancialObservation)
    assert result.metric_values == ["1250"]
    assert result.metric_units == ["EUR millions"]
    assert (typed.company_id, typed.metric_id, typed.fiscal_year) == (
        "NVIDIA_CORPORATION", "REVENUE", 2024
    )
    assert (typed.source_filing, typed.currency, typed.scale, typed.claim_id) == (
        "2025-10-K.pdf", "EUR", "millions", "claim_staging_1"
    )


def test_neo4j_projection_maps_financial_observation_and_build_scope():
    item = observation(130497, fact_year=2025, disclosure_year=2025)

    class Session:
        query = ""
        params = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def run(self, query, **params):
            self.query = query
            self.params = params
            return [{
                "node_names": ["NVIDIA Corporation", "Revenue"],
                "node_labels": ["Company", "FinancialMetric"],
                "relationships": ["REPORTS_METRIC"],
                "causal_strengths": ["DISCLOSED_ONLY"],
                "evidence": ["Total revenue from the filing table."],
                "pages": [80],
                "years": [2025],
                "evidence_ids": [item.claim_id],
                "evidence_build_ids": ["build_test"],
                "filings": ["2025-10-K.pdf"],
                "metric_values": [item.value],
                "metric_units": [item.unit],
                "financial_observation": item.to_dict(),
                "causal_forms": ["FINANCIAL_RELATION"],
                "hops": 1,
            }]

    class Driver:
        def __init__(self):
            self.last_session = Session()

        def session(self):
            return self.last_session

    driver = Driver()
    finder = CausalPathFinder(driver)
    [result] = finder.find_metric_disclosures(
        "REVENUE",
        year_start=2025,
        year_end=2025,
        source_filing="2025-10-K.pdf",
        company_id="NVIDIA_CORPORATION",
        build_id="build_test",
    )
    assert result.metric_values == [130497]
    assert result.metric_units == ["USD millions"]
    assert result.financial_observations[0] == item
    assert "FinancialObservation" in driver.last_session.query
    assert "observation.fiscal_year" in driver.last_session.query
    assert driver.last_session.params["build_id"] == "build_test"
