import json
import unittest

from strategic_graphrag.ontology.entity_registry import resolve_entity
from strategic_graphrag.pipeline.financial_table_extractor import (
    _numeric_values,
    _periods_for_evidence,
    extract_financial_table_triples,
)
from strategic_graphrag.pipeline.pipeline import KnowledgeGraphPipeline
from strategic_graphrag.ontology.intent_classifier import extract_financial_entities_from_query
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.engine.graph_rag_engine import CausalPath, GraphRAGEngine
from strategic_graphrag.engine.retrieval import QueryRouter, personalized_pagerank
from strategic_graphrag.schema.financial_observation import build_financial_observations
from strategic_graphrag.provenance import evidence_identity, normalize_evidence
from scripts.plan_incremental_update import build_plan


class _FakePage:
    def __init__(self, rows):
        self._rows = rows

    def extract_tables(self):
        return [self._rows]


class PipelineContractTests(unittest.TestCase):
    def test_evidence_identity_is_stable_across_runtime_metadata(self):
        base = dict(
            document_sha256="a" * 64,
            filename="2025-10-K.pdf",
            page=42,
            evidence_text="Sales, general and administrative expenses 3,491 2,654",
            source_id="NVIDIA_CORPORATION",
            relation_type="REPORTS_METRIC",
            target_id="SG_AND_A_EXPENSE",
        )
        first = evidence_identity(**base)
        second = evidence_identity(**{**base, "evidence_text": "Sales, general and\nadministrative expenses 3,491 2,654"})
        self.assertEqual(first, second)
        self.assertTrue(first.claim_id.startswith("claim_v2_"))
        self.assertNotEqual(first.claim_id, evidence_identity(**{**base, "page": 43}).claim_id)

    def test_evidence_normalization_preserves_words_and_numbers(self):
        self.assertEqual(normalize_evidence("  Revenue\n  130,497  "), "Revenue 130,497")

    def test_accounting_parentheses_preserve_negative_sign(self):
        self.assertEqual(
            _numeric_values("Inventories (2,554) (98)"),
            ["-2554", "-98"],
        )

    def test_sga_change_columns_are_not_metric_values(self):
        evidence = "Sales, general and administrative expenses 3,491 2,654 837 32 %"
        self.assertEqual(_numeric_values(evidence), ["3491", "2654", "837"])

        page_text = "\n".join([
            "Operating Expenses",
            "Year Ended",
            "$ %",
            "Jan 26, 2025 Jan 28, 2024 Change Change",
            "($ in millions)",
            evidence,
        ])
        triples = extract_financial_table_triples(
            _FakePage([["Sales, general and administrative expenses", "3,491", "2,654", "837", "32", "%"]]),
            page_text,
            2025,
        )
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        self.assertEqual(triple["target"], "SG_AND_A_EXPENSE")
        self.assertEqual(triple["metric_unit"], "USD millions")
        self.assertEqual(triple["metric_period"], "2025")
        self.assertEqual(
            json.loads(triple["metric_values_json"]),
            [
                {"period": "2025", "value": "3491"},
                {"period": "2024", "value": "2654"},
            ],
        )

    def test_split_header_periods_are_positionally_recovered(self):
        evidence = "Sales, general and administrative expenses 2,440 2,166 274 13 %"
        page_text = "\n".join([
            "Year Ended",
            "January 29, January 30, $ %",
            "2023 2022 Change Change",
            "($ in millions)",
            evidence,
        ])
        self.assertEqual(_periods_for_evidence(page_text, evidence, 2023), [2023, 2022])

    def test_metric_registry_contains_tax_and_pretax_metrics(self):
        self.assertEqual(resolve_entity("income before income tax", "FinancialMetric"), ("PRETAX_INCOME", "FinancialMetric"))
        self.assertEqual(resolve_entity("income tax expense", "FinancialMetric"), ("INCOME_TAX_EXPENSE", "FinancialMetric"))

    def test_evidence_span_tolerates_pdf_line_breaks(self):
        text = "The increase in sales, general and administrative expenses\nwas driven by compensation."
        start, end = KnowledgeGraphPipeline._evidence_span(
            text,
            "The increase in sales, general and administrative expenses was driven by compensation.",
        )
        self.assertEqual(start, 0)
        self.assertEqual(text[start:end].replace("\n", " "), "The increase in sales, general and administrative expenses was driven by compensation.")

    def test_sga_query_resolves_specific_metric_anchor(self):
        anchors = extract_financial_entities_from_query(
            "What sales, general and administrative expense did NVIDIA report?"
        )
        self.assertIn("SG_AND_A_EXPENSE", anchors)
        self.assertEqual(
            parse_query(
                "Compare sales, general and administrative expense in 2023, 2024, and 2025"
            ).target_metric,
            "SG_AND_A_EXPENSE",
        )

    def test_adaptive_retrieval_skips_vector_for_exact_metric(self):
        self.assertEqual(
            GraphRAGEngine._resolve_retrieval_mode(
                "Compare revenue in 2023 and 2025", "auto", "REVENUE"
            ),
            "hybrid_temporal",
        )
        self.assertEqual(
            GraphRAGEngine._resolve_retrieval_mode(
                "What risks could affect NVIDIA?", "auto"
            ),
            "hybrid",
        )

    def test_four_baselines_are_explicitly_routable(self):
        for mode in ("vector", "graph", "hybrid", "hybrid_temporal"):
            decision = QueryRouter.route("test question", mode)
            self.assertEqual(decision.mode, mode)
            self.assertEqual(decision.confidence, 1.0)
        self.assertEqual(
            QueryRouter.route("How did revenue change between 2023 and 2025?", "auto", "REVENUE").mode,
            "hybrid_temporal",
        )

    def test_ppr_discovers_bridge_without_reversing_path_semantics(self):
        scores = personalized_pagerank(
            [("EXPORT_CONTROL", "MARKET_ACCESS"), ("MARKET_ACCESS", "REVENUE")],
            ["EXPORT_CONTROL"],
        )
        self.assertGreater(scores["MARKET_ACCESS"], 0)
        self.assertAlmostEqual(sum(scores.values()), 1.0, places=6)

    def test_metric_claim_expands_to_period_observations(self):
        triple = {
            "relation": "REPORTS_METRIC",
            "target": "REVENUE",
            "metric_unit": "USD millions",
            "metric_values_json": json.dumps([
                {"period": "2025", "value": "130,497"},
                {"period": "2024", "value": "60,922"},
            ]),
            "table_name": "Consolidated Statements of Income",
            "row_label": "Revenue",
            "statement_type": "FINANCIAL_STATEMENTS",
        }
        observations = build_financial_observations(
            triple,
            claim_id="claim_v2_example",
            company_id="NVIDIA_CORPORATION",
            metric_id="REVENUE",
            source_filing="2025-10-K.pdf",
            page=91,
            filing_year=2025,
            section="FINANCIAL_STATEMENTS",
        )
        self.assertEqual(len(observations), 2)
        self.assertEqual(observations[0]["fiscal_year"], 2025)
        self.assertEqual(observations[0]["value"], 130497.0)
        self.assertEqual(observations[0]["currency"], "USD")
        self.assertEqual(observations[0]["scale"], "millions")
        self.assertEqual(observations[0]["claim_id"], "claim_v2_example")

    def test_percentage_only_rows_are_not_misread_as_currency(self):
        revenue = build_financial_observations(
            {
                "relation": "REPORTS_METRIC",
                "target": "REVENUE",
                "metric_unit": "USD millions",
                "metric_values_json": json.dumps([{"period": "2024", "value": "100.0"}]),
                "evidence_sentence": "Revenue 100.0 % 100.0 %",
            },
            claim_id="claim_revenue_ratio",
            company_id="nvidia_corporation",
            metric_id="revenue",
            source_filing="2024-10-K.pdf",
            page=39,
            filing_year=2024,
            section="MD_AND_A",
        )
        self.assertEqual(revenue, [])
        net_margin = build_financial_observations(
            {
                "relation": "REPORTS_METRIC",
                "target": "NET_INCOME",
                "metric_unit": "USD millions",
                "metric_values_json": json.dumps([{"period": "2024", "value": "48.9"}]),
                "evidence_sentence": "Net income 48.9 % 16.2 %",
            },
            claim_id="claim_net_margin",
            company_id="nvidia_corporation",
            metric_id="net_income",
            source_filing="2024-10-K.pdf",
            page=39,
            filing_year=2024,
            section="MD_AND_A",
        )
        self.assertEqual(net_margin[0]["metric_id"], "net_margin")
        self.assertEqual(net_margin[0]["unit"], "percent")
        legacy_path = CausalPath(
            path_id="legacy",
            nodes=["NVIDIA_CORPORATION", "REVENUE"],
            node_labels=["Company", "FinancialMetric"],
            relationships=["REPORTS_METRIC"],
            causal_strengths=["DISCLOSED_ONLY"],
            evidence=["Revenue 100.0 % 100.0 %"],
            pages=[39],
            years=[2024],
            evidence_ids=["claim_revenue_ratio"],
            filings=["2024-10-K.pdf"],
            total_hops=1,
        )
        self.assertTrue(GraphRAGEngine._is_percentage_denominator_path(legacy_path))
        self.assertEqual(
            GraphRAGEngine._resolve_retrieval_mode(
                "How do export controls impact NVIDIA revenue?", "auto", "REVENUE"
            ),
            "hybrid",
        )

    def test_incremental_plan_only_rebuilds_changed_files(self):
        manifest = {
            "active_filings": [
                {"filename": "2023-10-K.pdf", "sha256": "same"},
                {"filename": "2024-10-K.pdf", "sha256": "old"},
            ]
        }
        current = {"2023-10-K.pdf": "same", "2024-10-K.pdf": "new"}
        self.assertEqual(
            build_plan(manifest, current=current)["requires_rebuild"],
            ["2024-10-K.pdf"],
        )


if __name__ == "__main__":
    unittest.main()
