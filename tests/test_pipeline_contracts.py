import json
import tempfile
import unittest

from strategic_graphrag.ontology.entity_registry import resolve_entity
from strategic_graphrag.pipeline.financial_table_extractor import (
    _numeric_values,
    _periods_for_evidence,
    extract_financial_table_triples,
)
from strategic_graphrag.pipeline.pipeline import KnowledgeGraphPipeline, PipelineConfig
from strategic_graphrag.ontology.intent_classifier import extract_financial_entities_from_query
from strategic_graphrag.engine.query_understanding import parse_query
from strategic_graphrag.engine.graph_rag_engine import CausalPath, GraphRAGEngine
from strategic_graphrag.engine.evidence_quality import (
    apply_directness_ranking,
    audit_documents,
    contains_absence_claim,
    select_answer_evidence,
    semantic_scope,
)
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
    def test_table_candidate_identity_and_pending_queue_are_round_trip_safe(self):
        triple = {
            "table_id": "Consolidated Statements of Income",
            "row_id": "REVENUE",
            "metric_values_json": json.dumps([
                {"period": "2025", "value": "100"},
                {"period": "2024", "value": "90"},
            ]),
            "row_evidence": "Revenue 100 90",
            "source": "",
            "subject_resolution_status": "PENDING_COMPANY_REVIEW",
            "target": "REVENUE",
        }
        first = KnowledgeGraphPipeline._annotate_table_candidate(
            dict(triple), filename="2025-10-K.pdf", page_num=52
        )
        second = KnowledgeGraphPipeline._annotate_table_candidate(
            dict(triple), filename="2025-10-K.pdf", page_num=52
        )
        self.assertEqual(first["candidate_id"], second["candidate_id"])
        self.assertEqual(first["queue_id"], first["candidate_id"])
        self.assertEqual(first["column_ids"], ["2025", "2024"])
        self.assertIn("2025-10-K.pdf:52", first["row_id"])

        with tempfile.TemporaryDirectory() as directory:
            queue_path = f"{directory}/pending.jsonl"
            pipeline = KnowledgeGraphPipeline.__new__(KnowledgeGraphPipeline)
            pipeline.config = PipelineConfig(pending_table_queue_path=queue_path)
            record = pipeline._table_review_record(
                first,
                review_status="UNLABELED_CANDIDATE",
                reason="MISSING_VERIFIED_COMPANY_ID",
            )
            self.assertEqual(pipeline._write_pending_table_queue([record]), 1)
            self.assertEqual(pipeline._write_pending_table_queue([record]), 0)
            rows = [json.loads(line) for line in open(queue_path, encoding="utf-8")]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["review_status"], "UNLABELED_CANDIDATE")
            self.assertEqual(rows[0]["gold"]["cell_supported"], "")

    def test_document_registry_binds_company_to_filename_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            registry_path = f"{directory}/documents.json"
            with open(registry_path, "w", encoding="utf-8") as handle:
                json.dump({"documents": [{
                    "filename": "2025-10-K.pdf",
                    "document_sha256": "a" * 64,
                    "company_id": "nvidia",
                    "review_status": "VERIFIED",
                }]}, handle)
            pipeline = KnowledgeGraphPipeline.__new__(KnowledgeGraphPipeline)
            pipeline.config = PipelineConfig(document_registry_path=registry_path)
            company_id, source = pipeline._resolve_registered_company(
                filename="2025-10-K.pdf",
                document_sha256="a" * 64,
            )
            self.assertEqual(company_id, "NVIDIA_CORPORATION")
            self.assertEqual(source, "document_registry")
            pending, pending_source = pipeline._resolve_registered_company(
                filename="2025-10-K.pdf",
                document_sha256="b" * 64,
            )
            self.assertIsNone(pending)
            self.assertEqual(pending_source, "pending_registry_review")

    @staticmethod
    def _path(path_id, nodes, relationships, evidence):
        hops = len(relationships)
        return CausalPath(
            path_id=path_id,
            nodes=nodes,
            node_labels=["RiskFactor"] * len(nodes),
            relationships=relationships,
            causal_strengths=["CONFIRMED_CAUSAL"] * hops,
            evidence=evidence,
            pages=[1] * hops,
            years=[2025] * hops,
            evidence_ids=[f"claim_{path_id}_{index}" for index in range(hops)],
            filings=["2025-10-K.pdf"] * hops,
            total_hops=hops,
            aggregate_score=0.8,
        )

    def test_direct_mitigation_outranks_tangential_climate_path(self):
        direct = self._path(
            "direct",
            ["SUPPLY_CHAIN_DIVERSIFICATION", "SUPPLY_CHAIN_DISRUPTION"],
            ["MITIGATES"],
            ["Supplier diversification mitigates supply chain disruption risk."],
        )
        tangent = self._path(
            "tangent",
            ["CLIMATE_CHANGE", "SUPPLY_CHAIN_DISRUPTION"],
            ["CAUSES"],
            ["Climate change may cause supply chain disruption."],
        )
        ranked = apply_directness_ranking(
            [direct, tangent],
            "How does NVIDIA mitigate supply chain risks?",
            "MITIGATION_STRATEGY",
        )
        ranked.sort(key=GraphRAGEngine._path_sort_key)
        self.assertEqual(ranked[0].path_id, "direct")
        self.assertEqual(ranked[0].evidence_role, "ANSWER_CRITICAL")
        self.assertEqual(ranked[1].evidence_role, "BACKGROUND_CONTEXT")

    def test_structural_relation_requires_explicit_predicate_support(self):
        direct = self._path(
            "direct_produces",
            ["NVIDIA_CORPORATION", "DRIVE_PLATFORM"],
            ["PRODUCES"],
            ["NVIDIA built the NVIDIA DRIVE software stack for autonomous driving."],
        )
        weak = self._path(
            "weak_produces",
            ["NVIDIA_CORPORATION", "GPU"],
            ["PRODUCES"],
            ["Our full-stack includes the CUDA programming model that runs on all NVIDIA GPUs."],
        )
        ranked = apply_directness_ranking(
            [weak, direct],
            "What evidence-backed PRODUCES relationship connects NVIDIA_CORPORATION and DRIVE_PLATFORM?",
            "CAUSAL_CHAIN",
        )
        ranked.sort(key=GraphRAGEngine._path_sort_key)
        self.assertEqual(ranked[0].evidence_role, "ANSWER_CRITICAL")
        self.assertEqual(ranked[0].path_id, "direct_produces")
        self.assertEqual(weak.evidence_role, "BACKGROUND_CONTEXT")

    def test_stronger_produces_variant_becomes_primary_evidence(self):
        path = self._path(
            "variant_priority",
            ["NVIDIA_CORPORATION", "DRIVE_PLATFORM"],
            ["PRODUCES"],
            ["We offer NVIDIA DRIVE as a software solution."],
        )
        path.evidence_variants = [[
            {
                "evidence": "We offer NVIDIA DRIVE as a software solution.",
                "page": 15,
                "year": 2025,
                "evidence_id": "claim_offer",
                "filing": "2025-10-K.pdf",
            },
            {
                "evidence": "We built full software stacks that run on top of our GPUs, including NVIDIA DRIVE for autonomous driving.",
                "page": 4,
                "year": 2025,
                "evidence_id": "claim_built",
                "filing": "2025-10-K.pdf",
            },
        ]]
        apply_directness_ranking(
            [path],
            "What evidence-backed PRODUCES relationship connects NVIDIA_CORPORATION and DRIVE_PLATFORM?",
            "CAUSAL_CHAIN",
        )
        self.assertEqual(path.evidence_ids, ["claim_built"])
        self.assertEqual(path.pages, [4])

    def test_produces_rejects_membership_and_composition_wording(self):
        membership = self._path(
            "membership",
            ["NVIDIA_CORPORATION", "OMNIVERSE_PLATFORM"],
            ["PRODUCES"],
            ["The Graphics segment includes Omniverse Enterprise software."],
        )
        composition = self._path(
            "composition",
            ["NVIDIA_CORPORATION", "OMNIVERSE_PLATFORM"],
            ["PRODUCES"],
            ["We offer a simulation solution based on NVIDIA Omniverse software."],
        )
        apply_directness_ranking(
            [membership, composition],
            "What evidence-backed PRODUCES relationship connects NVIDIA_CORPORATION and OMNIVERSE_PLATFORM?",
            "CAUSAL_CHAIN",
        )
        self.assertEqual(membership.evidence_role, "BACKGROUND_CONTEXT")
        self.assertEqual(composition.evidence_role, "BACKGROUND_CONTEXT")

    def test_produces_accepts_explicit_enumerated_product(self):
        path = self._path(
            "enumerated",
            ["NVIDIA_CORPORATION", "DRIVE_PLATFORM"],
            ["PRODUCES"],
            ["We built full software stacks, including NVIDIA DRIVE for autonomous driving."],
        )
        apply_directness_ranking(
            [path],
            "What evidence-backed PRODUCES relationship connects NVIDIA_CORPORATION and DRIVE_PLATFORM?",
            "CAUSAL_CHAIN",
        )
        self.assertEqual(path.evidence_role, "ANSWER_CRITICAL")

    def test_absence_guard_does_not_reject_local_qualification(self):
        self.assertTrue(contains_absence_claim("The filings do not contain this disclosure."))
        self.assertFalse(contains_absence_claim("The evidence claim does not explicitly state production."))
        self.assertFalse(contains_absence_claim("The graph structure should not be characterized as a causal pathway."))

    def test_single_graph_hop_can_contain_embedded_text_mechanism(self):
        path = self._path(
            "embedded",
            ["EXPORT_CONTROL", "REVENUE"],
            ["DECREASES"],
            ["License restrictions reduce market access, resulting in lower revenue."],
        )
        self.assertEqual(path.total_hops, 1)
        self.assertEqual(semantic_scope(path), "EMBEDDED_MECHANISM")

    def test_negative_audit_never_turns_a_search_miss_into_pdf_absence(self):
        result = audit_documents(
            "Do the filings document a realized revenue loss from export controls?",
            [{
                "document": "NVIDIA is subject to competition in several markets.",
                "metadata": {"source_filing": "2025-10-K.pdf", "page": 20},
            }],
            scope="2025-10-K.pdf",
        )
        self.assertEqual(result["status"], "NO_MATCH_AFTER_INDEXED_CORPUS_AUDIT")
        self.assertIn("bounded retrieval result", result["safe_absence_statement"])
        self.assertNotIn("filing contains no", result["safe_absence_statement"].lower())

    def test_retrieval_only_negative_audit_is_explicitly_skipped(self):
        class NoCorpusRead:
            def corpus_documents(self, source_filing=None):
                raise AssertionError("retrieval-only mode must not scan the corpus")

        engine = GraphRAGEngine.__new__(GraphRAGEngine)
        result = engine._negative_evidence_audit(
            "What risks are disclosed?",
            vector_engine=NoCorpusRead(),
            vector_hits=[],
            source_filing=None,
            skip=True,
        )

        self.assertEqual(result["status"], "SKIPPED")
        self.assertTrue(result["not_applicable"])
        self.assertFalse(result["performed"])
        self.assertEqual(result["chunks_scanned"], 0)
        self.assertIn("no filing-wide absence conclusion", result["safe_absence_statement"])

    def test_fallback_metadata_has_common_numeric_contract(self):
        engine = GraphRAGEngine.__new__(GraphRAGEngine)
        engine.llm = None
        response = engine._fallback_response("unknown question", ["UNKNOWN"])
        self.assertEqual(response["metadata"]["avg_score"], 0.0)
        self.assertEqual(response["metadata"]["total_candidates"], 0)

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

    def test_table_metric_queries_resolve_canonical_targets(self):
        self.assertEqual(
            parse_query(
                "How did NVIDIA's reported CASH_AND_CASH_EQUIVALENTS change across fiscal 2023, 2024, and 2025?"
            ).target_metric,
            "CASH_AND_CASH_EQUIVALENTS",
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
        explicit_edge = QueryRouter.route(
            "What evidence-backed PRODUCES relationship connects NVIDIA_CORPORATION and DRIVE_PLATFORM?",
            "auto",
        )
        self.assertEqual(explicit_edge.mode, "graph")
        self.assertEqual(explicit_edge.confidence, 0.96)

    def test_answer_critical_path_survives_wide_candidate_pool(self):
        critical = self._path(
            "critical",
            ["NVIDIA_CORPORATION", "DRIVE_PLATFORM"],
            ["PRODUCES"],
            ["We built and introduced NVIDIA DRIVE for autonomous driving."],
        )
        background = self._path(
            "background",
            ["NVIDIA_CORPORATION", "DRIVE_PLATFORM"],
            ["PRODUCES"],
            ["Our software stack runs on GPUs, including NVIDIA DRIVE."],
        )
        paths = [critical, background]
        for path in paths:
            path.evidence_role = "ANSWER_CRITICAL" if path is critical else "BACKGROUND_CONTEXT"
        selected = select_answer_evidence(paths, 1)
        self.assertEqual(selected, [critical])

    def test_answer_selection_preserves_requested_years(self):
        older = self._path("older", ["A", "B"], ["CAUSES"], ["Older evidence."])
        older.years = [2023]
        newer = self._path("newer", ["A", "B"], ["CAUSES"], ["Newer evidence."])
        newer.years = [2025]
        older.evidence_role = newer.evidence_role = "MECHANISM_SUPPORT"
        selected = select_answer_evidence([newer, older], 2, required_years=[2023, 2025])
        self.assertEqual({selected[0].years[0], selected[1].years[0]}, {2023, 2025})

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
