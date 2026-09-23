import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from strategic_graphrag.build_identity import require_same_build_id
from strategic_graphrag.evidence_bundle import EvidenceBundle, EvidenceItem, from_graph_paths, from_vector_hits
from strategic_graphrag.evaluation.metric_spec import summarize_binary, wilson_interval
from strategic_graphrag.document_layer import Document, Page, _tables, normalize_text
from strategic_graphrag.engine.query_understanding import QueryPlan, parse_query
from strategic_graphrag.engine.graph_rag_engine import CausalPath, GraphRAGEngine
from strategic_graphrag.pipeline.financial_table_extractor import (
    TableParseError,
    extract_financial_table_triples,
)
from strategic_graphrag.response_contract import classify_outcome, is_abstention
from strategic_graphrag.schema.financial_observation import parse_numeric_value
from strategic_graphrag.api import server


class ReconstructionContractTests(unittest.TestCase):
    def test_document_layer_page_conservation_and_fail_closed_ocr(self):
        document = Document(
            document_id="demo",
            filename="demo.pdf",
            pdf_sha256="a" * 64,
            total_pages=2,
            pages=[
                Page(1, "1", 612, 792, "text", "text", parse_status="PARSED"),
                Page(2, None, 612, 792, "", "", parse_status="OCR_REQUIRED"),
            ],
        )
        coverage = document.coverage()
        self.assertTrue(coverage["conservation_holds"])
        self.assertEqual(coverage["successful_parse"], 1)
        self.assertEqual(coverage["pending"], 1)
        self.assertFalse(coverage["ocr_supported"])

    def test_document_normalization_preserves_line_structure(self):
        self.assertEqual(normalize_text(" A\u00a0  B\r\n\r\n C "), "A B\nC")

    def test_query_plan_does_not_default_unknown_question_to_causal(self):
        plan = parse_query("Tell me something about the filing")
        self.assertIsInstance(plan, QueryPlan)
        self.assertEqual(plan.analysis_type, "UNCLASSIFIED")
        self.assertEqual(plan.task_type, "UNCLASSIFIED")
        self.assertFalse(plan.relation_types)

    def test_query_plan_separates_disclosure_and_fact_period(self):
        plan = parse_query("What was revenue in the 2025 annual report for FY2024?")
        self.assertEqual(plan.document_scope, "2025-10-K.pdf")
        self.assertEqual(plan.disclosure_as_of, "FY2025")
        self.assertEqual(plan.fact_period, "FY2024")

    def test_build_identity_mixing_fails_closed(self):
        self.assertEqual(require_same_build_id([
            {"build_id": "build_a"}, {"build_id": "build_a"}
        ]), "build_a")
        with self.assertRaises(ValueError):
            require_same_build_id([
                {"build_id": "build_a"}, {"build_id": "build_b"}
            ])
        with self.assertRaises(ValueError):
            require_same_build_id([{"build_id": "build_a"}, {}])

    def test_evidence_bundle_rejects_mixed_builds(self):
        bundle = EvidenceBundle(items=[
            EvidenceItem("a", "one", build_id="build_a"),
            EvidenceItem("b", "two", build_id="build_a"),
        ], build_id="build_a")
        bundle.assert_consistent()
        mixed = EvidenceBundle(items=[
            EvidenceItem("a", "one", build_id="build_a"),
            EvidenceItem("b", "two", build_id="build_b"),
        ])
        with self.assertRaises(ValueError):
            mixed.assert_consistent()

    def test_evidence_bundle_does_not_relabel_unbound_asset(self):
        bundle = from_vector_hits(
            [{"document": "legacy", "metadata": {"chunk_id": "legacy:1"}}],
            build_id="request_side_label",
        )
        self.assertIsNone(bundle.build_id)
        self.assertEqual(bundle.status, "UNBOUND")
        self.assertIsNone(bundle.items[0].build_id)
        with self.assertRaises(ValueError):
            bundle.assert_strictly_bound()

    def test_vector_bundle_retains_page_and_build_identity(self):
        bundle = from_vector_hits([{
            "rank": 1,
            "document": "Revenue text",
            "rank_score": 1.0,
            "metadata": {
                "chunk_id": "2025-10-K.pdf:80:0",
                "source_filing": "2025-10-K.pdf",
                "page": 80,
                "build_id": "build_a",
            },
        }])
        self.assertEqual(bundle.items[0].physical_page, 80)
        self.assertEqual(bundle.build_id, "build_a")

    def test_graph_bundle_uses_per_evidence_build_identity(self):
        bundle = from_graph_paths([{
            "evidence_ids": ["claim_a", "claim_b"],
            "evidence": ["Revenue was 100 million.", "Revenue was 60 million."],
            "pages": [38, 79],
            "years": [2025, 2024],
            "filings": ["2025-10-K.pdf", "2025-10-K.pdf"],
            "evidence_build_ids": ["build_a", "build_a"],
        }], build_id="build_a")
        self.assertEqual(bundle.status, "OK")
        self.assertEqual(bundle.build_id, "build_a")
        self.assertEqual([item.build_id for item in bundle.items], ["build_a", "build_a"])

    def test_wilson_interval_does_not_claim_certainty_from_one_success(self):
        interval = wilson_interval(1, 1)
        self.assertIsNotNone(interval)
        self.assertLess(interval[0], 1.0)
        self.assertEqual(summarize_binary([])["status"], "NOT_RUN")

    def test_numeric_parser_rejects_non_finite_strings(self):
        for value in ("NaN", "Infinity", "-Infinity", "inf"):
            self.assertIsNone(parse_numeric_value(value))

    def test_response_contract_does_not_answer_metric_year_refusals_or_model_errors(self):
        refusal = {"answer": "The filing does not provide revenue for FY2025."}
        self.assertEqual(classify_outcome(refusal), "ABSTAINED")
        self.assertTrue(is_abstention(refusal))
        self.assertEqual(
            classify_outcome({"answer": "[Generation error: LLM call failed]"}),
            "MODEL_ERROR",
        )

    def test_vector_api_converts_generation_sentinel_to_model_error(self):
        class ErrorVector:
            def retrieve_with_metadata(self, query, *, k, source_filing=None):
                return {
                    "status": "OK",
                    "hits": [{
                        "rank": 1,
                        "document": "Revenue was 100 million.",
                        "metadata": {
                            "chunk_id": "chunk_api_generation",
                            "source_filing": "2025-10-K.pdf",
                            "page": 38,
                            "build_id": "build_a",
                        },
                    }],
                }

            def generate(self, query, documents):
                return "[Generation error: LLM call failed]"

        with patch.object(server, "get_vector_engine", return_value=ErrorVector()):
            result = asyncio.run(server.vector_query(server.QueryRequest(
                question="What was revenue?",
                synthesize=True,
            )))
        payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        self.assertEqual(payload["execution_status"], "MODEL_ERROR")
        self.assertEqual(payload["answer_status"], "NOT_REQUESTED")
        self.assertEqual(payload["outcome"], "MODEL_ERROR")

    def test_table_failure_is_explicit_and_cell_ids_bind_document_identity(self):
        class BrokenPage:
            def extract_tables(self):
                raise IndexError("ragged table")

        class TablePage:
            def extract_tables(self):
                return [[["Header"], ["Value"]]]

        failed, error = _tables(BrokenPage(), 3, "sha_a")
        self.assertEqual(failed, [])
        self.assertIn("IndexError", error)
        with self.assertRaises(TableParseError):
            extract_financial_table_triples(BrokenPage(), "Header\nValue", 2025)

        first, first_error = _tables(TablePage(), 3, "sha_a")
        second, second_error = _tables(TablePage(), 3, "sha_b")
        self.assertIsNone(first_error)
        self.assertIsNone(second_error)
        self.assertNotEqual(first[0].table_id, second[0].table_id)
        self.assertNotEqual(first[0].cells[0].cell_id, second[0].cells[0].cell_id)

    def test_grounding_binds_numeric_negation_and_exact_cited_hop(self):
        path = CausalPath(
            path_id="p",
            nodes=["A", "B", "C"],
            node_labels=["RiskFactor", "Mechanism", "FinancialMetric"],
            relationships=["CAUSES", "REPORTS_METRIC"],
            causal_strengths=["CONFIRMED_CAUSAL", "DISCLOSED_ONLY"],
            evidence=["A causes B.", "Revenue was 100 million."],
            pages=[1, 2],
            years=[2025, 2025],
            evidence_ids=["claim_hop_0", "claim_hop_1"],
            filings=["2025-10-K.pdf", "2025-10-K.pdf"],
            total_hops=2,
        )

        def validate(statement, claim_id, page, **extra):
            return GraphRAGEngine._validate_report_grounding(
                f"{statement} [EvidenceClaim: {claim_id}; p.{page}]",
                [path],
                structured_report={
                    "claims": [{
                        "statement": statement,
                        "evidence_claim_ids": [claim_id],
                        "pages": [page],
                        "fiscal_years": [2025],
                        "source_filings": ["2025-10-K.pdf"],
                        **extra,
                    }]
                },
            )

        borrowed = validate(
            "Revenue was 100 million.", "claim_hop_0", 1,
            numeric_fields={"value": "100"},
        )
        self.assertEqual(borrowed["status"], "UNSUPPORTED")

        forged_number = validate(
            "Revenue was 999 million.", "claim_hop_1", 2,
            numeric_fields={"value": "999"}, unit="million",
        )
        self.assertEqual(forged_number["status"], "UNSUPPORTED")

        negated = validate(
            "Revenue was not 100 million.", "claim_hop_1", 2,
            numeric_fields={"value": "100"}, unit="million",
        )
        self.assertEqual(negated["status"], "UNSUPPORTED")

        scale_mismatch = validate(
            "Revenue was 100 billion.", "claim_hop_1", 2,
        )
        self.assertEqual(scale_mismatch["status"], "UNSUPPORTED")
        self.assertTrue(any(
            item.get("scale_mismatch") for item in scale_mismatch["claim_support_failures"][0]["diagnostics"]
        ))

    def test_query_plan_scope_is_passed_to_vector_storage(self):
        class Vector:
            def __init__(self):
                self.source_filing = None

            def retrieve_with_metadata(self, query, *, k, source_filing):
                self.source_filing = source_filing
                return {"status": "NO_HITS", "hits": [], "collection": "test"}

        vector = Vector()
        engine = GraphRAGEngine.__new__(GraphRAGEngine)
        engine.reranker = None
        engine.llm = SimpleNamespace(
            provider="none", default_model="none",
            last_success_provider=None, last_success_model=None,
        )
        engine.model_name = "none"
        with patch.object(GraphRAGEngine, "_ensure_connection", return_value=False):
            result = engine.query(
                "What was revenue in the 2025 annual report for FY2024?",
                retrieval_mode="vector",
                vector_engine=vector,
                synthesize=False,
                use_llm_anchors=False,
            )
        self.assertEqual(vector.source_filing, "2025-10-K.pdf")
        self.assertEqual(result["metadata"]["query_plan"]["fact_period"], "FY2024")

    def test_query_plan_recognizes_fy_fact_in_later_filing_alias(self):
        plan = parse_query("What was FY2024 revenue in the 2025 filing?")
        self.assertEqual(plan.document_scope, "2025-10-K.pdf")
        self.assertEqual(plan.fact_period, "FY2024")
        self.assertEqual(plan.disclosure_as_of, "FY2025")


if __name__ == "__main__":
    unittest.main()
