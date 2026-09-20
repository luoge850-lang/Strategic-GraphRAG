import json
import asyncio
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from strategic_graphrag.engine.evidence_quality import apply_directness_ranking
from strategic_graphrag.engine.graph_rag_engine import CausalPath, GraphRAGEngine
from strategic_graphrag.response_contract import (
    classify_outcome,
    has_substantive_claims,
    is_abstention,
    is_answer_attempted,
    response_state,
)
from strategic_graphrag.pipeline.extractor import TripleExtractor
from strategic_graphrag.pipeline.financial_table_extractor import (
    _numeric_values,
    extract_financial_table_triples,
)
from strategic_graphrag.api import server


class _FakePage:
    def __init__(self, rows):
        self.rows = rows

    def extract_tables(self):
        return [self.rows]


def _path(evidence, *, role="ANSWER_CRITICAL"):
    path = CausalPath(
        path_id="p1",
        nodes=["NVIDIA_CORPORATION", "DRIVE_PLATFORM"],
        node_labels=["Company", "Product"],
        relationships=["PRODUCES"],
        causal_strengths=["DISCLOSED_ONLY"],
        evidence=[evidence],
        pages=[17],
        years=[2025],
        evidence_ids=["claim_v2_p1"],
        filings=["2025-10-K.pdf"],
        total_hops=1,
        evidence_role=role,
    )
    return path


class EvidenceAndOutcomeContractTests(unittest.TestCase):
    def test_api_does_not_return_http_200_for_dependency_outcome(self):
        class _BrokenEngine:
            def query(self, *args, **kwargs):
                return {
                    "query": "Q",
                    "answer": "[CONNECTION ERROR]",
                    "outcome": "DEPENDENCY_ERROR",
                    "paths": [],
                    "evidence_sentences": [],
                    "structured_report": {"status": "DEPENDENCY_ERROR"},
                    "metadata": {
                        "outcome": "DEPENDENCY_ERROR",
                        "error": {"code": "DEPENDENCY_ERROR", "dependency": "neo4j"},
                    },
                }

        with patch.object(server, "get_graph_engine", return_value=_BrokenEngine()):
            with self.assertRaises(HTTPException) as raised:
                asyncio.run(server.graphrag_query(server.QueryRequest(
                    question="Q",
                    retrieval_mode="graph",
                )))
        self.assertEqual(raised.exception.status_code, 503)
        self.assertEqual(raised.exception.detail["outcome"], "DEPENDENCY_ERROR")

    def test_negative_guard_with_retained_fact_is_not_grounding_exempt(self):
        guarded = {
            "status": "NEGATIVE_CLAIM_GUARD",
            "executive_summary": "The filing reports a disclosed product relationship.",
            "claims": [{"statement": "NVIDIA reports a product relationship."}],
        }
        pure_refusal = {
            "status": "NEGATIVE_CLAIM_GUARD",
            "executive_summary": "The indexed corpus audit was unavailable, so no absence conclusion is permitted.",
            "claims": [],
        }
        self.assertTrue(has_substantive_claims(guarded))
        self.assertFalse(has_substantive_claims(pure_refusal))

    def test_outcome_contract_prioritizes_structured_status_and_legacy_fallback(self):
        self.assertEqual(
            classify_outcome({"outcome": "DEPENDENCY_ERROR", "answer": "[CONNECTION ERROR]"}),
            "DEPENDENCY_ERROR",
        )
        self.assertEqual(
            classify_outcome({
                "structured_report": {"status": "NEGATIVE_CLAIM_GUARD"},
                "answer": "",
            }),
            "ABSTAINED",
        )
        self.assertEqual(
            classify_outcome({
                "structured_report": {
                    "status": "NEGATIVE_CLAIM_GUARD",
                    "claims": [{"statement": "A retained factual claim."}],
                },
                "answer": "A retained factual claim.",
            }),
            "PARTIALLY_ANSWERED",
        )
        self.assertEqual(
            classify_outcome({"answer": "The provided context does not contain any information about this metric."}),
            "ABSTAINED",
        )
        self.assertTrue(is_abstention({"answer": "The provided context does not contain any information."}))

    def test_mixed_fact_and_refusal_remains_substantive(self):
        payload = {
            "answer": (
                "Revenue was 100 million. The indexed corpus audit was unavailable, "
                "so no absence conclusion is permitted."
            )
        }
        self.assertTrue(has_substantive_claims(payload))
        self.assertEqual(response_state(payload)["answer_status"], "ANSWERED")

    def test_response_contract_separates_retrieval_only_and_failures(self):
        retrieval_only = response_state({
            "structured_report": {"status": "RETRIEVAL_ONLY"},
            "answer": "Retrieved evidence trace.",
        })
        self.assertEqual(retrieval_only["answer_status"], "NOT_REQUESTED")
        self.assertFalse(is_answer_attempted({
            "structured_report": {"status": "RETRIEVAL_ONLY"},
            "answer": "Retrieved evidence trace.",
        }))
        self.assertFalse(is_abstention({
            "structured_report": {"status": "RETRIEVAL_ONLY"},
            "answer": "Retrieved evidence trace.",
        }))
        self.assertEqual(response_state({}, http_status=200)["execution_status"], "CONTRACT_ERROR")
        self.assertEqual(response_state({"answer": "ignored"}, http_status=503)["execution_status"], "DEPENDENCY_ERROR")
        self.assertEqual(response_state({"answer": "ignored"}, http_status=504)["execution_status"], "TIMEOUT")
        self.assertEqual(response_state({"answer": "ignored"}, http_status=429)["execution_status"], "RATE_LIMITED")
        self.assertEqual(response_state({"answer": "ignored"}, http_status=401)["execution_status"], "AUTH_ERROR")
        self.assertEqual(response_state({"answer_status": "UNKNOWN"})["execution_status"], "CONTRACT_ERROR")

    def test_grounding_rejects_correct_id_with_wrong_page_file_and_year(self):
        path = _path("NVIDIA introduced DRIVE products.")
        report = {
            "status": "GENERATED",
            "executive_summary": "NVIDIA reports a product relationship.",
            "claims": [{
                "statement": "NVIDIA reports a product relationship.",
                "evidence_claim_ids": ["claim_v2_p1"],
                "pages": [99],
                "fiscal_years": [2024],
                "source_filings": ["2024-10-K.pdf"],
            }],
        }
        canonical = GraphRAGEngine._canonicalize_report_citations(report, [path])
        grounding = GraphRAGEngine._validate_report_grounding(
            canonical["narrative"],
            [path],
            structured_report=canonical,
        )
        self.assertEqual(grounding["status"], "UNSUPPORTED")
        self.assertTrue(grounding["claim_citation_mismatches"])

    def test_grounding_rejects_real_evidence_that_does_not_support_relation(self):
        path = _path(
            "Another company produces Jetson platform products that compete with NVIDIA Corporation."
        )
        apply_directness_ranking(
            [path],
            "What evidence-backed PRODUCES relationship connects NVIDIA_CORPORATION and DRIVE_PLATFORM?",
            "CAUSAL_CHAIN",
        )
        report = {
            "status": "GENERATED",
            "executive_summary": "NVIDIA produces DRIVE_PLATFORM.",
            "claims": [{
                "statement": "NVIDIA produces DRIVE_PLATFORM.",
                "evidence_claim_ids": ["claim_v2_p1"],
                "pages": [17],
                "fiscal_years": [2025],
                "source_filings": ["2025-10-K.pdf"],
            }],
        }
        grounding = GraphRAGEngine._validate_report_grounding(
            "NVIDIA produces DRIVE_PLATFORM [EvidenceClaim: claim_v2_p1; p.17]",
            [path],
            structured_report=report,
            query="What evidence-backed PRODUCES relationship connects NVIDIA_CORPORATION and DRIVE_PLATFORM?",
            intent="CAUSAL_CHAIN",
        )
        self.assertEqual(grounding["status"], "UNSUPPORTED")
        self.assertTrue(grounding["claim_support_failures"])

    def test_grounding_rejects_numeric_mismatch_and_mixed_citation_support(self):
        revenue_path = _path("Revenue 100 million was reported.")
        unrelated_path = CausalPath(
            path_id="p2",
            nodes=["NVIDIA_CORPORATION", "SUPPLY_CHAIN_RISK"],
            node_labels=["Company", "RiskFactor"],
            relationships=["EXPOSED_TO"],
            causal_strengths=["DISCLOSED_ONLY"],
            evidence=["NVIDIA disclosed supply-chain exposure."],
            pages=[18],
            years=[2025],
            evidence_ids=["claim_v2_p2"],
            filings=["2025-10-K.pdf"],
            total_hops=1,
            evidence_role="BACKGROUND_CONTEXT",
        )
        report = {
            "status": "GENERATED",
            "executive_summary": "Revenue was 101 million.",
            "claims": [{
                "statement": "Revenue was 101 million.",
                "evidence_claim_ids": ["claim_v2_p1", "claim_v2_p2"],
                "pages": [17, 18],
                "fiscal_years": [2025],
                "source_filings": ["2025-10-K.pdf"],
            }],
        }
        grounding = GraphRAGEngine._validate_report_grounding(
            "Revenue was 101 million [EvidenceClaim: claim_v2_p1; p.17]",
            [revenue_path, unrelated_path],
            structured_report=report,
            query="What revenue was reported?",
            intent="FINANCIAL_METRIC",
        )
        self.assertEqual(grounding["status"], "UNSUPPORTED")
        self.assertTrue(grounding["claim_support_failures"])
        self.assertTrue(grounding["claim_citation_mismatches"] or grounding["claim_support_failures"])

    def test_grounding_rejects_summary_claim_not_present_in_structured_claims(self):
        path = _path("NVIDIA produces DRIVE platform products.")
        report = {
            "status": "GENERATED",
            "executive_summary": "NVIDIA produces DRIVE platform products. Revenue increased.",
            "claims": [{
                "statement": "NVIDIA produces DRIVE platform products.",
                "evidence_claim_ids": ["claim_v2_p1"],
                "pages": [17],
                "fiscal_years": [2025],
                "source_filings": ["2025-10-K.pdf"],
            }],
        }
        grounding = GraphRAGEngine._validate_report_grounding(
            "NVIDIA produces DRIVE platform products [EvidenceClaim: claim_v2_p1; p.17]",
            [path],
            structured_report=report,
        )
        self.assertEqual(grounding["status"], "UNSUPPORTED")
        self.assertTrue(any(
            item["reason"] == "SUMMARY_NOT_COVERED_BY_CLAIMS"
            for item in grounding["claim_support_failures"]
        ))

    def test_produces_subject_binding_rejects_competitor_sentence(self):
        supported, _ = TripleExtractor._evidence_supports_relation(
            "NVIDIA_CORPORATION",
            "JETSON_PLATFORM",
            "PRODUCES",
            "NVIDIA Corporation produces Jetson platform products.",
        )
        rejected, reason = TripleExtractor._evidence_supports_relation(
            "NVIDIA_CORPORATION",
            "JETSON_PLATFORM",
            "PRODUCES",
            "Another company produces Jetson platform products that compete with NVIDIA Corporation.",
        )
        self.assertTrue(supported)
        self.assertFalse(rejected)
        self.assertIn("PRODUCTION", reason)

    def test_percentage_rows_keep_both_period_values(self):
        self.assertEqual(_numeric_values("Gross margin 75.0% 72.7%"), ["75.0", "72.7"])
        page_text = "\n".join([
            "Percentage of revenue",
            "Year Ended",
            "2025 2024",
            "Gross margin 75.0% 72.7%",
        ])
        triples = extract_financial_table_triples(
            _FakePage([["Gross margin", "75.0%", "72.7%"]]),
            page_text,
            2025,
            company_id="OTHER_COMPANY",
        )
        self.assertEqual(json.loads(triples[0]["metric_values_json"]), [
            {"period": "2025", "value": "75.0"},
            {"period": "2024", "value": "72.7"},
        ])

    def test_table_subject_is_pending_without_verified_company_input(self):
        page_text = "\n".join([
            "Year Ended",
            "2025 2024",
            "Revenue 100 90",
        ])
        triples = extract_financial_table_triples(
            _FakePage([["Revenue", "100", "90"]]),
            page_text,
            2025,
        )
        self.assertEqual(triples[0]["source"], "")
        self.assertEqual(triples[0]["subject_resolution_status"], "PENDING_COMPANY_REVIEW")
        self.assertEqual(triples[0]["company_id"], None)


if __name__ == "__main__":
    unittest.main()
