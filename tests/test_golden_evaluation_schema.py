import unittest

from scripts.run_golden_evaluation import (
    _answerable,
    _bootstrap_ci,
    _collapse_question_variants,
    _dataset_status,
    _gold_evidence_ids,
    _gold_pages,
    _f1,
    _is_abstention,
    _judge_answer,
    _judge_context,
    _retrieved_ids,
    _retrieved_pages,
    _source_filing,
)


class GoldenEvaluationSchemaTests(unittest.TestCase):
    def test_human_review_fields_are_preferred_over_candidate_fields(self):
        item = {
            "question": "Q",
            "gold_evidence_ids": ["claim-human"],
            "gold_pages": [80],
            "candidate_evidence_claim_ids": ["claim-candidate"],
            "candidate_pages": [81],
            "answerable": False,
            "candidate_answerable": True,
            "source_filing": "2025-10-K.pdf",
            "candidate_source_filing": "2024-10-K.pdf",
        }

        self.assertEqual(_gold_evidence_ids(item), {"claim-human"})
        self.assertEqual(_gold_pages(item), {80})
        self.assertFalse(_answerable(item))
        self.assertEqual(_source_filing(item), "2025-10-K.pdf")

    def test_legacy_candidate_schema_remains_supported(self):
        item = {
            "evidence_claim_ids": ["claim-old"],
            "pages": [12],
            "candidate_answerable": True,
            "candidate_source_filing": "2025-10-K.pdf",
        }

        self.assertEqual(_gold_evidence_ids(item), {"claim-old"})
        self.assertEqual(_gold_pages(item), {12})
        self.assertTrue(_answerable(item))
        self.assertEqual(_source_filing(item), "2025-10-K.pdf")

    def test_dataset_status_does_not_call_pending_rows_human_gold(self):
        self.assertEqual(
            _dataset_status([{"review_status": "HUMAN_REVIEW_PENDING"}]),
            "MIXED_OR_UNREVIEWED",
        )
        self.assertEqual(
            _dataset_status([{"review_status": "HUMAN_REVIEWED"}]),
            "HUMAN_REVIEWED",
        )
        self.assertEqual(
            _dataset_status([{"review_status": "HUMAN_REVIEWED_DERIVED_QUESTION_LEVEL"}]),
            "HUMAN_REVIEWED_DERIVED_QUESTION_LEVEL",
        )

    def test_question_level_view_unions_valid_evidence_variants(self):
        rows = [
            {
                "id": "GQ-A",
                "question": "What is the answer?",
                "answerable": False,
                "gold_evidence_ids": [],
                "gold_pages": [],
            },
            {
                "id": "GQ-B",
                "question": "What is the answer?",
                "answerable": True,
                "gold_evidence_ids": ["claim-valid"],
                "gold_pages": [4],
                "reference_answer": "A supported answer.",
            },
        ]

        grouped = _collapse_question_variants(rows)
        self.assertEqual(len(grouped), 1)
        self.assertTrue(grouped[0]["answerable"])
        self.assertEqual(grouped[0]["gold_evidence_ids"], ["claim-valid"])
        self.assertEqual(grouped[0]["gold_pages"], [4])
        self.assertTrue(grouped[0]["variant_label_conflict"])
        self.assertEqual(grouped[0]["variant_count"], 2)

    def test_graph_evidence_variants_are_separate_from_primary_paths(self):
        result = {
            "paths": [{
                "evidence_ids": ["claim-primary"],
                "pages": [7],
                "evidence_variants": [[
                    {"evidence_id": "claim-primary", "page": 7},
                    {"evidence_id": "claim-gold", "page": 4},
                ]],
            }],
        }

        self.assertEqual(_retrieved_ids(result), {"claim-primary"})
        self.assertEqual(
            _retrieved_ids(result, include_variants=True),
            {"claim-primary", "claim-gold"},
        )
        self.assertEqual(_retrieved_pages(result), {7})
        self.assertEqual(
            _retrieved_pages(result, include_variants=True),
            {4, 7},
        )

    def test_vector_pages_use_common_retrieval_metadata(self):
        result = {
            "metadata": {
                "retrieval": {
                    "hits": [
                        {"metadata": {"page": 4}},
                        {"metadata": {"page": 6}},
                    ],
                },
            },
        }
        self.assertEqual(_retrieved_pages(result, mode="vector"), {4, 6})

    def test_abstention_detection_uses_structured_status(self):
        self.assertTrue(_is_abstention({
            "structured_report": {"status": "NEGATIVE_CLAIM_GUARD"},
            "answer": "",
        }))
        self.assertFalse(_is_abstention({
            "structured_report": {"status": "RETRIEVAL_ONLY"},
            "answer": "Retrieved evidence trace.",
        }))

    def test_f1_requires_matched_precision_and_recall_units(self):
        self.assertAlmostEqual(_f1(0.1192, 0.8571), 0.2093, places=4)
        self.assertIsNone(_f1(None, 0.5))

    def test_bootstrap_ci_is_deterministic_and_labeled(self):
        first = _bootstrap_ci([1.0, 3.0, 5.0], resamples=50)
        second = _bootstrap_ci([1.0, 3.0, 5.0], resamples=50)
        self.assertEqual(first, second)
        self.assertEqual(first["n"], 3)
        self.assertIn("row-level", first["method"])

    def test_judge_context_keeps_primary_and_variant_provenance(self):
        context = _judge_context({
            "paths": [{
                "evidence": ["primary text"],
                "evidence_ids": ["claim-primary"],
                "pages": [7],
                "evidence_role": "ANSWER_CRITICAL",
                "evidence_variants": [[
                    {"evidence_id": "claim-primary", "page": 7, "evidence": "primary text"},
                    {"evidence_id": "claim-gold", "page": 4, "evidence": "gold text"},
                ]],
            }],
        })
        self.assertIn("claim-primary", context)
        self.assertIn("claim-gold", context)
        self.assertIn("primary", context)
        self.assertIn("variant", context)

    def test_answer_judge_keeps_five_score_contract(self):
        class FakeJudge:
            @staticmethod
            def chat_with_fallback(**kwargs):
                return '{"faithfulness": 5, "answer_relevance": 4, "completeness": 3, "citation_correctness": 2, "justification": "bounded"}'

        result = _judge_answer(
            "Q",
            "[EvidenceClaim: claim-1; p.4] context",
            "A",
            FakeJudge(),
            answerable=True,
            reference_answer="Reference",
            gold_evidence_ids=["claim-1"],
            gold_pages=[4],
        )
        self.assertEqual(
            set(result),
            {"faithfulness", "answer_relevance", "completeness", "citation_correctness", "justification"},
        )
        self.assertEqual(result["citation_correctness"], 2.0)


if __name__ == "__main__":
    unittest.main()
