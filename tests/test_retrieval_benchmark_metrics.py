import unittest

from scripts.evaluate_retrieval_benchmark import (
    _bootstrap_ci,
    _metrics,
    _mrr,
    _ndcg,
    _paired_comparison,
    _page_key,
    _score_record,
    _stratified_metrics,
)


class RetrievalBenchmarkMetricTests(unittest.TestCase):
    def test_page_key_normalizes_pdf_suffix(self):
        self.assertEqual(_page_key("2025-10-K.pdf", 80), "2025-10-K#80")

    def test_mrr_uses_first_relevant_rank(self):
        self.assertEqual(_mrr(["2025-10-K#4", "2025-10-K#80"], {"2025-10-K#80"}), 0.5)
        self.assertEqual(_mrr(["2025-10-K#4"], {"2025-10-K#80"}), 0.0)

    def test_ndcg_is_one_for_ideal_order(self):
        self.assertEqual(_ndcg(["2025-10-K#4", "2025-10-K#80"], {"2025-10-K#4", "2025-10-K#80"}, 2), 1.0)

    def test_score_record_computes_page_metrics_and_abstention(self):
        item = {
            "id": "SILVER-001",
            "question_type": "direct_relation",
            "answerable": True,
            "expected_page_keys": ["2025-10-K#4", "2025-10-K#80"],
        }
        result = {
            "paths": [
                {
                    "filings": ["2025-10-K.pdf", "2025-10-K.pdf"],
                    "pages": [4, 80],
                    "evidence_ids": ["claim-1", "claim-2"],
                }
            ],
            "metadata": {"retrieval": {"status": "ok"}},
        }
        row = _score_record(item, "graph", result, 12.345, 5)
        self.assertEqual(row["hit_rate_at_5"], 1.0)
        self.assertEqual(row["recall_at_5"], 1.0)
        self.assertEqual(row["ndcg_at_5"], 1.0)
        self.assertEqual(row["mrr"], 1.0)
        self.assertEqual(row["retrieved_evidence_ids"], ["claim-1", "claim-2"])

        unsupported = {
            "id": "SILVER-002",
            "question_type": "unsupported",
            "answerable": False,
            "requires_abstention": True,
        }
        abstained = _score_record(
            unsupported,
            "vector",
            {"answer": "Insufficient evidence", "structured_report": {"status": "NO_HITS"}},
            1.0,
            5,
        )
        self.assertTrue(abstained["abstention_correct"])

    def test_metrics_include_macro_scores_and_question_level_ci(self):
        rows = [
            {
                "id": "Q1",
                "question_type": "direct_relation",
                "answerable": True,
                "error": None,
                "precision_at_5": 0.2,
                "hit_rate_at_5": 1.0,
                "recall_at_5": 0.5,
                "ndcg_at_5": 0.5,
                "mrr": 1.0,
                "runtime_latency_ms": 10.0,
            },
            {
                "id": "Q2",
                "question_type": "multi_hop",
                "answerable": True,
                "error": None,
                "precision_at_5": 0.4,
                "hit_rate_at_5": 1.0,
                "recall_at_5": 1.0,
                "ndcg_at_5": 1.0,
                "mrr": 0.5,
                "runtime_latency_ms": 20.0,
            },
            {
                "id": "Q3",
                "question_type": "unsupported",
                "answerable": False,
                "error": None,
                "abstention_correct": True,
                "runtime_latency_ms": 30.0,
            },
        ]
        metrics = _metrics(rows, 5, bootstrap_resamples=50, bootstrap_seed=7)
        self.assertEqual(metrics["answerable_questions"], 2)
        self.assertEqual(metrics["unsupported_questions"], 1)
        self.assertEqual(metrics["recall_at_k"]["5"], 0.75)
        self.assertEqual(metrics["hit_rate_at_k"]["5"], 1.0)
        ci = metrics["bootstrap_ci_95"]["recall_at_k"]["5"]
        self.assertEqual(ci["unit"], "question_id")
        self.assertEqual(ci["question_count"], 2)
        self.assertEqual(ci["resamples"], 50)
        self.assertEqual(ci["seed"], 7)
        self.assertLessEqual(ci["lower"], 0.75)
        self.assertGreaterEqual(ci["upper"], 0.75)

    def test_stratified_metrics_keep_question_type_counts_separate(self):
        rows = [
            {
                "id": "Q1",
                "question_type": "direct_relation",
                "answerable": True,
                "error": None,
                "precision_at_5": 0.2,
                "recall_at_5": 0.5,
                "ndcg_at_5": 0.5,
                "mrr": 1.0,
                "runtime_latency_ms": 10.0,
            },
            {
                "id": "Q2",
                "question_type": "multi_hop",
                "answerable": True,
                "error": None,
                "precision_at_5": 0.4,
                "recall_at_5": 1.0,
                "ndcg_at_5": 1.0,
                "mrr": 0.5,
                "runtime_latency_ms": 20.0,
            },
        ]
        stratified = _stratified_metrics(
            {"graph": rows},
            5,
            bootstrap_resamples=10,
            bootstrap_seed=11,
        )
        self.assertEqual(stratified["graph"]["direct_relation"]["answerable_questions"], 1)
        self.assertEqual(stratified["graph"]["multi_hop"]["recall_at_k"]["5"], 1.0)

    def test_paired_comparison_reports_win_tie_loss_by_question_id(self):
        mode_a = [
            {"id": "Q1", "answerable": True, "error": None, "recall_at_5": 0.5, "ndcg_at_5": 0.5},
            {"id": "Q2", "answerable": True, "error": None, "recall_at_5": 0.7, "ndcg_at_5": 0.8},
            {"id": "Q3", "answerable": False, "error": None, "recall_at_5": None, "ndcg_at_5": None},
        ]
        mode_b = [
            {"id": "Q1", "answerable": True, "error": None, "recall_at_5": 0.7, "ndcg_at_5": 0.5},
            {"id": "Q2", "answerable": True, "error": None, "recall_at_5": 0.7, "ndcg_at_5": 0.3},
            {"id": "Q3", "answerable": True, "error": "Timeout", "recall_at_5": 0.9, "ndcg_at_5": 0.9},
        ]
        rows = {"graph": mode_a, "vector": mode_b}
        recall = _paired_comparison(rows, "graph", "vector", "recall_at_5")
        self.assertEqual(recall["unit_of_comparison"], "question_id")
        self.assertEqual(recall["eligible_questions"], 2)
        self.assertEqual(recall["excluded_questions"], 1)
        self.assertEqual(recall["win_tie_loss"], {"mode_a_wins": 0, "ties": 1, "mode_b_wins": 1})

        ndcg = _paired_comparison(rows, "graph", "vector", "ndcg_at_5")
        self.assertEqual(ndcg["win_tie_loss"], {"mode_a_wins": 1, "ties": 1, "mode_b_wins": 0})

    def test_bootstrap_ci_has_deterministic_and_empty_boundaries(self):
        self.assertIsNone(_bootstrap_ci([], resamples=20, seed=3))
        one = _bootstrap_ci([1.0], resamples=20, seed=3)
        self.assertEqual(one["lower"], 1.0)
        self.assertEqual(one["upper"], 1.0)
        first = _bootstrap_ci([0.0, 0.5, 1.0], resamples=100, seed=3)
        second = _bootstrap_ci([0.0, 0.5, 1.0], resamples=100, seed=3)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first["lower"], 0.0)
        self.assertLessEqual(first["upper"], 1.0)
        with self.assertRaises(ValueError):
            _bootstrap_ci([1.0], resamples=0, seed=3)


if __name__ == "__main__":
    unittest.main()
