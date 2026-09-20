import unittest

from scripts.evaluate_table_quality import evaluate, normalize_gold_rows


class TableQualityMetricTests(unittest.TestCase):
    def test_metrics_separate_numeric_unit_and_evidence_errors(self):
        gold = [{
            "id": "claim-1",
            "company_id": "NVIDIA_CORPORATION",
            "fiscal_year": 2025,
            "metric_id": "REVENUE",
            "value": "100",
            "unit": "USD millions",
            "source_filing": "2025-10-K.pdf",
            "page": 52,
            "row_label": "Revenue",
            "column_label": "2025",
            "table_name": "Statements",
            "evidence_text": "Revenue 100",
            "review_status": "HUMAN_REVIEWED",
        }]
        predicted = [{
            **gold[0],
            "value": "-100",
            "unit": "USD thousands",
            "evidence_text": "",
            "evidence_sentence": "Revenue -100",
        }]
        report = evaluate(gold, predicted)
        self.assertEqual(report["row_retrieval"]["f1"], 1.0)
        self.assertEqual(report["numeric_exact_match"]["recall"], 0.0)
        self.assertEqual(report["unit_accuracy"]["recall"], 0.0)
        self.assertEqual(report["table_cell_accuracy"], 0.0)
        self.assertEqual(report["evidence_alignment"]["rate"], 0.0)

    def test_candidate_gold_is_not_independent_human_gold(self):
        row = {"id": "claim-1", "value": "1", "review_status": "AUTO_GENERATED_CANDIDATE"}
        report = evaluate([row], [row])
        self.assertFalse(report["gold_is_independent_human"])

    def test_annotation_queue_gold_is_flattened_and_unsupported_rows_are_excluded(self):
        queued = [
            {
                "id": "cell-1",
                "review_status": "HUMAN_REVIEWED",
                "reviewer": "reviewer_a",
                "gold": {
                    "company_id": "nvidia_corporation",
                    "fiscal_year": 2025,
                    "metric_id": "revenue",
                    "value": 100,
                    "unit": "USD millions",
                    "source_filing": "2025-10-K.pdf",
                    "page": 52,
                    "row_label": "Revenue",
                    "column_label": "2025",
                    "table_name": "Statements",
                    "evidence_text": "Revenue 100",
                    "cell_supported": True,
                },
            },
            {
                "id": "cell-2",
                "review_status": "HUMAN_REVIEWED",
                "reviewer": "reviewer_a",
                "gold": {"cell_supported": False},
            },
        ]
        gold = normalize_gold_rows(queued)
        predicted = [{"id": "cell-1", "value": 100}, {"id": "cell-2", "value": 1}]
        report = evaluate(gold, predicted)
        self.assertEqual(report["candidate_gold_labels"]["supported_rows_evaluated"], 1)
        self.assertEqual(report["candidate_gold_labels"]["unsupported_rows_excluded_from_gold_set"], 1)
        self.assertEqual(report["row_retrieval"]["precision"], 0.5)
        self.assertEqual(report["row_retrieval"]["recall"], 1.0)


if __name__ == "__main__":
    unittest.main()
