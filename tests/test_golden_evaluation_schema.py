import unittest

from scripts.run_golden_evaluation import (
    _answerable,
    _dataset_status,
    _gold_evidence_ids,
    _gold_pages,
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


if __name__ == "__main__":
    unittest.main()
