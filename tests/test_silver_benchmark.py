import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_silver_benchmark import _direct_item, _multi_item, _unsupported_items


class SilverBenchmarkTests(unittest.TestCase):
    def test_direct_item_is_explicitly_silver_and_page_linked(self):
        row = _direct_item(
            {
                "claim_id": "claim_v2_abc",
                "claim_text": "NVIDIA reports revenue.",
                "page": 80,
                "source_id": "nvidia_corporation",
                "target_id": "revenue",
                "relation": "REPORTS_METRIC",
            },
            "2025-10-K.pdf",
            2025,
            1,
        )
        self.assertEqual(row["benchmark_status"], "AUTO_GENERATED_SILVER")
        self.assertEqual(row["expected_page_keys"], ["2025-10-K#80"])
        self.assertEqual(row["expected_evidence_ids"], ["claim_v2_abc"])
        self.assertNotIn("review_status", row)

    def test_multi_hop_preserves_both_evidence_units(self):
        row = _multi_item(
            {
                "source_id": "event",
                "middle_id": "risk",
                "target_id": "revenue",
                "relation1": "CAUSES",
                "relation2": "DECREASES",
                "claim1_id": "claim-1",
                "claim1_text": "Event causes risk.",
                "page1": 17,
                "claim2_id": "claim-2",
                "claim2_text": "Risk decreases revenue.",
                "page2": 18,
            },
            "2024-10-K.pdf",
            2024,
            2,
        )
        self.assertEqual(row["expected_evidence_ids"], ["claim-1", "claim-2"])
        self.assertEqual(row["expected_page_keys"], ["2024-10-K#17", "2024-10-K#18"])

    def test_unsupported_rows_are_abstention_cases(self):
        rows = _unsupported_items(10)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(not row["answerable"] for row in rows))
        self.assertTrue(all(row["requires_abstention"] for row in rows))
        self.assertTrue(all(row["expected_evidence_ids"] == [] for row in rows))


if __name__ == "__main__":
    unittest.main()
