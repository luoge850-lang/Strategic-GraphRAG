import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.export_table_annotation_queue import export


class _Manager:
    def _read(self, query, **params):
        return [{
            "id": "FO_1",
            "company_id": "NVIDIA_CORPORATION",
            "metric_id": "REVENUE",
            "fiscal_year": 2025,
            "value": 100.0,
            "raw_value": "100",
            "unit": "USD millions",
            "source_filing": "2025-10-K.pdf",
            "page": 52,
            "row_label": "Revenue",
            "column_label": "2025",
            "table_name": "Statements",
            "statement_type": "FINANCIAL_TABLE",
            "claim_id": "claim-1",
            "evidence_sentence": "Revenue 100",
            "table_context": "Year Ended 2025",
        }]


class TableAnnotationQueueTests(unittest.TestCase):
    def test_export_is_explicitly_not_gold(self):
        with patch("scripts.export_table_annotation_queue.get_schema_manager", return_value=_Manager()):
            rows = export(1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_status"], "UNLABELED_CANDIDATE")
        self.assertEqual(rows[0]["gold"]["metric_id"], "")


if __name__ == "__main__":
    unittest.main()
