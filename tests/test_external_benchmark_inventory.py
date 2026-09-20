import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.prepare_external_benchmarks import _validate


class ExternalBenchmarkInventoryTests(unittest.TestCase):
    def test_financebench_schema_is_checked_without_calling_it_gold_for_nvidia(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "financebench.jsonl"
            path.write_text(json.dumps({
                "financebench_id": 1,
                "question": "q",
                "answer": "a",
                "evidence": [],
                "doc_name": "DOC",
            }) + "\n", encoding="utf-8")
            report = _validate("financebench_questions", path, "jsonl")
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["rows"], 1)

    def test_missing_required_fields_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "finqa.json"
            path.write_text(json.dumps([{"id": "x"}]), encoding="utf-8")
            report = _validate("finqa_test", path, "json")
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(report["schema_errors"], 1)


if __name__ == "__main__":
    unittest.main()
