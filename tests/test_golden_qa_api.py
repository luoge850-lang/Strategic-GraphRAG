import json
import tempfile
import unittest
from pathlib import Path

from strategic_graphrag.api.server import (
    golden_qa_summary,
    read_golden_qa,
    update_golden_qa,
)


class GoldenQAApiTests(unittest.TestCase):
    def _rows(self):
        return [
            {
                "id": "GQ-001",
                "question": "What changed?",
                "candidate_expected_answer": "Candidate answer",
                "candidate_evidence_claim_ids": ["claim-1"],
                "review_status": "HUMAN_REVIEW_PENDING",
                "reference_answer": "",
                "gold_evidence_ids": [],
                "gold_pages": [],
                "relevant_evidence_grades": {},
                "answerable": None,
                "requires_abstention": None,
                "reviewer": "",
                "review_notes": "",
            },
            {
                "id": "GQ-002",
                "question": "Can this be answered from the filing?",
                "candidate_expected_answer": "",
                "candidate_evidence_claim_ids": [],
                "review_status": "HUMAN_REVIEW_PENDING",
                "reference_answer": "",
                "gold_evidence_ids": [],
                "gold_pages": [],
                "relevant_evidence_grades": {},
                "answerable": None,
                "requires_abstention": None,
                "reviewer": "",
                "review_notes": "",
            },
        ]

    def _path(self, directory: str) -> Path:
        path = Path(directory) / "golden_qa.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self._rows()),
            encoding="utf-8",
        )
        return path

    def test_summary_counts_only_human_reviewed_rows(self):
        rows = self._rows()
        rows[0]["review_status"] = "HUMAN_REVIEWED"
        self.assertEqual(golden_qa_summary(rows), {"total": 2, "reviewed": 1, "pending": 1})

    def test_draft_update_is_atomic_and_preserves_candidate_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._path(directory)
            before_candidate = read_golden_qa(path)[0]["candidate_expected_answer"]
            updated, rows = update_golden_qa(
                "GQ-001",
                {
                    "reference_answer": "The filing does not provide enough information.",
                    "answerable": False,
                    "requires_abstention": True,
                    "reviewer": "reviewer_01",
                    "review_status": "IN_PROGRESS",
                },
                path,
            )

            self.assertEqual(updated["review_status"], "IN_PROGRESS")
            self.assertEqual(updated["candidate_expected_answer"], before_candidate)
            self.assertEqual(rows[1]["id"], "GQ-002")
            self.assertEqual(read_golden_qa(path)[0]["candidate_expected_answer"], before_candidate)

    def test_complete_answerable_row_requires_evidence_and_grades(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._path(directory)
            updated, _ = update_golden_qa(
                "GQ-001",
                {
                    "reference_answer": "Revenue increased in fiscal 2025.",
                    "gold_evidence_ids": ["claim-1"],
                    "gold_pages": [80],
                    "relevant_evidence_grades": {"claim-1": 2},
                    "answerable": True,
                    "requires_abstention": False,
                    "reviewer": "reviewer_01",
                    "review_status": "HUMAN_REVIEWED",
                },
                path,
            )
            self.assertEqual(updated["review_status"], "HUMAN_REVIEWED")
            self.assertEqual(golden_qa_summary(read_golden_qa(path))["reviewed"], 1)

    def test_invalid_complete_row_and_unknown_id_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._path(directory)
            with self.assertRaises(ValueError):
                update_golden_qa("GQ-001", {"gold_pages": [0]}, path)
            with self.assertRaises(ValueError):
                update_golden_qa(
                    "GQ-001",
                    {
                        "reference_answer": "Incomplete evidence.",
                        "answerable": True,
                        "requires_abstention": False,
                        "reviewer": "reviewer_01",
                        "review_status": "HUMAN_REVIEWED",
                    },
                    path,
                )
            with self.assertRaises(KeyError):
                update_golden_qa("missing", {"review_status": "IN_PROGRESS"}, path)


if __name__ == "__main__":
    unittest.main()
