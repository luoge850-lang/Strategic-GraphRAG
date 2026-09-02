import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_human_golden_qa import HUMAN_FIELDS, prepare_human_golden_qa


class HumanGoldenQATemplateTests(unittest.TestCase):
    def test_template_is_separate_and_blank_without_changing_candidates(self):
        candidates = [
            {
                "id": "GQ-001",
                "question": "Which evidence supports the relation?",
                "expected_answer": "Candidate answer",
                "answerable": True,
                "evidence_claim_ids": ["claim-1"],
                "review_status": "AUTO_GENERATED_REGRESSION_CANDIDATE",
            },
            {
                "id": "GQ-002",
                "question": "Is this supported by the filing?",
                "expected_answer": "Another candidate answer",
                "answerable": False,
                "evidence_claim_ids": [],
                "review_status": "AUTO_GENERATED_REGRESSION_CANDIDATE",
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            input_path = temp_root / "candidate.jsonl"
            output_path = temp_root / "evaluation" / "golden_qa_human_v1.jsonl"
            input_path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates),
                encoding="utf-8",
            )
            original_input = input_path.read_bytes()

            count = prepare_human_golden_qa(input_path, output_path)

            self.assertEqual(count, len(candidates))
            self.assertEqual(input_path.read_bytes(), original_input)
            reviewed_rows = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(reviewed_rows), len(candidates))

            for candidate, reviewed in zip(candidates, reviewed_rows):
                self.assertEqual(reviewed["id"], candidate["id"])
                self.assertEqual(reviewed["question"], candidate["question"])
                self.assertEqual(reviewed["candidate_id"], candidate["id"])
                self.assertEqual(reviewed["candidate_expected_answer"], candidate["expected_answer"])
                self.assertEqual(reviewed["candidate_answerable"], candidate["answerable"])
                self.assertEqual(
                    reviewed["candidate_review_status"],
                    "AUTO_GENERATED_REGRESSION_CANDIDATE",
                )
                self.assertEqual(reviewed["review_status"], "HUMAN_REVIEW_PENDING")
                self.assertNotIn("expected_answer", reviewed)
                self.assertNotIn("candidate_answer", reviewed)

            self.assertEqual(reviewed_rows[0]["reference_answer"], "")
            self.assertEqual(reviewed_rows[0]["gold_evidence_ids"], [])
            self.assertEqual(reviewed_rows[0]["relevant_evidence_grades"], {})
            self.assertIsNone(reviewed_rows[0]["answerable"])
            self.assertIsNone(reviewed_rows[0]["requires_abstention"])
            self.assertEqual(reviewed_rows[0]["reviewer"], "")
            self.assertEqual(reviewed_rows[0]["review_notes"], "")
            self.assertTrue(all(not reviewed_rows[0][field] for field in HUMAN_FIELDS))

    def test_candidate_input_cannot_be_selected_as_output(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate_path = Path(directory) / "candidate.jsonl"
            candidate_path.write_text("{}\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                prepare_human_golden_qa(candidate_path, candidate_path)


if __name__ == "__main__":
    unittest.main()
