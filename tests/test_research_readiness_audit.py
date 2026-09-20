import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_research_readiness import audit


ROOT = Path(__file__).resolve().parents[1]


class ResearchReadinessAuditTests(unittest.TestCase):
    def test_current_checkout_reports_real_research_blockers(self):
        report = audit(ROOT)

        self.assertEqual(report["status"], "NOT_READY")
        self.assertEqual(report["facts"]["post_repair_annotation"]["rows"], 30)
        self.assertEqual(report["facts"]["post_repair_annotation"]["labeled"], 30)
        ai_assisted = report["facts"]["ai_assisted_annotation"]
        self.assertEqual(ai_assisted["rows"], 30)
        self.assertEqual(ai_assisted["labeled"], 30)
        self.assertEqual(ai_assisted["annotator_counts"], {"gpt5.6_sol_ai_01": 30})
        self.assertEqual(ai_assisted["label_counts"]["relation_correct"]["true"], 22)
        self.assertEqual(ai_assisted["label_counts"]["relation_correct"]["uncertain"], 7)
        self.assertEqual(ai_assisted["label_counts"]["relation_correct"]["false"], 1)
        self.assertEqual(ai_assisted["label_counts"]["evidence_supports_relation"]["true"], 22)
        self.assertFalse(ai_assisted["is_independent_human_golden_qa"])
        ai_check = next(
            item for item in report["checks"]
            if item["name"] == "ai_assisted_annotation_not_human_golden_qa"
        )
        self.assertFalse(ai_check["blocking"])
        self.assertEqual(ai_check["status"], "PASS")
        human_check = next(
            item for item in report["checks"]
            if item["name"] == "human_golden_qa_available"
        )
        self.assertEqual(human_check["status"], "PASS")
        self.assertNotIn("human_golden_qa_available", report["blocking_failures"])
        benchmark = report["facts"].get("retrieval_benchmark") or {}
        if benchmark.get("question_count", 0) >= 30:
            self.assertNotIn("retrieval_benchmark_has_multiple_questions", report["blocking_failures"])
        else:
            self.assertIn("retrieval_benchmark_has_multiple_questions", report["blocking_failures"])
        self.assertIn("extraction_run_is_repeatable", report["blocking_failures"])

    def test_temporal_inventory_detects_stale_and_rebuilt_snapshots(self):
        # Keep the stale-state regression independent of mutable local reports.
        for fact_count, expected_status in ((383, "BLOCKED"), (381, "PASS")):
            with self.subTest(fact_count=fact_count), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                reports = root / "reports"
                reports.mkdir()
                snapshot = {
                    "graph": {
                        "nodes_by_label": [{"label": "TemporalFact", "count": fact_count}],
                        "active_claims": [{"doc_id": "2025-10-K", "claims": 381}],
                    }
                }
                (reports / "neo4j_snapshot_post_rebuild_fixture.json").write_text(
                    json.dumps(snapshot), encoding="utf-8"
                )
                report = audit(root)
                check = next(
                    item for item in report["checks"]
                    if item["name"] == "derived_temporal_model_matches_active_claims"
                )
                self.assertEqual(check["status"], expected_status)

    def test_audit_is_fail_closed_when_required_files_are_missing(self):
        report = audit(ROOT / "does-not-exist")

        self.assertEqual(report["status"], "NOT_READY")
        self.assertIn("corpus_manifest_present", report["blocking_failures"])
        self.assertIn("post_rebuild_snapshot_present", report["blocking_failures"])

    def test_candidate_never_satisfies_human_golden_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            candidate_path = temp_root / "data" / "evaluation" / "golden_qa_v2.jsonl"
            candidate_path.parent.mkdir(parents=True)
            candidate_path.write_text(
                json.dumps(
                    {
                        "id": "GQ-001",
                        "answerable": True,
                        "review_status": "AUTO_GENERATED_REGRESSION_CANDIDATE",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            report = audit(temp_root)

            self.assertIn("human_golden_qa_available", report["blocking_failures"])
            self.assertEqual(report["facts"]["human_gold"]["rows"], 0)
            self.assertEqual(report["facts"]["human_gold"]["answerable"], 0)
            self.assertEqual(
                report["facts"]["golden_qa_candidate"]["dataset_status"],
                "AUTO_GENERATED_REGRESSION_CANDIDATE",
            )
            self.assertFalse(report["facts"]["golden_qa_candidate"]["eligible_as_human_gold"])

    def test_thirty_reviewed_human_rows_pass_direct_human_qa_check(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            human_path = temp_root / "evaluation" / "golden_qa_human_v1.jsonl"
            human_path.parent.mkdir(parents=True)
            rows = [
                {
                    "id": f"GQ-{index:03d}",
                    "reference_answer": "A reviewed answer.",
                    "gold_evidence_ids": [f"claim-{index}"],
                    "relevant_evidence_grades": {f"claim-{index}": 2},
                    "answerable": index % 2 == 0,
                    "requires_abstention": index % 2 != 0,
                    "reviewer": "reviewer_a",
                    "review_notes": "Reviewed against the filing.",
                    "review_status": "HUMAN_REVIEWED",
                }
                for index in range(1, 31)
            ]
            human_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit(temp_root)
            human_check = next(
                item
                for item in report["checks"]
                if item["name"] == "human_golden_qa_available"
            )

            self.assertEqual(human_check["status"], "PASS")
            self.assertNotIn("human_golden_qa_available", report["blocking_failures"])
            self.assertEqual(
                report["facts"]["human_gold"]["source"],
                str(Path("evaluation") / "golden_qa_human_v1.jsonl"),
            )
            self.assertEqual(
                report["facts"]["human_gold"]["status_counts"],
                {"HUMAN_REVIEWED": 30},
            )
            self.assertEqual(report["facts"]["human_gold"]["invalid_rows"], 0)

    def test_review_status_alone_cannot_satisfy_human_qa(self):
        with tempfile.TemporaryDirectory() as directory:
            temp_root = Path(directory)
            human_path = temp_root / "evaluation" / "golden_qa_human_v1.jsonl"
            human_path.parent.mkdir(parents=True)
            rows = [
                {"id": f"GQ-{index:03d}", "review_status": "HUMAN_REVIEWED"}
                for index in range(1, 31)
            ]
            human_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = audit(temp_root)

            self.assertIn("human_golden_qa_available", report["blocking_failures"])
            self.assertEqual(report["facts"]["human_gold"]["invalid_rows"], 30)
