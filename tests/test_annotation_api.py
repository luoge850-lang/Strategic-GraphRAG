import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from strategic_graphrag.api.server import (
    EXTRACTION_SAMPLE_PATH,
    ExtractionAnnotationPatch,
    extraction_sample_path,
    extraction_sample_summary,
    patch_extraction_sample,
    read_extraction_sample,
    update_extraction_sample,
)


class ExtractionAnnotationApiTests(unittest.TestCase):
    def _sample(self, directory: str) -> Path:
        path = Path(directory) / "sample.jsonl"
        rows = [
            {
                "claim_id": "claim_a",
                "labels": {
                    "source_entity_correct": None,
                    "target_entity_correct": None,
                    "relation_correct": None,
                    "evidence_supports_relation": None,
                    "missing_gold_relations": None,
                },
                "annotation_status": "UNLABELED",
                "annotator": None,
                "notes": "",
            },
            {"claim_id": "claim_b", "annotation_status": "UNLABELED"},
        ]
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        return path

    def test_update_preserves_other_rows_and_writes_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._sample(directory)
            updated, rows = update_extraction_sample(
                "claim_a",
                {
                    "source_entity_correct": True,
                    "target_entity_correct": "uncertain",
                    "relation_correct": False,
                    "evidence_supports_relation": True,
                    "missing_gold_relations": [],
                    "annotation_status": "LABELED",
                    "annotator": "tester",
                    "notes": "checked",
                },
                path,
            )

            self.assertEqual(updated["labels"]["relation_correct"], False)
            self.assertEqual(updated["annotation_status"], "LABELED")
            self.assertEqual(rows[1]["claim_id"], "claim_b")
            loaded = read_extraction_sample(path)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["annotator"], "tester")
            self.assertEqual(loaded[1]["annotation_status"], "UNLABELED")

    def test_unknown_claim_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._sample(directory)
            with self.assertRaises(KeyError):
                update_extraction_sample("missing", {"relation_correct": True}, path)

    def test_invalid_label_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._sample(directory)
            with self.assertRaises(ValueError):
                update_extraction_sample("claim_a", {"relation_correct": "yes"}, path)

    def test_labeled_annotation_requires_annotator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._sample(directory)
            with self.assertRaises(ValueError):
                update_extraction_sample(
                    "claim_a",
                    {
                        "source_entity_correct": True,
                        "target_entity_correct": True,
                        "relation_correct": True,
                        "evidence_supports_relation": True,
                        "annotation_status": "LABELED",
                    },
                    path,
                )

    def test_named_sample_paths_are_explicit_and_separate(self):
        self.assertEqual(extraction_sample_path("baseline"), EXTRACTION_SAMPLE_PATH)
        self.assertNotEqual(extraction_sample_path("baseline"), extraction_sample_path("2025_post_repair_v2"))
        human_path = extraction_sample_path("2025_post_repair_human_v1")
        self.assertTrue(human_path.is_file())
        self.assertNotEqual(human_path, extraction_sample_path("2025_post_repair_v2"))
        summary = extraction_sample_summary(read_extraction_sample(human_path))
        self.assertEqual(summary["total"], 30)
        self.assertEqual(summary["labeled"] + summary["unlabeled"], 30)
        with self.assertRaises(ValueError):
            extraction_sample_path("arbitrary_path")

    def test_baseline_endpoint_is_read_only(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(patch_extraction_sample("claim_a", ExtractionAnnotationPatch(relation_correct=True)))
        self.assertEqual(context.exception.status_code, 403)

    def test_historical_v2_endpoint_is_read_only(self):
        with self.assertRaises(HTTPException) as context:
            asyncio.run(
                patch_extraction_sample(
                    "claim_a",
                    ExtractionAnnotationPatch(relation_correct=True),
                    sample="2025_post_repair_v2",
                )
            )
        self.assertEqual(context.exception.status_code, 403)


if __name__ == "__main__":
    unittest.main()
