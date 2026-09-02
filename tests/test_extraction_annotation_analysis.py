import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_extraction_annotations import analyze
from scripts.evaluate_extraction_quality import write_or_preserve_annotation_sample


def row(claim_id, source=True, target=True, relation=True, evidence=True, uncertain=False):
    def value(correct):
        return "uncertain" if uncertain and correct else correct

    return {
        "claim_id": claim_id,
        "doc_id": "2025-10-K",
        "page": 1,
        "source_id": "nvidia_corporation",
        "relation_type": "PRODUCES",
        "target_id": "gpu",
        "evidence": f"evidence {claim_id}",
        "extraction_method": "RULE_EXTRACTION",
        "annotation_status": "LABELED",
        "labels": {
            "source_entity_correct": value(source),
            "target_entity_correct": value(target),
            "relation_correct": value(relation),
            "evidence_supports_relation": value(evidence),
            "missing_gold_relations": [],
        },
    }


class ExtractionAnnotationAnalysisTests(unittest.TestCase):
    def test_uncertain_is_counted_as_incorrect_and_recall_stays_unmeasured(self):
        report = analyze([row("a"), row("b", relation=False, evidence=False), row("c", uncertain=True)])
        metrics = report["precision_like_sample_estimates"]
        self.assertEqual(metrics["relation_correct_rate"], 0.3333)
        self.assertEqual(metrics["exact_triple_correct_rate"], 0.3333)
        self.assertEqual(report["annotation_protocol"]["recall_status"], "NOT_MEASURED")

    def test_duplicate_group_tracks_different_evidence_and_label_agreement(self):
        first = row("a")
        second = row("b", relation=False, evidence=False)
        second["evidence"] = first["evidence"] + " different"
        report = analyze([first, second])
        duplicate = report["duplicate_audit"]
        self.assertEqual(duplicate["duplicate_logical_triple_groups"], 1)
        self.assertEqual(duplicate["duplicate_groups_with_different_evidence"], 1)
        self.assertEqual(duplicate["mixed_label_groups"], 1)
        self.assertEqual(duplicate["duplicate_pair_label_agreement"], 0.0)

    def test_existing_sample_is_preserved_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.jsonl"
            existing = {"claim_id": "labeled", "annotation_status": "LABELED"}
            path.write_text(json.dumps(existing) + "\n", encoding="utf-8")
            generated = [{"claim_id": "new", "annotation_status": "UNLABELED"}]
            rows, mode = write_or_preserve_annotation_sample(path, generated)
            self.assertEqual(mode, "PRESERVED_EXISTING_ANNOTATION")
            self.assertEqual(rows, [existing])


if __name__ == "__main__":
    unittest.main()
