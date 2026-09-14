import inspect
import unittest

from strategic_graphrag.pipeline.extractor import TripleExtractor
from strategic_graphrag.pipeline.pipeline import (
    KnowledgeGraphPipeline,
    _get_filter_rejection_counts,
)


class _LegacyExtractor:
    pass


class ExtractionRejectionDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        # Filtering is deterministic and does not require an LLM provider.
        self.extractor = TripleExtractor.__new__(TripleExtractor)

    @staticmethod
    def _triple(
        source="SUPPLY_CHAIN_DISRUPTION",
        source_category="RiskFactor",
        target="REVENUE",
        target_category="FinancialMetric",
        relation="DECREASES",
        evidence="Supply chain disruption may reduce revenue.",
    ):
        return {
            "source": source,
            "source_category": source_category,
            "target": target,
            "target_category": target_category,
            "relation": relation,
            "evidence_sentence": evidence,
        }

    def test_filter_rejection_counts_reset_on_each_call(self):
        rejected = self._triple(source="", evidence="Missing source entity.")
        self.assertEqual(self.extractor.filter_triples([rejected], ""), [])
        self.assertEqual(
            self.extractor.get_filter_rejection_counts(),
            {"EMPTY_FIELD": 1},
        )

        valid = self._triple()
        self.assertEqual(
            len(self.extractor.filter_triples([valid], valid["evidence_sentence"])),
            1,
        )
        self.assertEqual(self.extractor.get_filter_rejection_counts(), {})

    def test_non_verbatim_evidence_is_rejected_and_counted(self):
        text = "Supply chain disruption may reduce revenue."
        candidate = self._triple(
            evidence="Supply chain disruption could reduce revenue.",
        )
        self.assertEqual(self.extractor.filter_triples([candidate], text), [])
        self.assertEqual(
            self.extractor.get_filter_rejection_counts(),
            {"NON_VERBATIM_EVIDENCE": 1},
        )

    def test_invalid_relation_and_category_pair_are_counted(self):
        invalid_relation = self._triple(
            source="NVIDIA_CORPORATION",
            source_category="Company",
            relation="NOT_A_RELATION",
            evidence="NVIDIA Corporation causes revenue.",
        )
        invalid_pair = self._triple(
            source="NVIDIA_CORPORATION",
            source_category="Company",
            relation="CAUSES",
            evidence="NVIDIA Corporation causes revenue.",
        )
        text = "NVIDIA Corporation causes revenue."
        self.assertEqual(
            self.extractor.filter_triples([invalid_relation, invalid_pair], text),
            [],
        )
        self.assertEqual(
            self.extractor.get_filter_rejection_counts(),
            {
                "INVALID_ENTITY_CATEGORY_RELATION": 1,
                "INVALID_RELATION": 1,
            },
        )

    def test_page_18_style_candidate_can_still_pass(self):
        candidate = self._triple()
        accepted = self.extractor.filter_triples([candidate], candidate["evidence_sentence"])
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]["source"], "SUPPLY_CHAIN_DISRUPTION")
        self.assertEqual(accepted[0]["target"], "REVENUE")
        self.assertEqual(self.extractor.get_filter_rejection_counts(), {})

    def test_pipeline_filter_snapshot_is_additive_and_backward_compatible(self):
        class CurrentExtractor:
            @staticmethod
            def get_filter_rejection_counts():
                return {"EVIDENCE_DIRECTION_UNSUPPORTED": 2}

        existing = {"page": 18, "strict_triples": 0, "evidence_spans": 0}
        page_stats = dict(existing)
        page_stats["filter_rejection_counts"] = _get_filter_rejection_counts(
            CurrentExtractor()
        )
        self.assertEqual(existing, {"page": 18, "strict_triples": 0, "evidence_spans": 0})
        self.assertEqual(
            page_stats["filter_rejection_counts"],
            {"EVIDENCE_DIRECTION_UNSUPPORTED": 2},
        )
        self.assertEqual(_get_filter_rejection_counts(_LegacyExtractor()), {})
        self.assertIn('"filter_rejection_counts"', inspect.getsource(KnowledgeGraphPipeline.process_pdf))


if __name__ == "__main__":
    unittest.main()
