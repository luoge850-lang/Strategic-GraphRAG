import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.audit_graph_semantic_consistency import (
    _normalize_text,
    _resolve_pdf_path,
    audit_records,
)


def _record(
    claim_id: str,
    *,
    page: int = 1,
    quote: str = "Supply constraints increase operating costs.",
    relation: str = "INCREASES",
    source_category: str = "RiskFactor",
    target_category: str = "FinancialMetric",
) -> dict:
    claim = {
        "id": claim_id,
        "doc_id": "2025-10-K",
        "text": quote,
        "relation_id": f"rel-{claim_id}",
        "relation_type": relation,
        "source_id": "supply_chain",
        "target_id": "operating_cost",
        "page": page,
        "fiscal_year": 2025,
        "verification_status": "VERBATIM",
        "evidence_char_start": 0,
        "evidence_char_end": len(quote),
        "chunk_id": f"2025-10-K.pdf:{page}:0",
    }
    source = {"id": "supply_chain", "name": "Supply Chain"}
    target = {"id": "operating_cost", "name": "Operating Cost"}
    edge = {
        "type": relation,
        "properties": {
            "id": f"rel-{claim_id}",
            "evidence_id": claim_id,
            "source_filing": "2025-10-K.pdf",
            "source_page": page,
            "year": 2025,
            "evidence_sentence": quote,
            "source_category": source_category,
            "target_category": target_category,
        },
        "source": source,
        "source_labels": [source_category],
        "target": target,
        "target_labels": [target_category],
    }
    return {
        "claim": claim,
        "evidence": {"id": f"sentence-{claim_id}", "text": quote, "page": page, "doc_id": "2025-10-K"},
        "source": source,
        "source_labels": [source_category],
        "target": target,
        "target_labels": [target_category],
        "linkage": {
            "supported_by": True,
            "about_source": True,
            "about_target": True,
            "native_relation_edge": True,
        },
        "edges": [edge],
    }


class GraphSemanticConsistencyAuditTests(unittest.TestCase):
    def test_pdf_resolver_searches_both_allowlisted_directories(self):
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            pdfs = root / "data" / "pdfs"
            pdfs_other = root / "data" / "pdfs_other"
            pdfs.mkdir(parents=True)
            pdfs_other.mkdir()
            current_pdf = pdfs / "2025-10-K.pdf"
            other_pdf = pdfs_other / "2024-10-K.pdf"
            current_pdf.write_bytes(b"placeholder")
            other_pdf.write_bytes(b"placeholder")

            resolved, candidates = _resolve_pdf_path(
                "2024-10-K.pdf", [pdfs, pdfs_other]
            )
            self.assertEqual(resolved, other_pdf)
            self.assertEqual(candidates, [pdfs / "2024-10-K.pdf", other_pdf])

            missing, missing_candidates = _resolve_pdf_path(
                "missing.pdf", [pdfs, pdfs_other]
            )
            self.assertIsNone(missing)
            self.assertEqual(
                missing_candidates,
                [pdfs / "missing.pdf", pdfs_other / "missing.pdf"],
            )

    def test_repeated_triples_are_classified_without_becoming_failures(self):
        records = [
            _record("c1"),
            _record("c2"),
            _record("c3", page=2, quote="The same supply constraints raise operating costs."),
        ]
        page_texts = {
            ("2025-10-K", 1): _normalize_text(records[0]["claim"]["text"]),
            ("2025-10-K", 2): _normalize_text(records[2]["claim"]["text"]),
        }

        report = audit_records(records, page_texts)

        self.assertEqual(report["status"], "WARN")
        duplicate = report["sections"]["normalized_triples"]
        self.assertEqual(duplicate["duplicate_normalized_triple_groups"], 1)
        self.assertEqual(duplicate["same_evidence_duplicate_groups"], 1)
        self.assertEqual(duplicate["different_evidence_duplicate_groups"], 1)
        self.assertEqual(duplicate["samples"][0]["different_evidence_duplicate"], True)
        self.assertIn("cannot prove semantic consistency", duplicate["note"])
        self.assertEqual(report["counts"]["quote_mismatches"], 0)

    def test_missing_evidence_and_invalid_relation_are_failures(self):
        record = _record(
            "bad",
            relation="MITIGATES",
            source_category="RiskFactor",
            target_category="FinancialMetric",
        )
        record["evidence"] = {}
        record["linkage"]["about_target"] = False
        record["edges"][0]["properties"]["evidence_sentence"] = "not present in the page"

        report = audit_records(record and [record], {("2025-10-K", 1): "different text"})

        self.assertEqual(report["status"], "FAIL")
        self.assertGreater(report["sections"]["field_completeness"]["missing_field_instances"], 0)
        self.assertEqual(report["sections"]["relation_direction_and_type"]["ontology_direction_type_conflicts"], 1)
        self.assertGreater(report["sections"]["quote_traceability"]["quote_mismatches"], 0)


if __name__ == "__main__":
    unittest.main()
