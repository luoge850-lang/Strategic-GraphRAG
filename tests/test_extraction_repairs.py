import json
import unittest

from strategic_graphrag.ontology.entity_registry import resolve_entity
from strategic_graphrag.pipeline.financial_table_extractor import (
    _table_context,
    extract_financial_table_triples,
)


class _FakePage:
    def __init__(self, tables):
        self._tables = tables

    def extract_tables(self):
        return [self._tables]


class ExtractionRepairTests(unittest.TestCase):
    def test_generic_product_alias_does_not_become_specific_model(self):
        self.assertEqual(resolve_entity("GeForce"), ("GEFORCE_GPU", "Product"))
        self.assertEqual(resolve_entity("GeForce RTX 4090"), ("GEFORCE_RTX_4090", "Product"))
        self.assertEqual(resolve_entity("graphics card"), ("GPU", "Product"))
        self.assertEqual(resolve_entity("H800"), ("H800_GPU", "Product"))
        self.assertEqual(resolve_entity("H800 China Export GPU"), ("H800_CHINA_EXPORT_GPU", "Product"))

    def test_drive_subproducts_have_distinct_canonical_ids(self):
        self.assertEqual(resolve_entity("DRIVE Sim"), ("DRIVE_SIM", "Product"))
        self.assertEqual(resolve_entity("DRIVE Orin"), ("DRIVE_ORIN", "Product"))
        self.assertEqual(resolve_entity("NVIDIA DRIVE"), ("DRIVE_PLATFORM", "Product"))

    def test_table_evidence_keeps_header_context_and_row_values(self):
        row = "Net income $ 10,000 $ 8,000"
        page_text = "\n".join([
            "Consolidated Statements of Income",
            "Year Ended",
            "2025 2024",
            "($ in millions)",
            row,
        ])
        context = _table_context(page_text, row)
        self.assertIn("Consolidated Statements of Income", context)
        self.assertIn("Year Ended", context)
        self.assertIn(row, context)

        triples = extract_financial_table_triples(
            _FakePage([["Net income", "$ 10,000", "$ 8,000"]]),
            page_text,
            2025,
        )
        self.assertEqual(len(triples), 1)
        triple = triples[0]
        self.assertEqual(triple["evidence_sentence"], row)
        self.assertIn("Consolidated Statements of Income", triple["table_context"])
        self.assertEqual(triple["row_evidence"], row)
        self.assertEqual(
            json.loads(triple["metric_values_json"]),
            [
                {"period": "2025", "value": "10000"},
                {"period": "2024", "value": "8000"},
            ],
        )
        self.assertEqual(
            triple["table_name"],
            "Consolidated Statements of Income",
        )

    def test_table_name_prefers_summary_heading_over_nearby_prose(self):
        page_text = "\n".join([
            "Termination of the Arm Share Purchase Agreement",
            "A paragraph about the transaction.",
            "Fiscal Year 2023 Summary",
            "Year Ended",
            "2023 2022 Change",
            "Operating expenses $ 11,132 $ 7,434 Up 50%",
        ])
        triples = extract_financial_table_triples(
            _FakePage([["Operating expenses", "$ 11,132", "$ 7,434"]]),
            page_text,
            2023,
        )
        self.assertEqual(triples[0]["table_name"], "Fiscal Year 2023 Summary")

    def test_table_name_does_not_report_scale_or_continued_marker(self):
        page_text = "\n".join([
            "(Continued)",
            "Year Ended",
            "2025 2024",
            "(In millions)",
            "Net income $ 10,000 $ 8,000",
        ])
        triples = extract_financial_table_triples(
            _FakePage([["Net income", "$ 10,000", "$ 8,000"]]),
            page_text,
            2025,
        )
        self.assertEqual(triples[0]["table_name"], "UNKNOWN_TABLE")


if __name__ == "__main__":
    unittest.main()
