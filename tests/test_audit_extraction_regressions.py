import json
import unittest

from strategic_graphrag.ontology.entity_registry import resolve_entity
from strategic_graphrag.pipeline.financial_table_extractor import extract_financial_table_triples


class Page:
    def extract_tables(self):
        return [[["Revenue", "$ 130,497", "$ 60,922", "$ 26,974"]]]


class AuditExtractionRegressions(unittest.TestCase):
    def test_three_year_income_statement_preserves_oldest_value(self):
        # Text transcribed from the 2025 filing, physical PDF page 52.
        text = "\n".join([
            "Consolidated Statements of Income",
            "(In millions, except per share data)",
            "Year Ended",
            "Jan 26, 2025 Jan 28, 2024 Jan 29, 2023",
            "Revenue $ 130,497 $ 60,922 $ 26,974",
        ])
        facts = extract_financial_table_triples(Page(), text, 2025)
        self.assertEqual(len(facts), 1)
        values = json.loads(facts[0]["metric_values_json"])
        self.assertEqual([{k: v[k] for k in ("period", "value")} for v in values], [
            {"period": "2025", "value": "130497"},
            {"period": "2024", "value": "60922"},
            {"period": "2023", "value": "26974"},
        ])
        self.assertEqual([v["column_label"] for v in values], ["Jan 26, 2025", "Jan 28, 2024", "Jan 29, 2023"])
        self.assertEqual(values[0]["period_end"], "2025-01-26")
        self.assertEqual(values[1]["period_end"], "2024-01-28")

    def test_generic_pandemic_does_not_assert_specific_event(self):
        for word in ("pandemic", "pandemics"):
            self.assertEqual(resolve_entity(word), ("PANDEMIC", "Event"))
        for word in ("COVID-19", "covid"):
            self.assertEqual(resolve_entity(word), ("COVID_19", "Event"))
