import unittest
from pathlib import Path

from scripts.render_diagrams import _parse


class DiagramContractTests(unittest.TestCase):
    def test_each_mermaid_source_has_nodes_and_edges(self):
        directory = Path("docs/diagrams")
        for source in directory.glob("*.mmd"):
            nodes, edges, _classes = _parse(source.read_text(encoding="utf-8"))
            self.assertGreaterEqual(len(nodes), 5, source.name)
            self.assertGreaterEqual(len(edges), 4, source.name)

    def test_generated_svg_is_present_for_each_source(self):
        for source in Path("docs/diagrams").glob("*.mmd"):
            svg = source.with_suffix(".svg")
            self.assertTrue(svg.exists(), svg)
            self.assertIn("<svg", svg.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
