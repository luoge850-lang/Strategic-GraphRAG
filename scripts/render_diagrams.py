"""Render the repository's small Mermaid subset to deterministic SVG.

The project keeps Mermaid as the editable source. This renderer intentionally
supports only the flowchart constructs used by the checked-in diagrams so the
SVGs can be regenerated without a browser, network access, or a hidden design
tool. It is a presentation renderer, not a general Mermaid implementation.
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path


NODE_RE = re.compile(r"(?P<id>[A-Za-z][A-Za-z0-9_]*)\[(?P<label>[^\]]+)\]")
EDGE_RE = re.compile(
    r"(?P<source>[A-Za-z][A-Za-z0-9_]*)(?:\[[^\]]+\])?\s*"
    r"(?P<arrow>-->|-\.\s*[^.]*\s*\.->)\s*"
    r"(?:\|[^|]*\|\s*)?(?P<target>[A-Za-z][A-Za-z0-9_]*)(?:\[[^\]]+\])?"
)
CLASS_RE = re.compile(r"class\s+(?P<ids>[A-Za-z0-9_,]+)\s+(?P<class>implemented|boundary)")


def _parse(source: str):
    nodes = {}
    edges = []
    classes = {}
    for line in source.splitlines():
        for match in NODE_RE.finditer(line):
            nodes.setdefault(match.group("id"), match.group("label"))
        edge = EDGE_RE.search(line)
        if edge:
            nodes.setdefault(edge.group("source"), edge.group("source"))
            nodes.setdefault(edge.group("target"), edge.group("target"))
            edges.append((edge.group("source"), edge.group("target"), edge.group("arrow").startswith("-.")))
        class_match = CLASS_RE.search(line)
        if class_match:
            for node_id in class_match.group("ids").split(","):
                classes[node_id] = class_match.group("class")
    return nodes, edges, classes


def _label_lines(label: str) -> list[str]:
    return [html.escape(part.strip()) for part in label.replace("<br/>", "\n").splitlines()]


def render(source_path: Path, output_path: Path) -> None:
    nodes, edges, classes = _parse(source_path.read_text(encoding="utf-8"))
    if not nodes:
        raise ValueError(f"no supported Mermaid nodes found in {source_path}")
    node_ids = list(nodes)
    columns = max(1, min(4, len(node_ids)))
    width, height = 290, 86
    gap_x, gap_y = 46, 40
    margin_x, margin_y = 40, 44
    rows = (len(node_ids) + columns - 1) // columns
    svg_width = margin_x * 2 + columns * width + (columns - 1) * gap_x
    svg_height = margin_y * 2 + rows * height + (rows - 1) * gap_y
    positions = {
        node_id: (
            margin_x + (index % columns) * (width + gap_x),
            margin_y + (index // columns) * (height + gap_y),
        )
        for index, node_id in enumerate(node_ids)
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_width}" height="{svg_height}" '
        f'viewBox="0 0 {svg_width} {svg_height}" role="img" aria-labelledby="title desc">',
        f"<title>{html.escape(source_path.stem)}</title>",
        "<desc>Deterministic SVG rendering of the checked-in Mermaid source.</desc>",
        "<defs><marker id=\"arrow\" markerWidth=\"10\" markerHeight=\"10\" refX=\"9\" refY=\"3\" orient=\"auto\"><path d=\"M0,0 L0,6 L9,3 z\" fill=\"#526071\"/></marker></defs>",
        "<style>.node{stroke-width:2}.implemented{fill:#e8f1ff;stroke:#2b5dab}.boundary{fill:#fff5d6;stroke:#a56b00}.unknown{fill:#f4f5f7;stroke:#526071}.label{font-family:Arial,sans-serif;font-size:16px;fill:#10213b}.edge{stroke:#526071;stroke-width:2;fill:none;marker-end:url(#arrow)}.dashed{stroke-dasharray:8 6}</style>",
    ]
    for source, target, dashed in edges:
        if source not in positions or target not in positions:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        x1, y1 = sx + width, sy + height / 2
        x2, y2 = tx, ty + height / 2
        if x2 < x1:
            x1, x2 = sx, tx + width
        parts.append(f'<path class="edge{" dashed" if dashed else ""}" d="M{x1:.1f},{y1:.1f} C{(x1+x2)/2:.1f},{y1:.1f} {(x1+x2)/2:.1f},{y2:.1f} {x2:.1f},{y2:.1f}"/>')
    for node_id, label in nodes.items():
        x, y = positions[node_id]
        node_class = classes.get(node_id, "unknown")
        parts.append(f'<rect class="node {node_class}" x="{x}" y="{y}" width="{width}" height="{height}" rx="12"/>')
        lines = _label_lines(label)
        start_y = y + height / 2 - (len(lines) - 1) * 10
        text = [f'<text class="label" text-anchor="middle" x="{x + width/2}" y="{start_y:.1f}">']
        for index, line in enumerate(lines):
            text.append(f'<tspan x="{x + width/2}" dy="{0 if index == 0 else 20}">{line}</tspan>')
        text.append("</text>")
        parts.extend(text)
    parts.append("</svg>\n")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render checked-in Mermaid flowcharts to deterministic SVG")
    parser.add_argument("--directory", type=Path, default=Path("docs/diagrams"))
    args = parser.parse_args()
    for source_path in sorted(args.directory.glob("*.mmd")):
        render(source_path, source_path.with_suffix(".svg"))
        print(source_path.with_suffix(".svg"))


if __name__ == "__main__":
    main()
