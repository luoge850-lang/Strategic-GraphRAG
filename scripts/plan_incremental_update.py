"""Compare the active PDF files with a saved corpus manifest.

This planner is deliberately read-only. It identifies which filings would need
re-extraction and prevents an unchanged corpus from being rebuilt by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ACTIVE_PDFS = [
    ROOT / "data" / "pdfs_other" / "2023-10-K.pdf",
    ROOT / "data" / "pdfs_other" / "2024-10-K.pdf",
    ROOT / "data" / "pdfs" / "2025-10-K.pdf",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_plan(manifest: dict, current: dict[str, str] | None = None) -> dict:
    previous = {
        item["filename"]: item["sha256"]
        for item in manifest.get("active_filings", [])
    }
    if current is None:
        current = {
            path.name: sha256(path)
            for path in ACTIVE_PDFS
            if path.exists()
        }
    return {
        "schema": "strategic-graphrag-incremental-plan/v1",
        "unchanged": sorted(name for name, value in current.items() if previous.get(name) == value),
        "changed": sorted(name for name, value in current.items() if name in previous and previous[name] != value),
        "added": sorted(name for name in current if name not in previous),
        "removed": sorted(name for name in previous if name not in current),
        "requires_rebuild": sorted(name for name, value in current.items() if previous.get(name) != value),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    plan = build_plan(manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
