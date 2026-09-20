"""Download and validate public finance QA benchmark samples.

This script only registers dataset structure and provenance. It deliberately
does not run the NVIDIA GraphRAG against questions whose source documents are
not in the active corpus. That external evaluation requires a separate,
versioned document index and must not be mixed with the NVIDIA 10-K results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests


ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "external" / "benchmarks"
SOURCES = {
    "financebench_questions": {
        "url": "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl",
        "path": DATA_ROOT / "financebench_open_source.jsonl",
        "format": "jsonl",
    },
    "financebench_documents": {
        "url": "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_document_information.jsonl",
        "path": DATA_ROOT / "financebench_document_information.jsonl",
        "format": "jsonl",
    },
    "finqa_test": {
        "url": "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/test.json",
        "path": DATA_ROOT / "finqa_test.json",
        "format": "json",
    },
    "tatqa_dev": {
        "url": "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_dev.json",
        "path": DATA_ROOT / "tatqa_dataset_dev.json",
        "format": "json",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(source: Dict[str, Any], *, refresh: bool) -> Dict[str, Any]:
    path = Path(source["path"])
    if refresh or not path.exists():
        response = requests.get(source["url"], timeout=60)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    return {
        "url": source["url"],
        "path": str(path.relative_to(ROOT)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "format": source["format"],
    }


def _load(path: Path, file_format: str) -> Any:
    if file_format == "jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(name: str, path: Path, file_format: str) -> Dict[str, Any]:
    value = _load(path, file_format)
    rows = value if isinstance(value, list) else []
    errors: List[str] = []
    if name == "financebench_questions":
        required = {"financebench_id", "question", "answer", "evidence", "doc_name"}
        errors = [str(sorted(required - set(row))) for row in rows if not required.issubset(row)]
    elif name == "financebench_documents":
        required = {"doc_name", "doc_link", "doc_period", "company"}
        errors = [str(sorted(required - set(row))) for row in rows if not required.issubset(row)]
    elif name == "finqa_test":
        required = {"id", "table", "qa"}
        errors = [str(sorted(required - set(row))) for row in rows if not required.issubset(row)]
    elif name == "tatqa_dev":
        required = {"table", "questions"}
        errors = [str(sorted(required - set(row))) for row in rows if not required.issubset(row)]
    return {
        "rows": len(rows),
        "status": "PASS" if rows and not errors else "FAIL",
        "schema_errors": len(errors),
        "error_examples": errors[:5],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports" / "external_benchmark_inventory_2026-09-19.json",
    )
    args = parser.parse_args()
    report = {
        "schema": "strategic-graphrag-external-benchmark-inventory/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_status": "REGISTERED_NOT_RUN_AGAINST_ACTIVE_NVIDIA_CORPUS",
        "reason": "The current index contains only NVIDIA 2023-2025 10-K files; external benchmark documents require a separate index.",
        "datasets": {},
    }
    for name, source in SOURCES.items():
        metadata = _download(source, refresh=args.refresh)
        metadata.update(_validate(name, Path(source["path"]), source["format"]))
        report["datasets"][name] = metadata
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] == "PASS" for item in report["datasets"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
