"""Fail-fast runtime contract for the API and Hybrid retrieval."""

from __future__ import annotations

import importlib
import json
import platform
import sys


REQUIRED = (
    "fastapi", "uvicorn", "neo4j", "chromadb", "onnxruntime",
    "pymupdf", "pdfplumber", "dotenv", "pydantic",
)


def main() -> int:
    checks = {}
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            checks[name] = {
                "ready": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as exc:
            checks[name] = {"ready": False, "error": f"{type(exc).__name__}: {exc}"}
    report = {
        "python": sys.executable,
        "python_version": platform.python_version(),
        "ready": all(item["ready"] for item in checks.values()),
        "dependencies": checks,
    }
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
