"""Shared build identity for PDF, graph, vector, numeric, and cache artifacts.

The identity is intentionally content-derived and secret-free.  It is not a
deployment lock by itself; it is the join key that lets a query or report
prove which corpus, source code, prompt, model, embedding configuration, and
dependency lock produced an artifact.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "strategic-graphrag-build-identity/v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(root: Path = ROOT) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
            text=True, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _fingerprint_files(paths: Iterable[Path], *, root: Path = ROOT) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(item) for item in paths), key=lambda item: str(item)):
        if not path.is_file():
            continue
        try:
            name = path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            name = path.name
        digest.update(name.encode("utf-8"))
        digest.update(sha256_file(path).encode("ascii"))
    return digest.hexdigest()


def source_fingerprint(root: Path = ROOT) -> str:
    paths = []
    for folder in ("strategic_graphrag", "scripts", "tests", "frontend/src"):
        paths.extend((root / folder).rglob("*.py"))
        paths.extend((root / folder).rglob("*.ts"))
        paths.extend((root / folder).rglob("*.tsx"))
        paths.extend((root / folder).rglob("*.css"))
    paths.extend(root.glob("requirements*.txt"))
    # Documentation is deliberately excluded: changing README wording must
    # not invalidate a graph/vector/numeric build that used the same runtime
    # code, corpus, prompts, and dependency lock.
    paths.extend([root / ".env.example"])
    return _fingerprint_files(paths, root=root)


def _safe_env(name: str, default: str = "") -> str:
    # Never include API keys, passwords, URIs, or other secret-bearing values.
    return str(os.getenv(name, default) or "")


@dataclass(frozen=True)
class BuildIdentity:
    build_id: str
    schema: str
    corpus_id: str
    source_tree_sha256: str
    pdfs: Dict[str, Dict[str, Any]]
    parser_version: str
    parser_config_hash: str
    prompt_version: str
    extraction_provider: str
    extraction_model: str
    query_model: str
    report_model: str
    embedding_backend: str
    embedding_model: str
    dependency_lock_sha256: Optional[str]
    git_commit: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_build_identity(
    pdf_paths: Iterable[str | Path],
    *,
    corpus_id: str = "nvidia-10k-2023-2025-v3",
    parser_version: str = "pdfplumber-document-layer/v1",
    parser_config_hash: str = "",
    prompt_version: Optional[str] = None,
    extraction_provider: Optional[str] = None,
    extraction_model: Optional[str] = None,
    query_model: Optional[str] = None,
    report_model: Optional[str] = None,
    embedding_backend: Optional[str] = None,
    embedding_model: Optional[str] = None,
    dependency_lock: Optional[str | Path] = None,
    root: Path = ROOT,
) -> BuildIdentity:
    pdfs: Dict[str, Dict[str, Any]] = {}
    for raw_path in pdf_paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        pdfs[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    lock_path = Path(dependency_lock) if dependency_lock else root / "requirements-lock-2026-09-19.txt"
    lock_hash = sha256_file(lock_path) if lock_path.exists() else None
    identity_payload = {
        "schema": SCHEMA,
        "corpus_id": corpus_id,
        "source_tree_sha256": source_fingerprint(root),
        "pdfs": pdfs,
        "parser_version": parser_version,
        "parser_config_hash": parser_config_hash,
        "prompt_version": prompt_version or _safe_env("GRAPHRAG_PROMPT_VERSION", "v2-evidence-claim-1"),
        "extraction_provider": extraction_provider or _safe_env("LLM_PROVIDER", "unknown"),
        "extraction_model": extraction_model or _safe_env("LLM_EXTRACTION_MODEL", _safe_env("LLM_MODEL", "unknown")),
        "query_model": query_model or _safe_env("LLM_QUERY_MODEL", _safe_env("LLM_MODEL", "unknown")),
        "report_model": report_model or _safe_env("LLM_REPORT_MODEL", _safe_env("LLM_MODEL", "unknown")),
        "embedding_backend": embedding_backend or _safe_env("GRAPH_EMBEDDING_BACKEND", "chroma_onnx"),
        "embedding_model": embedding_model or _safe_env("GRAPH_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "dependency_lock_sha256": lock_hash,
    }
    encoded = json.dumps(identity_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    build_id = f"build_{hashlib.sha256(encoded).hexdigest()[:16]}"
    return BuildIdentity(build_id=build_id, **identity_payload, git_commit=git_sha(root))


def build_id_from_env(default: Optional[str] = None) -> Optional[str]:
    value = str(os.getenv("GRAPHRAG_BUILD_ID", "") or "").strip()
    return value or default


def require_same_build_id(artifacts: Iterable[Mapping[str, Any]]) -> str:
    """Fail closed when graph/vector/numeric artifacts were mixed."""
    values = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            values.add("UNKNOWN")
            continue
        value = str(artifact.get("build_id") or "").strip()
        values.add(value or "UNKNOWN")
    if len(values) != 1 or "UNKNOWN" in values:
        raise ValueError(f"artifacts do not share exactly one build_id: {sorted(values)}")
    return next(iter(values))


__all__ = [
    "BuildIdentity", "SCHEMA", "make_build_identity", "build_id_from_env",
    "require_same_build_id", "source_fingerprint", "sha256_file",
]
