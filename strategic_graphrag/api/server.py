# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: FastAPI Backend Server
==========================================
REST API for the Strategic-GraphRAG system.

Endpoints:
  POST /query          — GraphRAG financial analysis
  POST /query/vector   — Vector RAG baseline (for comparison)
  GET  /graph/statistics — Knowledge graph statistics
  GET  /graph/subgraph  — Subgraph for visualization
  GET  /evidence/{id}   — Evidence trace for a specific path
"""

import asyncio
import os
import sys
import json
import logging
import hmac
import threading
import time
import tempfile
import uuid
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from typing import Any, Optional, List, Dict, Literal, Union

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr
from dotenv import load_dotenv
from pathlib import Path
from ..response_contract import (
    ANSWER_STATUSES,
    EXECUTION_STATUSES,
    GROUNDING_STATUSES,
    OUTCOMES,
    apply_response_contract,
    response_state,
)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

# ── App Setup ──
app = FastAPI(
    title="Strategic-GraphRAG API",
    description="Temporal Causal Knowledge Graph Framework for Financial Risk Inference",
    version="3.1.0",
)

_cors_origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        "http://127.0.0.1:8000,http://localhost:8000,http://127.0.0.1:5173,http://localhost:5173,http://127.0.0.1:4173,http://localhost:4173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

_API_AUTH_ENABLED = os.getenv("API_AUTH_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
_API_KEY = os.getenv("API_KEY", "").strip()
_RATE_LIMIT_PER_MINUTE = max(
    int(os.getenv("RATE_LIMIT_PER_MINUTE", "60") or 60),
    1,
)
_RATE_BUCKETS = defaultdict(deque)
_AUTH_EXEMPT_PATHS = {"/", "/health", "/health/live", "/health/ready", "/docs", "/openapi.json", "/redoc"}
_QUERY_CACHE_TTL_SECONDS = max(int(os.getenv("QUERY_CACHE_TTL_SECONDS", "300") or 300), 0)
_QUERY_CACHE_MAX_ENTRIES = max(int(os.getenv("QUERY_CACHE_MAX_ENTRIES", "128") or 128), 1)
_QUERY_CACHE = OrderedDict()
_GRAPH_CACHE_TTL_SECONDS = max(int(os.getenv("GRAPH_CACHE_TTL_SECONDS", "60") or 60), 0)
_GRAPH_CACHE_MAX_ENTRIES = max(int(os.getenv("GRAPH_CACHE_MAX_ENTRIES", "16") or 16), 1)
_GRAPH_CACHE = OrderedDict()
_STARTED_AT = time.time()
_READINESS = {"status": "unknown", "checked_at": None, "dependencies": {}}


@app.middleware("http")
async def request_guard(request: Request, call_next):
    """Add request tracing, basic abuse protection, and optional API auth."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()

    if _API_AUTH_ENABLED and request.url.path not in _AUTH_EXEMPT_PATHS:
        provided_key = request.headers.get("X-API-Key", "")
        if not _API_KEY or not hmac.compare_digest(provided_key, _API_KEY):
            return JSONResponse(
                status_code=401,
                content={
                    "execution_status": "AUTH_ERROR",
                    "answer_status": "NOT_REQUESTED",
                    "grounding_status": "NOT_EXECUTED",
                    "outcome": "AUTH_ERROR",
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "A valid X-API-Key is required.",
                        "request_id": request_id,
                    }
                },
                headers={"X-Request-ID": request_id},
            )

    client_key = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _RATE_BUCKETS[client_key]
    while bucket and now - bucket[0] >= 60:
        bucket.popleft()
    if request.url.path not in {"/", "/health", "/health/live", "/health/ready"} and len(bucket) >= _RATE_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={
                "execution_status": "RATE_LIMITED",
                "answer_status": "NOT_REQUESTED",
                "grounding_status": "NOT_EXECUTED",
                "outcome": "RATE_LIMITED",
                "error": {
                    "code": "RATE_LIMITED",
                    "message": "Too many requests. Retry later.",
                    "request_id": request_id,
                }
            },
            headers={"X-Request-ID": request_id, "Retry-After": "60"},
        )
    bucket.append(now)

    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error request_id=%s", request_id)
        response = JSONResponse(
            status_code=500,
            content={
                "execution_status": "INTERNAL_ERROR",
                "answer_status": "NOT_REQUESTED",
                "grounding_status": "NOT_EXECUTED",
                "outcome": "INTERNAL_ERROR",
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "The request could not be completed.",
                    "request_id": request_id,
                }
            },
        )

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = str(
        round((time.perf_counter() - started) * 1000, 2)
    )
    return response


@app.exception_handler(HTTPException)
async def structured_http_error(request: Request, exc: HTTPException):
    """Avoid leaking provider/database internals through API errors."""
    request_id = getattr(request.state, "request_id", uuid.uuid4().hex)
    if isinstance(exc.detail, dict):
        error = dict(exc.detail)
        error.setdefault("code", "HTTP_ERROR")
        error.setdefault("message", "The request could not be completed.")
    elif exc.status_code >= 500:
        error = {
            "code": "INTERNAL_SERVER_ERROR",
            "message": "The request could not be completed.",
        }
    else:
        error = {"code": "HTTP_ERROR", "message": str(exc.detail)}
    error["request_id"] = request_id
    state = response_state(
        {"outcome": error.get("outcome"), "error": error},
        http_status=exc.status_code,
    )
    error.update({
        "execution_status": state["execution_status"],
        "answer_status": state["answer_status"],
        "grounding_status": state["grounding_status"],
    })
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "execution_status": state["execution_status"],
            "answer_status": state["answer_status"],
            "grounding_status": state["grounding_status"],
            "outcome": state["outcome"],
            "status_provenance": state["status_provenance"],
            "error": error,
        },
        headers={"X-Request-ID": request_id},
    )


@app.exception_handler(RequestValidationError)
async def structured_validation_error(request: Request, exc: RequestValidationError):
    """Keep FastAPI request-shape failures in the same machine contract."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=422,
        content={
            "execution_status": "VALIDATION_ERROR",
            "answer_status": "NOT_REQUESTED",
            "grounding_status": "NOT_EXECUTED",
            "outcome": "VALIDATION_ERROR",
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "The request did not satisfy the API schema.",
                "request_id": request_id,
                "details": exc.errors(),
            },
        },
        headers={"X-Request-ID": request_id},
    )

# Serve the built Vite application when available.  The public/index.html
# fallback keeps the API usable before a frontend build has been produced.
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
EXTRACTION_SAMPLE_PATH = PROJECT_ROOT / "evaluation" / "annotation" / "extraction_sample_v1.jsonl"
EXTRACTION_SAMPLE_PATHS = {
    "baseline": EXTRACTION_SAMPLE_PATH,
    "2025_post_repair_v2": PROJECT_ROOT / "evaluation" / "annotation" / "extraction_sample_2025_post_repair_v2.jsonl",
    "2025_post_repair_human_v1": PROJECT_ROOT / "evaluation" / "annotation" / "extraction_sample_2025_post_repair_human_v1.jsonl",
}
HUMAN_EXTRACTION_SAMPLE_KEY = "2025_post_repair_human_v1"
_EXTRACTION_SAMPLE_LOCK = threading.Lock()
TABLE_QUALITY_PATH = PROJECT_ROOT / "evaluation" / "annotation" / "table_quality_candidate_2026-09-19.jsonl"
_TABLE_QUALITY_LOCK = threading.Lock()
_TABLE_QUALITY_GOLD_FIELDS = (
    "company_id",
    "fiscal_year",
    "metric_id",
    "value",
    "unit",
    "source_filing",
    "page",
    "row_label",
    "column_label",
    "table_name",
    "evidence_text",
    "cell_supported",
)
_TABLE_QUALITY_STATUS = {"UNLABELED_CANDIDATE", "IN_PROGRESS", "HUMAN_REVIEWED"}
_TABLE_QUALITY_SOURCE_FILES = {
    "2023-10-K.pdf": PROJECT_ROOT / "data" / "pdfs_other" / "2023-10-K.pdf",
    "2024-10-K.pdf": PROJECT_ROOT / "data" / "pdfs_other" / "2024-10-K.pdf",
    "2025-10-K.pdf": PROJECT_ROOT / "data" / "pdfs" / "2025-10-K.pdf",
}
# Prefer the claim-ID-v2 worklist generated from the current graph. Keep the
# older v1 path as a compatibility fallback, but never merge the two datasets.
_GOLDEN_QA_V2_PATH = PROJECT_ROOT / "evaluation" / "golden_qa_human_v2.jsonl"
_GOLDEN_QA_V1_PATH = PROJECT_ROOT / "evaluation" / "golden_qa_human_v1.jsonl"
GOLDEN_QA_PATH = _GOLDEN_QA_V2_PATH if _GOLDEN_QA_V2_PATH.exists() else _GOLDEN_QA_V1_PATH
_GOLDEN_QA_LOCK = threading.Lock()
_GOLDEN_QA_STATUSES = {"HUMAN_REVIEW_PENDING", "IN_PROGRESS", "HUMAN_REVIEWED"}


_ANNOTATION_LABEL_FIELDS = (
    "source_entity_correct",
    "target_entity_correct",
    "relation_correct",
    "evidence_supports_relation",
)
_ANNOTATION_PATCH_FIELDS = set(_ANNOTATION_LABEL_FIELDS) | {
    "missing_gold_relations",
    "annotation_status",
    "annotator",
    "notes",
}


def read_extraction_sample(path: Optional[Path] = None) -> List[Dict]:
    """Read the JSONL annotation sample without mutating the source file."""
    sample_path = path or EXTRACTION_SAMPLE_PATH
    with sample_path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on annotation sample line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Annotation sample line {line_number} is not an object")
            rows.append(row)
    return rows


def extraction_sample_path(sample: str = "baseline") -> Path:
    """Resolve a named annotation set without allowing arbitrary file paths."""
    try:
        return EXTRACTION_SAMPLE_PATHS[sample]
    except KeyError as exc:
        allowed = ", ".join(sorted(EXTRACTION_SAMPLE_PATHS))
        raise ValueError(f"Unknown extraction sample '{sample}'. Choose one of: {allowed}") from exc


def extraction_sample_summary(rows: List[Dict]) -> Dict[str, int]:
    labeled = sum(1 for row in rows if row.get("annotation_status") == "LABELED")
    return {"total": len(rows), "labeled": labeled, "unlabeled": len(rows) - labeled}


def read_table_quality(path: Optional[Path] = None) -> List[Dict]:
    """Read the table-quality queue without changing candidate or gold values."""
    queue_path = path or TABLE_QUALITY_PATH
    with queue_path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on table-quality line {line_number}") from exc
            if not isinstance(row, dict) or not row.get("queue_id"):
                raise ValueError(f"Invalid table-quality row on line {line_number}")
            rows.append(row)
    return rows


def table_quality_summary(rows: List[Dict]) -> Dict[str, int]:
    reviewed = sum(row.get("review_status") == "HUMAN_REVIEWED" for row in rows)
    in_progress = sum(row.get("review_status") == "IN_PROGRESS" for row in rows)
    return {
        "total": len(rows),
        "reviewed": reviewed,
        "in_progress": in_progress,
        "pending": len(rows) - reviewed,
    }


def _normalize_table_gold(gold: Dict[str, Any]) -> Dict[str, Any]:
    unknown = set(gold) - set(_TABLE_QUALITY_GOLD_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported table gold fields: {sorted(unknown)}")
    normalized = dict(gold)
    for field in ("fiscal_year", "page"):
        value = normalized.get(field)
        if value in (None, ""):
            continue
        try:
            normalized[field] = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be an integer") from exc
        if field == "page" and normalized[field] < 1:
            raise ValueError("page must be a positive integer")
    source_filing = normalized.get("source_filing")
    if source_filing not in (None, "") and source_filing not in _TABLE_QUALITY_SOURCE_FILES:
        raise ValueError("source_filing must reference an allowlisted active filing")
    if "value" in normalized and normalized["value"] not in (None, ""):
        value = normalized["value"]
        if isinstance(value, bool):
            raise ValueError("value must be numeric or a raw numeric string")
        try:
            normalized["value"] = float(value)
        except (TypeError, ValueError):
            if not isinstance(value, str):
                raise ValueError("value must be numeric or a raw numeric string")
    if "cell_supported" in normalized:
        value = normalized["cell_supported"]
        if value not in (True, False, "uncertain", None, ""):
            raise ValueError("cell_supported must be true, false, uncertain, or null")
    return normalized


def update_table_quality(queue_id: str, updates: Dict[str, Any], path: Optional[Path] = None) -> tuple[Dict, List[Dict]]:
    """Atomically update one table annotation while preserving system predictions."""
    allowed = {"gold", "reviewer", "review_notes", "review_status"}
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError(f"Unsupported table annotation fields: {sorted(unknown)}")
    if "gold" in updates:
        if not isinstance(updates["gold"], dict):
            raise ValueError("gold must be an object")
        updates = {**updates, "gold": _normalize_table_gold(updates["gold"])}
    status = updates.get("review_status")
    if status is not None and status not in _TABLE_QUALITY_STATUS:
        raise ValueError(f"review_status must be one of: {', '.join(sorted(_TABLE_QUALITY_STATUS))}")

    queue_path = path or TABLE_QUALITY_PATH
    with _TABLE_QUALITY_LOCK:
        rows = read_table_quality(queue_path)
        updated_row = None
        for row in rows:
            if row.get("queue_id") != queue_id:
                continue
            if "gold" in updates:
                gold = dict(row.get("gold") or {})
                gold.update(updates["gold"])
                row["gold"] = _normalize_table_gold(gold)
            for field in ("reviewer", "review_notes", "review_status"):
                if field in updates:
                    row[field] = updates[field]
            if row.get("review_status") == "HUMAN_REVIEWED":
                gold = row.get("gold") or {}
                if not str(row.get("reviewer") or "").strip():
                    raise ValueError("reviewer is required before marking HUMAN_REVIEWED")
                if gold.get("cell_supported") not in (True, False):
                    raise ValueError("cell_supported must be true or false before marking HUMAN_REVIEWED")
                if gold.get("cell_supported") is True:
                    missing = [field for field in _TABLE_QUALITY_GOLD_FIELDS if field != "cell_supported" and gold.get(field) in (None, "")]
                    if missing:
                        raise ValueError(f"supported cells require gold fields: {missing}")
            updated_row = row
            break
        if updated_row is None:
            raise KeyError(queue_id)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="", dir=str(queue_path.parent),
                prefix=f".{queue_path.name}.", suffix=".tmp", delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, queue_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    return updated_row, rows


def update_extraction_sample(
    claim_id: str,
    updates: Dict,
    path: Optional[Path] = None,
) -> tuple[Dict, List[Dict]]:
    """Patch one row and atomically replace the JSONL file.

    Unchanged source lines are copied byte-for-byte; only the matched JSON object
    is serialized again. The temporary file lives beside the source for an
    atomic os.replace on the same filesystem.
    """
    unknown = set(updates) - _ANNOTATION_PATCH_FIELDS
    if unknown:
        raise ValueError(f"Unsupported annotation fields: {sorted(unknown)}")
    for field in _ANNOTATION_LABEL_FIELDS:
        if field in updates:
            value = updates[field]
            if value is not None and not isinstance(value, bool) and value != "uncertain":
                raise ValueError(f"{field} must be true, false, uncertain, or null")
    if "missing_gold_relations" in updates:
        values = updates["missing_gold_relations"]
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("missing_gold_relations must be a string array")
    if updates.get("annotation_status") == "LABELED" and not str(updates.get("annotator") or "").strip():
        raise ValueError("annotator is required before an annotation can be marked LABELED")

    sample_path = path or EXTRACTION_SAMPLE_PATH
    with _EXTRACTION_SAMPLE_LOCK:
        with sample_path.open("r", encoding="utf-8", newline="") as handle:
            raw_lines = handle.readlines()

        updated_row = None
        updated_lines = []
        rows = []
        for line in raw_lines:
            if not line.strip():
                updated_lines.append(line)
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Annotation sample contains a non-object row")
            if row.get("claim_id") == claim_id:
                labels = dict(row.get("labels") or {})
                for field in _ANNOTATION_LABEL_FIELDS:
                    if field in updates:
                        labels[field] = updates[field]
                if "missing_gold_relations" in updates:
                    labels["missing_gold_relations"] = updates["missing_gold_relations"]
                row["labels"] = labels
                for field in ("annotation_status", "annotator", "notes"):
                    if field in updates:
                        row[field] = updates[field]
                updated_row = row
                line_ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                updated_lines.append(json.dumps(row, ensure_ascii=False) + line_ending)
            else:
                updated_lines.append(line)
            rows.append(row)

        if updated_row is None:
            raise KeyError(claim_id)

        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=str(sample_path.parent),
                prefix=f".{sample_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.writelines(updated_lines)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, sample_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()

    return updated_row, rows


def read_golden_qa(path: Optional[Path] = None) -> List[Dict]:
    """Read the independent human Golden QA work file."""
    qa_path = path or GOLDEN_QA_PATH
    with qa_path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on Golden QA line {line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Golden QA line {line_number} is not an object")
            rows.append(row)
    return rows


def golden_qa_summary(rows: List[Dict]) -> Dict[str, int]:
    reviewed = sum(row.get("review_status") == "HUMAN_REVIEWED" for row in rows)
    return {"total": len(rows), "reviewed": reviewed, "pending": len(rows) - reviewed}


def _validate_golden_qa_row(row: Dict) -> None:
    """Validate fields required before a row can become human Golden QA."""
    if row.get("review_status") != "HUMAN_REVIEWED":
        return
    missing = []
    if not str(row.get("reviewer") or "").strip():
        missing.append("reviewer")
    if not str(row.get("reference_answer") or "").strip():
        missing.append("reference_answer")
    if not isinstance(row.get("answerable"), bool):
        missing.append("answerable")
    if not isinstance(row.get("requires_abstention"), bool):
        missing.append("requires_abstention")
    evidence_ids = row.get("gold_evidence_ids")
    if not isinstance(evidence_ids, list) or any(
        not isinstance(value, str) or not value.strip() for value in evidence_ids
    ):
        missing.append("gold_evidence_ids")
    gold_pages = row.get("gold_pages")
    if not isinstance(gold_pages, list) or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in gold_pages
    ):
        missing.append("gold_pages")
    grades = row.get("relevant_evidence_grades")
    if not isinstance(grades, dict) or any(
        not isinstance(value, int) or isinstance(value, bool) or value not in {0, 1, 2}
        for value in grades.values()
    ):
        missing.append("relevant_evidence_grades")
    if isinstance(evidence_ids, list) and isinstance(grades, dict) and not set(evidence_ids).issubset(grades):
        missing.append("relevant_evidence_grades(missing gold IDs)")
    if row.get("answerable") is True:
        if not evidence_ids:
            missing.append("gold_evidence_ids(nonempty for answerable row)")
        if not gold_pages:
            missing.append("gold_pages(nonempty for answerable row)")
        if row.get("requires_abstention") is not False:
            missing.append("requires_abstention(false for answerable row)")
    if row.get("answerable") is False and row.get("requires_abstention") is not True:
        missing.append("requires_abstention(true for unanswerable row)")
    if missing:
        raise ValueError(
            "Cannot mark Golden QA as HUMAN_REVIEWED; missing or invalid: "
            f"{sorted(set(missing))}"
        )


def update_golden_qa(
    qa_id: str,
    updates: Dict,
    path: Optional[Path] = None,
) -> tuple[Dict, List[Dict]]:
    """Update only the independent human QA file with an atomic replacement."""
    allowed = {
        "reference_answer",
        "gold_evidence_ids",
        "gold_pages",
        "relevant_evidence_grades",
        "answerable",
        "requires_abstention",
        "reviewer",
        "review_notes",
        "review_status",
    }
    unknown = set(updates) - allowed
    if unknown:
        raise ValueError(f"Unsupported Golden QA fields: {sorted(unknown)}")
    if "review_status" in updates and updates["review_status"] not in _GOLDEN_QA_STATUSES:
        raise ValueError(f"review_status must be one of: {sorted(_GOLDEN_QA_STATUSES)}")
    if "gold_evidence_ids" in updates and (
        not isinstance(updates["gold_evidence_ids"], list)
        or any(
            not isinstance(value, str) or not value.strip()
            for value in updates["gold_evidence_ids"]
        )
    ):
        raise ValueError("gold_evidence_ids must be a string array")
    if "gold_pages" in updates and (
        not isinstance(updates["gold_pages"], list)
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in updates["gold_pages"]
        )
    ):
        raise ValueError("gold_pages must be an integer array")
    if "relevant_evidence_grades" in updates:
        grades = updates["relevant_evidence_grades"]
        if not isinstance(grades, dict) or any(
            not isinstance(key, str)
            or not isinstance(value, int)
            or isinstance(value, bool)
            or value not in {0, 1, 2}
            for key, value in grades.items()
        ):
            raise ValueError("relevant_evidence_grades must map string IDs to 0, 1, or 2")
    for field in ("answerable", "requires_abstention"):
        if field in updates and updates[field] is not None and not isinstance(updates[field], bool):
            raise ValueError(f"{field} must be true, false, or null")

    qa_path = path or GOLDEN_QA_PATH
    with _GOLDEN_QA_LOCK:
        with qa_path.open("r", encoding="utf-8", newline="") as handle:
            raw_lines = handle.readlines()
        updated_row = None
        updated_lines = []
        rows = []
        for line in raw_lines:
            if not line.strip():
                updated_lines.append(line)
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("Golden QA contains a non-object row")
            if row.get("id") == qa_id:
                row.update(updates)
                _validate_golden_qa_row(row)
                updated_row = row
                line_ending = "\r\n" if line.endswith("\r\n") else "\n" if line.endswith("\n") else ""
                updated_lines.append(json.dumps(row, ensure_ascii=False) + line_ending)
            else:
                updated_lines.append(line)
            rows.append(row)
        if updated_row is None:
            raise KeyError(qa_id)
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=str(qa_path.parent),
                prefix=f".{qa_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.writelines(updated_lines)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, qa_path)
        finally:
            if temp_path is not None and temp_path.exists():
                temp_path.unlink()
    return updated_row, rows


def get_frontend_index() -> Path:
    dist_index = FRONTEND_ROOT / "dist" / "index.html"
    public_index = FRONTEND_ROOT / "public" / "index.html"
    return dist_index if dist_index.exists() else public_index


# Vite emits JavaScript/CSS into ``dist/assets``.  Returning index.html from
# the root route is not enough: without this mount the browser receives the
# shell but every hashed asset URL returns 404.
FRONTEND_ASSETS = FRONTEND_ROOT / "dist" / "assets"
if FRONTEND_ASSETS.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="frontend-assets")

# ── Lazy-loaded engines ──
_graph_engine = None
_vector_engine = None
_schema_manager = None
_graph_engine_lock = threading.Lock()
_vector_engine_lock = threading.Lock()
_schema_manager_lock = threading.Lock()
_warmup_task = None


def get_graph_engine():
    global _graph_engine
    if _graph_engine is None:
        with _graph_engine_lock:
            if _graph_engine is None:
                from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine
                _graph_engine = GraphRAGEngine()
    return _graph_engine


def get_vector_engine():
    global _vector_engine
    if _vector_engine is None:
        with _vector_engine_lock:
            if _vector_engine is None:
                from strategic_graphrag.engine.vector_rag_baseline import VectorRAGBaseline
                _vector_engine = VectorRAGBaseline()
    return _vector_engine


def get_schema_manager():
    global _schema_manager
    if _schema_manager is None:
        with _schema_manager_lock:
            if _schema_manager is None:
                from strategic_graphrag.schema.manager import SchemaManager
                candidate = SchemaManager()
                if not candidate.connect():
                    raise RuntimeError("Neo4j connection failed")
                _schema_manager = candidate
    return _schema_manager


@app.on_event("startup")
async def warm_runtime_dependencies():
    """Warm Hybrid retrieval off the request path; readiness stays non-blocking."""
    global _warmup_task
    _warmup_task = asyncio.create_task(run_in_threadpool(get_vector_engine))


def resolve_active_filing(requested: Optional[str] = None) -> Optional[str]:
    """Resolve the corpus scope used by graph-facing endpoints."""
    return requested or os.getenv("GRAPH_ACTIVE_FILING", "").strip() or None


def resolve_source_filing(
    requested: Optional[str],
    cross_filing: bool,
) -> Optional[str]:
    """Resolve an explicit all-filings request without hiding it behind env.

    The UI uses ``cross_filing=true`` for the all-years view.  A single-filing
    request continues to fall back to GRAPH_ACTIVE_FILING for backwards
    compatibility, while an explicit cross-filing request is always global.
    """
    return None if cross_filing else resolve_active_filing(requested)


def _graph_cache_get(key: str):
    cached = _GRAPH_CACHE.get(key)
    if not cached:
        return None
    if time.monotonic() - cached[0] > _GRAPH_CACHE_TTL_SECONDS:
        _GRAPH_CACHE.pop(key, None)
        return None
    _GRAPH_CACHE.move_to_end(key)
    return deepcopy(cached[1])


def _graph_cache_put(key: str, value):
    if _GRAPH_CACHE_TTL_SECONDS <= 0:
        return
    _GRAPH_CACHE[key] = (time.monotonic(), deepcopy(value))
    _GRAPH_CACHE.move_to_end(key)
    while len(_GRAPH_CACHE) > _GRAPH_CACHE_MAX_ENTRIES:
        _GRAPH_CACHE.popitem(last=False)


def _load_strict_subgraph(entity: Optional[str], limit: int, source_filing: Optional[str]):
    """Fetch strict evidence-backed edges without a nodes x edges Cartesian product."""
    mgr = get_schema_manager()
    rows = mgr._read(
        """
        MATCH (claim:EvidenceClaim)-[:ABOUT_SOURCE]->(source)
        MATCH (claim)-[:ABOUT_TARGET]->(target)
        MATCH (source)-[rel]->(target)
        WHERE rel.evidence_id = claim.id
          AND claim.verification_status = 'VERBATIM'
          AND ($source_filing IS NULL OR
               coalesce(rel.source_filing, rel.filing, '') = $source_filing)
          AND ($entity IS NULL OR
               toLower(coalesce(source.name, source.id, '')) CONTAINS toLower($entity) OR
               toLower(coalesce(target.name, target.id, '')) CONTAINS toLower($entity))
        RETURN
          coalesce(source.id, elementId(source)) AS source_id,
          coalesce(source.name, source.id, elementId(source)) AS source_name,
          labels(source) AS source_labels,
          coalesce(target.id, elementId(target)) AS target_id,
          coalesce(target.name, target.id, elementId(target)) AS target_name,
          labels(target) AS target_labels,
          type(rel) AS relationship_type,
          claim.id AS evidence_id
        ORDER BY claim.filing_fiscal_year DESC, claim.page, claim.id
        LIMIT $edge_limit
        """,
        entity=entity,
        source_filing=source_filing,
        edge_limit=limit * 2,
    )
    nodes: Dict[str, Dict] = {}
    edges: List[Dict] = []
    seen_edges = set()
    for row in rows:
        for side in ("source", "target"):
            node_id = row[f"{side}_id"]
            nodes[node_id] = {
                "id": node_id,
                "name": row[f"{side}_name"],
                "labels": row[f"{side}_labels"],
            }
        edge = {
            "source": row["source_id"],
            "target": row["target_id"],
            "type": row["relationship_type"],
            "evidence_id": row["evidence_id"],
        }
        key = tuple(edge.values())
        if key not in seen_edges:
            seen_edges.add(key)
            edges.append(edge)
    return {"nodes": list(nodes.values())[:limit], "edges": edges[: limit * 2]}


# =============================================================================
# Request/Response Models
# =============================================================================

class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language financial question")
    max_paths: int = Field(default=10, ge=1, le=30)
    year_filter: Optional[int] = Field(
        default=None,
        description="Backward-compatible minimum fiscal year filter",
    )
    year_start: Optional[int] = Field(default=None, ge=1900, le=2100)
    year_end: Optional[int] = Field(default=None, ge=1900, le=2100)
    source_filing: Optional[str] = Field(
        default=None,
        max_length=200,
        description="Optional filing scope; defaults to GRAPH_ACTIVE_FILING",
    )
    retrieval_mode: Literal["auto", "vector", "graph", "hybrid", "hybrid_temporal"] = Field(
        default="auto",
        description="Adaptive router or one of four comparable retrieval baselines",
    )
    vector_top_k: int = Field(default=5, ge=1, le=20)
    cross_filing: bool = Field(
        default=False,
        description="Explicitly search all indexed filings; disabled by default",
    )
    use_cache: bool = Field(
        default=True,
        description="Reuse an identical successful query within the bounded TTL cache",
    )
    synthesize: bool = Field(
        default=True,
        description="When false, run retrieval and return evidence traces without sending context to an external LLM",
    )


class QueryResponse(BaseModel):
    query: str
    intent: str
    intent_display: str
    answer: str
    execution_status: str = "SUCCEEDED"
    answer_status: str = "ABSTAINED"
    grounding_status: str = "NOT_EXECUTED"
    outcome: str = "ABSTAINED"
    status_provenance: Dict = Field(default_factory=dict)
    structured_report: Optional[Dict] = None
    paths: List[Dict]
    evidence_sentences: List[str]
    metadata: Dict


class VectorQueryResponse(BaseModel):
    query: str
    answer: str
    documents: List[str]
    execution_status: str = "SUCCEEDED"
    answer_status: str = "NOT_REQUESTED"
    grounding_status: str = "NOT_EXECUTED"
    outcome: str = "PARTIALLY_ANSWERED"
    status_provenance: Dict = Field(default_factory=dict)
    metadata: Dict = Field(default_factory=dict)


class GraphStats(BaseModel):
    total_nodes: int
    total_relationships: int
    graph_nodes: int = 0
    graph_relationships: int = 0
    by_label: Dict[str, int]
    by_relationship: Dict[str, int]
    source_filing: Optional[str] = None


class TemporalEvent(BaseModel):
    target: Optional[str] = None
    relation: str
    strength: Optional[str] = None
    year: int
    evidence: Optional[str] = None
    page: Optional[int] = None
    filing: Optional[str] = None
    evidence_id: Optional[str] = None


class TemporalChangeResponse(BaseModel):
    id: str
    change_type: str
    source_id: str
    relation_type: str
    target_id: str
    from_year: int
    to_year: int
    from_value: Optional[float] = None
    to_value: Optional[float] = None
    absolute_delta: Optional[float] = None
    percent_delta: Optional[float] = None
    earlier_claim_id: str
    later_claim_id: str
    semantics: str


class FinancialObservationResponse(BaseModel):
    id: str
    company_id: str
    metric_id: str
    fiscal_period: str
    fiscal_year: int
    value: float
    raw_value: str
    unit: str
    source_filing: str
    page: int
    claim_id: str
    table_name: str
    statement_type: str


class TemporalFactResponse(BaseModel):
    id: str
    fact_key: str
    source_id: str
    relation_type: str
    target_id: str
    source_filing: str
    page: int
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    recorded_from: str
    recorded_to: Optional[str] = None
    is_current_record: bool
    invalidation_status: str
    claim_id: str


class SubgraphRequest(BaseModel):
    entity_ids: List[str] = Field(default_factory=list, description="Focus entity IDs")
    max_nodes: int = Field(default=50, ge=10, le=200)


AnnotationLabel = Union[StrictBool, Literal["uncertain"]]


class ExtractionAnnotationPatch(BaseModel):
    source_entity_correct: Optional[AnnotationLabel] = None
    target_entity_correct: Optional[AnnotationLabel] = None
    relation_correct: Optional[AnnotationLabel] = None
    evidence_supports_relation: Optional[AnnotationLabel] = None
    missing_gold_relations: Optional[List[StrictStr]] = None
    annotation_status: Optional[StrictStr] = None
    annotator: Optional[StrictStr] = None
    notes: Optional[StrictStr] = None

    model_config = ConfigDict(extra="forbid")


class GoldenQAPatch(BaseModel):
    reference_answer: Optional[StrictStr] = None
    gold_evidence_ids: Optional[List[StrictStr]] = None
    gold_pages: Optional[List[int]] = None
    relevant_evidence_grades: Optional[Dict[StrictStr, int]] = None
    answerable: Optional[StrictBool] = None
    requires_abstention: Optional[StrictBool] = None
    reviewer: Optional[StrictStr] = None
    review_notes: Optional[StrictStr] = None
    review_status: Optional[StrictStr] = None

    model_config = ConfigDict(extra="forbid")


class TableQualityPatch(BaseModel):
    gold: Optional[Dict[str, Any]] = None
    reviewer: Optional[StrictStr] = None
    review_notes: Optional[StrictStr] = None
    review_status: Optional[StrictStr] = None

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/")
async def root():
    index_path = get_frontend_index()
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "service": "Strategic-GraphRAG API",
        "version": "1.0.0",
        "endpoints": ["POST /query", "POST /query/vector", "GET /graph/statistics", "GET /graph/subgraph", "GET /evidence/{entity_id}", "GET /graph/temporal/{risk_id}"],
    }


@app.get("/annotation")
@app.get("/annotation/")
async def annotation_page():
    """Serve the standalone annotation UI without exposing it in the main demo nav."""
    index_path = get_frontend_index()
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend build not found")


@app.get("/golden-qa")
@app.get("/golden-qa/")
async def golden_qa_page():
    """Serve the standalone human Golden QA review UI."""
    index_path = get_frontend_index()
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend build not found")


@app.get("/table-qa")
@app.get("/table-qa/")
async def table_quality_page():
    """Serve the independent table-cell annotation UI."""
    index_path = get_frontend_index()
    if index_path.exists():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="Frontend build not found")


@app.get("/evaluation/extraction-sample")
async def get_extraction_sample(sample: str = "baseline"):
    try:
        path = extraction_sample_path(sample)
        rows = read_extraction_sample(path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Extraction annotation sample not found")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"sample": sample, "rows": rows, **extraction_sample_summary(rows)}


@app.patch("/evaluation/extraction-sample/{claim_id}")
async def patch_extraction_sample(
    claim_id: str,
    req: ExtractionAnnotationPatch,
    sample: str = "baseline",
):
    try:
        path = extraction_sample_path(sample)
        if sample != HUMAN_EXTRACTION_SAMPLE_KEY:
            raise HTTPException(
                status_code=403,
                detail=(
                    "This extraction sample is read-only. The original 60-row baseline and historical "
                    "2025_post_repair_v2 are immutable; annotate only 2025_post_repair_human_v1."
                ),
            )
        updated_row, rows = update_extraction_sample(
            claim_id,
            req.model_dump(exclude_unset=True),
            path,
        )
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Extraction annotation sample not found")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Claim not found: {claim_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"sample": sample, "row": updated_row, **extraction_sample_summary(rows)}


@app.get("/evaluation/golden-qa")
async def get_golden_qa():
    try:
        rows = read_golden_qa()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Golden QA file not found")
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"rows": rows, **golden_qa_summary(rows)}


@app.patch("/evaluation/golden-qa/{qa_id}")
async def patch_golden_qa(qa_id: str, req: GoldenQAPatch):
    try:
        updated_row, rows = update_golden_qa(qa_id, req.model_dump(exclude_unset=True))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Golden QA file not found")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Golden QA item not found: {qa_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"row": updated_row, **golden_qa_summary(rows)}


@app.get("/evaluation/table-quality")
async def get_table_quality():
    try:
        rows = read_table_quality()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Table-quality queue not found")
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"rows": rows, **table_quality_summary(rows)}


@app.patch("/evaluation/table-quality/{queue_id}")
async def patch_table_quality(queue_id: str, req: TableQualityPatch):
    try:
        updated_row, rows = update_table_quality(queue_id, req.model_dump(exclude_unset=True))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Table-quality queue not found")
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Table-quality row not found: {queue_id}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return {"row": updated_row, **table_quality_summary(rows)}


@app.get("/evaluation/table-quality/source/{filename}")
async def get_table_quality_source(filename: str):
    source_path = _TABLE_QUALITY_SOURCE_FILES.get(filename)
    if source_path is None or not source_path.exists():
        raise HTTPException(status_code=404, detail="Source filing not found")
    return FileResponse(source_path, media_type="application/pdf", filename=filename)


@app.post("/query", response_model=QueryResponse)
async def graphrag_query(req: QueryRequest):
    """
    Execute a full GraphRAG inference pipeline.
    Returns structured causal analysis with evidence provenance.
    """
    try:
        cache_key = json.dumps(
            req.model_dump(exclude={"use_cache"}), sort_keys=True, ensure_ascii=False
        )
        now = time.monotonic()
        cached = _QUERY_CACHE.get(cache_key) if req.use_cache else None
        if cached and now - cached[0] <= _QUERY_CACHE_TTL_SECONDS:
            result = deepcopy(cached[1])
            result.setdefault("metadata", {})["cache"] = {
                "hit": True,
                "age_ms": round((now - cached[0]) * 1000, 2),
                "ttl_seconds": _QUERY_CACHE_TTL_SECONDS,
            }
            _QUERY_CACHE.move_to_end(cache_key)
            return QueryResponse(**apply_response_contract(result, provenance_note="api_cache_hit"))
        if cached:
            _QUERY_CACHE.pop(cache_key, None)
        engine = get_graph_engine()
        year_start = req.year_start if req.year_start is not None else req.year_filter
        if req.year_end is not None and year_start is not None and req.year_end < year_start:
            raise HTTPException(status_code=422, detail="year_end must be >= year_start")
        vector_engine = None
        if req.retrieval_mode in {"auto", "vector", "hybrid", "hybrid_temporal"}:
            try:
                vector_engine = get_vector_engine()
            except Exception as e:
                logger.warning("Vector engine unavailable; continuing in degraded hybrid mode: %s", e)
        result = await run_in_threadpool(
            engine.query,
            req.question,
            top_k=req.max_paths,
            year_start=year_start,
            year_end=req.year_end,
            source_filing=resolve_source_filing(req.source_filing, req.cross_filing),
            cross_filing=req.cross_filing,
            retrieval_mode=req.retrieval_mode,
            vector_engine=vector_engine,
            vector_top_k=req.vector_top_k,
            synthesize=req.synthesize,
        )
        result = apply_response_contract(result, provenance_note="api_engine_result")
        engine_status = str(result.get("execution_status") or "").upper()
        if engine_status != "SUCCEEDED":
            status_code = {
                "DEPENDENCY_ERROR": 503,
                "MODEL_ERROR": 502,
                "TIMEOUT": 504,
                "RATE_LIMITED": 429,
                "AUTH_ERROR": 401,
                "VALIDATION_ERROR": 422,
                "CONTRACT_ERROR": 500,
                "INTERNAL_ERROR": 500,
            }.get(engine_status, 500)
            error = result.get("metadata", {}).get("error") or {
                "code": engine_status,
                "message": "The query could not be completed.",
            }
            error = dict(error)
            error["outcome"] = result.get("outcome")
            raise HTTPException(status_code=status_code, detail=error)
        result.setdefault("metadata", {})["cache"] = {
            "hit": False,
            "ttl_seconds": _QUERY_CACHE_TTL_SECONDS,
        }
        cacheable = not str(result.get("answer") or "").startswith("[CONNECTION ERROR]")
        if req.use_cache and _QUERY_CACHE_TTL_SECONDS > 0 and cacheable:
            _QUERY_CACHE[cache_key] = (time.monotonic(), deepcopy(result))
            _QUERY_CACHE.move_to_end(cache_key)
            while len(_QUERY_CACHE) > _QUERY_CACHE_MAX_ENTRIES:
                _QUERY_CACHE.popitem(last=False)
        return QueryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "outcome": "INTERNAL_ERROR"},
        )


@app.post("/query/vector", response_model=VectorQueryResponse)
async def vector_query(req: QueryRequest):
    """
    Execute standard Vector RAG for comparison.
    """
    try:
        engine = get_vector_engine()
        retrieval = engine.retrieve_with_metadata(
            req.question,
            k=5,
            source_filing=resolve_source_filing(req.source_filing, req.cross_filing),
        )
        if retrieval.get("status") == "ERROR":
            raise HTTPException(
                status_code=503,
                detail={
                    "code": "DEPENDENCY_ERROR",
                    "outcome": "DEPENDENCY_ERROR",
                    "dependency": "chroma",
                    "message": "Vector retrieval is unavailable.",
                },
            )
        docs = [hit.get("document", "") for hit in retrieval.get("hits", [])]
        execution_status = "SUCCEEDED"
        generation_error = None
        if req.synthesize and docs:
            try:
                answer = engine.generate(req.question, docs)
            except TimeoutError as exc:
                answer = "[TIMEOUT] Vector synthesis timed out."
                execution_status = "TIMEOUT"
                generation_error = type(exc).__name__
            except Exception as exc:
                answer = "[MODEL ERROR] Vector synthesis failed."
                execution_status = "MODEL_ERROR"
                generation_error = type(exc).__name__
        elif docs:
            answer = "\n\n".join(docs[:5])
        else:
            answer = "[INSUFFICIENT EVIDENCE] No vector chunks were retrieved."
        answer_status = (
            "NOT_REQUESTED" if not req.synthesize
            else "ABSTAINED" if not docs
            else "ANSWERED" if execution_status == "SUCCEEDED"
            else "NOT_REQUESTED"
        )
        result = {
            "query": req.question,
            "answer": answer,
            "documents": docs,
            "metadata": {
                "retrieval": retrieval,
                "error": {"code": execution_status, "detail": generation_error} if generation_error else None,
            },
        }
        return VectorQueryResponse(**apply_response_contract(
            result,
            execution_status=execution_status,
            answer_status=answer_status,
            grounding_status="NOT_EXECUTED",
            provenance_note="api_vector_result",
        ))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Vector query error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"code": "INTERNAL_ERROR", "outcome": "INTERNAL_ERROR"},
        )


@app.get("/graph/statistics", response_model=GraphStats)
async def graph_statistics(
    source_filing: Optional[str] = Query(None, max_length=200),
    cross_filing: bool = Query(False),
):
    """
    Get knowledge graph statistics.
    """
    try:
        scoped_filing = None if cross_filing else resolve_active_filing(source_filing)
        cache_key = f"stats:{scoped_filing or 'all'}"
        stats = _graph_cache_get(cache_key)
        if stats is None:
            mgr = get_schema_manager()
            stats = await asyncio.wait_for(
                run_in_threadpool(mgr.stats, scoped_filing),
                timeout=float(os.getenv("GRAPH_ENDPOINT_TIMEOUT_SECONDS", "12")),
            )
            _graph_cache_put(cache_key, stats)
        return GraphStats(
            total_nodes=stats.get("total_nodes", 0),
            total_relationships=stats.get("total_rels", 0),
            graph_nodes=stats.get("graph_nodes", 0),
            graph_relationships=stats.get("graph_relationships", 0),
            by_label=stats.get("by_label", {}),
            by_relationship=stats.get("by_relationship", {}),
            source_filing=scoped_filing,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/subgraph")
async def get_subgraph(
    entity: Optional[str] = Query(None, description="Focus entity name/ID"),
    limit: int = Query(500, ge=10, le=2000),
    source_filing: Optional[str] = Query(
        None,
        max_length=200,
        description="Optional filing scope; defaults to GRAPH_ACTIVE_FILING",
    ),
    cross_filing: bool = Query(False),
):
    """
    Get subgraph data for visualization (nodes + edges in JSON).
    If entity is provided, returns strict incident EvidenceClaim edges.
    Otherwise returns a bounded strict graph sample.
    """
    source_filing = None if cross_filing else resolve_active_filing(source_filing)
    try:
        cache_key = f"subgraph:{source_filing or 'all'}:{entity or ''}:{limit}"
        cached = _graph_cache_get(cache_key)
        if cached is not None:
            return cached
        result = await asyncio.wait_for(
            run_in_threadpool(_load_strict_subgraph, entity, limit, source_filing),
            timeout=float(os.getenv("GRAPH_ENDPOINT_TIMEOUT_SECONDS", "12")),
        )
        _graph_cache_put(cache_key, result)
        return result
        # Legacy Cypher retained below temporarily for rollback reference; the
        # strict query above is the only reachable production path.
        if entity:
            cypher = """
            MATCH (n)
            WHERE toLower(coalesce(n.name, n.id)) CONTAINS toLower($entity)
               OR toLower(n.id) CONTAINS toLower($entity)
            WITH n LIMIT 1
            MATCH (n)-[rels*1..2]-(m)
            WHERE NOT n:Sentence AND NOT m:Sentence
              AND ALL(rel IN rels WHERE
                ($source_filing IS NULL OR
                 coalesce(rel.source_filing, rel.filing, '') = $source_filing)
                AND rel.evidence_id IS NOT NULL
                AND EXISTS {
                    MATCH (claim:EvidenceClaim {id: rel.evidence_id})
                    WHERE claim.verification_status = 'VERBATIM'
                })
            UNWIND rels AS rel
            WITH collect(DISTINCT n) + collect(DISTINCT m) AS allNodes, collect(DISTINCT rel) AS allRels
            UNWIND allNodes AS nd
            UNWIND allRels AS rel
            WITH DISTINCT nd, rel
            WHERE startNode(rel) = nd OR endNode(rel) = nd
            WITH DISTINCT
                [x IN collect(DISTINCT nd) | {id: x.id, name: coalesce(x.name, x.id), labels: labels(x)}] AS nodes,
                [x IN collect(DISTINCT rel) | {source: startNode(x).id, target: endNode(x).id, type: type(x), evidence_id: x.evidence_id}] AS edges
            RETURN nodes, edges
            LIMIT 1
            """
            with mgr.driver.session() as session:
                result = session.run(
                    cypher,
                    entity=entity,
                    source_filing=source_filing,
                )
                records = list(result)
                if records and records[0]:
                    data = records[0].data()
                    nodes_raw = data.get("nodes", [])
                    edges_raw = data.get("edges", [])
                    # Preserve separate evidence-backed relation instances.
                    # Two claims can support the same endpoint/type pair.
                    seen_n = set(); nodes = []
                    for nd in nodes_raw:
                        if nd["id"] not in seen_n: seen_n.add(nd["id"]); nodes.append(nd)
                    seen_e = set(); edges = []
                    for e in edges_raw:
                        k = f'{e["source"]}|{e["target"]}|{e["type"]}|{e.get("evidence_id") or ""}'
                        if k not in seen_e: seen_e.add(k); edges.append(e)
                    return {"nodes": nodes[:limit], "edges": edges[:limit * 2]}
        else:
            # Full graph sample: get nodes with highest degree relationships
            cypher = """
            MATCH (n)-[r]->(m)
            WHERE NOT n:Sentence AND NOT m:Sentence
              AND ($source_filing IS NULL OR
                   coalesce(r.source_filing, r.filing, '') = $source_filing)
              AND r.evidence_id IS NOT NULL
              AND EXISTS {
                  MATCH (claim:EvidenceClaim {id: r.evidence_id})
                  WHERE claim.verification_status = 'VERBATIM'
              }
            WITH n, r, m
            WITH collect(DISTINCT n) + collect(DISTINCT m) AS allNodes, collect(DISTINCT r) AS allRels
            UNWIND allNodes AS nd
            UNWIND allRels AS rel
            WITH DISTINCT nd, rel
            WHERE (startNode(rel) = nd OR endNode(rel) = nd)
            WITH DISTINCT
                [x IN collect(DISTINCT nd) | {id: x.id, name: coalesce(x.name, x.id), labels: labels(x)}] AS nodes,
                [x IN collect(DISTINCT rel) | {source: startNode(x).id, target: endNode(x).id, type: type(x), evidence_id: x.evidence_id}] AS edges
            RETURN nodes, edges
            LIMIT 1
            """
            with mgr.driver.session() as session:
                result = session.run(cypher, source_filing=source_filing)
                records = list(result)
                if records and records[0]:
                    data = records[0].data()
                    nodes_raw = data.get("nodes", [])
                    edges_raw = data.get("edges", [])
                    seen_n = set(); nodes = []
                    for nd in nodes_raw:
                        if nd["id"] not in seen_n: seen_n.add(nd["id"]); nodes.append(nd)
                    seen_e = set(); edges = []
                    for e in edges_raw:
                        k = f'{e["source"]}|{e["target"]}|{e["type"]}|{e.get("evidence_id") or ""}'
                        if k not in seen_e: seen_e.add(k); edges.append(e)
                    return {"nodes": nodes[:limit], "edges": edges[:limit * 2]}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"nodes": [], "edges": []}


@app.get("/evidence/{entity_id}")
async def get_evidence(
    entity_id: str,
    limit: int = Query(10, ge=1, le=50),
    source_filing: Optional[str] = Query(None, max_length=200),
    cross_filing: bool = Query(False),
):
    """
    Get evidence sentences for a specific entity.
    """
    try:
        engine = get_graph_engine()
        evidence = engine.path_finder.find_evidence_for_entity(
            entity_id,
            limit=limit,
            source_filing=resolve_source_filing(source_filing, cross_filing),
        )
        return {"entity_id": entity_id, "evidence": evidence}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/temporal/{risk_id}", response_model=List[TemporalEvent])
async def temporal_evolution(
    risk_id: str,
    limit: int = Query(20, ge=1, le=100),
    source_filing: Optional[str] = Query(None, max_length=200),
):
    """Return year-anchored evidence links for one risk factor."""
    try:
        engine = get_graph_engine()
        rows = engine.path_finder.find_temporal_evolution(
            risk_id,
            source_filing=resolve_active_filing(source_filing),
        )
        return rows[:limit]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/graph/temporal-changes/{entity_id}", response_model=List[TemporalChangeResponse])
async def temporal_changes(entity_id: str, limit: int = Query(50, ge=1, le=200)):
    """Return observed cross-filing changes backed by two EvidenceClaims."""
    try:
        rows = await asyncio.wait_for(
            run_in_threadpool(
                get_schema_manager()._read,
                """
                MATCH (earlier:EvidenceClaim)-[:HAS_TEMPORAL_CHANGE]->(change:TemporalChange {model_version:'bitemporal_fact_v2'})-[:CHANGES_TO]->(later:EvidenceClaim)
                WHERE toLower(change.source_id) CONTAINS toLower($entity_id)
                   OR toLower(change.target_id) CONTAINS toLower($entity_id)
                RETURN change.id AS id, change.change_type AS change_type,
                       change.source_id AS source_id, change.relation_type AS relation_type,
                       change.target_id AS target_id,
                       change.earlier_year AS from_year, change.later_year AS to_year,
                       change.from_value AS from_value, change.to_value AS to_value,
                       change.absolute_delta AS absolute_delta, change.percent_delta AS percent_delta,
                       earlier.id AS earlier_claim_id, later.id AS later_claim_id,
                       change.semantics AS semantics
                ORDER BY from_year, to_year, id LIMIT $limit
                """,
                entity_id=entity_id,
                limit=limit,
            ),
            timeout=float(os.getenv("GRAPH_ENDPOINT_TIMEOUT_SECONDS", "12")),
        )
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/graph/financial-observations/{metric_id}",
    response_model=List[FinancialObservationResponse],
)
async def financial_observations(
    metric_id: str,
    year_start: Optional[int] = Query(None, ge=1900, le=2100),
    year_end: Optional[int] = Query(None, ge=1900, le=2100),
    limit: int = Query(100, ge=1, le=500),
):
    """Return period-specific numeric observations with claim provenance."""
    if year_start is not None and year_end is not None and year_end < year_start:
        raise HTTPException(status_code=422, detail="year_end must be >= year_start")
    rows = await run_in_threadpool(
        get_schema_manager()._read,
        """
        MATCH (observation:FinancialObservation)-[:OBSERVES_METRIC]->(metric:FinancialMetric)
        MATCH (observation)-[:SUPPORTED_BY_CLAIM]->(claim:EvidenceClaim)
        WHERE (metric.id=$metric_id OR toLower(metric.name)=toLower($metric_id))
          AND ($year_start IS NULL OR observation.fiscal_year >= $year_start)
          AND ($year_end IS NULL OR observation.fiscal_year <= $year_end)
        RETURN observation.id AS id, observation.company_id AS company_id,
               observation.metric_id AS metric_id,
               observation.fiscal_period AS fiscal_period,
               observation.fiscal_year AS fiscal_year,
               observation.value AS value, observation.raw_value AS raw_value,
               observation.unit AS unit, observation.source_filing AS source_filing,
               observation.page AS page, claim.id AS claim_id,
               observation.table_name AS table_name,
               observation.statement_type AS statement_type
        ORDER BY fiscal_year, source_filing, page, id LIMIT $limit
        """,
        metric_id=metric_id,
        year_start=year_start,
        year_end=year_end,
        limit=limit,
    )
    return rows


@app.get(
    "/graph/temporal-facts/{entity_id}",
    response_model=List[TemporalFactResponse],
)
async def temporal_facts(
    entity_id: str,
    current_only: bool = Query(False),
    limit: int = Query(100, ge=1, le=500),
):
    """Return bitemporal disclosure versions without deleting history."""
    rows = await run_in_threadpool(
        get_schema_manager()._read,
        """
        MATCH (fact:TemporalFact {model_version:'bitemporal_fact_v2'})
        MATCH (fact)-[:SUPPORTED_BY_CLAIM]->(claim:EvidenceClaim)
        WHERE (toLower(fact.source_id) CONTAINS toLower($entity_id)
            OR toLower(fact.target_id) CONTAINS toLower($entity_id))
          AND (NOT $current_only OR fact.is_current_record=true)
        RETURN fact.id AS id, fact.fact_key AS fact_key,
               fact.source_id AS source_id, fact.relation_type AS relation_type,
               fact.target_id AS target_id, fact.source_filing AS source_filing,
               fact.page AS page, fact.valid_from AS valid_from,
               fact.valid_to AS valid_to, toString(fact.recorded_from) AS recorded_from,
               toString(fact.recorded_to) AS recorded_to,
               fact.is_current_record AS is_current_record,
               fact.invalidation_status AS invalidation_status,
               claim.id AS claim_id
        ORDER BY fact.disclosure_order, fact.page, fact.id LIMIT $limit
        """,
        entity_id=entity_id,
        current_only=current_only,
        limit=limit,
    )
    return rows


@app.get("/health/live")
async def health_live():
    """Process liveness only; never waits for Aura, embeddings, or an LLM."""
    return {
        "status": "alive",
        "version": app.version,
        "uptime_seconds": round(time.time() - _STARTED_AT, 2),
    }


def _probe_readiness() -> Dict:
    dependencies: Dict[str, Dict] = {}
    try:
        rows = get_schema_manager()._read("RETURN 1 AS ok")
        dependencies["neo4j"] = {"ready": bool(rows and rows[0].get("ok") == 1)}
    except Exception as exc:
        dependencies["neo4j"] = {"ready": False, "error": type(exc).__name__}
    if _vector_engine is None:
        dependencies["vector"] = {"ready": False, "status": "warming"}
    else:
        try:
            dependencies["vector"] = _vector_engine.diagnostics()
        except Exception as exc:
            dependencies["vector"] = {"ready": False, "error": type(exc).__name__}
    try:
        from strategic_graphrag.llm_provider import get_llm
        llm = get_llm()
        dependencies["llm"] = {
            "ready": bool(llm.available),
            "provider": llm.provider,
            "model": llm.default_model,
        }
    except Exception as exc:
        dependencies["llm"] = {"ready": False, "error": type(exc).__name__}
    return {
        "status": "ready" if all(item.get("ready") for item in dependencies.values()) else "degraded",
        "checked_at": time.time(),
        "dependencies": dependencies,
        "active_filing": resolve_active_filing(),
        "auth_enabled": _API_AUTH_ENABLED,
    }


@app.get("/health/ready")
async def health_ready():
    """Bounded dependency readiness probe for deployment and acceptance tests."""
    global _READINESS
    try:
        _READINESS = await asyncio.wait_for(
            run_in_threadpool(_probe_readiness),
            timeout=float(os.getenv("READINESS_TIMEOUT_SECONDS", "12")),
        )
    except asyncio.TimeoutError:
        _READINESS = {
            "status": "degraded",
            "checked_at": time.time(),
            "dependencies": {"probe": {"ready": False, "error": "timeout"}},
        }
    status_code = 200 if _READINESS["status"] == "ready" else 503
    return JSONResponse(status_code=status_code, content=_READINESS)


@app.get("/health")
async def health_check():
    """Backward-compatible cheap status; use /health/ready for live dependencies."""
    return {
        "status": "healthy",
        "liveness": "alive",
        "readiness": _READINESS,
        "version": app.version,
        "uptime_seconds": round(time.time() - _STARTED_AT, 2),
    }


# ── Run ──

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
