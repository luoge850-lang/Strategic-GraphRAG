# -*- coding: utf-8 -*-
"""Auditable JSONL cache for structured LLM extraction responses.

The cache stores only the structured JSON returned by an extraction request.
The complete input prompt is represented by a SHA-256 digest in the request
metadata; the raw PDF prompt is not stored as a separate cache field.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


CACHE_SCHEMA_VERSION = "llm_response_cache_v1"
DEFAULT_CACHE_PATH = "evaluation/cache/llm_extraction_v1.jsonl"
VALID_CACHE_MODES = ("off", "record", "replay")


class LLMResponseCacheError(RuntimeError):
    """Base class for fail-closed response-cache errors."""


class LLMResponseCacheMiss(LLMResponseCacheError):
    """Raised when replay cannot find the requested frozen response."""


class LLMResponseCacheCorrupt(LLMResponseCacheError):
    """Raised when a cache record is malformed or ambiguous."""


@dataclass(frozen=True)
class CacheRequest:
    """The non-secret, reproducibility-relevant identity of one LLM call."""

    operation: str
    request_provider: str
    request_model: str
    temperature: float
    max_tokens: int
    prompt_sha256: str
    schema_version: str = CACHE_SCHEMA_VERSION

    @property
    def key(self) -> str:
        material = {
            "operation": self.operation,
            "request_provider": self.request_provider,
            "request_model": self.request_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_sha256": self.prompt_sha256,
            "schema_version": self.schema_version,
        }
        encoded = json.dumps(
            material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return f"{self.schema_version}:{hashlib.sha256(encoded).hexdigest()}"

    def metadata(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "request_provider": self.request_provider,
            "request_model": self.request_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "prompt_sha256": self.prompt_sha256,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class CacheHit:
    """A frozen response and the route that produced it during record mode."""

    key: str
    response: Any
    metadata: Dict[str, Any]


class LLMResponseCache:
    """Versioned append-only JSONL cache with explicit off/record/replay modes."""

    def __init__(
        self,
        path: str = DEFAULT_CACHE_PATH,
        mode: str = "off",
        schema_version: str = CACHE_SCHEMA_VERSION,
    ) -> None:
        normalized_mode = str(mode or "off").strip().lower()
        if normalized_mode not in VALID_CACHE_MODES:
            raise ValueError(
                f"Invalid LLM response cache mode {mode!r}; "
                f"expected one of {VALID_CACHE_MODES}."
            )
        if schema_version != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported LLM response cache schema {schema_version!r}; "
                f"expected {CACHE_SCHEMA_VERSION!r}."
            )

        self.mode = normalized_mode
        self.path = Path(path)
        self.schema_version = schema_version
        self._records: Dict[str, Dict[str, Any]] = {}
        self._hits = 0
        self._misses = 0
        self._writes = 0
        self._corrupt_records = 0
        self._duplicate_keys = 0

        # off is deliberately side-effect free: do not inspect, create, or
        # touch the configured path in this mode.
        if self.mode != "off":
            self._load()

    @classmethod
    def from_env(
        cls,
        mode: Optional[str] = None,
        path: Optional[str] = None,
    ) -> "LLMResponseCache":
        return cls(
            path=path or os.getenv("LLM_RESPONSE_CACHE_PATH", DEFAULT_CACHE_PATH),
            mode=mode or os.getenv("LLM_RESPONSE_CACHE_MODE", "off"),
        )

    @staticmethod
    def prompt_sha256(prompt: str, system_prompt: str = "") -> str:
        """Hash the complete role-separated prompt without storing its text."""
        prompt_material = {
            "system_prompt": str(system_prompt or ""),
            "user_prompt": str(prompt or ""),
        }
        encoded = json.dumps(
            prompt_material, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def make_request(
        cls,
        operation: str,
        request_provider: str,
        request_model: str,
        temperature: float,
        max_tokens: int,
        prompt: str,
        system_prompt: str = "",
    ) -> CacheRequest:
        try:
            normalized_temperature = float(temperature)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cache temperature must be numeric.") from exc
        if normalized_temperature != normalized_temperature or normalized_temperature in (
            float("inf"),
            float("-inf"),
        ):
            raise ValueError("Cache temperature must be finite.")
        try:
            normalized_max_tokens = int(max_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("Cache max_tokens must be an integer.") from exc
        return CacheRequest(
            operation=str(operation),
            request_provider=str(request_provider),
            request_model=str(request_model),
            temperature=normalized_temperature,
            max_tokens=normalized_max_tokens,
            prompt_sha256=cls.prompt_sha256(prompt, system_prompt),
        )

    @classmethod
    def make_key(
        cls,
        operation: str,
        request_provider: str,
        request_model: str,
        temperature: float,
        max_tokens: int,
        prompt: str,
        system_prompt: str = "",
    ) -> str:
        """Return the stable key used by the JSONL record."""
        return cls.make_request(
            operation=operation,
            request_provider=request_provider,
            request_model=request_model,
            temperature=temperature,
            max_tokens=max_tokens,
            prompt=prompt,
            system_prompt=system_prompt,
        ).key

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, 1):
                    if not raw_line.strip():
                        self._fail_corrupt(line_number, "blank JSONL record")
                    try:
                        record = json.loads(raw_line)
                    except json.JSONDecodeError as exc:
                        self._fail_corrupt(line_number, f"invalid JSON ({exc.msg})")
                    self._validate_record(record, line_number)
                    key = record["key"]
                    if key in self._records:
                        self._duplicate_keys += 1
                        raise LLMResponseCacheCorrupt(
                            f"Duplicate cache key at {self.path}:{line_number}: {key}"
                        )
                    self._records[key] = record
        except OSError as exc:
            raise LLMResponseCacheError(
                f"Unable to read LLM response cache {self.path}: {exc}"
            ) from exc

    def _fail_corrupt(self, line_number: int, reason: str) -> None:
        self._corrupt_records += 1
        raise LLMResponseCacheCorrupt(
            f"Corrupt LLM response cache record at {self.path}:{line_number}: {reason}"
        )

    def _validate_record(self, record: Any, line_number: int) -> None:
        if not isinstance(record, dict):
            self._fail_corrupt(line_number, "record must be a JSON object")
        required = {"key", "metadata", "response", "created_at"}
        missing = sorted(required - set(record))
        if missing:
            self._fail_corrupt(line_number, f"missing fields: {', '.join(missing)}")
        if record.get("schema_version") != self.schema_version:
            self._fail_corrupt(line_number, "schema_version does not match cache schema")
        if not isinstance(record.get("key"), str) or not record["key"]:
            self._fail_corrupt(line_number, "key must be a non-empty string")
        metadata = record.get("metadata")
        if not isinstance(metadata, dict):
            self._fail_corrupt(line_number, "metadata must be a JSON object")
        required_metadata = {
            "operation",
            "request_provider",
            "request_model",
            "temperature",
            "max_tokens",
            "prompt_sha256",
            "schema_version",
        }
        missing_metadata = sorted(required_metadata - set(metadata))
        if missing_metadata:
            self._fail_corrupt(
                line_number,
                f"metadata missing fields: {', '.join(missing_metadata)}",
            )
        if metadata.get("schema_version") != self.schema_version:
            self._fail_corrupt(line_number, "metadata schema_version does not match")
        if not isinstance(record.get("response"), (dict, list)):
            self._fail_corrupt(line_number, "response must be a structured JSON object or array")
        try:
            expected_key = CacheRequest(
                operation=str(metadata["operation"]),
                request_provider=str(metadata["request_provider"]),
                request_model=str(metadata["request_model"]),
                temperature=float(metadata["temperature"]),
                max_tokens=int(metadata["max_tokens"]),
                prompt_sha256=str(metadata["prompt_sha256"]),
                schema_version=self.schema_version,
            ).key
        except (TypeError, ValueError, OverflowError) as exc:
            self._fail_corrupt(line_number, f"invalid request metadata ({exc})")
        if record["key"] != expected_key:
            self._fail_corrupt(line_number, "key does not match request metadata")

    def lookup(self, request: CacheRequest) -> Optional[CacheHit]:
        """Look up a request, or raise on a replay miss."""
        if self.mode == "off":
            return None
        record = self._records.get(request.key)
        if record is None:
            self._misses += 1
            if self.mode == "replay":
                raise LLMResponseCacheMiss(
                    "LLM response cache replay miss: "
                    f"key={request.key}, path={self.path}. "
                    "Replay is fail-closed and will not call an external provider."
                )
            return None
        self._hits += 1
        return CacheHit(
            key=request.key,
            response=copy.deepcopy(record["response"]),
            metadata=copy.deepcopy(record["metadata"]),
        )

    def store(
        self,
        request: CacheRequest,
        response: Any,
        actual_provider: Optional[str] = None,
        actual_model: Optional[str] = None,
    ) -> bool:
        """Append a successful response once; never overwrite a frozen record."""
        if self.mode != "record":
            return False
        if not isinstance(response, (dict, list)):
            raise LLMResponseCacheError(
                f"Refusing to cache non-structured response for key {request.key}"
            )
        existing = self._records.get(request.key)
        if existing is not None:
            return False

        metadata = request.metadata()
        metadata.update(
            {
                "actual_provider": str(actual_provider or request.request_provider),
                "actual_model": str(actual_model or request.request_model),
                "response_sha256": hashlib.sha256(
                    json.dumps(
                        response,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
            }
        )
        record = {
            "schema_version": self.schema_version,
            "key": request.key,
            "metadata": metadata,
            "response": copy.deepcopy(response),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            raise LLMResponseCacheError(
                f"Unable to append LLM response cache {self.path}: {exc}"
            ) from exc
        self._records[request.key] = record
        self._writes += 1
        return True

    def stats(self) -> Dict[str, Any]:
        """Return JSON-serializable cache diagnostics for extraction reports."""
        return {
            "mode": self.mode,
            "path": str(self.path),
            "schema_version": self.schema_version,
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "key_count": len(self._records),
            "corrupt_records": self._corrupt_records,
            "duplicate_keys": self._duplicate_keys,
        }
