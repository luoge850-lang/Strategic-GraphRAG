"""Versioned response contract shared by the engine, API and evaluators.

``outcome`` remains as a compatibility field, but it is derived from three
orthogonal dimensions:

* ``execution_status`` — whether the requested operation ran successfully;
* ``answer_status`` — whether an answer task was requested and what it did;
* ``grounding_status`` — whether the displayed answer was checked against
  evidence.

The legacy parser is intentionally conservative. An unknown or empty response
is a contract failure, not a safe abstention, and an HTTP failure always takes
precedence over a body that claims success.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional


CONTRACT_VERSION = "response-contract/v2"

EXECUTION_STATUSES = frozenset({
    "SUCCEEDED",
    "DEPENDENCY_ERROR",
    "MODEL_ERROR",
    "TIMEOUT",
    "RATE_LIMITED",
    "AUTH_ERROR",
    "VALIDATION_ERROR",
    "CONTRACT_ERROR",
    "INTERNAL_ERROR",
})

ANSWER_STATUSES = frozenset({
    "NOT_REQUESTED",
    "ANSWERED",
    "PARTIALLY_ANSWERED",
    "ABSTAINED",
})

GROUNDING_STATUSES = frozenset({
    "VERIFIED",
    "FAILED",
    "NOT_EXECUTED",
    "NOT_APPLICABLE",
    "INSUFFICIENT",
})

OUTCOMES = frozenset({
    "ANSWERED",
    "PARTIALLY_ANSWERED",
    "ABSTAINED",
    "DEPENDENCY_ERROR",
    "MODEL_ERROR",
    "TIMEOUT",
    "RATE_LIMITED",
    "AUTH_ERROR",
    "VALIDATION_ERROR",
    "CONTRACT_ERROR",
    "INTERNAL_ERROR",
})

ABSTENTION_STATUSES = frozenset({
    "INSUFFICIENT_EVIDENCE",
    "INSUFFICIENT_DIRECT_EVIDENCE",
    "INSUFFICIENT_TEMPORAL_EVIDENCE",
    "NO_HITS",
    "NO_RESULTS",
})

_REFUSAL_RE = re.compile(
    r"(?:insufficient\s+(?:evidence|direct\s+evidence|temporal\s+evidence)|"
    r"no\s+(?:verified\s+)?(?:causal\s+path(?:way|ways)?|vector\s+chunks?|"
    r"information|conclusion)|"
    r"does\s+not\s+(?:contain|disclose|provide|include|establish)|"
    r"(?:no|without)\s+(?:[a-z]+\s+)?absence\s+conclusion|"
    r"audit\s+(?:was\s+)?unavailable|"
    r"cannot\s+(?:establish|determine|be\s+completed)|"
    r"not\s+(?:supported|permitted|available|established|reported|disclosed|provided|found)|"
    r"(?:no|without)\s+.{0,80}\b(?:information|evidence|data|value|revenue|sales|income|metric|fiscal\s+year|20\d{2})\b|"
    r"\b(?:revenue|sales|income|expense|margin|metric|20\d{2})\b.{0,40}\b(?:is|are|was|were)?\s*(?:not|never)\s+(?:reported|disclosed|provided|available|found)|"
    r"grounding\s+failure|connection\s+error|model\s+error)",
    re.IGNORECASE,
)

_POSITIVE_FACT_RE = re.compile(
    r"(?:(?<!\d)(?!(?:19|20)\d{2}\b)\$?\d[\d,]*(?:\.\d+)?\s*(?:%|million|billion|thousand)?|"
    r"\b(?:increased|decreased|produces|produced|introduced|"
    r"caused|causes|decreases|increases|affects|affected)\b)",
    re.IGNORECASE,
)

_GENERATION_ERROR_RE = re.compile(
    r"^\s*\[?\s*(?:generation\s+error|model\s+error|llm\s+unavailable|synthesis\s+error)\b",
    re.IGNORECASE,
)


def is_generation_error_text(value: Any) -> bool:
    """Identify provider failure sentinels returned as text instead of exceptions."""
    return bool(_GENERATION_ERROR_RE.search(str(value or "")))


def _structured_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    report = payload.get("structured_report")
    return report if isinstance(report, dict) else {}


def _status_sources(payload: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    metadata = payload.get("metadata")
    report = _structured_report(payload)
    yield payload
    if isinstance(metadata, dict):
        yield metadata
    if isinstance(report, dict):
        yield report


def _split_sentences(value: Any) -> list[str]:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _is_pure_refusal(value: Any) -> bool:
    text = re.sub(r"^\s*(?:\[[^\]]+\]\s*)+", "", str(value or "")).strip()
    if not text or not _REFUSAL_RE.search(text):
        return False
    # A concrete value or positive factual predicate means the fragment is
    # mixed content and must remain subject to grounding validation.
    # Metric names and years inside a refusal are not answers. Only a
    # concrete value or affirmative predicate turns the fragment into mixed
    # content; negated predicates remain refusals.
    if re.search(r"\b(?:no|not|never|cannot|unable|unavailable)\b", text, re.IGNORECASE):
        positive = _POSITIVE_FACT_RE.search(text)
        if positive and not re.search(
            r"(?:no|not|never)\s+(?:\w+\s+){0,5}" + re.escape(positive.group(0)),
            text,
            re.IGNORECASE,
        ):
            return False
        return True
    return not bool(_POSITIVE_FACT_RE.search(text))


def substantive_fragments(value: Any) -> list[str]:
    """Return factual-looking fragments while retaining mixed responses."""
    return [
        sentence
        for sentence in _split_sentences(value)
        if not _is_pure_refusal(sentence)
    ]


def _has_substantive_text(value: Any) -> bool:
    return bool(substantive_fragments(value))


def has_substantive_claims(payload_or_report: Dict[str, Any]) -> bool:
    """Return whether a response/report contains factual or analytical text."""
    report = (
        _structured_report(payload_or_report)
        if "structured_report" in payload_or_report
        else payload_or_report
    )
    for claim in report.get("claims", []) or []:
        if isinstance(claim, dict) and _has_substantive_text(claim.get("statement")):
            return True
    for field in ("executive_summary", "answer", "narrative"):
        if _has_substantive_text(report.get(field)):
            return True
    if "structured_report" in payload_or_report:
        return _has_substantive_text(payload_or_report.get("answer"))
    return False


def _first_valid(payload: Dict[str, Any], field: str, allowed: Iterable[str]) -> Optional[str]:
    allowed_set = set(allowed)
    for source in _status_sources(payload):
        if field not in source or source.get(field) is None:
            continue
        normalized = str(source.get(field)).upper().strip()
        if normalized in allowed_set:
            return normalized
    return None


def _unknown_explicit_status(payload: Dict[str, Any], field: str, allowed: Iterable[str]) -> Optional[str]:
    allowed_set = set(allowed)
    for source in _status_sources(payload):
        if field not in source or source.get(field) is None:
            continue
        normalized = str(source.get(field)).upper().strip()
        if normalized not in allowed_set:
            return normalized or "EMPTY"
    return None


def _http_execution_status(http_status: Optional[int]) -> Optional[str]:
    if http_status is None:
        return None
    if http_status == 429:
        return "RATE_LIMITED"
    if http_status in {401, 403}:
        return "AUTH_ERROR"
    if http_status in {408, 504}:
        return "TIMEOUT"
    if http_status == 503:
        return "DEPENDENCY_ERROR"
    if http_status == 502:
        return "MODEL_ERROR"
    if http_status >= 500:
        return "INTERNAL_ERROR"
    if http_status >= 400:
        return "VALIDATION_ERROR"
    return "SUCCEEDED"


def _legacy_execution(payload: Dict[str, Any]) -> Optional[str]:
    report = _structured_report(payload)
    report_status = str(report.get("status") or "").upper()
    answer = str(payload.get("answer") or "")
    upper_answer = answer.upper()
    if report_status in {"SYNTHESIS_ERROR", "INVALID_JSON", "INVALID_SCHEMA"}:
        return "MODEL_ERROR"
    if report_status == "GROUNDING_FAILURE" or "[GROUNDING FAILURE]" in upper_answer:
        return "VALIDATION_ERROR"
    if "[CONNECTION ERROR]" in upper_answer:
        return "DEPENDENCY_ERROR"
    if "TIMEOUT" in upper_answer or "TIMED OUT" in upper_answer:
        return "TIMEOUT"
    if is_generation_error_text(answer):
        return "MODEL_ERROR"
    return None


def _legacy_answer_status(payload: Dict[str, Any], execution_status: str) -> Optional[str]:
    if execution_status != "SUCCEEDED":
        return "NOT_REQUESTED"
    report = _structured_report(payload)
    status = str(report.get("status") or "").upper()
    if status == "RETRIEVAL_ONLY" or payload.get("answer_status") == "NOT_REQUESTED":
        return "NOT_REQUESTED"
    if status in ABSTENTION_STATUSES or status.startswith("INSUFFICIENT"):
        return "ABSTAINED"
    if status == "NEGATIVE_CLAIM_GUARD":
        return "PARTIALLY_ANSWERED" if has_substantive_claims(report) else "ABSTAINED"
    answer = str(payload.get("answer") or "")
    if has_substantive_claims(payload) or has_substantive_claims(report):
        return "ANSWERED"
    if _is_pure_refusal(answer) or _is_pure_refusal(report.get("executive_summary")):
        return "ABSTAINED"
    return None


def _legacy_grounding_status(payload: Dict[str, Any], answer_status: str) -> str:
    for source in _status_sources(payload):
        grounding = source.get("grounding")
        if isinstance(grounding, dict):
            value = str(grounding.get("status") or "").upper()
            if value == "UNSUPPORTED":
                return "FAILED"
            if value in GROUNDING_STATUSES:
                return value
    if answer_status == "NOT_REQUESTED":
        return "NOT_EXECUTED"
    if answer_status == "ABSTAINED":
        return "NOT_APPLICABLE"
    return "NOT_EXECUTED"


def response_state(
    payload: Optional[Dict[str, Any]],
    *,
    http_status: Optional[int] = None,
    error: Any = None,
) -> Dict[str, Any]:
    """Derive the authoritative three-dimensional response state."""
    payload = payload if isinstance(payload, dict) else {}
    provenance = {
        "contract_version": CONTRACT_VERSION,
        "source": "explicit_fields",
        "legacy_compatibility": False,
        "uncertain": False,
        "notes": [],
    }

    unknown_execution = _unknown_explicit_status(payload, "execution_status", EXECUTION_STATUSES)
    unknown_answer = _unknown_explicit_status(payload, "answer_status", ANSWER_STATUSES)
    unknown_grounding = _unknown_explicit_status(payload, "grounding_status", GROUNDING_STATUSES)
    if unknown_execution or unknown_answer or unknown_grounding:
        provenance["uncertain"] = True
        provenance["notes"].append("unknown_explicit_status")
        return {
            "execution_status": "CONTRACT_ERROR",
            "answer_status": "NOT_REQUESTED",
            "grounding_status": "NOT_EXECUTED",
            "outcome": "CONTRACT_ERROR",
            "status_provenance": provenance,
        }

    http_execution = _http_execution_status(http_status)
    explicit_execution = _first_valid(payload, "execution_status", EXECUTION_STATUSES)
    if http_execution and (http_status or 0) >= 400:
        execution_status = http_execution
        provenance["notes"].append("http_status_precedence")
    elif error is not None:
        execution_status = "TIMEOUT" if isinstance(error, TimeoutError) else "DEPENDENCY_ERROR"
        provenance["notes"].append("exception")
    elif explicit_execution:
        execution_status = explicit_execution
    else:
        legacy_execution = _legacy_execution(payload)
        if legacy_execution:
            execution_status = legacy_execution
            provenance["source"] = "legacy_compatibility"
            provenance["legacy_compatibility"] = True
        elif _first_valid(payload, "outcome", OUTCOMES) in EXECUTION_STATUSES:
            execution_status = _first_valid(payload, "outcome", OUTCOMES) or "CONTRACT_ERROR"
            provenance["source"] = "legacy_compatibility"
            provenance["legacy_compatibility"] = True
        elif not payload:
            execution_status = "CONTRACT_ERROR"
            provenance["uncertain"] = True
            provenance["notes"].append("empty_payload")
        else:
            execution_status = "SUCCEEDED"

    explicit_answer = _first_valid(payload, "answer_status", ANSWER_STATUSES)
    if execution_status != "SUCCEEDED":
        answer_status = explicit_answer if explicit_answer and explicit_answer != "ANSWERED" else "NOT_REQUESTED"
    elif explicit_answer:
        answer_status = explicit_answer
    else:
        answer_status = _legacy_answer_status(payload, execution_status)
        if answer_status is None:
            answer_status = "NOT_REQUESTED"
            execution_status = "CONTRACT_ERROR"
            provenance["source"] = "legacy_compatibility"
            provenance["uncertain"] = True
            provenance["notes"].append("no_answer_status_or_legacy_answer")

    explicit_grounding = _first_valid(payload, "grounding_status", GROUNDING_STATUSES)
    grounding_status = explicit_grounding or _legacy_grounding_status(payload, answer_status)
    if execution_status != "SUCCEEDED" and not explicit_grounding:
        grounding_status = "NOT_EXECUTED"

    explicit_outcome = _first_valid(payload, "outcome", OUTCOMES)
    if execution_status != "SUCCEEDED":
        outcome = execution_status
    elif grounding_status == "FAILED" and answer_status != "NOT_REQUESTED":
        outcome = "VALIDATION_ERROR"
    elif answer_status == "NOT_REQUESTED":
        # Retrieval-only is not a normal answer refusal. Keep a compatibility
        # outcome while exposing NOT_REQUESTED as the authoritative field.
        outcome = "PARTIALLY_ANSWERED"
    else:
        outcome = answer_status

    if explicit_outcome and explicit_outcome != outcome:
        provenance["notes"].append("outcome_rederived_from_authoritative_statuses")

    return {
        "execution_status": execution_status,
        "answer_status": answer_status,
        "grounding_status": grounding_status,
        "outcome": outcome,
        "status_provenance": provenance,
    }


def apply_response_contract(
    payload: Dict[str, Any],
    *,
    execution_status: Optional[str] = None,
    answer_status: Optional[str] = None,
    grounding_status: Optional[str] = None,
    provenance_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Attach authoritative statuses to an engine/API response in place."""
    if execution_status:
        payload["execution_status"] = execution_status
    if answer_status:
        payload["answer_status"] = answer_status
    if grounding_status:
        payload["grounding_status"] = grounding_status
    state = response_state(payload)
    if provenance_note:
        state["status_provenance"]["notes"].append(provenance_note)
    payload.update({
        "execution_status": state["execution_status"],
        "answer_status": state["answer_status"],
        "grounding_status": state["grounding_status"],
        "outcome": state["outcome"],
    })
    payload.setdefault("metadata", {})
    payload["metadata"].update({
        "execution_status": state["execution_status"],
        "answer_status": state["answer_status"],
        "grounding_status": state["grounding_status"],
        "outcome": state["outcome"],
        "status_provenance": state["status_provenance"],
    })
    report = payload.get("structured_report")
    if isinstance(report, dict):
        report.update({
            "execution_status": state["execution_status"],
            "answer_status": state["answer_status"],
            "grounding_status": state["grounding_status"],
            "outcome": state["outcome"],
        })
    return payload


def classify_outcome(
    payload: Optional[Dict[str, Any]],
    *,
    http_status: Optional[int] = None,
    error: Any = None,
) -> str:
    """Return the compatibility outcome derived from the full response state."""
    return response_state(payload, http_status=http_status, error=error)["outcome"]


def is_successful_execution(
    payload: Optional[Dict[str, Any]],
    *,
    http_status: Optional[int] = None,
) -> bool:
    return response_state(payload, http_status=http_status)["execution_status"] == "SUCCEEDED"


def is_answer_attempted(
    payload: Optional[Dict[str, Any]],
    *,
    http_status: Optional[int] = None,
) -> bool:
    return response_state(payload, http_status=http_status)["answer_status"] != "NOT_REQUESTED"


def is_abstention(
    payload: Optional[Dict[str, Any]],
    *,
    http_status: Optional[int] = None,
) -> bool:
    """Count only normal, successful answer-task refusals as abstentions."""
    state = response_state(payload, http_status=http_status)
    return (
        state["execution_status"] == "SUCCEEDED"
        and state["answer_status"] == "ABSTAINED"
        and state["grounding_status"] != "FAILED"
    )
