"""Deterministic identifiers and corpus identity for evidence-grounded GraphRAG."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass


CLAIM_ID_VERSION = "v2"


def normalize_evidence(text: str) -> str:
    """Normalize PDF whitespace without changing lexical evidence content."""
    value = unicodedata.normalize("NFKC", str(text or ""))
    value = value.replace("\u00ad", "")
    return re.sub(r"\s+", " ", value).strip()


def _digest(kind: str, *parts: object) -> str:
    payload = "\x1f".join(str(part or "").strip() for part in parts)
    return f"{kind}_{CLAIM_ID_VERSION}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


@dataclass(frozen=True)
class EvidenceIdentity:
    relation_id: str
    sentence_id: str
    claim_id: str


def evidence_identity(
    *,
    document_sha256: str,
    filename: str,
    page: int,
    evidence_text: str,
    source_id: str,
    relation_type: str,
    target_id: str,
) -> EvidenceIdentity:
    """Return stable IDs for one exact, evidence-backed graph assertion.

    ``document_sha256`` is authoritative. ``filename`` is retained only as an
    explicit fallback for dry/unit contexts where no source file exists.
    Chunk IDs and character offsets are deliberately excluded: they describe
    a parser run and may change while the source assertion remains identical.
    """
    document_key = str(document_sha256 or "").strip().lower() or f"filename:{filename}"
    quote = normalize_evidence(evidence_text)
    claim_id = _digest(
        "claim",
        document_key,
        int(page),
        quote,
        str(source_id).upper(),
        str(relation_type).upper(),
        str(target_id).upper(),
    )
    # The current graph models one provenance Sentence node per claim. Bind
    # its ID to the stable claim so one verbatim row supporting multiple
    # triples does not violate the Sentence uniqueness constraint.
    sentence_id = _digest("sent", claim_id)
    relation_id = _digest("rel", claim_id)
    return EvidenceIdentity(relation_id, sentence_id, claim_id)
