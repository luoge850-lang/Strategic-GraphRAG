"""Evidence prioritisation and absence-claim audit helpers.

The functions in this module are deterministic by design.  They do not turn
semantic similarity into causality and they never translate a bounded search
miss into a claim that a filing contains no evidence.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Set


STOPWORDS = {
    "a", "about", "actual", "an", "and", "are", "as", "at", "be", "by",
    "did", "do", "does", "document", "documents", "filing", "filings", "for",
    "from", "how", "in", "is", "it", "nvidia", "of", "on", "or", "the",
    "their", "to", "was", "were", "what", "which", "with",
}

CONCEPT_GROUPS = {
    "supply_chain": (
        "supply chain", "supplier", "foundry", "manufactur", "wafer", "packaging",
        "capacity constraint", "component", "inventory",
    ),
    "mitigation": (
        "mitigat", "manage risk", "risk management", "diversif", "alternative source",
        "second source", "multiple supplier", "reduce risk", "capacity commitment",
        "purchase commitment", "safety stock", "resilien",
    ),
    "export_control": (
        "export control", "export restriction", "license requirement", "licensing requirement",
        "trade restriction", "entity list", "china license",
    ),
    "financial_impact": (
        "revenue", "sales", "gross margin", "operating income", "profit", "earnings",
        "results of operations", "financial results", "charge",
    ),
    "realized_impact": (
        "adversely affected", "material impact", "materially affected", "reduced revenue",
        "lost revenue", "revenue loss", "actual impact", "incurred", "charge",
    ),
    "climate": ("climate", "weather", "carbon", "greenhouse gas"),
}

RELATION_TERMS = {
    "MITIGATION_STRATEGY": {"MITIGATES", "IMPLEMENTS"},
    "FINANCIAL_IMPACT": {"CAUSES", "INCREASES", "DECREASES", "CONSTRAINS", "TRIGGERS", "REPORTS_METRIC"},
    "REGULATORY_ANALYSIS": {"CONSTRAINS", "CAUSES", "DECREASES", "INCREASES"},
    "SUPPLY_CHAIN": {"DEPENDS_ON", "EXPOSED_TO", "CAUSES", "TRIGGERS", "MITIGATES"},
    "CAUSAL_CHAIN": {"CAUSES", "TRIGGERS", "AMPLIFIES", "AGGRAVATES", "DECREASES", "INCREASES", "MITIGATES"},
}


def _normalise(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").lower()).strip()


def query_terms(query: str) -> Set[str]:
    return {
        token
        for token in re.findall(r"[a-z][a-z0-9-]{2,}", _normalise(query))
        if token not in STOPWORDS
    }


def query_concepts(query: str) -> Set[str]:
    lowered = _normalise(query)
    return {
        name
        for name, phrases in CONCEPT_GROUPS.items()
        if any(phrase in lowered for phrase in phrases)
    }


def _path_text(path: Any) -> str:
    values: List[str] = []
    for field in ("nodes", "relationships", "evidence"):
        values.extend(str(value) for value in getattr(path, field, []) or [])
    return _normalise(" ".join(values).replace("_", " "))


def score_path_directness(path: Any, query: str, intent: str) -> Dict[str, Any]:
    """Score whether a path directly answers the question, not just resembles it."""
    text = _path_text(path)
    terms = query_terms(query)
    concepts = query_concepts(query)
    relations = {str(value).upper() for value in getattr(path, "relationships", []) or []}
    preferred = RELATION_TERMS.get(intent, RELATION_TERMS["CAUSAL_CHAIN"])

    term_overlap = sum(1 for term in terms if term in text) / max(len(terms), 1)
    concept_hits = 0
    for concept in concepts:
        if any(phrase in text for phrase in CONCEPT_GROUPS[concept]):
            concept_hits += 1
    concept_coverage = concept_hits / max(len(concepts), 1)
    relation_alignment = 1.0 if relations.intersection(preferred) else 0.0

    # A climate branch is relevant only when the user actually asked about climate.
    tangent_penalty = 0.0
    if "climate" not in concepts and any(phrase in text for phrase in CONCEPT_GROUPS["climate"]):
        tangent_penalty = 0.35

    score = max(
        0.0,
        min(1.0, 0.40 * concept_coverage + 0.35 * relation_alignment + 0.25 * term_overlap - tangent_penalty),
    )
    if relation_alignment >= 1.0 and concept_coverage >= 0.75 and score >= 0.68:
        role = "ANSWER_CRITICAL"
    elif score >= 0.38:
        role = "MECHANISM_SUPPORT"
    else:
        role = "BACKGROUND_CONTEXT"
    return {
        "score": round(score, 4),
        "role": role,
        "term_overlap": round(term_overlap, 4),
        "concept_coverage": round(concept_coverage, 4),
        "relation_alignment": round(relation_alignment, 4),
        "tangent_penalty": round(tangent_penalty, 4),
    }


def apply_directness_ranking(paths: Sequence[Any], query: str, intent: str) -> List[Any]:
    for path in paths:
        diagnostic = score_path_directness(path, query, intent)
        setattr(path, "evidence_role", diagnostic["role"])
        breakdown = getattr(path, "score_breakdown", {})
        breakdown.update({
            "directness": diagnostic["score"],
            "query_term_overlap": diagnostic["term_overlap"],
            "query_concept_coverage": diagnostic["concept_coverage"],
            "relation_alignment": diagnostic["relation_alignment"],
            "tangent_penalty": diagnostic["tangent_penalty"],
        })
        path.score_breakdown = breakdown
        path.aggregate_score = round(0.65 * float(getattr(path, "aggregate_score", 0.0)) + 0.35 * diagnostic["score"], 4)
    return list(paths)


def select_answer_evidence(paths: Sequence[Any], limit: int) -> List[Any]:
    """Fill answer context by proof strength: critical, mechanism, then background."""
    ordered: List[Any] = []
    for role in ("ANSWER_CRITICAL", "MECHANISM_SUPPORT", "BACKGROUND_CONTEXT"):
        ordered.extend(path for path in paths if getattr(path, "evidence_role", "BACKGROUND_CONTEXT") == role)
    return ordered[: max(limit, 0)]


def semantic_scope(path: Any) -> str:
    text = _normalise(" ".join(getattr(path, "evidence", []) or []))
    markers = (" because ", " which may ", " resulting in ", " result in ", " lead to ", " through ", " thereby ")
    return "EMBEDDED_MECHANISM" if any(marker in f" {text} " for marker in markers) else "ATOMIC_RELATION"


def audit_documents(
    query: str,
    documents: Iterable[Dict[str, Any]],
    *,
    semantic_hits: Iterable[Dict[str, Any]] = (),
    scope: str = "all indexed filings",
    max_matches: int = 12,
) -> Dict[str, Any]:
    """Audit every indexed chunk before allowing a scoped absence statement."""
    docs = list(documents)
    concepts = query_concepts(query)
    terms = query_terms(query)
    matches: List[Dict[str, Any]] = []
    total_matches = 0

    for item in docs:
        text = _normalise(item.get("document"))
        concept_hits = [
            concept for concept in concepts
            if any(phrase in text for phrase in CONCEPT_GROUPS[concept])
        ]
        term_hits = [term for term in terms if term in text]
        required_concepts = 1 if len(concepts) <= 1 else 2
        if len(concept_hits) >= required_concepts or (not concepts and len(term_hits) >= 2):
            total_matches += 1
            metadata = item.get("metadata") or {}
            if len(matches) < max_matches:
                matches.append({
                    "source_filing": metadata.get("source_filing") or metadata.get("doc_id"),
                    "page": metadata.get("page"),
                    "chunk_id": metadata.get("chunk_id"),
                    "matched_concepts": concept_hits,
                    "excerpt": str(item.get("document") or "")[:360],
                    "match_type": "CORPUS_LEXICAL",
                })

    if not matches:
        for hit in semantic_hits:
            distance = hit.get("distance")
            if not isinstance(distance, (int, float)) or distance > 0.45:
                continue
            metadata = hit.get("metadata") or {}
            matches.append({
                "source_filing": metadata.get("source_filing") or metadata.get("doc_id"),
                "page": metadata.get("page"),
                "chunk_id": metadata.get("chunk_id"),
                "matched_concepts": [],
                "excerpt": str(hit.get("document") or "")[:360],
                "match_type": "CORPUS_SEMANTIC",
                "distance": round(float(distance), 4),
            })
            total_matches += 1
            if len(matches) >= max_matches:
                break

    if not docs:
        status = "AUDIT_UNAVAILABLE"
        safe = "The indexed corpus was unavailable, so no absence conclusion is permitted."
    elif matches:
        status = "POTENTIAL_EVIDENCE_FOUND"
        safe = "Potentially relevant corpus evidence was found; an absence conclusion is not permitted."
    else:
        status = "NO_MATCH_AFTER_INDEXED_CORPUS_AUDIT"
        safe = (
            f"No matching passage was found after scanning {len(docs)} indexed chunks in {scope}; "
            "this is a bounded retrieval result, not proof that the underlying filings contain no such evidence."
        )
    return {
        "performed": bool(docs),
        "status": status,
        "scope": scope,
        "chunks_scanned": len(docs),
        "query_concepts": sorted(concepts),
        "matches": matches,
        "total_matches": total_matches,
        "safe_absence_statement": safe,
    }


ABSENCE_PATTERNS = re.compile(
    r"\b(?:filings?|corpus|graph|evidence).{0,80}(?:do(?:es)? not|did not|no |cannot|could not|without)\b|"
    r"\bno (?:actual|realized|material|direct) (?:impact|effect|loss|evidence)\b",
    re.IGNORECASE,
)


def contains_absence_claim(text: str) -> bool:
    return bool(ABSENCE_PATTERNS.search(str(text or "")))
