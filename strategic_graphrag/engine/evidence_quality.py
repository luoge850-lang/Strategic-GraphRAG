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

DIRECT_STRUCTURAL_RELATIONS = {"PRODUCES", "OPERATES_IN"}


def _explicit_query_relations(query: str) -> Set[str]:
    """Return ontology relations literally named by a debug/Gold query."""
    try:
        from ..ontology.relation_inference import VALID_RELATIONS
    except ImportError:
        VALID_RELATIONS = set()
    query_upper = str(query or "").upper().replace("-", "_")
    return {
        relation
        for relation in VALID_RELATIONS
        if re.search(rf"\b{re.escape(relation)}\b", query_upper)
    }


def _node_aliases(node: Any) -> Set[str]:
    """Extract useful surface aliases from a canonical graph node ID."""
    raw_value = str(node or "")
    raw = raw_value.replace("_", " ")
    aliases = {
        token.lower()
        for token in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", raw)
        if len(token) >= 3
    }
    aliases.update(
        token.lower()
        for token in re.split(r"[_\s]+", raw_value)
        if len(token) >= 3
    )
    normalized = re.sub(r"\s+", " ", raw).strip().lower()
    if normalized:
        aliases.add(normalized)
    return aliases


def _relation_evidence_support(
    relation: str,
    source: Any,
    target: Any,
    evidence: Any,
) -> float:
    """Score lexical support for a structural edge without inferring causality.

    This is intentionally conservative.  Collection membership (``includes``),
    hosting/runtime language (``runs on``), and composition (``based on``) do
    not establish that the company produces the queried target.
    """
    text = _normalise(evidence)
    if not text:
        return 0.0
    aliases = sorted(_node_aliases(target), key=len, reverse=True)
    target_match = next(
        (match for alias in aliases if (match := re.search(rf"\b{re.escape(alias)}\b", text))),
        None,
    )
    if target_match is None:
        compact = re.sub(r"[^a-z0-9]", "", str(target or "").lower())
        # Some legacy IDs concatenate several words (for example
        # PROFESSIONALVIZMARKET).  A conservative prefix can still bind the
        # evidence to the intended entity when the filing uses its natural
        # surface name; never use a one- or two-character prefix.
        for width in range(min(len(compact), 14), 3, -1):
            alias = compact[:width]
            target_match = re.search(rf"\b{re.escape(alias)}\b", text)
            if target_match:
                break
    if target_match is None:
        return 0.0

    relation = str(relation or "").upper()
    if relation == "REPORTS_METRIC":
        # A verified table/numeric edge is already the structural proof of a
        # disclosure.  Unlike PRODUCES/OPERATES_IN it does not require a
        # causal verb in the row text; requiring one incorrectly rejects
        # valid evidence such as ``Total revenue $130,497 ...``.
        return 1.0 if re.search(r"(?<!\d)[\$€£]?\d[\d,]*(?:\.\d+)?", text) else 0.0
    if relation == "PRODUCES":
        predicate = re.compile(
            r"\b(?:built|develop\w*|launch\w*|introduc\w*|offer\w*|"
                r"provid\w*|manufactur\w*|produc\w*)\b"
        )
        blockers = re.compile(
                r"\b(?:includes?|based on|built on|runs? on|running|within|derived from)\b"
        )
        source_aliases = [
            alias
            for alias in _node_aliases(source)
            if alias not in {"company", "corporation", "inc", "corp"}
        ]
        source_match = next(
            (
                re.search(rf"\b{re.escape(alias)}\b", text)
                for alias in sorted(source_aliases, key=len, reverse=True)
            ),
            None,
        )
        source_position = source_match.start() if source_match else -1
        source_end = source_match.end() if source_match else -1

        def _source_before(position: int) -> bool:
            sentence_start = max(
                text.rfind(".", 0, position),
                text.rfind(";", 0, position),
                text.rfind(":", 0, position),
            ) + 1
            subject = text[sentence_start:position]
            return any(
                re.search(rf"\b{re.escape(alias)}\b", subject)
                for alias in sorted(source_aliases, key=len, reverse=True)
            )

        for match in predicate.finditer(text):
            reporting_span = text[source_end:match.start()] if source_position >= 0 else ""
            # ``NVIDIA said another company produces Jetson`` contains the
            # source before the predicate, but NVIDIA is the reporting speaker
            # rather than the grammatical producer. Do not treat source-first
            # order as subject binding when a reporting clause introduces a
            # different producer.
            if re.search(
                r"\b(?:said|reported|stated|noted|disclosed|announced|believes|expects)\b",
                reporting_span,
            ) and re.search(
                r"\b(?:another|other|a|the)\s+(?:company|manufacturer|vendor|"
                r"competitor|party|supplier)\b",
                reporting_span,
            ):
                continue
            between_target = text[match.end():target_match.end()]
            enumerated_source = (
                "including" in between_target
                and _source_before(target_match.end())
            )
            # Passive voice binds the producer after the predicate:
            # ``DRIVE is produced by NVIDIA``.
            if match.start() > target_match.start():
                sentence_end = min(
                    item for item in (
                        text.find(".", match.end()),
                        text.find(";", match.end()),
                    ) if item >= 0
                ) if any(
                    item >= 0 for item in (
                        text.find(".", match.end()),
                        text.find(";", match.end()),
                    )
                ) else len(text)
                passive_tail = text[match.end():sentence_end]
                if re.search(
                    r"\bby\s+(?:the\s+)?(?:" + "|".join(
                        re.escape(alias) for alias in sorted(source_aliases, key=len, reverse=True)
                    ) + r")\b",
                    passive_tail,
                ):
                    return 0.5 if re.search(r"\b(?:may|might|could|expected to)\b", passive_tail) else 1.0
                continue
            if not _source_before(match.start()) and not enumerated_source:
                # A target and predicate can be real filing evidence while
                # still describing another producer.  Do not bind the edge to
                # the queried source merely because that source is mentioned
                # later in the sentence.
                continue
            between = text[match.end():target_match.start()]
            blocker = blockers.search(between)
            # A sentence can first describe how a stack runs on GPUs and then
            # enumerate products introduced by the company ("built ... run on
            # GPUs ..., including DRIVE").  The runtime clause must not block
            # the later enumerated target.  A blocker after the last
            # enumeration marker, however, still invalidates the relation.
            if blocker is not None:
                last_including = between.lower().rfind("including")
                if last_including < 0 or blocker.start() >= last_including:
                    continue
            if not blocker or blocker.start() < between.lower().rfind("including"):
                certainty_text = text[max(0, match.start() - 80):target_match.end()]
                return 0.5 if re.search(
                    r"\b(?:may|might|could|expected to)\b", certainty_text
                ) else 1.0
        return 0.0

    if relation == "OPERATES_IN":
        direct = re.search(
            r"\b(?:serve|serves|operate|operates|participate|compete)\w*\b"
            r".{0,120}\b(?:market|business|segment)\b",
            text,
        )
        if direct and direct.start() <= target_match.start() + 120:
            return 1.0
        if re.search(r"\b(?:market|platform|solutions?|business|segment|licenses?)\b", text):
            return 0.5
    return 0.0


def _relation_evidence_priority(relation: Any, evidence: Any) -> float:
    """Prefer stronger direct predicates when variants have equal support."""
    if str(relation or "").upper() != "PRODUCES":
        return 0.0
    text = _normalise(evidence)
    if re.search(r"\b(?:built|develop\w*|launch\w*|introduc\w*|manufactur\w*|produc\w*)\b", text):
        return 2.0
    if re.search(r"\b(?:offer\w*|provid\w*)\b", text):
        return 1.0
    return 0.0


def _path_evidence_options(path: Any) -> List[List[Dict[str, Any]]]:
    """Return primary and alternative provenance options for each hop."""
    options: List[List[Dict[str, Any]]] = []
    for index, _ in enumerate(getattr(path, "relationships", []) or []):
        primary = {
            "evidence": (getattr(path, "evidence", []) or [])[index]
            if index < len(getattr(path, "evidence", []) or []) else "",
            "page": (getattr(path, "pages", []) or [])[index]
            if index < len(getattr(path, "pages", []) or []) else 0,
            "year": (getattr(path, "years", []) or [])[index]
            if index < len(getattr(path, "years", []) or []) else 0,
            "evidence_id": (getattr(path, "evidence_ids", []) or [])[index]
            if index < len(getattr(path, "evidence_ids", []) or []) else "",
            "filing": (getattr(path, "filings", []) or [])[index]
            if index < len(getattr(path, "filings", []) or []) else "",
        }
        hop_options = [primary]
        variants = (getattr(path, "evidence_variants", []) or [])
        if index < len(variants):
            hop_options.extend(
                variant for variant in variants[index]
                if isinstance(variant, dict)
            )
        unique: List[Dict[str, Any]] = []
        for option in hop_options:
            if option not in unique:
                unique.append(option)
        options.append(unique)
    return options


def _query_endpoint_alignment(path: Any, query: str) -> float:
    """Check whether a direct relation path contains the named endpoints."""
    explicit_tokens = [
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", str(query or ""))
        if token not in _explicit_query_relations(query)
    ]
    # A query written with canonical IDs often contains both a short company
    # name (NVIDIA) and its full ID (NVIDIA_CORPORATION).  The short token is
    # redundant; keeping it would make exact endpoint matching impossible for
    # the canonical company node.  Do not apply substring matching to product
    # IDs: GPU must not silently match A800_GPU or GEFORCE_GPU.
    explicit_tokens = [
        token
        for token in explicit_tokens
        if not any(
            token != other
            and token in other.split("_")
            for other in explicit_tokens
        )
    ]
    if not explicit_tokens:
        return 0.0
    path_nodes = {
        str(node or "").upper().replace("-", "_").strip()
        for node in getattr(path, "nodes", []) or []
    }
    matched = 0
    for token in explicit_tokens:
        if token.upper().replace("-", "_").strip() in path_nodes:
            matched += 1
    return round(matched / len(explicit_tokens), 4)


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


def score_path_directness(
    path: Any,
    query: str,
    intent: str,
    preferred_relations: Iterable[str] = (),
) -> Dict[str, Any]:
    """Score whether a path directly answers the question, not just resembles it."""
    text = _path_text(path)
    terms = query_terms(query)
    concepts = query_concepts(query)
    relations = {str(value).upper() for value in getattr(path, "relationships", []) or []}
    explicit_relations = _explicit_query_relations(query)
    preferred = explicit_relations or {
        str(value).upper() for value in preferred_relations if value
    } or RELATION_TERMS.get(intent, RELATION_TERMS["CAUSAL_CHAIN"])

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

    relation_support = 0.0
    best_variants: List[Dict[str, Any]] = []
    options_by_hop = _path_evidence_options(path)
    for index, relation in enumerate(getattr(path, "relationships", []) or []):
        hop_options = options_by_hop[index] if index < len(options_by_hop) else []
        scored_options = [
            (
                _relation_evidence_support(
                    relation,
                    (getattr(path, "nodes", []) or [])[index]
                    if index < len(getattr(path, "nodes", []) or []) else "",
                    (getattr(path, "nodes", []) or [])[index + 1]
                    if index + 1 < len(getattr(path, "nodes", []) or []) else "",
                    option.get("evidence", ""),
                ),
                _relation_evidence_priority(relation, option.get("evidence", "")),
                option,
            )
            for option in hop_options
        ]
        if scored_options:
            best_score, _, best_option = max(scored_options, key=lambda item: (item[0], item[1]))
            relation_support = max(relation_support, best_score)
            best_variants.append(best_option)

    score = max(
        0.0,
        min(1.0, 0.40 * concept_coverage + 0.35 * relation_alignment + 0.25 * term_overlap - tangent_penalty),
    )
    explicit_structural = explicit_relations.intersection(DIRECT_STRUCTURAL_RELATIONS)
    endpoint_alignment = _query_endpoint_alignment(path, query)
    if explicit_structural and relation_alignment and endpoint_alignment < 1.0:
        role = "BACKGROUND_CONTEXT"
    elif explicit_structural and relation_alignment and relation_support <= 0.0:
        role = "BACKGROUND_CONTEXT"
    elif explicit_structural and relation_alignment and relation_support >= 1.0 and path.total_hops == 1:
        role = "ANSWER_CRITICAL"
    elif explicit_structural and relation_alignment and relation_support >= 0.5:
        role = "MECHANISM_SUPPORT"
    elif relation_alignment >= 1.0 and concept_coverage >= 0.75 and score >= 0.68:
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
        "relation_evidence_support": round(relation_support, 4),
        "endpoint_alignment": endpoint_alignment,
        "tangent_penalty": round(tangent_penalty, 4),
        "best_variants": best_variants,
    }


def apply_directness_ranking(
    paths: Sequence[Any],
    query: str,
    intent: str,
    preferred_relations: Iterable[str] = (),
) -> List[Any]:
    for path in paths:
        diagnostic = score_path_directness(path, query, intent, preferred_relations)
        setattr(path, "evidence_role", diagnostic["role"])
        breakdown = getattr(path, "score_breakdown", {})
        breakdown.update({
            "directness": diagnostic["score"],
            "query_term_overlap": diagnostic["term_overlap"],
            "query_concept_coverage": diagnostic["concept_coverage"],
            "relation_alignment": diagnostic["relation_alignment"],
            "relation_evidence_support": diagnostic["relation_evidence_support"],
            "endpoint_alignment": diagnostic["endpoint_alignment"],
            "tangent_penalty": diagnostic["tangent_penalty"],
        })
        path.score_breakdown = breakdown
        if diagnostic["role"] in {"ANSWER_CRITICAL", "MECHANISM_SUPPORT"}:
            best_variants = diagnostic.get("best_variants") or []
            for index, variant in enumerate(best_variants):
                if not isinstance(variant, dict):
                    continue
                if index < len(getattr(path, "evidence", []) or []):
                    path.evidence[index] = variant.get("evidence", path.evidence[index])
                if index < len(getattr(path, "pages", []) or []):
                    path.pages[index] = variant.get("page", path.pages[index])
                if index < len(getattr(path, "years", []) or []):
                    path.years[index] = variant.get("year", path.years[index])
                if index < len(getattr(path, "evidence_ids", []) or []):
                    path.evidence_ids[index] = variant.get("evidence_id", path.evidence_ids[index])
                if index < len(getattr(path, "filings", []) or []):
                    path.filings[index] = variant.get("filing", path.filings[index])
        path.aggregate_score = round(0.65 * float(getattr(path, "aggregate_score", 0.0)) + 0.35 * diagnostic["score"], 4)
    return list(paths)


def select_answer_evidence(
    paths: Sequence[Any],
    limit: int,
    required_years: Iterable[int] = (),
) -> List[Any]:
    """Fill answer context by proof strength while preserving required years.

    Ranking a bounded candidate pool by evidence role is safer than ranking
    only the first Top-K returned by Neo4j.  Temporal questions add one more
    constraint: the final context must retain the best available path for
    every requested year, otherwise a later truncation can silently turn a
    complete comparison into an incomplete one.
    """
    ordered: List[Any] = []
    for role in ("ANSWER_CRITICAL", "MECHANISM_SUPPORT", "BACKGROUND_CONTEXT"):
        ordered.extend(
            path
            for path in paths
            if getattr(path, "evidence_role", "BACKGROUND_CONTEXT") == role
        )

    selected: List[Any] = []
    selected_ids: Set[int] = set()
    for year in dict.fromkeys(int(value) for value in required_years if value is not None):
        matches = [path for path in ordered if year in (getattr(path, "years", []) or [])]
        if not matches:
            continue
        best = matches[0]
        marker = id(best)
        if marker not in selected_ids:
            selected.append(best)
            selected_ids.add(marker)

    for path in ordered:
        if len(selected) >= max(limit, 0):
            break
        if id(path) not in selected_ids:
            selected.append(path)
            selected_ids.add(id(path))
    return selected[: max(limit, 0)]


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
    r"\b(?:filings?|corpus).{0,80}(?:do(?:es)? not|did not|no |cannot|could not|without)"
    r"\s+(?:contain|disclose|provide|include|document|show|establish)\b|"
    r"\b(?:knowledge graph|indexed graph).{0,80}(?:do(?:es)? not|did not|no |cannot|could not|without)"
    r"\s+(?:contain|disclose|provide|include|document|show|establish)\b|"
    r"\bno (?:actual|realized|material|direct) (?:impact|effect|loss|evidence)\b",
    re.IGNORECASE,
)


def contains_absence_claim(text: str) -> bool:
    return bool(ABSENCE_PATTERNS.search(str(text or "")))
