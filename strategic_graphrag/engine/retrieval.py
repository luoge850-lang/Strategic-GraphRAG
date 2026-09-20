"""Retrieval routing and graph propagation for comparable RAG baselines."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional


RETRIEVAL_MODES = {"vector", "graph", "hybrid", "hybrid_temporal"}
EXPLICIT_RELATION_TOKENS = {
    "CAUSES", "TRIGGERS", "AMPLIFIES", "INCREASES", "DECREASES",
    "IMPLEMENTS", "MITIGATES", "CONSTRAINS", "EXPOSED_TO",
    "AFFECTS_SEGMENT", "CONSTRAINS_MARKET", "EXPOSED_THROUGH", "IMPACTS",
    "EXECUTES", "ADDRESSES", "OPERATES_IN", "PRODUCES", "COMPETES_WITH",
    "DEPENDS_ON", "REGULATED_BY", "SUPPLIES_TO", "REPORTS_METRIC",
}


@dataclass(frozen=True)
class RetrievalDecision:
    mode: str
    confidence: float
    reason: str
    use_vector: bool
    use_graph: bool
    use_temporal: bool
    use_ppr: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryRouter:
    """Deterministic, inspectable router shared by API and evaluation runs."""

    TEMPORAL = re.compile(
        r"\b(compare|change|changed|trend|evol|over time|year.over.year|yoy|between|since|from 20\d{2}|20\d{2}.+20\d{2})\b",
        re.IGNORECASE,
    )
    MULTIHOP = re.compile(
        r"\b(how|why|impact|affect|cause|lead to|mechanism|path|propagat|through|risk|mitigat|supply chain)\b",
        re.IGNORECASE,
    )
    FACTOID = re.compile(
        r"\b(what (?:was|is)|how much|reported|value|amount|page|cite|according to)\b",
        re.IGNORECASE,
    )

    @classmethod
    def route(
        cls,
        query: str,
        requested_mode: str = "auto",
        target_metric: Optional[str] = None,
    ) -> RetrievalDecision:
        requested = str(requested_mode or "auto").strip().lower().replace("+", "_")
        aliases = {"hybridtemporal": "hybrid_temporal", "temporal": "hybrid_temporal"}
        requested = aliases.get(requested, requested)
        if requested in RETRIEVAL_MODES:
            return cls._decision(requested, 1.0, "explicit_api_selection")

        text = str(query or "")
        temporal = bool(cls.TEMPORAL.search(text)) or len(re.findall(r"20\d{2}", text)) >= 2
        multihop = bool(cls.MULTIHOP.search(text))
        factoid = bool(cls.FACTOID.search(text))
        explicit_relation = any(
            re.search(rf"\b{re.escape(relation)}\b", text.upper())
            for relation in EXPLICIT_RELATION_TOKENS
        )
        explicit_entities = [
            token
            for token in re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", text)
            if token not in EXPLICIT_RELATION_TOKENS
        ]

        if explicit_relation and len(explicit_entities) >= 2 and not temporal:
            return cls._decision(
                "graph", 0.96,
                "explicit ontology relation and endpoints are best resolved from the evidence graph",
            )

        if temporal:
            return cls._decision(
                "hybrid_temporal", 0.92,
                "cross-period language requires vector recall, graph paths, and valid-time filtering",
            )
        if multihop:
            return cls._decision(
                "hybrid", 0.86,
                "causal or relationship question benefits from semantic anchors and graph expansion",
            )
        if target_metric:
            return cls._decision(
                "graph", 0.84,
                "canonical metric can be resolved through FinancialObservation and EvidenceClaim nodes",
            )
        if factoid:
            return cls._decision(
                "vector", 0.76,
                "local factoid does not require graph traversal",
            )
        return cls._decision(
            "hybrid", 0.60,
            "ambiguous analytical query uses the conservative combined baseline",
        )

    @staticmethod
    def _decision(mode: str, confidence: float, reason: str) -> RetrievalDecision:
        return RetrievalDecision(
            mode=mode,
            confidence=confidence,
            reason=reason,
            use_vector=mode in {"vector", "hybrid", "hybrid_temporal"},
            use_graph=mode in {"graph", "hybrid", "hybrid_temporal"},
            use_temporal=mode == "hybrid_temporal",
            use_ppr=mode in {"hybrid", "hybrid_temporal"},
        )


def personalized_pagerank(
    edges: Iterable[tuple[str, str]],
    seeds: Iterable[str],
    *,
    restart_probability: float = 0.20,
    iterations: int = 30,
) -> Dict[str, float]:
    """Dependency-free PPR over the strict evidence graph.

    Edges are treated as bidirectional for entity discovery. Direction is
    enforced later by the causal path query, preventing PPR from inventing a
    reverse causal claim while still allowing it to find bridge entities.
    """
    adjacency: Dict[str, set[str]] = {}
    for source, target in edges:
        if not source or not target or source == target:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    seed_nodes = [seed for seed in dict.fromkeys(seeds) if seed in adjacency]
    if not adjacency or not seed_nodes:
        return {}
    teleport = {node: 1.0 / len(seed_nodes) for node in seed_nodes}
    scores = {node: teleport.get(node, 0.0) for node in adjacency}
    for _ in range(max(iterations, 1)):
        updated = {
            node: restart_probability * teleport.get(node, 0.0)
            for node in adjacency
        }
        dangling = 0.0
        for node, score in scores.items():
            neighbours = adjacency[node]
            if not neighbours:
                dangling += score
                continue
            contribution = (1.0 - restart_probability) * score / len(neighbours)
            for neighbour in neighbours:
                updated[neighbour] += contribution
        if dangling:
            for seed, weight in teleport.items():
                updated[seed] += (1.0 - restart_probability) * dangling * weight
        scores = updated
    total = sum(scores.values()) or 1.0
    return {node: score / total for node, score in scores.items()}


class Neo4jPPRRetriever:
    """Load only strict EvidenceClaim edges and rank bridge entities with PPR."""

    def __init__(
        self,
        driver,
        *,
        cache_ttl_seconds: float = 300.0,
        cache_max_entries: int = 64,
    ):
        self.driver = driver
        # PPR reads the frozen strict graph and is safe to cache within one
        # API process. This bounded TTL cache reduces repeated Graph/Hybrid
        # latency without hiding graph changes forever.
        self.cache_ttl_seconds = max(float(cache_ttl_seconds), 0.0)
        self.cache_max_entries = max(int(cache_max_entries), 1)
        self._cache: OrderedDict[tuple[Any, ...], tuple[float, List[Dict[str, Any]]]] = OrderedDict()
        self._cache_lock = threading.RLock()

    @staticmethod
    def _cache_key(
        anchors: List[str],
        source_filing: Optional[str],
        year_start: Optional[int],
        year_end: Optional[int],
        limit: int,
    ) -> tuple[Any, ...]:
        return (
            tuple(sorted({str(anchor).casefold() for anchor in anchors if str(anchor).strip()})),
            source_filing,
            year_start,
            year_end,
            limit,
        )

    def _cached(self, key: tuple[Any, ...]) -> Optional[List[Dict[str, Any]]]:
        if self.cache_ttl_seconds <= 0:
            return None
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None
            created_at, value = entry
            if time.monotonic() - created_at >= self.cache_ttl_seconds:
                self._cache.pop(key, None)
                return None
            self._cache.move_to_end(key)
            return [dict(item) for item in value]

    def _store(self, key: tuple[Any, ...], value: List[Dict[str, Any]]) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        with self._cache_lock:
            self._cache[key] = (time.monotonic(), [dict(item) for item in value])
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_max_entries:
                self._cache.popitem(last=False)

    def rank(
        self,
        anchors: List[str],
        *,
        source_filing: Optional[str] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        limit: int = 12,
    ) -> List[Dict[str, Any]]:
        if not anchors:
            return []
        cache_key = self._cache_key(
            anchors, source_filing, year_start, year_end, max(limit, 0)
        )
        cached = self._cached(cache_key)
        if cached is not None:
            return cached
        with self.driver.session() as session:
            rows = list(session.run(
                """
                MATCH (source)-[rel]->(target)
                MATCH (claim:EvidenceClaim {id: rel.evidence_id})
                WHERE claim.verification_status='VERBATIM'
                  AND ($source_filing IS NULL OR coalesce(rel.source_filing, rel.filing, '')=$source_filing)
                  AND ($year_start IS NULL OR rel.year >= $year_start)
                  AND ($year_end IS NULL OR rel.year <= $year_end)
                RETURN coalesce(source.id, source.name) AS source_id,
                       coalesce(target.id, target.name) AS target_id
                """,
                source_filing=source_filing,
                year_start=year_start,
                year_end=year_end,
            ))
        edges = [(str(row["source_id"]), str(row["target_id"])) for row in rows]
        node_ids = {value for edge in edges for value in edge}
        normalized = {node.casefold(): node for node in node_ids}
        resolved: List[str] = []
        for anchor in anchors:
            key = str(anchor).casefold()
            if key in normalized:
                resolved.append(normalized[key])
                continue
            resolved.extend(
                node for node in node_ids
                if key in node.casefold() or node.casefold() in key
            )
        scores = personalized_pagerank(edges, resolved)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        ranked = [
            {"entity_id": entity_id, "ppr_score": round(score, 8)}
            for entity_id, score in ranked[: max(limit, 0)]
        ]
        self._store(cache_key, ranked)
        return ranked
