"""Local, immutable graph index used by isolated acceptance builds.

The production path remains Neo4j-backed.  This adapter deliberately reads a
staging JSON graph and implements the same narrow ``CausalPathFinder`` methods
used by ``GraphRAGEngine.query``.  It makes the end-to-end contract testable
without touching the active Neo4j instance when that dependency is unavailable.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .engine.graph_rag_engine import CausalPath


def _key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _year(value: Any) -> Optional[int]:
    match = re.search(r"20\d{2}", str(value or ""))
    return int(match.group(0)) if match else None


class StagingGraphIndex:
    """Read-only graph index with build identity checks on every edge."""

    def __init__(self, payload: Dict[str, Any]):
        self.build_id = str(payload.get("build_id") or "").strip()
        self.edges: List[Dict[str, Any]] = list(payload.get("edges") or [])
        if not self.build_id:
            raise ValueError("staging graph has no build_id")
        if any(str(edge.get("build_id") or "") != self.build_id for edge in self.edges):
            raise ValueError("staging graph contains an edge from another build")

    @classmethod
    def from_path(cls, path: str | Path) -> "StagingGraphIndex":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _eligible(
        self,
        edge: Dict[str, Any],
        *,
        relation_preference: Optional[Iterable[str]] = None,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        source_filing: Optional[str] = None,
    ) -> bool:
        relation = str(edge.get("relation") or "").upper()
        preferred = {str(value).upper() for value in relation_preference or [] if value}
        if preferred and relation not in preferred:
            return False
        if source_filing and str(edge.get("source_filing") or "") != source_filing:
            return False
        fact_year = _year(edge.get("fact_year") or edge.get("filing_year"))
        if year_start is not None and (fact_year is None or fact_year < year_start):
            return False
        if year_end is not None and (fact_year is None or fact_year > year_end):
            return False
        return True

    def find_text_anchors(self, query: str, limit: int = 8) -> List[str]:
        query_key = _key(query)
        ranked = []
        for edge in self.edges:
            for field in ("source", "target"):
                value = str(edge.get(field) or "")
                value_key = _key(value)
                if value_key and value_key in query_key:
                    ranked.append((len(value_key), value))
        return [value for _, value in sorted(set(ranked), key=lambda item: (-item[0], item[1]))[:limit]]

    def find_vector_evidence_anchors(self, vector_hits: List[Dict[str, Any]], limit: int = 12) -> List[str]:
        pairs = {
            (str((hit.get("metadata") or {}).get("source_filing") or ""), int((hit.get("metadata") or {}).get("page")))
            for hit in vector_hits or []
            if str((hit.get("metadata") or {}).get("page", "")).isdigit()
        }
        values = []
        for edge in self.edges:
            if (str(edge.get("source_filing") or ""), int(edge.get("page") or 0)) in pairs:
                values.extend([str(edge.get("source") or ""), str(edge.get("target") or "")])
        return list(dict.fromkeys(value for value in values if value))[:limit]

    @staticmethod
    def _matches_anchor(edge: Dict[str, Any], anchors: Iterable[str]) -> bool:
        keys = {_key(value) for value in anchors if value}
        return _key(edge.get("source")) in keys or _key(edge.get("target")) in keys

    @staticmethod
    def _to_path(edges: List[Dict[str, Any]], path_id: str) -> CausalPath:
        nodes = [str(edges[0].get("source") or "")]
        for edge in edges:
            nodes.append(str(edge.get("target") or ""))
        return CausalPath(
            path_id=path_id,
            nodes=nodes,
            node_labels=[str(edges[0].get("source_category") or "Entity")]
            + [str(edge.get("target_category") or "Entity") for edge in edges],
            relationships=[str(edge.get("relation") or "") for edge in edges],
            causal_strengths=[str(edge.get("causal_strength") or "DISCLOSED_ONLY") for edge in edges],
            evidence=[str(edge.get("evidence") or "") for edge in edges],
            pages=[int(edge.get("page") or 0) for edge in edges],
            years=[int(_year(edge.get("fact_year") or edge.get("filing_year")) or 0) for edge in edges],
            evidence_ids=[str(edge.get("claim_id") or "") for edge in edges],
            filings=[str(edge.get("source_filing") or "") for edge in edges],
            evidence_build_ids=[str(edge.get("build_id") or "") for edge in edges],
            metric_values=[edge.get("metric_value") for edge in edges],
            metric_units=[str(edge.get("unit") or edge.get("metric_unit") or "") or None for edge in edges],
            causal_forms=[str(edge.get("causal_form") or "FINANCIAL_RELATION") for edge in edges],
            total_hops=len(edges),
        )

    def find_paths(
        self,
        anchor_entities: List[str],
        max_hops: int = 4,
        intent: str = "CAUSAL_CHAIN",
        relation_preference: List[str] | None = None,
        year_constraint: int | None = None,
        year_start: int | None = None,
        year_end: int | None = None,
        source_filing: str | None = None,
        max_paths: int = 20,
    ) -> List[CausalPath]:
        start_year = year_start if year_start is not None else year_constraint
        eligible = [
            edge for edge in self.edges
            if self._eligible(
                edge,
                relation_preference=relation_preference,
                year_start=start_year,
                year_end=year_end,
                source_filing=source_filing,
            )
        ]
        paths: List[CausalPath] = []
        direct = [edge for edge in eligible if self._matches_anchor(edge, anchor_entities)]
        for index, edge in enumerate(direct):
            paths.append(self._to_path([edge], f"staging_{index:04d}"))

        # Bounded forward expansion is enough for the isolated acceptance
        # graph and preserves the production path shape without hidden joins.
        anchor_keys = {_key(value) for value in anchor_entities if value}
        frontier = [
            ([edge], str(edge.get("target") or ""))
            for edge in eligible
            if _key(edge.get("source")) in anchor_keys
        ]
        for depth in range(2, max_hops + 1):
            next_frontier = []
            for chain, endpoint in frontier:
                for edge in eligible:
                    if _key(edge.get("source")) != _key(endpoint):
                        continue
                    if any(edge is previous for previous in chain):
                        continue
                    extended = chain + [edge]
                    next_frontier.append((extended, str(edge.get("target") or "")))
                    if self._matches_anchor(edge, anchor_entities):
                        paths.append(self._to_path(extended, f"staging_{len(paths):04d}"))
            frontier = next_frontier
            if not frontier:
                break
        return paths[:max_paths]

    def find_metric_disclosures(
        self,
        metric_id: str,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None,
        source_filing: Optional[str] = None,
        limit: int = 100,
    ) -> List[CausalPath]:
        metric_key = _key(metric_id)
        edges = [
            edge for edge in self.edges
            if _key(edge.get("target")) == metric_key
            and str(edge.get("relation") or "").upper() == "REPORTS_METRIC"
            and self._eligible(
                edge,
                year_start=year_start,
                year_end=year_end,
                source_filing=source_filing,
            )
        ]
        return [self._to_path([edge], f"metric_staging_{index:04d}") for index, edge in enumerate(edges[:limit])]

    def find_evidence_for_entity(self, entity_name: str, limit: int = 10, source_filing: str | None = None) -> List[Dict[str, Any]]:
        key = _key(entity_name)
        return [
            {
                "evidence": edge.get("evidence"),
                "page": edge.get("page"),
                "filing": edge.get("source_filing"),
                "fiscal_year": edge.get("fact_year"),
                "evidence_id": edge.get("claim_id"),
                "connected_to": edge.get("target") if _key(edge.get("source")) == key else edge.get("source"),
            }
            for edge in self.edges
            if (not source_filing or edge.get("source_filing") == source_filing)
            and key in {_key(edge.get("source")), _key(edge.get("target"))}
        ][:limit]
