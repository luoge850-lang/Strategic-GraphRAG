"""Common evidence envelope for graph, vector, hybrid, and temporal results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional


SCHEMA = "evidence-bundle/v1"


@dataclass(frozen=True)
class EvidenceItem:
    evidence_id: str
    text: str
    document_id: Optional[str] = None
    source_filing: Optional[str] = None
    physical_page: Optional[int] = None
    location: Dict[str, Any] = field(default_factory=dict)
    fact_period: Optional[str] = None
    score: Optional[float] = None
    source_method: str = "unknown"
    build_id: Optional[str] = None


@dataclass
class EvidenceBundle:
    items: List[EvidenceItem] = field(default_factory=list)
    retrieval_mode: str = "unknown"
    build_id: Optional[str] = None
    status: str = "OK"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": SCHEMA,
            "status": self.status,
            "retrieval_mode": self.retrieval_mode,
            "build_id": self.build_id,
            "items": [asdict(item) for item in self.items],
        }

    def assert_consistent(self) -> None:
        ids = {item.build_id for item in self.items if item.build_id}
        if self.build_id and ids and ids != {self.build_id}:
            raise ValueError(f"evidence items mix build IDs: {sorted(ids)}")
        if len(ids) > 1:
            raise ValueError(f"evidence items mix build IDs: {sorted(ids)}")

    def assert_strictly_bound(self) -> None:
        """Require every returned asset to carry the same asset-derived ID."""
        if not self.items or any(not item.build_id for item in self.items):
            raise ValueError("evidence bundle contains UNKNOWN/UNBOUND asset versions")
        self.assert_consistent()
        if not self.build_id:
            raise ValueError("evidence bundle has no asset-derived build_id")


def _build_id(metadata: Mapping[str, Any]) -> Optional[str]:
    return str(metadata.get("build_id") or "").strip() or None


def from_vector_hits(
    hits: Iterable[Mapping[str, Any]],
    *,
    retrieval_mode: str = "vector",
    build_id: Optional[str] = None,
) -> EvidenceBundle:
    items: List[EvidenceItem] = []
    for hit in hits or []:
        metadata = hit.get("metadata") or {}
        evidence_id = str(metadata.get("chunk_id") or hit.get("chunk_id") or "").strip()
        if not evidence_id:
            continue
        # The caller's build_id is an expected identity, never an authority
        # that can relabel an old or incomplete stored asset.
        item_build_id = _build_id(metadata)
        items.append(EvidenceItem(
            evidence_id=evidence_id,
            text=str(hit.get("document") or ""),
            document_id=str(metadata.get("doc_id") or metadata.get("source_filing") or "") or None,
            source_filing=str(metadata.get("source_filing") or metadata.get("doc_id") or "") or None,
            physical_page=int(metadata["page"]) if str(metadata.get("page", "")).isdigit() else None,
            location={"chunk_id": evidence_id, "rank": hit.get("rank")},
            fact_period=str(metadata.get("fact_period") or "") or None,
            score=hit.get("rank_score"),
            source_method="vector",
            build_id=item_build_id,
        ))
    ids = {item.build_id for item in items if item.build_id}
    bundle = EvidenceBundle(
        items=items,
        retrieval_mode=retrieval_mode,
        build_id=next(iter(ids)) if len(ids) == 1 and all(item.build_id for item in items) else None,
        status="OK" if items and len(ids) == 1 and all(item.build_id for item in items) else "UNBOUND",
    )
    if build_id and bundle.build_id and build_id != bundle.build_id:
        raise ValueError(f"expected asset build_id {build_id!r}, got {bundle.build_id!r}")
    bundle.assert_consistent()
    return bundle


def from_graph_paths(
    paths: Iterable[Any],
    *,
    retrieval_mode: str = "graph",
    build_id: Optional[str] = None,
) -> EvidenceBundle:
    items: List[EvidenceItem] = []
    for path in paths or []:
        value = path if isinstance(path, Mapping) else getattr(path, "__dict__", {})
        evidence_ids = value.get("evidence_ids") or []
        evidence = value.get("evidence") or []
        evidence_build_ids = value.get("evidence_build_ids") or value.get("build_ids") or []
        pages = value.get("pages") or []
        years = value.get("years") or []
        filings = value.get("filings") or []
        score = value.get("score", value.get("aggregate_score"))
        for index, evidence_id in enumerate(evidence_ids):
            if not evidence_id:
                continue
            stored_build_id = (
                evidence_build_ids[index]
                if index < len(evidence_build_ids)
                else value.get("build_id")
            )
            items.append(EvidenceItem(
                evidence_id=str(evidence_id),
                text=str(evidence[index]) if index < len(evidence) else "",
                document_id=str(filings[index]).replace(".pdf", "") if index < len(filings) and filings[index] else None,
                source_filing=str(filings[index]) if index < len(filings) and filings[index] else None,
                physical_page=int(pages[index]) if index < len(pages) and str(pages[index]).isdigit() else None,
                location={"path_id": value.get("path_id"), "hop": index},
                fact_period=f"FY{years[index]}" if index < len(years) and str(years[index]).isdigit() else None,
                score=score,
                source_method="graph",
                # Only the stored per-evidence/path identity is authoritative.
                build_id=str(stored_build_id or "").strip() or None,
            ))
    ids = {item.build_id for item in items if item.build_id}
    bundle = EvidenceBundle(
        items=items,
        retrieval_mode=retrieval_mode,
        build_id=next(iter(ids)) if len(ids) == 1 and all(item.build_id for item in items) else None,
        status="OK" if items and len(ids) == 1 and all(item.build_id for item in items) else "UNBOUND",
    )
    if build_id and bundle.build_id and build_id != bundle.build_id:
        raise ValueError(f"expected asset build_id {build_id!r}, got {bundle.build_id!r}")
    bundle.assert_consistent()
    return bundle


__all__ = ["SCHEMA", "EvidenceItem", "EvidenceBundle", "from_vector_hits", "from_graph_paths"]
