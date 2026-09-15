"""Read-only semantic consistency audit for the current evidence graph.

The audit checks machine-observable consistency across EvidenceClaim nodes,
Sentence evidence, entity links, native business relationships, and local PDF
pages. It does not infer new claims and never sends a write query to Neo4j.
Repeated normalized triples are classified as same-evidence duplicates or
different-evidence duplicate groups; duplication alone is not an error.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import fitz
from dotenv import load_dotenv
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from strategic_graphrag.ontology.entity_registry import norm_id
from strategic_graphrag.ontology.relation_inference import validate_triple


EVIDENCE_RELATIONS = {
    "OBSERVED_IN",
    "REPORTS",
    "SUPPORTS",
    "BELONGS_TO",
    "SUPPORTED_BY",
    "ABOUT_SOURCE",
    "ABOUT_TARGET",
}

CLAIM_REQUIRED_FIELDS = (
    "id",
    "doc_id",
    "text",
    "relation_id",
    "relation_type",
    "source_id",
    "target_id",
    "page",
    "fiscal_year",
    "verification_status",
    "evidence_char_start",
    "evidence_char_end",
    "chunk_id",
)
EVIDENCE_REQUIRED_FIELDS = ("id", "text", "page", "doc_id")
ENTITY_REQUIRED_FIELDS = ("id",)
EDGE_REQUIRED_FIELDS = (
    "id",
    "evidence_id",
    "source_filing",
    "source_page",
    "year",
    "evidence_sentence",
)


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def _primary_label(labels: Any) -> str | None:
    if not isinstance(labels, (list, tuple)):
        return None
    for label in labels:
        if label and label not in {"Entity", "Node"}:
            return str(label)
    return str(labels[0]) if labels else None


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _empty_linkage() -> dict[str, bool]:
    return {
        "supported_by": False,
        "about_source": False,
        "about_target": False,
        "native_relation_edge": False,
    }


def _merge_claim_rows(raw_rows: list[dict[str, Any]], edge_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Combine Neo4j rows into one audit record per claim ID."""

    records: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        claim = raw.get("claim") or {}
        claim_id = str(claim.get("id") or f"__missing_claim_id_{index}")
        record = records.setdefault(
            claim_id,
            {
                "claim": dict(claim),
                "evidence": raw.get("evidence") or {},
                "source": raw.get("source") or {},
                "source_labels": raw.get("source_labels") or [],
                "target": raw.get("target") or {},
                "target_labels": raw.get("target_labels") or [],
                "linkage": _empty_linkage(),
                "edges": [],
            },
        )
        for object_key in ("claim", "evidence", "source", "target"):
            incoming = raw.get(object_key) or {}
            if not record[object_key] and incoming:
                record[object_key] = dict(incoming)
        for list_key in ("source_labels", "target_labels"):
            if not record[list_key] and raw.get(list_key):
                record[list_key] = list(raw[list_key])
        for linkage_key in record["linkage"]:
            record["linkage"][linkage_key] = bool(
                record["linkage"][linkage_key] or raw.get(f"has_{linkage_key}", False)
            )

    edges_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in edge_rows:
        properties = raw.get("relation_properties") or {}
        claim_id = raw.get("claim_id") or properties.get("evidence_id")
        if claim_id:
            edges_by_claim[str(claim_id)].append(
                {
                    "type": raw.get("relation"),
                    "properties": dict(properties),
                    "source": raw.get("source") or {},
                    "source_labels": raw.get("source_labels") or [],
                    "target": raw.get("target") or {},
                    "target_labels": raw.get("target_labels") or [],
                }
            )

    for claim_id, record in records.items():
        record["edges"] = edges_by_claim.get(claim_id, [])
        record["linkage"]["native_relation_edge"] = bool(record["edges"])
    return list(records.values())


def _resolve_pdf_path(filename: str, search_dirs: list[Path]) -> tuple[Path | None, list[Path]]:
    """Resolve one allowlisted filename without moving or copying any PDF."""

    candidates = [directory / filename for directory in search_dirs]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, candidates
    return None, candidates


def _load_pdf_pages(
    pdf_dirs: list[Path], documents: dict[str, dict[str, Any]]
) -> tuple[dict[tuple[str, int], str], list[dict[str, Any]], dict[str, Any]]:
    page_texts: dict[tuple[str, int], str] = {}
    issues: list[dict[str, Any]] = []
    resolution: dict[str, Any] = {
        "status": "PASS",
        "search_dirs": [str(directory) for directory in pdf_dirs],
        "found": [],
        "missing": [],
        "read_errors": [],
        "note": "found lists local files resolved by exact filename; missing lists filenames absent from every search directory.",
    }
    for doc_id, document in sorted(documents.items()):
        filename = str(document.get("filename") or f"{doc_id}.pdf")
        pdf_path, candidates = _resolve_pdf_path(filename, pdf_dirs)
        if pdf_path is None:
            missing = {
                "doc_id": doc_id,
                "filename": filename,
                "searched_paths": [str(candidate) for candidate in candidates],
            }
            resolution["missing"].append(missing)
            issues.append({**missing, "reason": "MISSING_PDF"})
            continue
        resolution["found"].append(
            {
                "doc_id": doc_id,
                "filename": filename,
                "path": str(pdf_path),
                "directory": str(pdf_path.parent),
            }
        )
        try:
            with fitz.open(pdf_path) as pdf:
                for page_number, page in enumerate(pdf, start=1):
                    page_texts[(doc_id, page_number)] = _normalize_text(page.get_text("text"))
        except Exception as exc:
            error = {
                "doc_id": doc_id,
                "filename": filename,
                "path": str(pdf_path),
                "reason": f"PDF_READ_ERROR: {type(exc).__name__}: {exc}",
            }
            resolution["read_errors"].append(error)
            issues.append(error)
    if resolution["missing"] or resolution["read_errors"]:
        resolution["status"] = "WARN"
    return page_texts, issues, resolution


def _field_value(properties: dict[str, Any], primary: str, alternate: str | None = None) -> Any:
    value = properties.get(primary)
    if not _present(value) and alternate:
        value = properties.get(alternate)
    return value


def _audit_field_completeness(records: list[dict[str, Any]]) -> dict[str, Any]:
    missing_counts: Counter[str] = Counter()
    samples: list[dict[str, Any]] = []
    for record in records:
        claim = record["claim"]
        missing: list[str] = []
        for field in CLAIM_REQUIRED_FIELDS:
            if not _present(claim.get(field)):
                missing.append(f"claim.{field}")
        evidence = record["evidence"]
        for field in EVIDENCE_REQUIRED_FIELDS:
            if not _present(evidence.get(field)):
                missing.append(f"evidence.{field}")
        for entity_name in ("source", "target"):
            entity = record[entity_name]
            labels = record[f"{entity_name}_labels"]
            for field in ENTITY_REQUIRED_FIELDS:
                if not _present(entity.get(field)):
                    missing.append(f"{entity_name}.{field}")
            if not _primary_label(labels):
                missing.append(f"{entity_name}.label")
        linkage = record["linkage"]
        for linkage_name, linked in linkage.items():
            if not linked:
                missing.append(f"linkage.{linkage_name}")
        for edge_index, edge in enumerate(record["edges"]):
            properties = edge["properties"]
            if not _present(edge.get("type")):
                missing.append(f"relation[{edge_index}].type")
            for field in EDGE_REQUIRED_FIELDS:
                alternate = "filing" if field == "source_filing" else "page" if field == "source_page" else None
                if not _present(_field_value(properties, field, alternate)):
                    missing.append(f"relation[{edge_index}].{field}")
        for field in missing:
            missing_counts[field] += 1
        if missing and len(samples) < 20:
            samples.append({"claim_id": claim.get("id"), "missing": missing})
    return {
        "status": "PASS" if not missing_counts else "FAIL",
        "claims_checked": len(records),
        "missing_field_instances": sum(missing_counts.values()),
        "missing_field_counts": dict(missing_counts),
        "samples": samples,
    }


def _audit_duplicates(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        claim = record["claim"]
        key = (
            norm_id(claim.get("source_id", "")),
            str(claim.get("relation_type", "") or "").upper(),
            norm_id(claim.get("target_id", "")),
        )
        if all(key):
            groups[key].append(record)

    duplicate_groups = []
    same_evidence_groups = 0
    different_evidence_groups = 0
    for key, group in groups.items():
        if len(group) < 2:
            continue
        fingerprints = Counter(
            (
                str(item["claim"].get("doc_id") or ""),
                item["claim"].get("page"),
                _normalize_text(item["claim"].get("text")),
            )
            for item in group
        )
        same_evidence = any(count > 1 for count in fingerprints.values())
        different_evidence = len(fingerprints) > 1
        same_evidence_groups += int(same_evidence)
        different_evidence_groups += int(different_evidence)
        if len(duplicate_groups) < 20:
            duplicate_groups.append(
                {
                    "normalized_triple": {
                        "source": key[0],
                        "relation": key[1],
                        "target": key[2],
                    },
                    "claim_ids": [item["claim"].get("id") for item in group],
                    "same_evidence_duplicate": same_evidence,
                    "different_evidence_duplicate": different_evidence,
                    "evidence_fingerprints": [
                        {"doc_id": fp[0], "page": fp[1], "quote": fp[2]}
                        for fp in fingerprints
                    ],
                }
            )
    return {
        "status": "WARN" if duplicate_groups else "PASS",
        "normalized_triple_groups": len(groups),
        "duplicate_normalized_triple_groups": sum(len(group) >= 2 for group in groups.values()),
        "same_evidence_duplicate_groups": same_evidence_groups,
        "different_evidence_duplicate_groups": different_evidence_groups,
        "samples": duplicate_groups,
        "note": (
            "This machine audit only classifies duplicate normalized triples; it cannot prove semantic "
            "consistency. Duplicate normalized triples are not independently treated as errors."
        ),
    }


def _audit_quotes(
    records: list[dict[str, Any]],
    page_texts: dict[tuple[str, int], str],
    pdf_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    missing_pdf: list[dict[str, Any]] = []
    checked = 0
    quote_checks = 0
    for record in records:
        claim = record["claim"]
        claim_id = claim.get("id")
        doc_id = str(claim.get("doc_id") or "")
        page = claim.get("page")
        try:
            page_number = int(page)
        except (TypeError, ValueError):
            continue
        page_text = page_texts.get((doc_id, page_number))
        if page_text is None:
            issue = {"claim_id": claim_id, "doc_id": doc_id, "page": page_number}
            if any(item.get("doc_id") == doc_id for item in pdf_issues):
                missing_pdf.append(issue)
            else:
                mismatches.append({**issue, "reason": "PAGE_NOT_FOUND_IN_LOCAL_PDF"})
            continue
        checked += 1
        claim_quote = _normalize_text(claim.get("text"))
        evidence_quote = _normalize_text(record["evidence"].get("text"))
        if claim_quote:
            quote_checks += 1
            if claim_quote not in page_text:
                mismatches.append({**{"claim_id": claim_id, "doc_id": doc_id, "page": page_number}, "reason": "CLAIM_QUOTE_NOT_ON_PAGE"})
        if claim_quote and evidence_quote and claim_quote != evidence_quote:
            mismatches.append({"claim_id": claim_id, "reason": "CLAIM_EVIDENCE_TEXT_MISMATCH"})
        if record["evidence"].get("page") not in (None, page_number):
            mismatches.append({"claim_id": claim_id, "reason": "EVIDENCE_PAGE_MISMATCH"})
        for edge_index, edge in enumerate(record["edges"]):
            edge_quote = _normalize_text(edge["properties"].get("evidence_sentence"))
            if edge_quote and edge_quote not in page_text:
                mismatches.append(
                    {
                        "claim_id": claim_id,
                        "relation_index": edge_index,
                        "reason": "RELATION_QUOTE_NOT_ON_PAGE",
                    }
                )
            if claim_quote and edge_quote and claim_quote != edge_quote:
                mismatches.append(
                    {
                        "claim_id": claim_id,
                        "relation_index": edge_index,
                        "reason": "CLAIM_RELATION_QUOTE_MISMATCH",
                    }
                )
    status = "FAIL" if mismatches else "WARN" if missing_pdf else "PASS"
    return {
        "status": status,
        "claims_with_local_pdf_page": checked,
        "quote_checks": quote_checks,
        "quote_mismatches": len(mismatches),
        "pdf_unavailable_for_claims": len(missing_pdf),
        "mismatches": mismatches[:20],
        "missing_pdf_samples": missing_pdf[:20],
        "pdf_issues": pdf_issues[:20],
    }


def _audit_relations(records: list[dict[str, Any]]) -> dict[str, Any]:
    ontology_conflicts: list[dict[str, Any]] = []
    linkage_conflicts: list[dict[str, Any]] = []
    relation_types = Counter()
    for record in records:
        claim = record["claim"]
        relation = str(claim.get("relation_type") or "").upper()
        source_id = str(record["source"].get("id") or claim.get("source_id") or "")
        target_id = str(record["target"].get("id") or claim.get("target_id") or "")
        source_category = _primary_label(record["source_labels"])
        target_category = _primary_label(record["target_labels"])
        if relation:
            relation_types[relation] += 1
        if source_category and target_category and relation and source_id and target_id:
            valid, reason = validate_triple(
                source_category,
                target_category,
                relation,
                source_id,
                target_id,
            )
            if not valid:
                ontology_conflicts.append(
                    {
                        "claim_id": claim.get("id"),
                        "source": source_id,
                        "source_category": source_category,
                        "relation": relation,
                        "target": target_id,
                        "target_category": target_category,
                        "reason": reason,
                    }
                )
        for edge_index, edge in enumerate(record["edges"]):
            properties = edge["properties"]
            edge_source = str(edge["source"].get("id") or "")
            edge_target = str(edge["target"].get("id") or "")
            conflicts: list[str] = []
            if edge.get("type") != relation:
                conflicts.append("relation_type_does_not_match_claim")
            if properties.get("evidence_id") != claim.get("id"):
                conflicts.append("evidence_id_does_not_match_claim")
            if claim.get("relation_id") and properties.get("id") != claim.get("relation_id"):
                conflicts.append("relation_id_does_not_match_edge_id")
            if edge_source != source_id:
                conflicts.append("edge_source_does_not_match_claim_source")
            if edge_target != target_id:
                conflicts.append("edge_target_does_not_match_claim_target")
            if _field_value(properties, "source_page", "page") != claim.get("page"):
                conflicts.append("edge_page_does_not_match_claim_page")
            if properties.get("year") != claim.get("fiscal_year"):
                conflicts.append("edge_year_does_not_match_claim_year")
            edge_source_category = str(properties.get("source_category") or "")
            edge_target_category = str(properties.get("target_category") or "")
            if edge_source_category and edge_source_category != source_category:
                conflicts.append("edge_source_category_does_not_match_node_label")
            if edge_target_category and edge_target_category != target_category:
                conflicts.append("edge_target_category_does_not_match_node_label")
            if conflicts and len(linkage_conflicts) < 50:
                linkage_conflicts.append(
                    {
                        "claim_id": claim.get("id"),
                        "relation_index": edge_index,
                        "conflicts": conflicts,
                    }
                )
    return {
        "status": "FAIL" if ontology_conflicts or linkage_conflicts else "PASS",
        "relation_type_counts": dict(relation_types),
        "ontology_direction_type_conflicts": len(ontology_conflicts),
        "linkage_direction_type_conflicts": len(linkage_conflicts),
        "ontology_samples": ontology_conflicts[:20],
        "linkage_samples": linkage_conflicts[:20],
        "note": "Ontology validation identifies obvious category/direction conflicts; it does not prove causal semantics.",
    }


def audit_records(
    records: list[dict[str, Any]],
    page_texts: dict[tuple[str, int], str] | None = None,
    pdf_issues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit already-loaded records; exposed for offline unit tests."""

    page_texts = page_texts or {}
    pdf_issues = pdf_issues or []
    field_audit = _audit_field_completeness(records)
    duplicate_audit = _audit_duplicates(records)
    quote_audit = _audit_quotes(records, page_texts, pdf_issues)
    relation_audit = _audit_relations(records)
    sections = {
        "field_completeness": field_audit,
        "normalized_triples": duplicate_audit,
        "quote_traceability": quote_audit,
        "relation_direction_and_type": relation_audit,
    }
    statuses = [section["status"] for section in sections.values()]
    overall = "FAIL" if "FAIL" in statuses else "WARN" if "WARN" in statuses else "PASS"
    return {
        "status": overall,
        "counts": {
            "claims": len(records),
            "native_business_edges": sum(len(record["edges"]) for record in records),
            "duplicate_normalized_triple_groups": duplicate_audit["duplicate_normalized_triple_groups"],
            "same_evidence_duplicate_groups": duplicate_audit["same_evidence_duplicate_groups"],
            "different_evidence_duplicate_groups": duplicate_audit["different_evidence_duplicate_groups"],
            "missing_field_instances": field_audit["missing_field_instances"],
            "quote_mismatches": quote_audit["quote_mismatches"],
            "ontology_direction_type_conflicts": relation_audit["ontology_direction_type_conflicts"],
            "linkage_direction_type_conflicts": relation_audit["linkage_direction_type_conflicts"],
        },
        "sections": sections,
        "limitations": [
            "Duplicate normalized triples are classified, not independently failed; repeated claims may be valid restatements.",
            "PDF quote checks use normalized text containment and do not establish semantic correctness or causal validity.",
            "Ontology direction/type checks cover explicit schema rules and obvious linkage conflicts only.",
            "This audit is not human Golden QA and does not measure answer quality or retrieval superiority.",
        ],
    }


CLAIM_QUERY = """
MATCH (c:EvidenceClaim)
OPTIONAL MATCH (c)-[:SUPPORTED_BY]->(e:Sentence)
OPTIONAL MATCH (c)-[:ABOUT_SOURCE]->(src)
OPTIONAL MATCH (c)-[:ABOUT_TARGET]->(tgt)
RETURN properties(c) AS claim,
       properties(e) AS evidence,
       properties(src) AS source,
       labels(src) AS source_labels,
       properties(tgt) AS target,
       labels(tgt) AS target_labels,
       e IS NOT NULL AS has_supported_by,
       src IS NOT NULL AS has_about_source,
       tgt IS NOT NULL AS has_about_target
"""

EDGE_QUERY = """
MATCH (src)-[r]->(tgt)
WHERE NOT type(r) IN $evidence_relations
  AND r.evidence_id IS NOT NULL
OPTIONAL MATCH (c:EvidenceClaim {id: r.evidence_id})
RETURN type(r) AS relation,
       properties(r) AS relation_properties,
       properties(src) AS source,
       labels(src) AS source_labels,
       properties(tgt) AS target,
       labels(tgt) AS target_labels,
       c.id AS claim_id
"""

DOCUMENT_QUERY = """
MATCH (d:Document)
RETURN properties(d) AS document
"""


def run_audit(root: Path = PROJECT_ROOT, pdf_dir: Path | None = None) -> dict[str, Any]:
    """Read current Neo4j and local PDFs, returning a machine-readable audit."""

    root = root.resolve()
    requested_pdf_dir = (pdf_dir or root / "data" / "pdfs").resolve()
    pdf_dirs: list[Path] = []
    for candidate in (
        requested_pdf_dir,
        root / "data" / "pdfs",
        root / "data" / "pdfs_other",
    ):
        resolved = candidate.resolve()
        if resolved not in pdf_dirs:
            pdf_dirs.append(resolved)
    load_dotenv(root / ".env")
    generated = datetime.now(timezone.utc).isoformat()
    base = {
        "schema": "strategic-graphrag-graph-semantic-consistency/v1",
        "generated_at_utc": generated,
        "scope": "all EvidenceClaim records and native business edges in current Neo4j",
        "read_only": True,
        "neo4j_write_queries": 0,
        "pdf_dir": str(requested_pdf_dir),
        "pdf_search_dirs": [str(directory) for directory in pdf_dirs],
    }
    try:
        uri = os.environ["NEO4J_URI"]
        username = os.environ["NEO4J_USERNAME"]
        password = os.environ["NEO4J_PASSWORD"]
    except KeyError as exc:
        return {
            **base,
            "status": "FAIL",
            "connection": {"status": "FAIL", "error": f"Missing Neo4j setting: {exc.args[0]}"},
            "counts": {},
            "sections": {},
            "limitations": ["Neo4j was not queried because required settings were missing."],
        }

    driver = None
    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            raw_claims = [record.data() for record in session.run(CLAIM_QUERY)]
            raw_edges = [
                record.data()
                for record in session.run(
                    EDGE_QUERY,
                    evidence_relations=sorted(EVIDENCE_RELATIONS),
                )
            ]
            raw_documents = [record.data() for record in session.run(DOCUMENT_QUERY)]
    except Exception as exc:
        return {
            **base,
            "status": "FAIL",
            "connection": {"status": "FAIL", "error": f"{type(exc).__name__}: {exc}"},
            "counts": {},
            "sections": {},
            "limitations": ["Neo4j read failed; no write query was attempted."],
        }
    finally:
        if driver is not None:
            driver.close()

    documents: dict[str, dict[str, Any]] = {}
    for row in raw_documents:
        document = row.get("document") or {}
        if document.get("doc_id"):
            documents[str(document["doc_id"])] = document
    records = _merge_claim_rows(raw_claims, raw_edges)
    page_texts, pdf_issues, pdf_resolution = _load_pdf_pages(pdf_dirs, documents)
    audit = audit_records(records, page_texts, pdf_issues)
    return {
        **base,
        "status": audit["status"],
        "connection": {"status": "PASS", "documents_loaded": len(documents)},
        "counts": {**audit["counts"], "documents": len(documents)},
        "sections": audit["sections"],
        "pdf_resolution": pdf_resolution,
        "limitations": audit["limitations"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Neo4j/PDF semantic consistency audit")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--pdf-dir", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "graph_semantic_consistency.json",
    )
    args = parser.parse_args()
    report = run_audit(args.root, args.pdf_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Keep console output portable on Windows code pages; the report file
    # above remains UTF-8 with the original evidence text preserved.
    print(json.dumps(report, ensure_ascii=True, indent=2))
    raise SystemExit(0 if report["status"] in {"PASS", "WARN"} else 2)


if __name__ == "__main__":
    main()
