"""Canonical acceptance ledger and graph expansion for the current build."""
import json,re
from pathlib import Path
from typing import Any,Dict,List
from strategic_graphrag.provenance import evidence_identity

def json_read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def json_write(path, value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")

def _int_year(value: Any, fallback: int) -> int:
    match = re.search(r"20\d{2}", str(value or ""))
    return int(match.group(0)) if match else fallback

def accept_candidates(package: Path, build_id: str) -> None:
    candidate = json_read(package / "candidate_extraction.json")
    edges: List[Dict[str, Any]] = []
    accepted: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for file_result in candidate["files"]:
        filename = str(file_result["filename"])
        doc_sha = str(file_result["document_sha256"])
        filing_year = int(file_result.get("year") or _int_year(filename, 0))
        for triple_index, triple in enumerate(file_result.get("accepted_triples") or []):
            from strategic_graphrag.pipeline.financial_candidate_guard import rejection_reason
            context_error = rejection_reason(triple)
            if context_error:
                pending.append({"filename": filename, "triple_index": triple_index, "reason": context_error, "triple": triple})
                continue
            source = str(triple.get("source") or "").strip()
            target = str(triple.get("target") or "").strip()
            relation = str(triple.get("relation") or "").strip().upper()
            evidence = str(triple.get("evidence_sentence") or triple.get("row_evidence") or "").strip()
            page = int(triple.get("source_page") or triple.get("page") or 0)
            if not source or not target or not relation or not evidence or page <= 0:
                pending.append({"filename": filename, "triple_index": triple_index, "reason": "missing_identity_or_evidence", "triple": triple})
                continue
            values: List[Dict[str, Any]] = []
            if relation == "REPORTS_METRIC":
                try:
                    raw_values = json.loads(triple.get("metric_values_json") or "[]")
                    values = [item for item in raw_values if isinstance(item, dict) and item.get("value") is not None]
                except (TypeError, ValueError, json.JSONDecodeError):
                    rejected.append({"filename": filename, "triple_index": triple_index, "reason": "invalid_metric_values_json"})
                    continue
            if not values:
                values = [{"period": str(triple.get("metric_period") or filing_year), "value": triple.get("metric_value")}]
            identity = evidence_identity(
                document_sha256=doc_sha,
                filename=filename,
                page=page,
                evidence_text=evidence,
                source_id=source,
                relation_type=relation,
                target_id=target,
            )
            for value_index, value in enumerate(values):
                fact_year = _int_year(value.get("period"), filing_year)
                edge = {
                    "edge_id": f"{identity.claim_id}:{value_index}",
                    "claim_id": identity.claim_id,
                    "relation_id": identity.relation_id,
                    "sentence_id": identity.sentence_id,
                    "build_id": build_id,
                    "source": source,
                    "source_category": triple.get("source_category") or "Entity",
                    "target": target,
                    "target_category": triple.get("target_category") or "Entity",
                    "relation": relation,
                    "causal_strength": triple.get("causal_strength") or "DISCLOSED_ONLY",
                    "causal_form": triple.get("causal_form") or "FINANCIAL_RELATION",
                    "evidence": evidence,
                    "page": page,
                    "fact_year": fact_year,
                    "filing_year": filing_year,
                    "source_filing": filename,
                    "document_sha256": doc_sha,
                    "chunk_id": triple.get("chunk_id"),
                    "table_instance_id": triple.get("table_instance_id"),
                    "metric_value": value.get("value"),
                    "metric_unit": triple.get("metric_unit") or triple.get("unit"),
                    "unit": triple.get("unit") or triple.get("metric_unit"),
                    "currency": triple.get("currency"),
                    "scale": triple.get("scale"),
                    "report_period": triple.get("report_period") or f"FY{filing_year}",
                    "source_row_id": triple.get("source_row_id"),
                    "statement_type": triple.get("statement_type"),
                    "source_section": triple.get("source_section"),
                    "table_name": triple.get("table_name"),
                    "row_label": triple.get("row_label"),
                    "comparability_status": triple.get("comparability_status"),
                    **{key:triple.get(key) for key in ('financial_basis','basis_evidence','source_row','table_context','eps_basis')},
                    **{key:value.get(key) for key in ('column_label','period_start','period_end','period_bounds_status')},
                }
                edges.append(edge)
            accepted.append({
                "filename": filename,
                "triple_index": triple_index,
                "claim_id": identity.claim_id,
                "expanded_values": len(values),
                "relation": relation,
            })
    graph = {
        "schema": "staging-evidence-graph/v1",
        "build_id": build_id,
        "edges": edges,
        "counts": {"accepted_triples": len(accepted), "expanded_fact_edges": len(edges), "pending": len(pending), "rejected": len(rejected)},
    }
    graph["pending"] = pending
    graph["rejected"] = rejected
    json_write(package / "graph.json", graph)
