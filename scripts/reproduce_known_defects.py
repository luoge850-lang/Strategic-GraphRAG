"""Capture the four acceptance defects before the next corrective patch.

This script intentionally exercises public-ish engine/API boundaries rather
than only helper functions.  Its output is a historical reproduction record;
it must be run before changing the corresponding implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategic_graphrag.api import server
from strategic_graphrag.engine.graph_rag_engine import CausalPath, GraphRAGEngine
from strategic_graphrag.evidence_bundle import from_graph_paths
from strategic_graphrag.engine.query_understanding import parse_query


def reproduce() -> dict:
    path = CausalPath(
        path_id="defect-scale",
        nodes=["NVIDIA_CORPORATION", "REVENUE"],
        node_labels=["Company", "FinancialMetric"],
        relationships=["REPORTS_METRIC"],
        causal_strengths=["DISCLOSED_ONLY"],
        evidence=["Revenue was 100 million."],
        pages=[38],
        years=[2025],
        evidence_ids=["claim_defect_scale"],
        filings=["2025-10-K.pdf"],
        total_hops=1,
    )
    scale_report = {
        "status": "GENERATED",
        "executive_summary": "Revenue was 100 billion.",
        "claims": [{
            "statement": "Revenue was 100 billion.",
            "evidence_claim_ids": ["claim_defect_scale"],
            "pages": [38],
            "fiscal_years": [2025],
            "source_filings": ["2025-10-K.pdf"],
        }],
    }
    scale_grounding = GraphRAGEngine._validate_report_grounding(
        "Revenue was 100 billion [EvidenceClaim: claim_defect_scale; p.38]",
        [path],
        structured_report=scale_report,
        query="What revenue was reported?",
        intent="FINANCIAL_METRIC",
    )

    class ErrorVector:
        build_id = "build_a"

        def retrieve_with_metadata(self, query, *, k, source_filing=None):
            return {
                "status": "OK",
                "hits": [{
                    "rank": 1,
                    "document": "Revenue was 100 million.",
                    "metadata": {
                        "chunk_id": "chunk_defect_generation",
                        "source_filing": "2025-10-K.pdf",
                        "page": 38,
                        "build_id": "build_a",
                    },
                }],
                "collection": "staging",
                "build_id": "build_a",
            }

        def generate(self, query, documents):
            return "[Generation error: LLM call failed]"

    async def run_vector_api():
        with patch.object(server, "get_vector_engine", return_value=ErrorVector()):
            response = await server.vector_query(server.QueryRequest(
                question="What was revenue?",
                synthesize=True,
            ))
        return response.model_dump() if hasattr(response, "model_dump") else response.dict()

    vector_api = asyncio.run(run_vector_api())
    bundle = from_graph_paths([{
        "path_id": "defect-build",
        "nodes": ["NVIDIA_CORPORATION", "REVENUE"],
        "relationships": ["REPORTS_METRIC"],
        "evidence_ids": ["claim_build_1", "claim_build_2"],
        "evidence": ["Revenue was 100 million.", "Revenue was 60 million."],
        "pages": [38, 79],
        "years": [2025, 2024],
        "filings": ["2025-10-K.pdf", "2025-10-K.pdf"],
        "evidence_build_ids": ["build_a", "build_a"],
    }], build_id="build_a")
    alias_query = "What was FY2024 revenue in the 2025 filing?"
    alias_plan = parse_query(alias_query)
    return {
        "schema": "known-defect-reproduction/v1",
        "defects": {
            "numeric_scale": {
                "input": "evidence=100 million; answer=100 billion",
                "observed_grounding_status": scale_grounding.get("status"),
                "observed": scale_grounding,
                "failure_reproduced": scale_grounding.get("status") == "VERIFIED",
            },
            "vector_generation_error": {
                "observed_execution_status": vector_api.get("execution_status"),
                "observed_answer_status": vector_api.get("answer_status"),
                "observed_outcome": vector_api.get("outcome"),
                "answer": vector_api.get("answer"),
                "failure_reproduced": (
                    vector_api.get("execution_status") == "SUCCEEDED"
                    and vector_api.get("answer_status") == "ANSWERED"
                ),
            },
            "graph_build_identity": {
                "observed_bundle_status": bundle.status,
                "observed_bundle_build_id": bundle.build_id,
                "item_build_ids": [item.build_id for item in bundle.items],
                "failure_reproduced": bundle.status == "UNBOUND",
            },
            "fy2024_in_2025_filing_alias": {
                "query": alias_query,
                "document_scope": alias_plan.document_scope,
                "fact_period": alias_plan.fact_period,
                "disclosure_as_of": alias_plan.disclosure_as_of,
                "fiscal_year_start": alias_plan.fiscal_year_start,
                "fiscal_year_end": alias_plan.fiscal_year_end,
                "failure_reproduced": not (
                    alias_plan.document_scope == "2025-10-K.pdf"
                    and alias_plan.fact_period == "FY2024"
                ),
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = reproduce()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "failure_reproduced": sum(
            int(item["failure_reproduced"])
            for item in result["defects"].values()
        ),
        "defect_count": len(result["defects"]),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
