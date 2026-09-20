"""Run the four retrieval baselines through one reproducible response contract."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from strategic_graphrag.engine.graph_rag_engine import GraphRAGEngine
from strategic_graphrag.engine.vector_rag_baseline import VectorRAGBaseline


MODES = ("vector", "graph", "hybrid", "hybrid_temporal")


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _retrieved_contexts(mode: str, result: dict) -> dict[str, object]:
    """Expose comparable IDs without pretending chunks and claims are identical."""
    metadata = result.get("metadata") or {}
    retrieval = metadata.get("retrieval") or {}
    hits = retrieval.get("hits") or retrieval.get("vector_hits") or []
    if mode == "vector":
        contexts = []
        for hit in hits:
            hit_metadata = hit.get("metadata") or {}
            context_id = hit_metadata.get("chunk_id") or hit.get("chunk_id")
            if context_id:
                contexts.append(
                    {
                        "id": str(context_id),
                        "unit_type": "chunk_id",
                        "page": hit_metadata.get("page"),
                        "filing": hit_metadata.get("doc_id") or hit_metadata.get("source_filing"),
                    }
                )
        return {"unit_type": "chunk_id", "contexts": contexts}

    contexts = []
    for path in result.get("paths", []) or []:
        filings = path.get("filings") or []
        pages = path.get("pages") or []
        for index, evidence_id in enumerate(path.get("evidence_ids") or []):
            if evidence_id:
                contexts.append(
                    {
                        "id": str(evidence_id),
                        "unit_type": "evidence_claim_id",
                        "page": pages[index] if index < len(pages) else None,
                        "filing": filings[index] if index < len(filings) else None,
                    }
                )
    return {"unit_type": "evidence_claim_id", "contexts": contexts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-filing")
    parser.add_argument("--cross-filing", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--synthesize",
        action="store_true",
        help="Opt in to LLM synthesis. The default retrieval-only mode does not send evidence externally.",
    )
    args = parser.parse_args()

    engine = GraphRAGEngine()
    vector = VectorRAGBaseline()
    records = []
    try:
        for question in args.question:
            for mode in MODES:
                result = engine.query(
                    question,
                    top_k=args.top_k,
                    source_filing=args.source_filing,
                    cross_filing=args.cross_filing,
                    retrieval_mode=mode,
                    vector_engine=vector,
                    synthesize=args.synthesize,
                    use_llm_anchors=args.synthesize,
                )
                metadata = result.get("metadata") or {}
                records.append({
                    "question": question,
                    "mode": mode,
                    "intent": result.get("intent"),
                    "answer": result.get("answer"),
                    "paths": result.get("paths", []),
                    "evidence_sentences": result.get("evidence_sentences", []),
                    "structured_report": result.get("structured_report"),
                    "router": metadata.get("router"),
                    "retrieval": metadata.get("retrieval"),
                    "temporal_fusion": metadata.get("temporal_fusion"),
                    "ppr": metadata.get("ppr"),
                    "latency_ms": metadata.get("latency_ms"),
                    "retrieved_contexts": _retrieved_contexts(mode, result),
                })
    finally:
        engine.close()

    report = {
        "schema": "strategic-graphrag-retrieval-baselines/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "modes": list(MODES),
        "question_count": len(args.question),
        "run_count": len(records),
        "run_config": {
            "top_k": args.top_k,
            "source_filing": args.source_filing,
            "cross_filing": args.cross_filing,
            "synthesize": args.synthesize,
            "llm_anchor_expansion": args.synthesize,
            "git_sha": _git_sha(),
            "python": platform.python_version(),
            "llm_provider": os.getenv("LLM_PROVIDER"),
            "llm_model": os.getenv("LLM_MODEL") or os.getenv("LLM_QUERY_MODEL"),
            "embedding_model": os.getenv("EMBEDDING_MODEL"),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = [
        {
            "question": item["question"],
            "mode": item["mode"],
            "paths": len(item["paths"]),
            "answer_status": (item["structured_report"] or {}).get("status"),
            "latency_ms": (item["latency_ms"] or {}).get("total_ms"),
            "ppr_entities": len((item.get("ppr") or {}).get("ranked_entities") or []),
        }
        for item in records
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
