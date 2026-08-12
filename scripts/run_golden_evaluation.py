"""Run retrieval and answer-level metrics against the evidence-linked QA set.

Default mode is a cheap structural smoke test. ``--judge`` enables an optional
LLM judge through the project's configured provider for Faithfulness and Answer
Relevance. Structural grounding is reported separately and must not be called
semantic faithfulness.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 2)
    index = (len(values) - 1) * percentile / 100
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    weight = index - lower
    return round(values[lower] * (1 - weight) + values[upper] * weight, 2)


def _mean(values: List[Optional[float]]) -> Optional[float]:
    usable = [value for value in values if value is not None]
    return round(statistics.mean(usable), 4) if usable else None


def _retrieved_ids(result: Dict[str, Any]) -> Set[str]:
    return {
        str(evidence_id)
        for path in result.get("paths", [])
        for evidence_id in path.get("evidence_ids", []) or []
        if evidence_id
    }


def _retrieved_pages(result: Dict[str, Any]) -> Set[int]:
    return {
        int(page)
        for path in result.get("paths", [])
        for page in path.get("pages", []) or []
        if str(page).isdigit()
    }


def _judge_answer(
    question: str,
    context: str,
    answer: str,
    llm: Any,
) -> Optional[Dict[str, Any]]:
    prompt = f"""Evaluate this evidence-grounded RAG answer. Return JSON only.
Score each item from 1 to 5.
faithfulness: factual statements are supported by the retrieved context;
answer_relevance: the answer directly addresses the question.
Do not reward unsupported detail or stylistic fluency.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context[:12000]}

ANSWER:
{answer[:8000]}

JSON shape:
{{"faithfulness": 1, "answer_relevance": 1, "justification": "short reason"}}
"""
    try:
        content = llm.chat_with_fallback(
            prompt=prompt,
            system_prompt="You are a strict academic RAG evaluator.",
            temperature=0.0,
            max_tokens=350,
        )
        parsed = json.loads((content or "").strip())
        return {
            "faithfulness": float(parsed["faithfulness"]),
            "answer_relevance": float(parsed["answer_relevance"]),
            "justification": str(parsed.get("justification", "")),
        }
    except Exception as exc:
        return {"error": type(exc).__name__}


def evaluate(
    dataset: List[Dict[str, Any]],
    base_url: str,
    limit: Optional[int],
    judge: bool,
) -> Dict[str, Any]:
    load_dotenv(ROOT / ".env")
    judge_llm = None
    if judge:
        from strategic_graphrag.llm_provider import get_llm

        judge_llm = get_llm()

    rows = dataset[:limit] if limit else dataset
    results = []
    for item in rows:
        started = time.perf_counter()
        row: Dict[str, Any] = {"id": item["id"], "question_type": item.get("question_type")}
        try:
            response = requests.post(
                f"{base_url.rstrip('/')}/query",
                json={
                    "question": item["question"],
                    "max_paths": 5,
                    "retrieval_mode": "hybrid",
                    "vector_top_k": 5,
                    "source_filing": item.get("source_filing"),
                },
                timeout=180,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000
            row["http_status"] = response.status_code
            row["latency_ms"] = round(elapsed_ms, 2)
            payload = response.json()
            if response.status_code >= 400:
                row["error"] = payload
                results.append(row)
                continue

            gold_ids = set(item.get("evidence_claim_ids", []) or [])
            gold_pages = {int(page) for page in item.get("pages", []) or [] if str(page).isdigit()}
            retrieved_ids = _retrieved_ids(payload)
            retrieved_pages = _retrieved_pages(payload)
            overlap_ids = gold_ids & retrieved_ids
            overlap_pages = gold_pages & retrieved_pages
            answerable = bool(item.get("answerable", True))
            row.update(
                {
                    "answerable": answerable,
                    "gold_evidence_ids": sorted(gold_ids),
                    "retrieved_evidence_ids": sorted(retrieved_ids),
                    "evidence_recall": round(len(overlap_ids) / len(gold_ids), 4) if gold_ids else None,
                    "evidence_precision": round(len(overlap_ids) / len(retrieved_ids), 4) if retrieved_ids else 0.0,
                    "page_recall": round(len(overlap_pages) / len(gold_pages), 4) if gold_pages else None,
                    "grounding_status": (payload.get("metadata", {}).get("grounding") or {}).get("status"),
                    "abstention_correct": (
                        not answerable
                        and any(marker in payload.get("answer", "") for marker in ("INSUFFICIENT", "GROUNDING FAILURE"))
                    ) if not answerable else None,
                }
            )
            if judge_llm is not None:
                context = "\n".join(payload.get("evidence_sentences", []))
                row["judge"] = _judge_answer(item["question"], context, payload.get("answer", ""), judge_llm)
        except Exception as exc:
            row["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)

    answerable_rows = [row for row in results if row.get("answerable") and not row.get("error")]
    judge_rows = [row.get("judge") for row in answerable_rows if isinstance(row.get("judge"), dict) and "faithfulness" in row["judge"]]
    latency = [row["latency_ms"] for row in results if "latency_ms" in row]
    return {
        "dataset": "golden_qa_v2",
        "dataset_status": "AUTO_GENERATED_REGRESSION_CANDIDATE_NOT_HUMAN_GOLD",
        "source_filing": "2025-10-K.pdf",
        "evaluation_protocol": "hybrid retrieval, structural metrics only; LLM judge disabled",
        "dataset_size": len(dataset),
        "evaluated": len(results),
        "judge_enabled": judge,
        "metrics": {
            "evidence_recall": _mean([row.get("evidence_recall") for row in answerable_rows]),
            "evidence_precision": _mean([row.get("evidence_precision") for row in answerable_rows]),
            "faithfulness_structural_proxy": _mean([
                1.0 if row.get("grounding_status") == "VERIFIED" else 0.0
                for row in answerable_rows
            ]),
            "faithfulness_llm_1_to_5": _mean([row.get("faithfulness") for row in judge_rows]),
            "answer_relevance_llm_1_to_5": _mean([row.get("answer_relevance") for row in judge_rows]),
            "abstention_accuracy": _mean([
                1.0 if row.get("abstention_correct") else 0.0
                for row in results
                if row.get("answerable") is False
            ]),
            "latency_ms": {
                "p50": _percentile(latency, 50),
                "p95": _percentile(latency, 95),
                "mean": _mean(latency),
            },
        },
        "metric_notes": {
            "evidence_recall_precision": "Exact EvidenceClaim ID overlap against the manually reviewable gold set.",
            "faithfulness_structural_proxy": "Runtime citation/grounding check; not a substitute for semantic faithfulness.",
            "faithfulness_llm_1_to_5": "Optional configured-provider judge; run with --judge and report model/version.",
            "answer_relevance_llm_1_to_5": "Optional configured-provider judge; run with --judge and report model/version.",
            "latency": "End-to-end HTTP latency, including retrieval and answer synthesis; cold and warm runs should be separated for publication.",
        },
        "rows": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the single-filing Golden QA set")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "evaluation" / "golden_qa_v2.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only the first N questions")
    parser.add_argument("--judge", action="store_true", help="Enable configured LLM judge for semantic metrics")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "evaluation" / "golden_qa_v2_results.json")
    args = parser.parse_args()
    report = evaluate(_load_jsonl(args.dataset), args.base_url, args.limit, args.judge)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"evaluated": report["evaluated"], "metrics": report["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
