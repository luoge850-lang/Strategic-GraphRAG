"""Machine-readable metric definitions and conservative proportion summaries."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class MetricSpec:
    metric_id: str
    definition: str
    evaluation_unit: str
    numerator: str
    denominator: str
    dataset_version: str
    exclusion_conditions: List[str] = field(default_factory=list)
    aggregation: str = "micro"
    uncertainty: str = "not_reported"
    scope: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


METRIC_SPECS: Dict[str, MetricSpec] = {
    "pdf_page_coverage": MetricSpec(
        "pdf_page_coverage", "Pages with a retained parse record", "physical_page",
        "pages with PARSED status", "total PDF pages", "document-layer/v1",
        ["do not count OCR_REQUIRED as parsed"], "micro", "exact_conservation", "PDF integrity only",
    ),
    "table_cell_binding": MetricSpec(
        "table_cell_binding", "Gold table cells whose value/unit/period/source location are all correct", "annotated_cell",
        "fully correct cells", "annotated cells", "table-quality/v1",
        ["exclude unreviewed candidates", "do not estimate recall from extracted rows"], "clustered_by_table", "wilson_or_clustered", "annotated table scope",
    ),
    "evidence_claim_precision": MetricSpec(
        "evidence_claim_precision", "Accepted claims that satisfy entity, relation, and evidence checks", "accepted_claim",
        "correct accepted claims", "reviewed accepted claims", "extraction-gold/v1",
        ["AI-assisted rows are not independent human gold"], "clustered_by_document", "wilson_or_clustered", "closed annotated scope only",
    ),
    "retrieval_recall_at_k": MetricSpec(
        "retrieval_recall_at_k", "Expected evidence units surfaced in top K", "question",
        "questions with expected unit surfaced", "answerable reviewed questions", "retrieval-gold/v1",
        ["exclude execution failures", "report page and evidence units separately"], "paired_by_question", "paired_bootstrap_or_exact", "benchmark-specific",
    ),
    "retrieval_hit_rate_at_k": MetricSpec(
        "retrieval_hit_rate_at_k", "Questions with at least one expected evidence unit in top K", "question",
        "answerable questions with any expected unit surfaced", "answerable reviewed questions", "retrieval-gold/v1",
        ["exclude execution failures", "do not call this Recall@K"], "paired_by_question", "paired_interval", "benchmark-specific",
    ),
    "answer_correctness": MetricSpec(
        "answer_correctness", "Answer matches the independent label including numeric/unit/period constraints", "question",
        "correct answers", "successful answer requests", "answer-gold/v1",
        ["dependency/model errors are not answers", "do not convert abstentions into correct answers"], "paired_by_question", "paired_interval", "benchmark-specific",
    ),
    "correct_abstention": MetricSpec(
        "correct_abstention", "Unsupported or unanswerable questions are explicitly abstained", "question",
        "correct abstentions", "successful unanswerable questions", "answer-gold/v1",
        ["exclude execution failures", "partial answers require their own label"], "paired_by_question", "wilson_or_clustered", "benchmark-specific",
    ),
    "execution_success_rate": MetricSpec(
        "execution_success_rate", "Requests completed without a dependency/model/timeout error", "request",
        "SUCCEEDED requests", "all attempted requests", "response-contract/v2",
        ["never remove failures from the denominator"], "micro", "wilson", "engineering run only",
    ),
    "latency_ms": MetricSpec(
        "latency_ms", "Wall-clock request latency including dependency calls", "request",
        "measured milliseconds", "successful or all requests as declared", "runtime/v1",
        ["report cache hit/miss separately", "do not infer load capacity from a smoke run"], "distribution", "quantiles", "deployment profile required",
    ),
}


def metric_registry() -> Dict[str, Dict[str, Any]]:
    return {key: spec.to_dict() for key, spec in METRIC_SPECS.items()}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Optional[List[float]]:
    """Return a two-sided 95% Wilson interval; never returns [1, 1] for n=0."""
    if total < 0 or successes < 0 or successes > total:
        raise ValueError("successes and total must satisfy 0 <= successes <= total")
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1.0 + (z * z / total)
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        (proportion * (1.0 - proportion) / total) + (z * z / (4 * total * total))
    ) / denominator
    return [round(max(0.0, centre - margin), 6), round(min(1.0, centre + margin), 6)]


def summarize_binary(values: Iterable[bool]) -> Dict[str, Any]:
    observed = [bool(value) for value in values]
    successes = sum(observed)
    total = len(observed)
    return {
        "successes": successes,
        "total": total,
        "proportion": round(successes / total, 6) if total else None,
        "interval": wilson_interval(successes, total),
        "status": "NOT_RUN" if total == 0 else "PASS",
    }


__all__ = ["MetricSpec", "METRIC_SPECS", "metric_registry", "wilson_interval", "summarize_binary"]
