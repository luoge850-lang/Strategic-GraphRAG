"""Run six comparable retrieval implementations on development families only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import shutil
import statistics
import sys
import tempfile
import threading
import time
import weakref
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import ONNXMiniLM_L6_V2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from strategic_graphrag.engine.query_understanding import parse_query

K0 = 60
FINAL_K = 10
ARM_BUDGET = 50
MODES = (
    "keyword_bm25",
    "dense_semantic",
    "keyword_dense_fusion_rrf",
    "fusion_graph_expansion",
    "fusion_graph_expansion_temporal",
    "fusion_temporal_only",
)
METHOD_NAMES_ZH = {
    "keyword_bm25": "关键词检索（BM25）",
    "dense_semantic": "语义向量检索",
    "keyword_dense_fusion_rrf": "关键词与语义融合检索（RRF）",
    "fusion_graph_expansion": "融合检索＋知识图谱扩展",
    "fusion_graph_expansion_temporal": "融合检索＋知识图谱扩展＋时间约束",
    "fusion_temporal_only": "融合检索＋时间约束（无图扩展诊断对照）",
}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _verify_immutable_artifacts(build_dir: Path) -> None:
    ledger_path = build_dir / "artifact_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    failures = []
    for item in ledger.get("artifacts") or []:
        relative = str(item.get("path") or "")
        target = build_dir / relative
        if not relative or not target.is_file():
            failures.append(f"missing:{relative or '<empty>'}")
            continue
        if target.stat().st_size != int(item.get("bytes", -1)) or _hash(target) != item.get("sha256"):
            failures.append(f"hash_mismatch:{relative}")
    if failures:
        raise RuntimeError("Candidate immutable artifacts changed: " + ", ".join(failures[:12]))


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).casefold())


def _page_key(metadata: dict[str, Any], fallback: str) -> str:
    filing = str(metadata.get("source_filing") or "").strip()
    page = metadata.get("page")
    if filing and str(page).isdigit():
        return f"{filing}#{int(page)}"
    return f"chunk#{fallback}"


def _rss() -> int:
    if sys.platform == "win32":
        import ctypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get_current_process = kernel.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_memory = psapi.GetProcessMemoryInfo
        get_memory.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
        get_memory.restype = ctypes.c_int
        return int(counters.WorkingSetSize) if get_memory(get_current_process(), ctypes.byref(counters), counters.cb) else -1
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return -1


def _system_memory_total() -> int | str:
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        call = ctypes.WinDLL("kernel32", use_last_error=True).GlobalMemoryStatusEx
        call.argtypes = [ctypes.POINTER(MEMORYSTATUSEX)]
        call.restype = ctypes.c_int
        return int(status.ullTotalPhys) if call(ctypes.byref(status)) else "NOT_AVAILABLE"
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return "NOT_AVAILABLE"


class PeakRSS:
    def __init__(self, interval_seconds: float = 0.01):
        self.interval = interval_seconds
        self.peak = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        while not self._stop.is_set():
            current = _rss()
            if current > 0:
                self.peak = max(self.peak, current)
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak = _rss()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        self.peak = max(self.peak, _rss())


class _CleanupLease:
    """Weak-referenceable lifetime token for scratch-resource cleanup."""


def _cleanup_verified_runtime(state: dict[str, Any]) -> None:
    """Close the mutable runtime copy and re-verify its immutable source on every exit path."""
    try:
        monitor = state.get("peak_monitor")
        if monitor is not None and monitor._thread is not None:
            monitor.__exit__(None, None, None)
    finally:
        try:
            client = state.get("client")
            if client is not None:
                client.close()
        finally:
            try:
                workspace = state.get("workspace")
                if workspace is not None:
                    workspace.cleanup()
            finally:
                _verify_immutable_artifacts(Path(state["build_dir"]))


class BM25:
    def __init__(self, documents: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [_tokenize(item["document"]) for item in documents]
        self.term_freq = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / max(len(self.lengths), 1)
        self.doc_freq: Counter[str] = Counter()
        for terms in self.term_freq:
            self.doc_freq.update(terms.keys())

    def search(self, query: str, limit: int = ARM_BUDGET) -> list[dict[str, Any]]:
        query_terms = _tokenize(query)
        n = len(self.documents)
        scores = [0.0] * n
        for term in query_terms:
            df = self.doc_freq.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
            for i, frequencies in enumerate(self.term_freq):
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / max(self.avg_length, 1.0))
                scores[i] += idf * tf * (self.k1 + 1) / norm
        ranked = sorted(range(n), key=lambda i: (-scores[i], i))
        return [
            {**self.documents[i], "score": scores[i], "method_rank": rank}
            for rank, i in enumerate(ranked[:limit], start=1)
            if scores[i] > 0
        ]


def _aggregate_pages(hits: list[dict[str, Any]], *, score_key: str = "score") -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(hits, start=1):
        key = _page_key(hit.get("metadata") or {}, str(hit.get("id") or ""))
        existing = pages.get(key)
        score = float(hit.get(score_key) or 0.0)
        if existing is None or score > existing["score"]:
            pages[key] = {
                "page_key": key,
                "id": hit.get("id"),
                "document": hit.get("document") or "",
                "metadata": hit.get("metadata") or {},
                "score": score,
                "source_rank": rank,
            }
    return pages


def _ranked(values: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(values.values(), key=lambda item: (-float(item["score"]), item["page_key"]))


def _rrf(*arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores: dict[str, float] = defaultdict(float)
    records: dict[str, dict[str, Any]] = {}
    for arm in arms:
        for rank, item in enumerate(arm, start=1):
            key = item["page_key"]
            scores[key] += 1.0 / (K0 + rank)
            records.setdefault(key, item)
    combined = []
    for key, score in scores.items():
        combined.append({**records[key], "score": score, "rrf_score": score})
    return sorted(combined, key=lambda item: (-item["score"], item["page_key"]))


def _graph_expand(
    fused: list[dict[str, Any]],
    edges_by_page: dict[str, list[dict[str, Any]]],
    page_documents: dict[str, dict[str, Any]],
    *,
    final_limit: int | None = FINAL_K,
) -> list[dict[str, Any]]:
    seeds = fused[:FINAL_K]
    seed_entities: set[str] = set()
    for item in seeds:
        for edge in edges_by_page.get(item["page_key"], []):
            seed_entities.update({str(edge.get("source") or "").casefold(), str(edge.get("target") or "").casefold()})
    seed_entities.discard("")
    scores = {item["page_key"]: float(item["score"]) for item in fused}
    for page_key, edges in edges_by_page.items():
        if page_key in scores:
            continue
        shared = sum(
            1 for edge in edges
            if str(edge.get("source") or "").casefold() in seed_entities
            or str(edge.get("target") or "").casefold() in seed_entities
        )
        if shared and page_key in page_documents:
            scores[page_key] = 0.5 * shared / (K0 + FINAL_K)
    output = []
    for page_key, score in scores.items():
        base = page_documents.get(page_key)
        if base:
            output.append({
                **base,
                "page_key": page_key,
                "score": score,
                "graph_expansion": page_key not in {item["page_key"] for item in fused},
            })
    ranked = sorted(output, key=lambda item: (-item["score"], item["page_key"]))
    return ranked if final_limit is None else ranked[:final_limit]


def _select_mode_hits(
    mode: str,
    query: str,
    bm25_pages: list[dict[str, Any]],
    dense_pages: list[dict[str, Any]],
    edges_by_page: dict[str, list[dict[str, Any]]],
    page_documents: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str, dict[str, Any]]:
    """Apply one arm; the temporal-only diagnostic must never execute graph expansion."""
    use_bm25 = mode in {"keyword_bm25", "keyword_dense_fusion_rrf", "fusion_graph_expansion", "fusion_graph_expansion_temporal", "fusion_temporal_only"}
    use_dense = mode in {"dense_semantic", "keyword_dense_fusion_rrf", "fusion_graph_expansion", "fusion_graph_expansion_temporal", "fusion_temporal_only"}
    stats: dict[str, Any] = {
        "bm25_unique_candidate_pages": len(bm25_pages) if use_bm25 else None,
        "dense_unique_candidate_pages": len(dense_pages) if use_dense else None,
        "fusion_unique_candidate_pages": None,
        "graph_seed_pages": None,
        "graph_expansion_candidate_pages": None,
        "graph_added_candidate_pages": None,
        "temporal_candidate_pages_before_filter": None,
        "temporal_candidate_pages_after_filter": None,
    }
    if mode == "keyword_bm25":
        hits = bm25_pages[:FINAL_K]
        return hits, None, "OK" if hits else "NO_HITS", {**stats, "returned_unique_pages": len(hits)}
    if mode == "dense_semantic":
        hits = dense_pages[:FINAL_K]
        return hits, None, "OK" if hits else "NO_HITS", {**stats, "returned_unique_pages": len(hits)}
    fused = _rrf(bm25_pages, dense_pages)
    stats["fusion_unique_candidate_pages"] = len(fused)
    if mode == "keyword_dense_fusion_rrf":
        hits = fused[:FINAL_K]
        return hits, None, "OK" if hits else "NO_HITS", {**stats, "returned_unique_pages": len(hits)}
    if mode == "fusion_temporal_only":
        hits, plan, status, time_stats = _temporal_filter(query, fused, edges_by_page)
        stats.update(time_stats)
        return hits, plan, status, {**stats, "returned_unique_pages": len(hits)}
    stats["graph_seed_pages"] = min(FINAL_K, len(fused))
    expanded = _graph_expand(
        fused,
        edges_by_page,
        page_documents,
        final_limit=None,
    )
    stats["graph_expansion_candidate_pages"] = len(expanded)
    stats["graph_added_candidate_pages"] = sum(bool(item.get("graph_expansion")) for item in expanded)
    if mode == "fusion_graph_expansion":
        hits = expanded[:FINAL_K]
        return hits, None, "OK" if hits else "NO_HITS", {**stats, "returned_unique_pages": len(hits)}
    if mode == "fusion_graph_expansion_temporal":
        hits, plan, status, time_stats = _temporal_filter(query, expanded, edges_by_page)
        stats.update(time_stats)
        return hits, plan, status, {**stats, "returned_unique_pages": len(hits)}
    raise ValueError(f"Unsupported retrieval method: {mode}")


def _temporal_filter(
    query: str,
    graph_hits: list[dict[str, Any]],
    edges_by_page: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any], str, dict[str, int]]:
    plan = parse_query(query).to_dict()
    years = sorted({int(value) for value in plan.get("fiscal_years", []) if str(value).isdigit()})
    document_scope = str(plan.get("document_scope") or "").strip()
    temporal_requested = bool(plan.get("temporal_required") or years or document_scope)
    if temporal_requested and not years and not document_scope:
        return [], plan, "INSUFFICIENT_TEMPORAL_SCOPE", {
            "temporal_candidate_pages_before_filter": len(graph_hits),
            "temporal_candidate_pages_after_filter": 0,
        }
    allowed = []
    for hit in graph_hits:
        edges = edges_by_page.get(hit["page_key"], [])
        if document_scope:
            edges = [edge for edge in edges if str(edge.get("source_filing") or "").casefold() == document_scope.casefold()]
        if years:
            edges = [edge for edge in edges if int(edge.get("fact_year") or 0) in years]
        if temporal_requested and not edges:
            continue
        allowed.append(hit)
    return allowed[:FINAL_K], plan, "OK" if allowed else "NO_HITS_AFTER_TEMPORAL_FILTER", {
        "temporal_candidate_pages_before_filter": len(graph_hits),
        "temporal_candidate_pages_after_filter": len(allowed),
    }


def _load_build(build_dir: Path) -> tuple[str, dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    _verify_immutable_artifacts(build_dir)
    identity = json.loads((build_dir / "build_identity.json").read_text(encoding="utf-8"))
    manifest = json.loads((build_dir / "vector_index_manifest.json").read_text(encoding="utf-8"))
    verification = json.loads((build_dir / "verification.json").read_text(encoding="utf-8"))
    validation = json.loads((build_dir / "phases" / "validated.json").read_text(encoding="utf-8"))
    if verification.get("status") != "PASS" or validation.get("status") != "PASS":
        raise RuntimeError("Candidate build lacks passing integrity verification and validation phase")
    if validation.get("build_id") != identity.get("build_id"):
        raise RuntimeError("Candidate validation phase is not bound to build identity")
    if identity.get("build_id") != manifest.get("build_id") or identity.get("build_id") != verification.get("build_id"):
        raise RuntimeError("Build identity mismatch across package artifacts")
    graph = json.loads((build_dir / "graph.json").read_text(encoding="utf-8"))
    accepted = json.loads((build_dir / "accepted_facts.json").read_text(encoding="utf-8"))
    if graph.get("build_id") != identity["build_id"] or accepted.get("build_id") != identity["build_id"]:
        raise RuntimeError("Graph facts are not bound to candidate build")
    return identity["build_id"], manifest, graph.get("edges") or [], accepted.get("counts") or {}, identity


def run(build_dir: Path, dataset_path: Path, output_path: Path, *, limit_families: int | None = None) -> dict[str, Any]:
    run_started_at = datetime.now(timezone.utc)
    run_started_clock = time.perf_counter()
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite raw matrix output or manifest: {output_path}")
    protocol = ROOT / "docs/financial_qa_candidate_protocol_v4.md"
    lock = ROOT / "requirements-lock-2026-09-19.txt"
    frozen_paths = {
        "runner": Path(__file__).resolve(),
        "query_parser": ROOT / "strategic_graphrag/engine/query_understanding.py",
        "summarizer": ROOT / "scripts/summarize_financial_candidate_run.py",
        "protocol": protocol,
        "dataset": dataset_path.resolve(),
        "dependency_lock": lock,
    }
    frozen_hashes_before = {key: _hash(path) for key, path in frozen_paths.items()}
    build_id, manifest, edges, accepted_counts, identity = _load_build(build_dir)
    query_model = ONNXMiniLM_L6_V2.MODEL_NAME
    if str(manifest.get("embedding_model") or "") != query_model:
        raise RuntimeError(
            f"Query embedding model mismatch: index={manifest.get('embedding_model')!r}, query={query_model!r}"
        )
    family_rows = _jsonl(dataset_path)
    if not family_rows:
        raise ValueError("No question candidates supplied")
    if limit_families is not None:
        if limit_families < 1:
            raise ValueError("limit_families must be positive")
        family_rows = family_rows[:limit_families]
    if any(row.get("partition") != "development" for row in family_rows):
        raise ValueError("This runner accepts development families only; never run the exposed test split")
    if any(row.get("label_status") != "AI_SOURCE_REVIEW_NOT_HUMAN" for row in family_rows):
        raise ValueError("Every development family must have source-reviewed labels and an explicit review tier")
    if any(row.get("gold_status") != "NOT_GOLD" for row in family_rows):
        raise ValueError("AI/source-reviewed labels must not be promoted to Gold")
    rows: list[dict[str, Any]] = []
    for family in family_rows:
        base = {key: value for key, value in family.items() if key != "candidate_variants"}
        rows.append(base)
        for variant in family.get("candidate_variants") or []:
            rows.append({**base, **variant, "family_id": family["family_id"]})
    vector_dir = build_dir / manifest["path"]
    # Chroma mutates its SQLite/HNSW persistence files during an apparently
    # read-only query. Query an isolated byte-for-byte copy so the verified,
    # immutable candidate package remains untouched.
    vector_workspace = tempfile.TemporaryDirectory(
        prefix="financial-qa-vector-", ignore_cleanup_errors=True
    )
    run_peak = PeakRSS()
    run_peak.__enter__()
    cleanup_state: dict[str, Any] = {
        "build_dir": build_dir,
        "workspace": vector_workspace,
        "client": None,
        "peak_monitor": run_peak,
    }
    cleanup_lease = _CleanupLease()
    cleanup_finalizer = weakref.finalize(cleanup_lease, _cleanup_verified_runtime, cleanup_state)
    runtime_vector_dir = Path(vector_workspace.name) / "vector_index"
    shutil.copytree(vector_dir, runtime_vector_dir)
    client = chromadb.PersistentClient(path=str(runtime_vector_dir))
    cleanup_state["client"] = client
    collection = client.get_collection(
        name=manifest["collection"],
        embedding_function=embedding_functions.DefaultEmbeddingFunction(),
    )
    if collection.count() != int(manifest.get("chunks_indexed", -1)):
        raise RuntimeError("Vector collection count disagrees with manifest")
    raw = collection.get(include=["documents", "metadatas"])
    documents = []
    by_page: dict[str, dict[str, Any]] = {}
    for i, document in enumerate(raw.get("documents") or []):
        item = {
            "id": raw["ids"][i],
            "document": document or "",
            "metadata": (raw.get("metadatas") or [])[i] or {},
        }
        documents.append(item)
        key = _page_key(item["metadata"], item["id"])
        by_page.setdefault(key, item)
    if len(documents) != int(manifest.get("chunks_indexed", -1)):
        raise RuntimeError("Fetched vector corpus count disagrees with manifest")
    bm25 = BM25(documents)
    edges_by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if str(edge.get("build_id") or "") != build_id:
            continue
        filing, page = str(edge.get("source_filing") or ""), edge.get("page")
        if filing and str(page).isdigit():
            edges_by_page[f"{filing}#{int(page)}"].append(edge)
    if not edges_by_page:
        raise RuntimeError("Candidate graph has no build-scoped page evidence")

    # Warm the local ONNX query embedding once; warm-up is recorded separately.
    warm_started = time.perf_counter()
    collection.query(query_texts=[rows[0]["question"]], n_results=1, include=["documents", "metadatas", "distances"])
    warmup_seconds = time.perf_counter() - warm_started

    schedule = []
    for row in rows:
        order = list(MODES)
        random.Random(f"20260924:{row['item_id']}").shuffle(order)
        schedule.extend((row, mode) for mode in order)
    records: list[dict[str, Any]] = []
    latencies: dict[str, list[float]] = defaultdict(list)
    with PeakRSS() as peak:
        for row, mode in schedule:
            started = time.perf_counter()
            rss_before = _rss()
            query = row["question"]
            dense_error = None
            dense_pages: list[dict[str, Any]] = []
            bm25_pages: list[dict[str, Any]] = []
            try:
                bm25_chunk_count = None
                dense_chunk_count = None
                if mode in {"keyword_bm25", "keyword_dense_fusion_rrf", "fusion_graph_expansion", "fusion_graph_expansion_temporal", "fusion_temporal_only"}:
                    bm25_raw = bm25.search(query, ARM_BUDGET)
                    bm25_chunk_count = len(bm25_raw)
                    bm25_pages = _ranked(_aggregate_pages(bm25_raw))[:ARM_BUDGET]
                if mode in {"dense_semantic", "keyword_dense_fusion_rrf", "fusion_graph_expansion", "fusion_graph_expansion_temporal", "fusion_temporal_only"}:
                    dense_result = collection.query(
                        query_texts=[query], n_results=min(ARM_BUDGET, len(documents)),
                        include=["documents", "metadatas", "distances"],
                    )
                    dense_raw = []
                    docs = (dense_result.get("documents") or [[]])[0] or []
                    metas = (dense_result.get("metadatas") or [[]])[0] or []
                    ids = (dense_result.get("ids") or [[]])[0] or []
                    distances = (dense_result.get("distances") or [[]])[0] or []
                    dense_chunk_count = len(docs)
                    for i, doc in enumerate(docs):
                        dense_raw.append({
                            "id": ids[i] if i < len(ids) else "",
                            "document": doc or "",
                            "metadata": metas[i] if i < len(metas) else {},
                            "distance": distances[i] if i < len(distances) else None,
                            "score": 1.0 / (1.0 + max(float(distances[i] or 0.0), 0.0)) if i < len(distances) else 0.0,
                        })
                    dense_pages = _ranked(_aggregate_pages(dense_raw))[:ARM_BUDGET]

                hits, plan, result_status, candidate_counts = _select_mode_hits(
                    mode, query, bm25_pages, dense_pages, edges_by_page, by_page
                )
                candidate_counts["bm25_retrieved_chunks"] = bm25_chunk_count
                candidate_counts["dense_retrieved_chunks"] = dense_chunk_count
                elapsed = time.perf_counter() - started
                latencies[mode].append(elapsed)
                selected = [
                    {
                        "rank": rank,
                        "page_key": hit["page_key"],
                        "document_id": hit.get("id"),
                        "score": hit.get("score"),
                        "metadata": hit.get("metadata"),
                        "excerpt": str(hit.get("document") or "")[:700],
                        "graph_expansion": bool(hit.get("graph_expansion", False)),
                    }
                    for rank, hit in enumerate(hits, start=1)
                ]
                error_text = None
                execution_status = "SUCCESS"
            except Exception as error:  # Preserve failed requests in the raw denominator.
                elapsed = time.perf_counter() - started
                latencies[mode].append(elapsed)
                selected = []
                plan = None
                result_status = "DEPENDENCY_ERROR"
                execution_status = "FAILED"
                error_text = f"{type(error).__name__}: {error}"
            rss_after = _rss()
            if plan is None:
                try:
                    plan = parse_query(query).to_dict()
                except Exception as error:
                    plan = {"status": "QUERY_PLAN_ERROR", "error": type(error).__name__}
            records.append({
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "request_id": f"{row['item_id']}:{mode}",
                "item_id": row["item_id"],
                "family_id": row["family_id"],
                "partition": row["partition"],
                "question_type": row["question_type"],
                "question": query,
                "method": mode,
                "method_name_zh": METHOD_NAMES_ZH[mode],
                "build_id": build_id,
                "query_plan": plan,
                "execution_status": execution_status,
                "result_status": result_status,
                "ranked_evidence": selected,
                "candidate_counts": candidate_counts if execution_status == "SUCCESS" else "PARTIAL_OR_UNAVAILABLE_AFTER_ERROR",
                "reference_answer": row.get("reference_answer", "NOT_AVAILABLE_PENDING_PDF_REVIEW"),
                "answerability_label": row.get("answerability_label", "NOT_REVIEWED"),
                "gold_page_grades": row.get("gold_page_grades", {}),
                "retrieval_scoring_status": row.get("retrieval_scoring_status", "NOT_RUN"),
                "label_tier": row.get("label_tier", "NOT_REVIEWED"),
                "answer": "NOT_RUN_GENERATION_DISABLED",
                "citations": "NOT_RUN_GENERATION_DISABLED",
                "latency_ms": round(elapsed * 1000, 3),
                "process_rss_before_bytes": rss_before if rss_before > 0 else "NOT_AVAILABLE",
                "process_rss_after_bytes": rss_after if rss_after > 0 else "NOT_AVAILABLE",
                "peak_process_rss_bytes": "RECORDED_AT_RUN_LEVEL",
                "llm_calls": 0,
                "embedding_model": manifest.get("embedding_model"),
                "local_embedding_inferences": 1 if mode != "keyword_bm25" else 0,
                "tokens": "NOT_APPLICABLE_GENERATION_DISABLED",
                "verified_cost": "NOT_APPLICABLE_LOCAL_ONLY",
                "recall_at_k": "NOT_RUN_REQUIRES_INDEPENDENT_RELEVANCE_LABELS",
                "ndcg_at_k": "NOT_RUN_REQUIRES_INDEPENDENT_RELEVANCE_LABELS",
                "mrr": "NOT_RUN_REQUIRES_INDEPENDENT_RELEVANCE_LABELS",
                "error": error_text,
                "candidate_filing_hint": row["candidate_filing"],
                "candidate_pages_hint": row["candidate_pages"],
            })
    frozen_hashes_after = {key: _hash(path) for key, path in frozen_paths.items()}
    if frozen_hashes_before != frozen_hashes_after:
        changed = [key for key in frozen_hashes_before if frozen_hashes_before[key] != frozen_hashes_after[key]]
        raise RuntimeError("Frozen experiment inputs changed during execution: " + ", ".join(changed))
    cleanup_finalizer()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=str) + "\n")
    raw_jsonl_sha256 = _hash(output_path)
    installed_versions = {}
    for distribution in ("chromadb", "onnxruntime", "tokenizers", "numpy", "pillow"):
        try:
            installed_versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed_versions[distribution] = "NOT_INSTALLED"
    runtime = {
        "os": platform.platform(),
        "python": sys.version,
        "cpu_count_logical": os.cpu_count() or "NOT_AVAILABLE",
        "memory_total_bytes": _system_memory_total(),
        "process_peak_rss_bytes": max(peak.peak, run_peak.peak) if max(peak.peak, run_peak.peak) > 0 else "NOT_AVAILABLE",
        "warmup_seconds_excluded_from_request_latency": warmup_seconds,
        "warmup_state": "local ONNX query embedding initialized once; result cache not used",
        "concurrency": 1,
        "cold_start_latency": "NOT_MEASURED",
        "model_generation_enabled": False,
        "query_embedding_model": query_model,
        "query_embedding_model_archive_sha256": ONNXMiniLM_L6_V2._MODEL_SHA256,
        "installed_runtime_versions": installed_versions,
        "llm_calls": 0,
        "network_calls": 0,
        "candidate_pool_chunks": len(documents),
        "evidence_budget_unique_pages": FINAL_K,
        "arm_candidate_budget": ARM_BUDGET,
        "bm25": {"k1": 1.5, "b": 0.75},
        "rrf_k0": K0,
        "graph_expansion": "从关键词与语义融合检索前 10 页做一跳共享端点扩展；只使用同构建身份的事实边",
        "temporal_filter": "structured parser fact-year/disclosure scope; fails closed if temporal scope unresolved",
        "latency_ms": {
            mode: {
                "n": len(values),
                "p50": round(statistics.median(values) * 1000, 3) if values else "NOT_RUN",
                "p95": round(sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)] * 1000, 3) if values else "NOT_RUN",
            }
            for mode, values in latencies.items()
        },
    }
    pdf_hashes = identity.get("pdfs") or {}
    run_completed_at = datetime.now(timezone.utc)
    run_wall_seconds = round(time.perf_counter() - run_started_clock, 3)
    manifest_out = {
        "schema": "financial-retrieval-raw-run/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_started_at_utc": run_started_at.isoformat(),
        "run_completed_at_utc": run_completed_at.isoformat(),
        "run_wall_clock_seconds": run_wall_seconds,
        "request_throughput_per_second_including_setup_and_serial_queries": round(
            len(records) / max(run_wall_seconds, 1e-9), 3
        ),
        "build_id": build_id,
        "source_fingerprint": identity.get("source_tree_sha256"),
        "protocol_sha256": _hash(protocol),
        "candidate_dataset_sha256": _hash(dataset_path),
        "runner_sha256": _hash(Path(__file__).resolve()),
        "dependency_lock_sha256": _hash(lock),
        "frozen_input_sha256_before": frozen_hashes_before,
        "frozen_input_sha256_after": frozen_hashes_after,
        "raw_jsonl_sha256": raw_jsonl_sha256,
        "source_pdf_hashes": pdf_hashes,
        "vector_index_chunks": len(documents),
        "accepted_triples": int(accepted_counts.get("accepted_triples") or 0),
        "expanded_fact_edges": int(accepted_counts.get("expanded_fact_edges") or len(edges)),
        "matrix_modes": list(MODES),
        "records": len(records),
        "candidate_family_count": len({row["family_id"] for row in family_rows}),
        "selected_family_ids": sorted({row["family_id"] for row in family_rows}),
        "limited_smoke_run": limit_families is not None,
        "candidate_question_count": len(rows),
        "quality_metrics": "AI/PDF source-reviewed development diagnostics; not human Gold or independent test",
        "cold_start_latency": "NOT_MEASURED",
        "runtime": runtime,
    }
    manifest_path.write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit-families", type=int, default=None, help="Optional smoke-run limit; omitted for the full frozen set")
    args = parser.parse_args()
    report = run(
        Path(args.build_dir).resolve(), Path(args.dataset).resolve(), Path(args.output).resolve(),
        limit_families=args.limit_families,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
