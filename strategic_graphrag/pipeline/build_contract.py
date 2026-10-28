from pathlib import Path
from typing import Any,Dict,List
from strategic_graphrag.document_layer import DocumentLayerReader
from strategic_graphrag.build_identity import make_build_identity
ROOT=Path(__file__).resolve().parents[2]

def identity_for(pdfs: List[Path], corpus_id: str):
    reader = DocumentLayerReader()
    return make_build_identity(
        pdfs,
        corpus_id=corpus_id,
        parser_version=reader.parser_version,
        parser_config_hash=reader.config_hash,
        prompt_version="not_applicable_rule_table_v1",
        extraction_provider="local_rules_and_tables",
        extraction_model="not_applicable",
        query_model="disabled_for_local_acceptance",
        report_model="disabled_for_local_acceptance",
        embedding_backend="chroma_onnx",
        embedding_model="all-MiniLM-L6-v2",
        root=ROOT,
    )

def embedding_provenance_matches(identity: Dict[str, Any], vector: Dict[str, Any]) -> bool:
    configured = str(vector.get("configured_embedding_backend") or vector.get("embedding_backend") or "").strip().lower()
    runtime = str(vector.get("runtime_embedding_backend") or vector.get("embedding_backend") or "").strip().lower()
    identity_backend = str(identity.get("embedding_backend") or "").strip().lower()
    identity_model = str(identity.get("embedding_model") or "").strip()
    manifest_model = str(vector.get("embedding_model") or "").strip()
    return (
        configured == runtime == identity_backend == "chroma_onnx"
        and vector.get("backend_consistent", True) is True
        and bool(identity_model)
        and identity_model == manifest_model
    )

def normalize_no_llm_candidate_metadata(results: List[Dict[str, Any]]) -> None:
    """Separate configured model defaults from models actually called."""
    for item in results:
        llm_stats = item.get("llm") or {}
        if int(llm_stats.get("calls") or 0) or int(llm_stats.get("network_calls") or 0):
            raise RuntimeError("no-LLM staging run unexpectedly made a model/network call")
        item["configured_llm_provider"] = item.pop("llm_provider", None)
        item["configured_llm_model"] = item.pop("llm_model", None)
        item["configured_prompt_version"] = item.pop("prompt_version", None)
        item["llm_provider"] = "not_used"
        item["llm_model"] = "not_used"
        item["prompt_version"] = "not_applicable_rule_table_v1"
        item["execution_extraction_mode"] = "local_rules_and_tables"
