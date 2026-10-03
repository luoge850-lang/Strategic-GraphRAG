import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_isolated_staging import (
    embedding_provenance_matches,
    identity_for,
    normalize_no_llm_candidate_metadata,
)
from strategic_graphrag.build_identity import _fingerprint_files


class IsolatedStagingIdentityTests(unittest.TestCase):
    def test_source_fingerprint_normalizes_checkout_newlines(self):
        with tempfile.TemporaryDirectory() as lf_dir, tempfile.TemporaryDirectory() as crlf_dir:
            lf_root = Path(lf_dir)
            crlf_root = Path(crlf_dir)
            lf_file = lf_root / "src" / "sample.py"
            crlf_file = crlf_root / "src" / "sample.py"
            lf_file.parent.mkdir()
            crlf_file.parent.mkdir()
            lf_file.write_bytes(b"first line\nsecond line\n")
            crlf_file.write_bytes(b"first line\r\nsecond line\r\n")

            self.assertEqual(
                _fingerprint_files([lf_file], root=lf_root),
                _fingerprint_files([crlf_file], root=crlf_root),
            )

    def test_identity_describes_the_executed_local_pipeline_not_machine_defaults(self):
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "deepseek",
            "LLM_MODEL": "machine-default-model",
            "GRAPH_EMBEDDING_BACKEND": "sentence_transformers",
        }):
            identity = identity_for([Path(__file__)], "identity-test").to_dict()

        self.assertEqual(identity["extraction_provider"], "local_rules_and_tables")
        self.assertEqual(identity["extraction_model"], "not_applicable")
        self.assertEqual(identity["query_model"], "disabled_for_local_acceptance")
        self.assertEqual(identity["report_model"], "disabled_for_local_acceptance")
        self.assertEqual(identity["embedding_backend"], "chroma_onnx")

    def test_embedding_backend_and_model_must_match_runtime_manifest(self):
        identity = {
            "embedding_backend": "chroma_onnx",
            "embedding_model": "all-MiniLM-L6-v2",
        }
        manifest = {
            "configured_embedding_backend": "chroma_onnx",
            "runtime_embedding_backend": "chroma_onnx",
            "embedding_model": "all-MiniLM-L6-v2",
            "backend_consistent": True,
        }
        self.assertTrue(embedding_provenance_matches(identity, manifest))

        wrong_backend = {**identity, "embedding_backend": "sentence_transformers"}
        wrong_model = {**manifest, "embedding_model": "different-model"}
        wrong_runtime = {**manifest, "runtime_embedding_backend": "sentence_transformers"}
        self.assertFalse(embedding_provenance_matches(wrong_backend, manifest))
        self.assertFalse(embedding_provenance_matches(identity, wrong_model))
        self.assertFalse(embedding_provenance_matches(identity, wrong_runtime))

    def test_candidate_report_distinguishes_configured_llm_from_actual_calls(self):
        candidates = [{
            "llm_provider": "deepseek",
            "llm_model": "configured-default",
            "prompt_version": "configured-prompt",
            "llm": {"calls": 0, "network_calls": 0},
        }]
        normalize_no_llm_candidate_metadata(candidates)
        self.assertEqual(candidates[0]["configured_llm_provider"], "deepseek")
        self.assertEqual(candidates[0]["llm_provider"], "not_used")
        self.assertEqual(candidates[0]["llm_model"], "not_used")
        self.assertEqual(candidates[0]["execution_extraction_mode"], "local_rules_and_tables")

        with self.assertRaisesRegex(RuntimeError, "unexpectedly made a model/network call"):
            normalize_no_llm_candidate_metadata([{"llm": {"calls": 1}}])


if __name__ == "__main__":
    unittest.main()
