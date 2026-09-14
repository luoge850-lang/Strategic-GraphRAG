import json
import tempfile
import unittest
from pathlib import Path

from strategic_graphrag.llm_response_cache import (
    LLMResponseCache,
    LLMResponseCacheCorrupt,
    LLMResponseCacheMiss,
)
from strategic_graphrag.pipeline.extractor import TripleExtractor


class FakeProvider:
    available = True
    provider = "fake"
    default_model = "fake-model"

    def __init__(self, response=None, available=True):
        self.response = response if response is not None else {"triples": []}
        self.available = available
        self.calls = 0
        self.network_calls = 0
        self.last_success_provider = "fake"
        self.last_success_model = "fake-model"

    def get_task_model(self, task):
        return "fake-model"

    def extract_json_with_fallback(self, *args, **kwargs):
        self.calls += 1
        self.network_calls += 1
        return self.response


class LLMResponseCacheTests(unittest.TestCase):
    @staticmethod
    def _request(cache, **overrides):
        values = {
            "operation": "extract_json_with_fallback",
            "request_provider": "fake",
            "request_model": "fake-model",
            "temperature": 0.0,
            "max_tokens": 3000,
            "prompt": "full prompt text",
        }
        values.update(overrides)
        return cache.make_request(**values)

    def test_key_changes_with_prompt_model_and_temperature(self):
        cache = LLMResponseCache(mode="off")
        base = self._request(cache)
        self.assertNotEqual(base.key, self._request(cache, prompt="changed").key)
        self.assertNotEqual(base.key, self._request(cache, request_model="other-model").key)
        self.assertNotEqual(base.key, self._request(cache, temperature=0.1).key)
        self.assertNotEqual(base.key, self._request(cache, max_tokens=3001).key)

    def test_record_then_replay_returns_identical_extractor_result(self):
        response = {
            "triples": [
                {
                    "source": "SUPPLY_CHAIN_DISRUPTION",
                    "source_category": "RiskFactor",
                    "target": "REVENUE",
                    "target_category": "FinancialMetric",
                    "relation": "decreases",
                    "evidence_sentence": (
                        "Supply chain disruption decreases revenue by affecting shipments."
                    ),
                }
            ]
        }
        text = response["triples"][0]["evidence_sentence"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "responses.jsonl")
            record_provider = FakeProvider(response=response)
            record_extractor = TripleExtractor(
                llm_provider=record_provider,
                response_cache=LLMResponseCache(path=path, mode="record"),
            )
            recorded = record_extractor.llm_extract(text)

            replay_provider = FakeProvider(response={"triples": []})
            replay_extractor = TripleExtractor(
                llm_provider=replay_provider,
                response_cache=LLMResponseCache(path=path, mode="replay"),
            )
            replayed = replay_extractor.llm_extract(text)

            self.assertEqual(recorded, replayed)
            self.assertEqual(record_provider.calls, 1)
            self.assertEqual(replay_provider.calls, 0)
            self.assertEqual(record_extractor.get_llm_stats()["cache"]["writes"], 1)
            self.assertEqual(replay_extractor.get_llm_stats()["cache"]["hits"], 1)
            self.assertEqual(replay_extractor.get_llm_stats()["network_calls"], 0)

    def test_replay_miss_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = LLMResponseCache(
                path=str(Path(temp_dir) / "missing.jsonl"), mode="replay"
            )
            with self.assertRaisesRegex(LLMResponseCacheMiss, "Replay is fail-closed"):
                cache.lookup(self._request(cache))

    def test_bad_json_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "bad.jsonl"
            path.write_text("not json\n", encoding="utf-8")
            with self.assertRaises(LLMResponseCacheCorrupt):
                LLMResponseCache(path=str(path), mode="replay")

    def test_duplicate_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicate.jsonl"
            record_cache = LLMResponseCache(path=str(path), mode="record")
            request = self._request(record_cache)
            self.assertTrue(record_cache.store(request, {"triples": []}))
            first = json.loads(path.read_text(encoding="utf-8"))
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(first) + "\n")
            with self.assertRaisesRegex(LLMResponseCacheCorrupt, "Duplicate cache key"):
                LLMResponseCache(path=str(path), mode="replay")

    def test_record_does_not_overwrite_existing_frozen_response(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "responses.jsonl")
            cache = LLMResponseCache(path=path, mode="record")
            request = self._request(cache)
            self.assertTrue(cache.store(request, {"triples": [{"id": "first"}]}))
            self.assertFalse(cache.store(request, {"triples": [{"id": "second"}]}))
            replay = LLMResponseCache(path=path, mode="replay")
            self.assertEqual(replay.lookup(request).response["triples"][0]["id"], "first")
            self.assertEqual(len(Path(path).read_text(encoding="utf-8").splitlines()), 1)

    def test_off_does_not_read_or_write_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "must-not-exist.jsonl"
            provider = FakeProvider()
            extractor = TripleExtractor(
                llm_provider=provider,
                response_cache=LLMResponseCache(path=str(path), mode="off"),
            )
            extractor.llm_extract("Revenue increased by demand for products.")
            stats = extractor.get_llm_stats()["cache"]
            self.assertFalse(path.exists())
            self.assertEqual(stats["mode"], "off")
            self.assertEqual(stats["hits"], 0)
            self.assertEqual(stats["misses"], 0)
            self.assertEqual(stats["writes"], 0)
            self.assertEqual(provider.calls, 1)


if __name__ == "__main__":
    unittest.main()
