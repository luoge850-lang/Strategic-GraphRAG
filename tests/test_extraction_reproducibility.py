import os
import unittest
from unittest.mock import Mock, patch

from strategic_graphrag.llm_provider import LLMProvider
from strategic_graphrag.pipeline.extractor import TripleExtractor


class FakeExtractionLLM:
    available = True
    provider = "fake"
    last_success_provider = "fake"
    last_success_model = "fake-model"

    def __init__(self):
        self.calls = []

    def get_task_model(self, task):
        return "fake-model"

    def extract_json_with_fallback(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return {"triples": []}


class FallbackProbe(LLMProvider):
    def __init__(self):
        self.provider = "deepseek"
        self.default_model = "deepseek-model"
        self.calls = []

    @property
    def available(self):
        return True

    def extract_json(self, *args, **kwargs):
        return None

    def switch_provider(self, provider, model=None):
        self.provider = provider
        self.default_model = model or f"{provider}-model"

    def chat(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return '{"ok": true}'


class ExtractionReproducibilityTests(unittest.TestCase):
    def test_extract_json_passes_temperature_to_chat(self):
        provider = LLMProvider.__new__(LLMProvider)
        provider.chat = Mock(return_value='{"triples": []}')

        result = provider.extract_json(
            "prompt",
            system_prompt="system",
            model="model",
            max_tokens=100,
            temperature=0.35,
        )

        self.assertEqual(result, {"triples": []})
        provider.chat.assert_called_once_with(
            prompt="prompt",
            system_prompt="system",
            model="model",
            max_tokens=100,
            temperature=0.35,
            json_mode=True,
        )

    def test_extract_json_with_fallback_passes_temperature_to_fallback_chat(self):
        provider = FallbackProbe()
        with patch.dict(os.environ, {"LLM_FALLBACK_PROVIDERS": "ollama"}, clear=False):
            result = provider.extract_json_with_fallback("prompt", temperature=0.6)

        self.assertEqual(result, {"ok": True})
        self.assertEqual(provider.calls[0][1]["temperature"], 0.6)

    def test_extractor_defaults_to_zero_and_records_temperature(self):
        llm = FakeExtractionLLM()
        with patch.dict(os.environ, {}, clear=True):
            extractor = TripleExtractor(llm_provider=llm)
            self.assertEqual(extractor.llm_extract("Revenue increased."), [])

        self.assertEqual(llm.calls[0][1]["temperature"], 0.0)
        self.assertEqual(extractor.get_llm_stats()["extraction_temperature"], 0.0)

    def test_extractor_accepts_explicit_temperature(self):
        llm = FakeExtractionLLM()
        with patch.dict(os.environ, {"LLM_EXTRACTION_TEMPERATURE": "0.7"}, clear=False):
            extractor = TripleExtractor(llm_provider=llm)
            extractor.llm_extract("Revenue increased.")

        self.assertEqual(llm.calls[0][1]["temperature"], 0.7)
        self.assertEqual(extractor.get_llm_stats()["extraction_temperature"], 0.7)

    def test_invalid_or_out_of_range_temperature_falls_back_to_zero(self):
        for raw in ("-0.1", "2.1", "not-a-number", "nan", "inf"):
            with self.subTest(raw=raw), patch.dict(
                os.environ, {"LLM_EXTRACTION_TEMPERATURE": raw}, clear=False
            ):
                extractor = TripleExtractor(llm_provider=FakeExtractionLLM())
                self.assertEqual(extractor.extraction_temperature, 0.0)


if __name__ == "__main__":
    unittest.main()
