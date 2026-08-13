# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Unified LLM Provider
========================================
Single interface: Gemini (free, REST API), Groq (free, OpenAI compat),
and DeepSeek (paid, OpenAI compat).

Usage:
    from strategic_graphrag.llm_provider import get_llm
    llm = get_llm()
    text = llm.chat("Extract triples from this text...")
    data = llm.extract_json("...", system_prompt="...")
"""

import os
import re
import json
import logging
from typing import Optional, Dict, List

import requests

logger = logging.getLogger("LLMProvider")

# ── Provider Configs ──

PROVIDER_CONFIG = {
    "gemini": {
        "api_key_env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "free": True,
        "type": "gemini_rest",  # Uses native REST API with ?key= param
    },
    "groq": {
        "api_key_env": "GROQ_API_KEY",
        "default_model": "llama-3.3-70b-versatile",
        "free": True,
        "type": "groq_native",  # Uses native groq SDK (already installed)
        "base_url": "https://api.groq.com/openai/v1",
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "default_model": "deepseek-v4-flash",
        "free": False,
        "type": "openai_compat",
        "base_url": "https://api.deepseek.com",
    },
    # ISSUE-FIX #3: Local Ollama — free, no API key, no rate limits.
    "ollama": {
        "api_key_env": "OLLAMA_API_KEY",
        "default_model": "qwen2.5:7b",
        "free": True,
        "type": "ollama",
        "base_url": "http://localhost:11434/v1",
    },
    # Local transformers model — loaded from disk via ModelScope
    # Stores model on D drive, runs on CPU, zero cost, no network needed after download.
    "local": {
        "api_key_env": "LOCAL_MODEL_PATH",  # path to model directory
        "default_model": "D:/modelscope_models/models/Qwen--Qwen2.5-3B-Instruct/snapshots/master",
        "free": True,
        "type": "local_transformers",
    },
}

PROVIDER_MODELS = {
    "gemini": {"default": "gemini-2.5-flash", "fast": "gemini-2.5-flash", "pro": "gemini-2.5-pro"},
    "groq": {"default": "llama-3.3-70b-versatile", "fast": "llama-3.1-8b-instant", "pro": "llama-3.3-70b-versatile"},
    "deepseek": {"default": "deepseek-v4-flash", "fast": "deepseek-v4-flash", "pro": "deepseek-v4-pro"},
    "ollama": {"default": "qwen2.5:7b", "fast": "qwen2.5:3b", "pro": "qwen2.5:14b"},
    "local": {"default": "Qwen2.5-7B-Instruct", "fast": "Qwen2.5-3B-Instruct", "pro": "Qwen2.5-14B-Instruct"},
}


class LLMProvider:
    """Unified LLM interface — Gemini REST, Groq/DeepSeek via OpenAI compat."""

    def __init__(
        self,
        provider: str = None,
        api_key: str = None,
        model: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2000,
    ):
        from dotenv import load_dotenv
        load_dotenv()

        self.provider = str(
            provider or os.getenv("LLM_PROVIDER", "deepseek")
        ).strip().lower()
        if self.provider not in PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider: {self.provider}. Options: {list(PROVIDER_CONFIG.keys())}")

        cfg = PROVIDER_CONFIG[self.provider]
        self.api_key = api_key or os.getenv(cfg["api_key_env"], "")
        # Resolve a model only within the selected provider.  A global
        # DeepSeek model must never leak into a Groq/Gemini fallback client.
        configured_model = os.getenv("LLM_MODEL", "").strip()
        known_models = set(PROVIDER_MODELS.get(self.provider, {}).values())
        requested_model = str(model or configured_model).strip()
        if requested_model in known_models:
            self.default_model = requested_model
        elif requested_model:
            # Never send a model identifier belonging to another provider.
            # This protects a new provider run from a stale LLM_MODEL value.
            logger.warning(
                "Model %s is not registered for provider %s; using %s",
                requested_model,
                self.provider,
                cfg["default_model"],
            )
            self.default_model = cfg["default_model"]
        else:
            self.default_model = cfg["default_model"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_free = cfg["free"]
        self._type = cfg["type"]
        self._base_url = cfg.get("base_url", "")

        self._client = None  # OpenAI client for groq/deepseek/ollama
        self._local_model = None  # transformers model for local provider
        self._local_tokenizer = None
        self.last_success_provider = None
        self.last_success_model = None

        # Ollama and local are keyless — always "available"
        self._available = (self._type in ("ollama", "local_transformers")) or \
                          (bool(self.api_key) and "your_" not in self.api_key)

        if self._available:
            if self._type in ("openai_compat", "ollama"):
                self._init_openai_client()
            elif self._type == "groq_native":
                self._init_groq_client()
            elif self._type == "local_transformers":
                self._init_local_model()
            # gemini_rest uses requests directly — no client init needed
        else:
            logger.warning(f"LLM '{self.provider}' not configured. Set {cfg['api_key_env']} in .env")

    def _init_openai_client(self):
        try:
            from openai import OpenAI
            # Ollama is local and keyless — use a dummy key
            api_key = self.api_key if self.api_key else "ollama"
            self._client = OpenAI(api_key=api_key, base_url=self._base_url)
            logger.info(f"LLM '{self.provider}' ready (model={self.default_model}, free={self.is_free})")
        except ImportError:
            logger.error("openai not installed. Run: pip install openai")
            self._available = False

    def _init_groq_client(self):
        try:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
            logger.info(f"LLM '{self.provider}' ready via native Groq SDK (model={self.default_model}, free={self.is_free})")
        except ImportError:
            logger.error("groq not installed. Run: pip install groq")
            self._available = False

    def _init_local_model(self):
        """Lazy-init: model is loaded on first chat() call to avoid memory hit at startup."""
        try:
            import torch  # noqa: F401
            self._available = True
            logger.info(f"LLM 'local' ready (model={self.default_model}, CPU inference, free)")
        except ImportError:
            logger.error("torch not installed. Run: pip install torch transformers")
            self._available = False

    def _ensure_local_model_loaded(self):
        """Load the model if not already in memory."""
        if self._local_model is not None and self._local_tokenizer is not None:
            return True
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = self.default_model  # path to model directory
            logger.info(f"Loading local model from {model_path}...")

            self._local_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._local_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                dtype=torch.float32,  # CPU-friendly
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            logger.info(f"Local model loaded. Params: {sum(p.numel() for p in self._local_model.parameters()) / 1e9:.1f}B")
            return True
        except Exception as e:
            logger.error(f"Failed to load local model: {e}")
            self._available = False
            return False

    @property
    def available(self) -> bool:
        if self._type in ("openai_compat", "groq_native", "ollama"):
            return self._available and self._client is not None
        if self._type == "local_transformers":
            return self._available  # Model loaded lazily on first call
        return self._available  # Gemini REST always available via requests

    # ── Main Chat Interface ──

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        json_mode: bool = False,
    ) -> Optional[str]:
        """Send a chat completion. Returns text or None."""
        if not self.available:
            return None

        if self._type == "gemini_rest":
            return self._chat_gemini(prompt, system_prompt, model, temperature, max_tokens, json_mode)
        elif self._type == "groq_native":
            return self._chat_groq(prompt, system_prompt, model, temperature, max_tokens, json_mode)
        elif self._type == "local_transformers":
            return self._chat_local(prompt, system_prompt, model, temperature, max_tokens, json_mode)
        else:
            return self._chat_openai(prompt, system_prompt, model, temperature, max_tokens, json_mode)

    # ── Gemini REST API ──

    def _chat_gemini(
        self, prompt: str, system_prompt: str, model: str,
        temperature: float, max_tokens: int, json_mode: bool,
    ) -> Optional[str]:
        model_name = model or self.default_model
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent"
        )

        # Build contents
        contents = []
        if system_prompt:
            contents.append({"role": "user", "parts": [{"text": system_prompt}]})
            contents.append({"role": "model", "parts": [{"text": "Understood."}]})
        contents.append({"role": "user", "parts": [{"text": prompt}]})

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature if temperature is not None else self.temperature,
                "maxOutputTokens": max_tokens or self.max_tokens,
            },
        }

        if json_mode:
            body["generationConfig"]["responseMimeType"] = "application/json"

        try:
            resp = requests.post(url, params={"key": self.api_key}, json=body, timeout=60)
            if resp.status_code != 200:
                logger.error(f"Gemini API error {resp.status_code}: {resp.text[:300]}")
                return None
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            logger.error(f"Gemini call failed: {e}")
            return None

    # ── Groq Native SDK ──

    def _chat_groq(
        self, prompt: str, system_prompt: str, model: str,
        temperature: float, max_tokens: int, json_mode: bool,
    ) -> Optional[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Groq has no native JSON mode — prepend instruction if needed.
        # Use the modified prompt in the actual message payload.
        if json_mode:
            prompt_clean = prompt + "\n\nReturn ONLY valid JSON. No markdown, no explanation."
        else:
            prompt_clean = prompt
        messages.append({"role": "user", "content": prompt_clean})

        try:
            resp = self._client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq call failed: {e}")
            return None

    # ── Local Transformers ──

    def _chat_local(
        self, prompt: str, system_prompt: str, model: str,
        temperature: float, max_tokens: int, json_mode: bool,
    ) -> Optional[str]:
        """Run inference on local transformers model (CPU)."""
        if not self._ensure_local_model_loaded():
            return None

        import torch

        # Build messages in Qwen chat format
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Apply chat template
        try:
            full_prompt = self._local_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback: simple concatenation
            sp = f"{system_prompt}\n\n" if system_prompt else ""
            full_prompt = f"{sp}User: {prompt}\nAssistant:"

        inputs = self._local_tokenizer(full_prompt, return_tensors="pt")
        # Truncate to avoid OOM on CPU (16GB RAM constraint)
        max_input = 1024
        if inputs["input_ids"].shape[1] > max_input:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -max_input:]

        with torch.no_grad():
            outputs = self._local_model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_tokens,
                temperature=temperature if temperature is not None else self.temperature,
                do_sample=(temperature or self.temperature) > 0,
                pad_token_id=self._local_tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        result = self._local_tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return result.strip() if result else None

    # ── OpenAI-compatible (DeepSeek) ──

    def _chat_openai(
        self, prompt: str, system_prompt: str, model: str,
        temperature: float, max_tokens: int, json_mode: bool,
    ) -> Optional[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        kwargs = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }

        if json_mode and self.provider == "deepseek":
            kwargs["response_format"] = {"type": "json_object"}

        # DeepSeek V4 Flash defaults to thinking mode.  For structured
        # extraction, thinking can consume the entire output budget before
        # emitting JSON.  Keep the fast model non-thinking by default, while
        # allowing an explicit environment override for experiments.
        if self.provider == "deepseek" and str(kwargs["model"]).startswith("deepseek-v4-flash"):
            thinking = os.getenv("DEEPSEEK_THINKING", "disabled").strip().lower()
            if thinking not in {"enabled", "disabled"}:
                thinking = "disabled"
            kwargs["extra_body"] = {"thinking": {"type": thinking}}

        try:
            resp = self._client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed ({self.provider}): {e}")
            return None

    # ── JSON Extraction ──

    def extract_json(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = None,
        max_tokens: int = None,
    ) -> Optional[dict]:
        """Chat with forced JSON, return parsed dict or None."""
        text = self.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model,
            max_tokens=max_tokens,
            json_mode=True,
        )
        if text is None:
            return None

        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            logger.warning(f"JSON parse failed: {text[:200]}...")
            return None

    # ── Convenience ──

    def get_model_name(self, tier: str = "default") -> str:
        return PROVIDER_MODELS.get(self.provider, {}).get(tier, self.default_model)

    def get_task_model(self, task: str) -> str:
        """Resolve a provider-safe model override for one pipeline task."""
        configured = os.getenv(f"LLM_{str(task).upper()}_MODEL", "").strip()
        return self.fallback_model_for(self.provider, configured or self.default_model)

    @classmethod
    def fallback_model_for(cls, provider: str, requested_model: str = None) -> str:
        """Resolve a model name without crossing provider boundaries."""
        cfg = PROVIDER_CONFIG.get(provider, {})
        default_model = cfg.get("default_model", "")
        if not requested_model:
            return default_model

        known_models = set(PROVIDER_MODELS.get(provider, {}).values())
        known_models.add(default_model)
        if requested_model in known_models:
            return requested_model

        logger.warning(
            "Model %s is not registered for fallback provider %s; using %s",
            requested_model, provider, default_model,
        )
        return default_model

    def switch_provider(self, provider: str, model: str = None):
        """Switch to a different provider at runtime."""
        self.__init__(provider=provider, model=model)
        return self

    # ── Auto-Fallback ──

    # Cross-provider fallback is opt-in.  This prevents a project configured
    # for DeepSeek from silently sending financial filing text to another
    # vendor.  Set LLM_FALLBACK_PROVIDERS=ollama,deepseek when explicitly
    # desired.
    FALLBACK_ORDER = []

    @classmethod
    def configured_fallback_order(cls, primary_provider: str) -> List[str]:
        """Return the explicitly allowed fallback providers."""
        raw = os.getenv("LLM_FALLBACK_PROVIDERS", "").strip()
        if not raw:
            return []
        providers = []
        for provider in raw.split(","):
            provider = provider.strip().lower()
            if provider in PROVIDER_CONFIG and provider != primary_provider and provider not in providers:
                providers.append(provider)
        return providers

    def chat_with_fallback(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        json_mode: bool = False,
    ) -> Optional[str]:
        """
        Try primary provider first, then fall back through alternatives.
        Each provider gets one attempt. Returns first successful result.
        """
        # Try primary first
        result = self.chat(prompt, system_prompt, model, temperature, max_tokens, json_mode)
        if result is not None:
            self.last_success_provider = self.provider
            self.last_success_model = model or self.default_model
            return result

        # Build fallback list (skip primary, skip unavailable)
        tried = {self.provider}
        original_provider = self.provider
        original_model = self.default_model

        for fb in self.configured_fallback_order(original_provider):
            if fb in tried:
                continue
            tried.add(fb)

            # Check if fallback is configured
            cfg = PROVIDER_CONFIG.get(fb, {})
            api_key = os.getenv(cfg.get("api_key_env", ""), "")
            if fb != "ollama" and (not api_key or "your_" in api_key):
                continue  # Not configured, skip

            logger.info(f"Falling back to {fb} (primary {original_provider} returned None)...")
            try:
                self.switch_provider(fb)
                if not self.available:
                    continue
                fb_model = self.fallback_model_for(fb, model)
                result = self.chat(prompt, system_prompt, fb_model, temperature, max_tokens, json_mode)
                if result is not None:
                    logger.info(f"Fallback {fb} succeeded")
                    success_provider = fb
                    success_model = fb_model
                    self.switch_provider(original_provider, model=original_model)
                    self.last_success_provider = success_provider
                    self.last_success_model = success_model
                    return result
            except Exception as e:
                logger.warning(f"Fallback {fb} error: {e}")
                continue

        # All fallbacks exhausted — restore original provider
        self.switch_provider(original_provider, model=original_model)
        self.last_success_provider = None
        self.last_success_model = None
        return None

    def extract_json_with_fallback(
        self,
        prompt: str,
        system_prompt: str = "",
        model: str = None,
        max_tokens: int = None,
    ) -> Optional[dict]:
        """extract_json with automatic provider fallback."""
        result = self.extract_json(
            prompt, system_prompt, model=model, max_tokens=max_tokens
        )
        if result is not None:
            self.last_success_provider = self.provider
            self.last_success_model = model or self.default_model
            return result

        original_provider = self.provider
        original_model = self.default_model
        tried = {original_provider}

        for fb in self.configured_fallback_order(original_provider):
            if fb in tried:
                continue
            tried.add(fb)
            cfg = PROVIDER_CONFIG.get(fb, {})
            api_key = os.getenv(cfg.get("api_key_env", ""), "")
            if fb != "ollama" and (not api_key or "your_" in api_key):
                continue

            logger.info(f"extract_json falling back to {fb}...")
            try:
                self.switch_provider(fb)
                if not self.available:
                    continue
                fallback_model = self.fallback_model_for(fb, model)
                result = self.chat(
                    prompt,
                    system_prompt,
                    model=fallback_model,
                    max_tokens=max_tokens,
                    json_mode=True,
                )
                if result is None:
                    continue
                text = result.strip()
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"```\s*$", "", text)
                try:
                    parsed = json.loads(text)
                    self.last_success_provider = fb
                    self.last_success_model = fallback_model
                    self.switch_provider(original_provider, model=original_model)
                    self.last_success_provider = fb
                    self.last_success_model = fallback_model
                    return parsed
                except json.JSONDecodeError:
                    match = re.search(r"\[.*\]", text, re.DOTALL)
                    if match:
                        try:
                            parsed = json.loads(match.group(0))
                            self.last_success_provider = fb
                            self.last_success_model = fallback_model
                            self.switch_provider(original_provider, model=original_model)
                            self.last_success_provider = fb
                            self.last_success_model = fallback_model
                            return parsed
                        except json.JSONDecodeError:
                            pass
                    continue
            except Exception:
                continue

        self.switch_provider(original_provider, model=original_model)
        self.last_success_provider = None
        self.last_success_model = None
        return None


# ── Singleton ──

_llm_instance: Optional[LLMProvider] = None


def get_llm(provider: str = None, model: str = None) -> LLMProvider:
    global _llm_instance
    if _llm_instance is None or provider is not None or model is not None:
        _llm_instance = LLMProvider(provider=provider, model=model)
    return _llm_instance
