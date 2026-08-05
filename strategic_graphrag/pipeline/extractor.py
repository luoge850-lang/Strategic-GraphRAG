# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Triple Extractor (LLM + Rule Dual-Engine)
=============================================================
Extracts (source, relation, target, evidence) triples from
financial text using a hybrid LLM + rule-based approach.

LLM Engine: Groq/Llama-3.3-70B for semantic understanding
Rule Engine: regex pattern matching + ontology dictionary lookup
"""

import os
import re
import json
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict

from ..ontology.entity_registry import (
    CANONICAL_MAP, resolve_entity, norm_id,
    is_banned, is_generic, is_competitor, is_non_risk,
    is_company_blacklisted, extract_mechanisms,
)
from ..ontology.relation_inference import (
    infer_relation, detect_causal_strength,
    validate_triple, VALID_RELATIONS, ENTITY_CATEGORIES,
    _is_concession, _is_negated, _is_counterfactual,
    _has_concession_pivot, _clean_signal,
)

logger = logging.getLogger("TripleExtractor")

# =============================================================================
# LLM Extraction Prompt
# =============================================================================

EXTRACTION_SYSTEM_PROMPT = """Extract financial causal triples from SEC 10-K text. Return ONLY a JSON array.

Entity categories: Company, RiskFactor, Strategy, FinancialMetric, Product, Market, Region, Regulation, Mechanism, Event.

Relation types: CAUSES, DECREASES, INCREASES, MITIGATES, EXPOSED_TO, TRIGGERS, AMPLIFIES, CONSTRAINS, IMPLEMENTS, PRODUCES, OPERATES_IN, DEPENDS_ON, COMPETES_WITH.

Critical rules:
- Entity names in UPPER_SNAKE_CASE (e.g., SUPPLY_CHAIN_DISRUPTION, REVENUE, NVIDIA_CORPORATION)
- evidence_sentence MUST be a verbatim quote from the text (30-500 chars)
- Extract 5-15 triples. Quality over quantity.
- Never use generic names like "employees", "customers", "management"

JSON format:
[{"source": "ENTITY", "source_category": "RiskFactor", "target": "ENTITY", "target_category": "FinancialMetric", "relation": "DECREASES", "evidence_sentence": "verbatim quote here"}]"""


class TripleExtractor:
    """
    Hybrid triple extractor combining LLM semantic extraction with
    rule-based pattern matching for financial knowledge graph construction.
    Supports Gemini (free), Groq (free), and DeepSeek (paid) via LLMProvider.
    """

    def __init__(self, llm_provider=None, model_name: str = None):
        """
        Args:
            llm_provider: LLMProvider instance (auto-created from env if None)
            model_name: LLM model identifier (auto-detected if None)
        """
        from ..llm_provider import get_llm

        if llm_provider is not None:
            self.llm = llm_provider
        else:
            self.llm = get_llm()

        self.model_name = model_name or self.llm.default_model
        self._llm_enabled = self.llm.available

        if self._llm_enabled:
            logger.info(f"LLM extraction enabled: {self.llm.provider}/{self.model_name}")
        else:
            logger.warning("No LLM provider configured. LLM extraction disabled. "
                           "Set GEMINI_API_KEY in .env for free Gemini access.")

    # ── LLM Extraction ──

    def llm_extract(self, text: str, max_tokens: int = 512) -> List[Dict]:
        """Extract triples using LLM with auto-fallback across providers."""
        if not self._llm_enabled:
            return []

        prompt = f"{EXTRACTION_SYSTEM_PROMPT}\n\nTEXT:\n{text[:2000]}\n\nReturn ONLY valid JSON array."

        # Use auto-fallback: primary → gemini → ollama → deepseek
        result = self.llm.extract_json_with_fallback(
            prompt,
            model=self.model_name,
        )
        if result is None:
            return []
        if isinstance(result, dict):
            result = [result]
        if isinstance(result, list):
            # Keep only triples whose evidence is a verbatim excerpt from the
            # input text.  A plausible LLM paraphrase is not provenance.
            valid = []
            for item in result:
                if not isinstance(item, dict):
                    continue
                if not item.get("source") or not item.get("target"):
                    continue
                evidence = str(item.get("evidence_sentence", "")).strip()
                if not self._evidence_is_verbatim(evidence, text):
                    logger.warning(
                        "Dropping LLM triple with non-verbatim evidence: %s -> %s",
                        item.get("source"), item.get("target"),
                    )
                    continue

                item["relation"] = str(item.get("relation", "")).strip().upper()
                valid_relation, _ = validate_triple(
                    str(item.get("source_category", "")).strip(),
                    str(item.get("target_category", "")).strip(),
                    item["relation"],
                    str(item.get("source", "")),
                    str(item.get("target", "")),
                )
                if not valid_relation:
                    continue
                item["evidence_sentence"] = evidence[:500]
                valid.append(item)
            return valid
        return []

    @staticmethod
    def _normalize_evidence_text(value: str) -> str:
        """Normalize whitespace while preserving the words of an excerpt."""
        return re.sub(r"\s+", " ", str(value or "")).strip()

    @classmethod
    def _evidence_is_verbatim(cls, evidence: str, source_text: str) -> bool:
        """Return True only when evidence is an exact normalized text span."""
        normalized_evidence = cls._normalize_evidence_text(evidence)
        normalized_source = cls._normalize_evidence_text(source_text)
        return (
            20 <= len(normalized_evidence) <= 500
            and bool(normalized_source)
            and normalized_evidence in normalized_source
        )

    # ── Rule-Based Extraction ──
    # Strategy context patterns: detect strategy implementation from SEC language
    STRATEGY_CONTEXT_PATTERNS: Dict[str, List[str]] = {
        "SUPPLY_CHAIN_DIVERSIFICATION": [
            r"diversif(?:y|ied|ying)\s+(?:our\s+)?suppl",
            r"multi[\s-]sourc",
            r"qualif(?:y|ied|ying)\s+(?:additional|new|alternative)\s+(?:suppl|manufactur|foundr)",
            r"expand(?:ed|ing)?\s+(?:our\s+)?(?:supplier|manufacturing|foundry)\s+(?:base|network|capacity)",
            r"second[\s-]sourc",
        ],
        "R_AND_D_INVESTMENT": [
            r"(?:increas|continu|significant|heav|substantial)(?:ed|ing|e|y)?\s+(?:investment|spending|expenditure)(?:\s+in)?\s+(?:research|development|r\s*&?\s*d)",
            r"r\s*&?\s*d\s+(?:investment|spending|expenditure|expense)",
            r"invest(?:ed|ing|ment)?\s+(?:in|heavily in)\s+(?:research|development|innovation)",
        ],
        "COST_OPTIMIZATION": [
            r"cost[\s-](?:reduction|optimization|saving|efficien)",
            r"(?:reduc|lower|cut|optimiz)(?:ed|ing|e)\s+(?:our\s+)?(?:cost|expense|spending)",
            r"(?:operating|operation)\s+efficien",
        ],
        "MARKET_EXPANSION": [
            r"(?:expand|enter|grow)(?:ed|ing)?\s+(?:into|in)\s+(?:new\s+)?(?:market|region|geograph|country)",
            r"geographic\s+(?:expansion|diversification|growth)",
            r"(?:international|global)\s+expansion",
        ],
        "STRATEGIC_ACQUISITION": [
            r"(?:acquis|merger|acquire)(?:ition|ed|ing|es)?\s+(?:of|strategy|to\s+expand)",
            r"strategic\s+(?:investment|acquisition|partnership)",
        ],
        "PRODUCT_DIVERSIFICATION": [
            r"product\s+(?:diversif|portfolio\s+exp|line\s+exp)",
            r"(?:expand|diversif)(?:ed|ing)?\s+(?:our\s+)?product\s+(?:portfolio|line|offering)",
            r"(?:new|additional)\s+product\s+(?:introduction|launch|line)",
        ],
        "SUPPLY_CHAIN_RESILIENCE": [
            r"supply\s+chain\s+(?:resilien|robust|strengthen|secure)",
            r"(?:build|strengthen|improve|enhance)(?:ing)?\s+(?:supply\s+chain|supplier\s+relationship)",
            r"long[\s-]term\s+supply\s+(?:agreement|contract|relationship)",
        ],
        "TECHNOLOGY_INVESTMENT": [
            r"(?:invest|spend)(?:ed|ing|ment)?\s+(?:in|heavily\s+in)\s+(?:new\s+)?(?:tech|platform|architecture|infrastructure)",
            r"(?:technology|platform|architecture)\s+(?:investment|upgrade|development)",
        ],
        "TALENT_ACQUISITION": [
            r"(?:hire|recruit|attract)(?:ed|ing)?\s+(?:top|key|skilled|technical|engineer)",
            r"(?:talent|workforce)\s+(?:acquisition|expansion|development|retention)",
        ],
        "REGULATORY_COMPLIANCE": [
            r"(?:regulatory|compliance|regulation)\s+(?:compliance|management|program|strategy)",
            r"compl(?:y|ied|iance)\s+(?:with|program)",
            r"(?:monitor|manage|address)(?:ing)?\s+(?:regulatory|compliance|export\s+control)",
        ],
        "CUSTOMER_FOCUS_STRATEGY": [
            r"customer[\s-](?:focus|centric|driven|relationship)",
            r"(?:deepen|strengthen|expand)(?:ing)?\s+(?:customer|client)\s+(?:relationship|engagement)",
            r"(?:close|direct)\s+(?:customer|client)\s+(?:relationship|engagement|collaboration)",
        ],
    }

    def rule_extract(self, text: str) -> List[Dict]:
        """
        Extract triples using deterministic rule matching against
        the canonical entity registry and relation inference rules.

        Enhanced with:
        - Context-aware strategy detection from SEC language patterns
        - Regulation→Risk causal chain detection
        - Company→Strategy implementation detection
        - Improved evidence sentence matching (causal language filtering)
        - Mechanism extraction from linguistic causal patterns
        """
        triples: List[Dict] = []
        text_lower = text.lower()

        # Step 1: Find all canonical entities in the text
        found_entities: Dict[str, Tuple[str, str]] = {}  # raw_key → (name, category)

        for key, (name, cat) in CANONICAL_MAP.items():
            if len(key) <= 2:
                continue
            if "_" in key:
                phrase = re.escape(key.replace("_", " "))
                if re.search(rf'\b{phrase}s?\b', text_lower):
                    found_entities[key] = (name, cat)
            else:
                if re.search(rf'\b{re.escape(key)}s?\b', text_lower):
                    found_entities[key] = (name, cat)

        # Step 1b: Detect NVIDIA (always include the primary company)
        if "nvidia" in text_lower or "nvidia_corporation" in found_entities:
            found_entities["nvidia"] = ("NVIDIA_CORPORATION", "Company")

        # Step 2: Extract mechanism patterns
        for mech_name, mech_cat in extract_mechanisms(text):
            found_entities[f"_mech_{mech_name}"] = (mech_name, mech_cat)

        # Step 2b: Context-aware strategy detection
        detected_strategies: Dict[str, Tuple[str, str]] = {}
        for strategy_name, patterns in self.STRATEGY_CONTEXT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    detected_strategies[strategy_name.lower()] = (strategy_name, "Strategy")
                    break

        # Step 3: Group by category
        risks = {k: v for k, v in found_entities.items() if v[1] == "RiskFactor"}
        metrics = {k: v for k, v in found_entities.items() if v[1] == "FinancialMetric"}
        strategies = {k: v for k, v in found_entities.items() if v[1] == "Strategy"}
        # Merge detected strategies with registry strategies
        for k, v in detected_strategies.items():
            if k not in strategies:
                strategies[k] = v
        markets = {k: v for k, v in found_entities.items() if v[1] == "Market"}
        regions = {k: v for k, v in found_entities.items() if v[1] == "Region"}
        products = {k: v for k, v in found_entities.items() if v[1] == "Product"}
        regulations = {k: v for k, v in found_entities.items() if v[1] == "Regulation"}
        events = {k: v for k, v in found_entities.items() if v[1] == "Event"}
        mechanisms = {k: v for k, v in found_entities.items() if v[1] == "Mechanism"}
        companies = {k: v for k, v in found_entities.items() if v[1] == "Company"}

        # Step 4: Generate candidate triples by category pairs
        MAX_OUT = 10  # Increased from 5 for richer extraction
        node_degree: Dict[str, int] = defaultdict(int)

        def _can_emit(entity_name: str) -> bool:
            n = norm_id(entity_name)
            if node_degree[n] >= MAX_OUT:
                return False
            node_degree[n] += 1
            return True

        def _make_triple(src_name, src_cat, tgt_name, tgt_cat, ctx_text="", ev_sent=""):
            """Construct a validated triple with evidence-relation cross-check."""
            if norm_id(src_name) == norm_id(tgt_name):
                return None
            # P2-FIX #6: Reject triples with no meaningful evidence.
            # Callers already check this but defense-in-depth against edge cases
            # where empty evidence leaks through.
            ev_raw = (ev_sent or ctx_text).strip()
            if len(ev_raw) < 20:
                return None
            ev_text = ev_raw[:500].lower()
            rel = infer_relation(src_cat, tgt_cat, ctx_text, src_name, tgt_name)
            cs = detect_causal_strength(ev_text)
            valid, reason = validate_triple(src_cat, tgt_cat, rel, src_name, tgt_name)
            if not valid:
                return None

            # ── Evidence-Relation Cross-Validation ──
            # Check if the evidence text actually contains keywords matching the
            # inferred relationship direction. If not, the evidence does not
            # support the claimed relationship → downgrade confidence.
            dir_keywords = TripleExtractor.RELATION_EVIDENCE_KEYWORDS.get(rel, [])
            if dir_keywords and ev_text:
                has_direction = any(kw in ev_text for kw in dir_keywords)
                if not has_direction:
                    # Evidence doesn't semantically match the relation direction.
                    # Downgrade: DIRECT → INDIRECT, INDIRECT → SPECULATIVE
                    downgrade_map = {
                        "DIRECT_CAUSALITY": "INDIRECT_CAUSALITY",
                        "INDIRECT_CAUSALITY": "RISK_ASSOCIATION",
                        "RISK_ASSOCIATION": "SPECULATIVE_RELATION",
                        "DISCLOSED_EXPOSURE": "RISK_ASSOCIATION",
                    }
                    cs = downgrade_map.get(cs, cs)
            return {
                "source": src_name, "source_category": src_cat,
                "target": tgt_name, "target_category": tgt_cat,
                "relation": rel,
                "causal_strength": cs,
                "evidence_sentence": ev_sent or ctx_text[:500],
            }

        # ═══ COMPANY → Strategy (IMPLEMENTS) ═══
        for ck, (cn, _) in companies.items():
            for sk, (sn, _) in strategies.items():
                ev = self._find_evidence(text, ck, sk)
                if ev and len(ev) > 20 and _can_emit(cn):
                    t = _make_triple(cn, "Company", sn, "Strategy", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Causal Chain: Risk → Mechanism → FinancialMetric ═══
        for rk, (rn, _) in risks.items():
            for mk, (mn, _) in mechanisms.items():
                ev = self._find_evidence(text, rk, mk)
                if ev and len(ev) > 20 and _can_emit(rn):
                    t = _make_triple(rn, "RiskFactor", mn, "Mechanism", ev, ev)
                    if t:
                        triples.append(t)
                # Mechanism → Financial (causal chain continuation)
                for fk, (fn, _) in metrics.items():
                    ev2 = self._find_evidence(text, mk, fk)
                    if ev2 and len(ev2) > 20 and _can_emit(mn):
                        t2 = _make_triple(mn, "Mechanism", fn, "FinancialMetric", ev2, ev2)
                        if t2:
                            triples.append(t2)

        # ═══ Risk → FinancialMetric (direct impact) ═══
        for rk, (rn, _) in risks.items():
            for fk, (fn, _) in metrics.items():
                ev = self._find_evidence(text, rk, fk)
                if ev and len(ev) > 30 and _can_emit(rn):
                    t = _make_triple(rn, "RiskFactor", fn, "FinancialMetric", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Risk → Risk (amplification chains) ═══
        risk_items = list(risks.items())
        for i, (rk, (rn, _)) in enumerate(risk_items):
            for j, (rk2, (rn2, _)) in enumerate(risk_items):
                if i >= j:
                    continue
                ev = self._find_evidence(text, rk, rk2)
                if ev and len(ev) > 20 and _can_emit(rn):
                    t = _make_triple(rn, "RiskFactor", rn2, "RiskFactor", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Strategy → Risk (MITIGATES) — with stronger evidence filtering ═══
        for sk, (sn, _) in strategies.items():
            for rk, (rn, _) in risks.items():
                ev = self._find_mitigation_evidence(text, sk, rk)
                if ev and len(ev) > 25 and _can_emit(sn):
                    t = _make_triple(sn, "Strategy", rn, "RiskFactor", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Regulation → Risk (CAUSES regulatory risk) ═══
        for reg_k, (reg_name, _) in regulations.items():
            for rk, (rn, _) in risks.items():
                ev = self._find_evidence(text, reg_k, rk)
                if ev and len(ev) > 20 and _can_emit(reg_name):
                    t = _make_triple(reg_name, "Regulation", rn, "RiskFactor", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Regulation → Market (CONSTRAINS) ═══
        for reg_k, (reg_name, _) in regulations.items():
            for mk, (mn, _) in markets.items():
                ev = self._find_evidence(text, reg_k, mk)
                if ev and len(ev) > 20 and _can_emit(reg_name):
                    t = _make_triple(reg_name, "Regulation", mn, "Market", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Market → Risk (market conditions create risks) ═══
        for mk, (mn, _) in markets.items():
            for rk, (rn, _) in risks.items():
                ev = self._find_evidence(text, mk, rk)
                if ev and len(ev) > 20 and _can_emit(mn):
                    t = _make_triple(mn, "Market", rn, "RiskFactor", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Region → Risk (geographic exposure) ═══
        for rk, (region_name, _) in regions.items():
            for risk_k, (risk_name, _) in risks.items():
                ev = self._find_evidence(text, rk, risk_k)
                if ev and len(ev) > 20 and _can_emit(region_name):
                    t = _make_triple(region_name, "Region", risk_name, "RiskFactor", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Event → Risk (events trigger risks) ═══
        for ek, (en, _) in events.items():
            for rk, (rn, _) in risks.items():
                ev = self._find_evidence(text, ek, rk)
                if ev and len(ev) > 20 and _can_emit(en):
                    t = _make_triple(en, "Event", rn, "RiskFactor", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Company → Market/Region (OPERATES_IN) ═══
        for ck, (cn, _) in companies.items():
            for mk, (mn, _) in markets.items():
                ev = self._find_evidence(text, ck, mk)
                if ev and len(ev) > 15 and _can_emit(cn):
                    t = _make_triple(cn, "Company", mn, "Market", ev, ev)
                    if t:
                        triples.append(t)

        # ═══ Company → Product (PRODUCES) ═══
        for ck, (cn, _) in companies.items():
            for pk, (pn, _) in products.items():
                ev = self._find_evidence(text, ck, pk)
                if ev and len(ev) > 15 and _can_emit(cn):
                    t = _make_triple(cn, "Company", pn, "Product", ev, ev)
                    if t:
                        triples.append(t)

        return triples

    @staticmethod
    def _find_mitigation_evidence(text: str, strategy_term: str, risk_term: str) -> str:
        """Find evidence sentence showing strategy ACTUALLY mitigates risk.
        Requires mitigation/address/reduce language, not just co-occurrence."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        st_lower = strategy_term.lower().replace("_", " ")
        tt_lower = risk_term.lower().replace("_", " ")

        # Strong mitigation language patterns
        mitigation_patterns = [
            r'\b(?:mitigat|address|reduc|manag|hedge|protect|'
            r'counter|offset|alleviat|minimiz|limit|control|'
            r'oversee|monitor|govern|framework|program)\w*\b',
            r'\b(?:to\s+(?:address|manage|mitigate|reduce))\b',
        ]

        for sent in sentences:
            sl = sent.lower().strip()
            if len(sl) < 15:
                continue
            # Check for mitigation language
            has_mitigation = any(re.search(p, sl) for p in mitigation_patterns)
            if not has_mitigation:
                continue
            # At least one term present, prefer both
            score = 0.0
            if st_lower in sl:
                score += 2.0
            if tt_lower in sl:
                score += 2.0
            # Bonus for strong causal + mitigation combo
            if re.search(r'\b(?:in order to|so as to|thereby|thus|therefor)\b', sl):
                score += 1.0
            if score >= 2.0:
                return sent[:500]

        # Fallback: regular evidence finder
        return TripleExtractor._find_evidence(text, strategy_term, risk_term)

    # ── Combined Extraction ──

    def extract(self, text: str) -> List[Dict]:
        """
        Extract triples using both LLM and rule-based engines,
        then merge and deduplicate.
        """
        # LLM extraction
        llm_triples = self.llm_extract(text)

        # Rule-based extraction
        rule_triples = self.rule_extract(text)

        # Merge and deduplicate by (source, relation, target) key
        seen: Set[Tuple[str, str, str]] = set()
        merged: List[Dict] = []

        # P1-FIX #3: LLM first — semantic extraction produces better evidence
        # sentences than rule-based co-occurrence matching.
        for triple_list in [llm_triples, rule_triples]:
            for t in triple_list:
                key = (
                    norm_id(str(t.get("source", ""))),
                    str(t.get("relation", "")).upper(),
                    norm_id(str(t.get("target", ""))),
                )
                if key not in seen:
                    seen.add(key)
                    # Ensure causal_strength is set
                    if "causal_strength" not in t or not t.get("causal_strength"):
                        ev = str(t.get("evidence_sentence", ""))
                        t["causal_strength"] = detect_causal_strength(ev)
                    merged.append(t)

        return merged

    # ── Filtering ──

    def filter_triples(self, triples: List[Dict], text: str = "") -> List[Dict]:
        """
        Filter extracted triples through quality gates:
        - Remove banned/generic entities
        - Reclassify mislabeled entities
        - Enforce hubness and cardinality limits
        """
        filtered: List[Dict] = []
        # Cardinality limits per relation type
        limits = {"DISCLOSES": 2, "EXPOSED_TO": 4}
        counts: Dict[str, int] = defaultdict(int)

        for t in triples:
            s_raw = str(t.get("source", "")).strip()
            t_raw = str(t.get("target", "")).strip()
            rel = str(t.get("relation", "")).strip().upper()
            s_cat = str(t.get("source_category", "")).strip()
            t_cat = str(t.get("target_category", "")).strip()

            # Basic validation
            if not s_raw or not t_raw or not rel:
                continue
            if rel not in VALID_RELATIONS:
                continue
            if norm_id(s_raw) == norm_id(t_raw):
                continue

            # Filter noise
            if is_banned(s_raw) or is_banned(t_raw):
                continue
            if is_generic(s_raw) or is_generic(t_raw):
                continue
            if t_cat == "RiskFactor" and is_non_risk(t_raw):
                continue
            if s_cat == "RiskFactor" and is_non_risk(s_raw):
                continue
            if is_company_blacklisted(s_raw) or is_company_blacklisted(t_raw):
                continue

            # Evidence is mandatory for the graph pipeline.  If a caller has
            # supplied source text, reject paraphrased or fabricated evidence.
            evidence = str(t.get("evidence_sentence", "")).strip()
            if text and not self._evidence_is_verbatim(evidence, text):
                continue

            # Reclassify competitors as Company + COMPETITION_RISK
            if is_competitor(s_raw):
                s_cat = "Company"
            if is_competitor(t_raw):
                t_cat = "Company"

            # Cardinality enforcement
            if rel in limits:
                if counts[rel] >= limits[rel]:
                    continue
                counts[rel] += 1

            # Canonicalize entity names
            sn, sc = resolve_entity(s_raw, s_cat)
            tn, tc = resolve_entity(t_raw, t_cat)

            valid_relation, _ = validate_triple(sc, tc, rel, sn, tn)
            if valid_relation:
                t["source"] = sn
                t["source_category"] = sc
                t["target"] = tn
                t["target_category"] = tc
                t["relation"] = rel
                filtered.append(t)

        return filtered

    # ── Helper: Evidence Sentence Finder ──

    # Relation-direction keyword verification — evidence must contain at least one
    # keyword that semantically matches the relationship direction
    RELATION_EVIDENCE_KEYWORDS = {
        "DECREASES": ["decrease", "decline", "reduce", "lower", "harm", "negatively",
                       "adversely affect", "could reduce", "may reduce", "impair",
                       "diminish", "negatively impact", "hurt", "deteriorate"],
        "INCREASES": ["increase", "grow", "raise", "higher", "improve", "positively",
                       "drive growth", "boost", "expand", "accelerate", "strengthen"],
        "CAUSES": ["cause", "lead to", "result in", "due to", "because", "driven by",
                    "creates", "generates", "produces", "trigger"],
        "MITIGATES": ["mitigate", "address", "reduce risk", "counter", "hedge",
                       "protect against", "manage risk", "offset", "alleviate"],
        "EXPOSED_TO": ["exposed to", "subject to", "vulnerable", "susceptible",
                        "face risk", "at risk of", "could be affected by"],
        "TRIGGERS": ["trigger", "activate", "set off", "spark", "prompt", "initiate"],
        "AMPLIFIES": ["amplify", "magnify", "worsen", "exacerbate", "compound",
                       "aggravate", "intensify", "deepen"],
        "CONSTRAINS": ["constrain", "restrict", "limit", "cap", "prevent", "prohibit",
                        "block", "bar", "hinder"],
        "IMPLEMENTS": ["implement", "adopt", "employ", "deploy", "utilize", "execute",
                        "apply", "practice", "invest in"],
    }

    @staticmethod
    def _find_evidence(text: str, source_term: str, target_term: str,
                       relation: str = "") -> str:
        """
        Find the best evidence sentence that contains BOTH entity terms
        AND (for causal relations) at least one relation-direction keyword.

        Returns empty string if no sentence meets the criteria.
        """
        if not text:
            return ""
        # P2-FIX #5: Quick pre-filter — skip sentence-level scanning entirely if
        # either entity term doesn't appear in the text at all. This avoids
        # O(sentences × entity_pairs) regex work for impossible matches.
        text_lower = text.lower()
        st_lower = source_term.lower().replace("_", " ")
        tt_lower = target_term.lower().replace("_", " ")
        if st_lower not in text_lower or tt_lower not in text_lower:
            return ""
        sentences = re.split(r'(?<=[.!?])\s+', text)

        # Get direction keywords for this relation type
        dir_keywords = TripleExtractor.RELATION_EVIDENCE_KEYWORDS.get(relation.upper(), [])
        is_causal = bool(dir_keywords)  # Only structural relations lack direction keywords

        best_sent = ""
        best_score = 0.0

        for sent in sentences:
            sl = sent.lower().strip()
            if len(sl) < 15:
                continue

            has_source = st_lower in sl
            has_target = tt_lower in sl

            # For short entity names, require word-boundary matching to avoid false positives
            if len(st_lower) <= 5 and has_source:
                has_source = bool(re.search(rf'\b{re.escape(st_lower)}\b', sl))
            if len(tt_lower) <= 5 and has_target:
                has_target = bool(re.search(rf'\b{re.escape(tt_lower)}\b', sl))

            # Both terms must be in the SAME sentence for a valid match
            if not (has_source and has_target):
                continue

            # P1-FIX #4: Reject concessive sentences ("Despite X, Y increased")
            # and negated sentences ("X did not affect Y").
            # These contain both entities but do NOT assert the expected causal direction.
            if _is_concession(sl) or _is_negated(sl) or _is_counterfactual(sl):
                continue

            # For causal relations: require at least one direction keyword
            has_direction = True
            if is_causal:
                has_direction = any(kw in sl for kw in dir_keywords)

            if has_direction:
                # Strong match: both entities + direction keyword (or structural relation)
                return sent[:500]

            # ISSUE-FIX #1: For causal relations, do NOT fall back to weak
            # entity co-occurrence.  "X and Y are mentioned in the same sentence"
            # is NOT evidence that X CAUSES/DECREASES/INCREASES Y.  Without a
            # direction keyword, the sentence does not support the relation.
            # Previously this collected weak matches as a fallback, producing
            # misleading evidence like "Our success depends on..." for INCREASES.
            if is_causal:
                continue  # skip — no direction keyword = no causal evidence

            # For structural relations only (PRODUCES, OPERATES_IN, etc.):
            # entity co-occurrence IS sufficient evidence.
            score = 1.0 + min(len(sl) / 500.0, 0.5)
            if score > best_score:
                best_score = score
                best_sent = sent[:500]

        # P0-FIX: When no sentence contains both entities, return empty string.
        # Previously returned text[:300] which fabricated "evidence" completely
        # unrelated to the triple. Callers already check `if ev and len(ev) > N`
        # so empty return naturally skips triple creation.
        if best_sent:
            return best_sent[:500]
        return ""


# =============================================================================
# Sentence-Level Evidence Extraction
# =============================================================================

def extract_sentences(text: str) -> List[Dict[str, any]]:
    """
    Extract individual sentences with metadata for evidence layer population.

    Returns list of dicts with: text, char_offset, sentence_index
    """
    try:
        import nltk
        from nltk.tokenize import sent_tokenize
        raw_sents = sent_tokenize(text)
    except (ImportError, LookupError):
        raw_sents = re.split(r'(?<=[.!?])\s+', text)

    sentences = []
    offset = 0
    for i, sent in enumerate(raw_sents):
        sent = sent.strip()
        if len(sent) < 10:
            offset += len(sent) + 1
            continue
        sentences.append({
            "text": sent,
            "char_offset": offset,
            "sentence_index": i,
        })
        offset += len(sent) + 1

    return sentences
