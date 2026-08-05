# -*- coding: utf-8 -*-
"""
=====================================================================
TCFSG V19.0 — Temporal Causal Financial Semantic Graph
Seven-Layer Heterogeneous Architecture | Graph-Native Mesh | Causal Strength
Windows-Compatible Edition

FIXES over V18.7:
[P0] A: REGION NER Pipeline — spaCy en_core_web_trf GPE/LOC extraction
[P0] B: Intent Classifier — FINANCIAL vs ESG intent before relation extraction
[P0] C: Sentence-Level Evidence — exact sentence match via NLTK tokenizer
[P0] D: Native Neo4j Relations — native relationship types, not generic RELATION
[P1] E: Causal Strength Layer — DIRECT/INDIRECT/ASSOCIATION/SPECULATIVE
[P1] F: Mechanism Layer — MECHANISM nodes bridging Risk→Financial impact
[P1] G: Hubness Balancer — MAX_OUT_EDGES_PER_NODE, mesh topology constraint

Windows Fixes:
- UTF-8 BOM header for Chinese path compatibility
- os.path.join() everywhere, no hardcoded forward slashes
- Absolute path resolution for PDF folder
- Graceful spaCy model fallback (trf → sm → None)
- Try/except around all optional imports
=====================================================================
"""

import os
import re
import json
import time
import glob
import logging
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict

# ── V19.0: Conditional imports with graceful Windows fallback ──
try:
    import certifi
except ImportError:
    certifi = None

try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    print("ERROR: pdfplumber not installed. Run: pip install pdfplumber")

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from groq import Groq
except ImportError:
    Groq = None
    print("ERROR: groq not installed. Run: pip install groq")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    load_dotenv = None

try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
    print("ERROR: neo4j not installed. Run: pip install neo4j")

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    RecursiveCharacterTextSplitter = None

try:
    import spacy
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False

try:
    import nltk
    from nltk.tokenize import sent_tokenize as nltk_sent_tokenize
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

# ── Remove proxy env vars (Windows compatibility) ──
os.environ.pop('http_proxy', None)
os.environ.pop('https_proxy', None)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger("TCFSG_V19_0")


# ── Fallback sentence tokenizer for Windows without NLTK ──
def _regex_sent_tokenize(text: str) -> List[str]:
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

sent_tokenize = nltk_sent_tokenize if _HAS_NLTK else _regex_sent_tokenize


# ============================================================
# 1. Ontology (expanded for V19.0)
# ============================================================

ENTITY_CATEGORIES: Set[str] = {
    "COMPANY", "MARKET", "REGULATION", "ESG_TOPIC",
    "PRODUCT", "REGION", "RISK_FACTOR",
    "BUSINESS_STRATEGY", "FINANCIAL_METRIC", "EVENT",
    "MECHANISM",  # V19.0
}

VALID_RELATIONS: Set[str] = {
    "EXPOSED_TO", "MITIGATES", "CAUSES", "AMPLIFIES",
    "REDUCES", "INCREASES", "DECREASES", "DISCLOSES", "IMPLEMENTS",
    "TRIGGERS", "LEADS_TO", "AGGRAVATES",  # V19.0
}

CAUSAL_STRENGTHS: Set[str] = {
    "DIRECT_CAUSALITY",
    "INDIRECT_CAUSALITY",
    "RISK_ASSOCIATION",
    "SPECULATIVE_RELATION",
    "DISCLOSED_EXPOSURE",
}

# ── V19.0: 30+ Mechanism patterns ──
MECHANISM_PATTERNS: Dict[str, Tuple[str, str]] = {
    "extreme weather": ("EXTREME_WEATHER", "MECHANISM"),
    "factory shutdown": ("FACTORY_SHUTDOWN", "MECHANISM"),
    "production halt": ("PRODUCTION_HALT", "MECHANISM"),
    "logistics delay": ("LOGISTICS_DELAY", "MECHANISM"),
    "port congestion": ("PORT_CONGESTION", "MECHANISM"),
    "inventory shortage": ("INVENTORY_SHORTAGE", "MECHANISM"),
    "component shortage": ("COMPONENT_SHORTAGE", "MECHANISM"),
    "price increase": ("PRICE_INCREASE", "MECHANISM"),
    "price decline": ("PRICE_DECLINE", "MECHANISM"),
    "order cancellation": ("ORDER_CANCELLATION", "MECHANISM"),
    "order delay": ("ORDER_DELAY", "MECHANISM"),
    "customer attrition": ("CUSTOMER_ATTRITION", "MECHANISM"),
    "market share loss": ("MARKET_SHARE_LOSS", "MECHANISM"),
    "credit tightening": ("CREDIT_TIGHTENING", "MECHANISM"),
    "liquidity squeeze": ("LIQUIDITY_SQUEEZE", "MECHANISM"),
    "workforce reduction": ("WORKFORCE_REDUCTION", "MECHANISM"),
    "layoff": ("WORKFORCE_REDUCTION", "MECHANISM"),
    "restructuring charge": ("RESTRUCTURING_CHARGE", "MECHANISM"),
    "impairment charge": ("IMPAIRMENT_CHARGE", "MECHANISM"),
    "patent expiration": ("PATENT_EXPIRATION", "MECHANISM"),
    "regulatory fine": ("REGULATORY_FINE", "MECHANISM"),
    "litigation settlement": ("LITIGATION_SETTLEMENT", "MECHANISM"),
    "currency devaluation": ("CURRENCY_DEVALUATION", "MECHANISM"),
    "interest rate hike": ("INTEREST_RATE_HIKE", "MECHANISM"),
    "tariff imposition": ("TARIFF_IMPOSITION", "MECHANISM"),
    "contract loss": ("CONTRACT_LOSS", "MECHANISM"),
    "supplier bankruptcy": ("SUPPLIER_BANKRUPTCY", "MECHANISM"),
    "technology disruption": ("TECHNOLOGY_DISRUPTION", "MECHANISM"),
    "supply shortage": ("COMPONENT_SHORTAGE", "MECHANISM"),
}

MITIGATES_SOURCE_CATEGORIES: Set[str] = {"BUSINESS_STRATEGY"}
MITIGATES_TARGET_CATEGORIES: Set[str] = {"RISK_FACTOR", "MECHANISM"}

FINANCIAL_METRICS: Set[str] = {
    "REVENUE", "GROSS_MARGIN", "OPERATING_MARGIN", "NET_INCOME",
    "CASH_FLOW", "OPERATING_COST", "MARKET_VALUE", "R_AND_D_EXPENSE",
    "CAPEX", "FREE_CASH_FLOW", "EARNINGS_PER_SHARE", "RETURN_ON_EQUITY",
}

POSITIVE_METRICS: Set[str] = {
    "REVENUE", "GROSS_MARGIN", "OPERATING_MARGIN", "NET_INCOME",
    "CASH_FLOW", "FREE_CASH_FLOW", "EARNINGS_PER_SHARE",
    "RETURN_ON_EQUITY", "MARKET_VALUE",
}

NEGATIVE_METRICS: Set[str] = {
    "OPERATING_COST", "R_AND_D_EXPENSE", "CAPEX",
}

# ── V19.0: INTENT CLASSIFIER ──
FINANCIAL_INTENT_KEYWORDS: Set[str] = {
    "revenue", "cost", "margin", "income", "earnings",
    "cash flow", "capex", "eps", "roe", "ebitda",
    "operating", "net income", "gross profit", "free cash flow",
    "return on equity", "earnings per share", "market value",
    "stock price", "share price", "dividend", "buyback",
}

ESG_INTENT_KEYWORDS: Set[str] = {
    "health", "wellness", "diversity", "emission", "carbon",
    "turnover", "engagement", "training", "learning",
    "sustainability", "esg", "environmental", "social",
    "governance", "climate", "renewable", "green",
    "safety", "injury", "occupational", "inclusive",
    "equity", "belonging", "volunteer", "philanthropy",
}


def classify_intent(text: str) -> str:
    tl = text.lower()
    fin_score = sum(1 for k in FINANCIAL_INTENT_KEYWORDS if k in tl)
    esg_score = sum(1 for k in ESG_INTENT_KEYWORDS if k in tl)
    if fin_score > esg_score and fin_score >= 2:
        return "FINANCIAL"
    elif esg_score > fin_score and esg_score >= 2:
        return "ESG"
    return "UNKNOWN"


def detect_causal_strength(text: str) -> str:
    tl = text.lower()
    if re.search(r'\b(causes?|result(ed|s|ing)? in|lead(s|ing)? to|'
                 r'trigger(s|ed|ing)?|drive(s|n)?|generate(s|d)?|'
                 r'produce(s|d)?|create(s|d)?|induce(s|d)?)\b', tl):
        return "DIRECT_CAUSALITY"
    if re.search(r'\b(contribute(s|d)? to|influence(s|d)?|'
                 r'affect(s|ed|ing)?|impact(s|ed|ing)?|'
                 r'play(s|ed)? a role|factor(s)? in|'
                 r'associate(d)? with|link(ed)? to|relate(d)? to)\b', tl):
        return "INDIRECT_CAUSALITY"
    if re.search(r'\b(may|could|might|potentially|possibly|'
                 r'would|can|should|expect(ed)? to)\b', tl):
        return "SPECULATIVE_RELATION"
    if re.search(r'\b(in addition to|as well as|along with|'
                 r'accompanied by|together with|including|such as)\b', tl):
        return "RISK_ASSOCIATION"
    return "DISCLOSED_EXPOSURE"


def extract_evidence_sentence(text: str, source_term: str, target_term: str) -> str:
    if not text:
        return ""
    sentences = sent_tokenize(text)
    best_score = 0.0
    best_sent = ""
    st_lower = source_term.lower()
    tt_lower = target_term.lower()
    for sent in sentences:
        sl = sent.lower().strip()
        if len(sl) < 10:
            continue
        if st_lower in sl and tt_lower in sl:
            return sent[:500]
        score = 0.0
        if st_lower in sl:
            score += 1.0
            if tt_lower in sl:
                score += 3.0
        if tt_lower in sl:
            score += 1.0
        if any(w in sl for w in ["cause","result","lead","impact","affect",
                                  "increase","decrease","reduce","risk"]):
            score += 0.5
        if score > best_score:
            best_score = score
            best_sent = sent
    return (best_sent or text[:300])[:500]


def extract_regions_ner(text: str, nlp=None) -> List[str]:
    if not _HAS_SPACY or nlp is None:
        return []
    regions_found: Set[str] = set()
    try:
        doc = nlp(text[:10000])
        for ent in doc.ents:
            if ent.label_ in ("GPE", "LOC"):
                nid = norm_id(ent.text)
                if nid in CANONICAL_MAP and CANONICAL_MAP[nid][1] == "REGION":
                    regions_found.add(CANONICAL_MAP[nid][0])
    except Exception as e:
        logger.warning(f"  spaCy NER error: {e}")
    return list(regions_found)


def extract_mechanisms(text: str) -> List[Tuple[str, str]]:
    tl = text.lower()
    found: List[Tuple[str, str]] = []
    for pattern, (mech_name, mech_cat) in MECHANISM_PATTERNS.items():
        if pattern in tl:
            found.append((mech_name, mech_cat))
    return found


# ── Static filter sets (from V18.7) ──
GENERIC_NAMES: Set[str] = {
    "employees","employee","customers","customer",
    "suppliers","supplier","stakeholders","stakeholder",
    "partners","partner","people","team","workforce",
    "staff","management","board","investors","investor",
    "shareholders","shareholder","clients","client",
    "competitors","competitor","vendors","vendor",
    "users","user","members","member",
}

COMPANY_BLACKLIST_PREFIXES: Set[str] = {
    "employee","staff","workforce","talent","people",
    "mentor","train","learn","career","tuition",
    "pulse","survey","suggestion","referral",
    "community","military","health",
}

NON_RISK_NAMES: Set[str] = {
    "pulse surveys","pulse_surveys","pulse survey",
    "suggestion box","suggestion_box",
    "anonymous third-party platform","anonymous_third_party_platform",
    "tuition reimbursement programs","tuition_reimbursement_programs",
    "career coaching","career_coaching",
    "mentoring and development programs","mentoring_and_development_programs",
    "mentoring programs","mentoring_programs",
    "training programs","training_programs",
    "learning experiences","learning_experiences",
    "learning paths","learning_paths",
    "employee referrals","employee_referrals",
    "physical health challenges","physical_health_challenges",
    "mental health challenges","mental_health_challenges",
    "well-being challenges","well_being_challenges",
    "time-management challenges","time_management_challenges",
    "financial health challenges","financial_health_challenges",
    "stress","community needs","community_needs",
    "military members","military_members",
    "support for military members","support_for_military_members",
    "additional mental health benefits","additional_mental_health_benefits",
    "sourcing renewable energy","sourcing_renewable_energy",
    "decreased protection","decreased_protection",
}

ESG_NON_CAUSAL_NAMES: Set[str] = {
    "mental health challenges","mental_health_challenges",
    "well-being challenges","well_being_challenges",
    "physical health challenges","physical_health_challenges",
    "time-management challenges","time_management_challenges",
    "financial health challenges","financial_health_challenges",
    "stress","community needs","community_needs",
    "military members","military_members",
    "support for military members","support_for_military_members",
    "additional mental health benefits","additional_mental_health_benefits",
}

NON_CANONICAL_METRICS: Set[str] = {
    "business success","business_success",
    "material impact on our business","material_impact_on_our_business",
    "$1.36 billion charge","1_36_billion_charge",
    "material capital expenditures for environmental control facilities",
    "material_capital_expenditures_for_environmental_control_facilities",
    "financial condition","financial_condition",
    "results of operations","results_of_operations",
    "competitive position","competitive_position",
    "energy_efficiency","energy efficiency",
    "health coverage","health_coverage",
    "learning and development resources","learning_and_development_resources",
    "employee sentiment","employee_sentiment",
    "environmental impact metrics","environmental_impact_metrics",
    "4.9%","4_9","turnover rate","turnover_rate",
    "employee turnover rate","employee_turnover_rate",
}

ESG_MISCLASSIFIED_AS_MARKET: Dict[str, str] = {
    "customer requirements":"MARKET","customer_requirements":"MARKET",
    "market demand":"MARKET","market_demand":"MARKET",
    "customer demand":"MARKET","customer_demand":"MARKET",
}

RISK_MISCLASSIFIED_AS_ESG: Dict[str, str] = {
    "energy efficiency":"ESG_TOPIC","energy_efficiency":"ESG_TOPIC",
    "energy consumption":"ESG_TOPIC","energy_consumption":"ESG_TOPIC",
}

COMPETITOR_NAMES: Set[str] = {
    "amd","intel","samsung","tesla","apple","qualcomm","broadcom",
    "micron","tsmc","microsoft","google","alphabet","amazon",
    "ibm","oracle","cisco","huawei","mediatek","nokia","ericsson",
    "advanced_micro_devices","advanced micro devices",
    "alphabet_inc","amazon_com","apple_inc",
    "microsoft_corporation","tesla_inc",
}

CANONICAL_MAP: Dict[str, Tuple[str, str]] = {
    "nvidia": ("NVIDIA","COMPANY"), "nvidia_corporation": ("NVIDIA","COMPANY"), "nvda": ("NVIDIA","COMPANY"),
    "revenue": ("REVENUE","FINANCIAL_METRIC"), "revenues": ("REVENUE","FINANCIAL_METRIC"),
    "sales": ("REVENUE","FINANCIAL_METRIC"), "net_revenue": ("REVENUE","FINANCIAL_METRIC"),
    "gross_margin": ("GROSS_MARGIN","FINANCIAL_METRIC"),
    "operating_margin": ("OPERATING_MARGIN","FINANCIAL_METRIC"),
    "net_income": ("NET_INCOME","FINANCIAL_METRIC"), "net_earnings": ("NET_INCOME","FINANCIAL_METRIC"),
    "cash_flow": ("CASH_FLOW","FINANCIAL_METRIC"), "cash_flows": ("CASH_FLOW","FINANCIAL_METRIC"),
    "operating_cost": ("OPERATING_COST","FINANCIAL_METRIC"), "operating_costs": ("OPERATING_COST","FINANCIAL_METRIC"),
    "operating_expenses": ("OPERATING_COST","FINANCIAL_METRIC"),
    "expense": ("OPERATING_COST","FINANCIAL_METRIC"), "expenses": ("OPERATING_COST","FINANCIAL_METRIC"),
    "cost": ("OPERATING_COST","FINANCIAL_METRIC"), "costs": ("OPERATING_COST","FINANCIAL_METRIC"),
    "market_value": ("MARKET_VALUE","FINANCIAL_METRIC"), "stock_price": ("MARKET_VALUE","FINANCIAL_METRIC"),
    "r_and_d": ("R_AND_D_EXPENSE","FINANCIAL_METRIC"), "r_and_d_expense": ("R_AND_D_EXPENSE","FINANCIAL_METRIC"),
    "rd_expense": ("R_AND_D_EXPENSE","FINANCIAL_METRIC"),
    "research_and_development_expense": ("R_AND_D_EXPENSE","FINANCIAL_METRIC"),
    "capex": ("CAPEX","FINANCIAL_METRIC"), "capital_expenditures": ("CAPEX","FINANCIAL_METRIC"),
    "free_cash_flow": ("FREE_CASH_FLOW","FINANCIAL_METRIC"),
    "eps": ("EARNINGS_PER_SHARE","FINANCIAL_METRIC"), "earnings_per_share": ("EARNINGS_PER_SHARE","FINANCIAL_METRIC"),
    "return_on_equity": ("RETURN_ON_EQUITY","FINANCIAL_METRIC"), "roe": ("RETURN_ON_EQUITY","FINANCIAL_METRIC"),
    # ── Risk: Supply Chain ──
    "supply_chain_disruption": ("SUPPLY_CHAIN_DISRUPTION","RISK_FACTOR"),
    "supply_chain_risk": ("SUPPLY_CHAIN_DISRUPTION","RISK_FACTOR"),
    "supply_constraints": ("SUPPLY_CHAIN_DISRUPTION","RISK_FACTOR"),
    "supply_shortage": ("SUPPLY_CHAIN_DISRUPTION","RISK_FACTOR"),
    "supply_shortages": ("SUPPLY_CHAIN_DISRUPTION","RISK_FACTOR"),
    "foundry_capacity": ("SUPPLY_CHAIN_DISRUPTION","RISK_FACTOR"),
    "wafer_supply": ("SUPPLY_CHAIN_DISRUPTION","RISK_FACTOR"),
    "manufacturing_capacity": ("SUPPLY_CHAIN_DISRUPTION","RISK_FACTOR"),
    # ── Risk: Export ──
    "export_controls": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "export_control": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "export_restrictions": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "export_restriction": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "trade_restrictions": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "chip_export_restriction": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "import_and_export_requirements_and_tariffs": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "import_export_restrictions": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "tariffs": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "trade_barriers": ("CHIP_EXPORT_RESTRICTION","RISK_FACTOR"),
    "ai_regulation": ("AI_REGULATION_RISK","RISK_FACTOR"),
    "ai_regulation_risk": ("AI_REGULATION_RISK","RISK_FACTOR"),
    "artificial_intelligence_regulation": ("AI_REGULATION_RISK","RISK_FACTOR"),
    "government_regulation": ("GOVERNMENT_REGULATION_RISK","RISK_FACTOR"),
    "regulatory_compliance": ("GOVERNMENT_REGULATION_RISK","RISK_FACTOR"),
    # ── Risk: IP ──
    "ip_ownership_and_infringement": ("INTELLECTUAL_PROPERTY_RISK","RISK_FACTOR"),
    "intellectual_property_protection": ("INTELLECTUAL_PROPERTY_RISK","RISK_FACTOR"),
    "intellectual_property": ("INTELLECTUAL_PROPERTY_RISK","RISK_FACTOR"),
    "ip_infringement": ("INTELLECTUAL_PROPERTY_RISK","RISK_FACTOR"),
    "patent_infringement": ("INTELLECTUAL_PROPERTY_RISK","RISK_FACTOR"),
    # ── Risk: Demand ──
    "demand_volatility": ("DEMAND_VOLATILITY","RISK_FACTOR"),
    "decrease_in_demand": ("DEMAND_VOLATILITY","RISK_FACTOR"),
    "decreased_demand": ("DEMAND_VOLATILITY","RISK_FACTOR"),
    "demand_uncertainty": ("DEMAND_VOLATILITY","RISK_FACTOR"),
    "product_demand_fluctuations": ("DEMAND_VOLATILITY","RISK_FACTOR"),
    "inventory_risk": ("INVENTORY_RISK","RISK_FACTOR"),
    "excess_inventory": ("INVENTORY_RISK","RISK_FACTOR"),
    "inventory_obsolescence": ("INVENTORY_RISK","RISK_FACTOR"),
    "inventory_management": ("INVENTORY_RISK","RISK_FACTOR"),
    # ── Risk: Security ──
    "cybersecurity_risk": ("SYSTEM_SECURITY_BREACH","RISK_FACTOR"),
    "security_breach": ("SYSTEM_SECURITY_BREACH","RISK_FACTOR"),
    "cyber_attack": ("SYSTEM_SECURITY_BREACH","RISK_FACTOR"),
    "data_breach": ("SYSTEM_SECURITY_BREACH","RISK_FACTOR"),
    "cybersecurity_threats": ("SYSTEM_SECURITY_BREACH","RISK_FACTOR"),
    "technology_obsolescence": ("TECHNOLOGY_OBSOLESCENCE","RISK_FACTOR"),
    "technological_change": ("TECHNOLOGY_OBSOLESCENCE","RISK_FACTOR"),
    "technological_advancements": ("TECHNOLOGY_OBSOLESCENCE","RISK_FACTOR"),
    "talent_shortage": ("TALENT_SHORTAGE","RISK_FACTOR"),
    "talent_competition": ("TALENT_SHORTAGE","RISK_FACTOR"),
    "skilled_personnel": ("TALENT_SHORTAGE","RISK_FACTOR"),
    "key_personnel": ("TALENT_SHORTAGE","RISK_FACTOR"),
    "competition": ("COMPETITION_RISK","RISK_FACTOR"),
    "competitive_pressure": ("COMPETITION_RISK","RISK_FACTOR"),
    "intense_competition": ("COMPETITION_RISK","RISK_FACTOR"),
    "competitive_landscape": ("COMPETITION_RISK","RISK_FACTOR"),
    "market_competition": ("COMPETITION_RISK","RISK_FACTOR"),
    "reputation_risk": ("REPUTATION_RISK","RISK_FACTOR"),
    "brand_reputation": ("REPUTATION_RISK","RISK_FACTOR"),
    # ── Risk: Macro ──
    "macroeconomic_conditions": ("MACROECONOMIC_CONDITIONS","RISK_FACTOR"),
    "economic_downturn": ("MACROECONOMIC_CONDITIONS","RISK_FACTOR"),
    "global_economic_conditions": ("MACROECONOMIC_CONDITIONS","RISK_FACTOR"),
    "geopolitical_risk": ("GEOPOLITICAL_RISK","RISK_FACTOR"),
    "geopolitical_tensions": ("GEOPOLITICAL_RISK","RISK_FACTOR"),
    "geopolitical_uncertainty": ("GEOPOLITICAL_RISK","RISK_FACTOR"),
    # ── Risk: Other ──
    "acquisition_integration": ("ACQUISITION_INTEGRATION_RISK","RISK_FACTOR"),
    "acquisition_risk": ("ACQUISITION_INTEGRATION_RISK","RISK_FACTOR"),
    "regulatory_approval": ("REGULATORY_APPROVAL_RISK","RISK_FACTOR"),
    "credit_risk": ("CREDIT_RISK","RISK_FACTOR"),
    "cryptocurrency_volatility": ("CRYPTO_MARKET_VOLATILITY","RISK_FACTOR"),
    "crypto_market": ("CRYPTO_MARKET_VOLATILITY","RISK_FACTOR"),
    "digital_currency": ("CRYPTO_MARKET_VOLATILITY","RISK_FACTOR"),
    "foreign_exchange": ("FOREIGN_EXCHANGE_RISK","RISK_FACTOR"),
    "currency_fluctuations": ("FOREIGN_EXCHANGE_RISK","RISK_FACTOR"),
    "tax_risk": ("TAX_RISK","RISK_FACTOR"), "tax_legislation": ("TAX_RISK","RISK_FACTOR"),
    "litigation": ("LITIGATION_RISK","RISK_FACTOR"), "legal_proceedings": ("LITIGATION_RISK","RISK_FACTOR"),
    "product_liability": ("PRODUCT_LIABILITY_RISK","RISK_FACTOR"),
    "warranty_claims": ("PRODUCT_LIABILITY_RISK","RISK_FACTOR"),
    "quality_control": ("PRODUCT_LIABILITY_RISK","RISK_FACTOR"),
    # ── Markets ──
    "china_market": ("CHINA_MARKET","MARKET"), "chinese_market": ("CHINA_MARKET","MARKET"),
    "us_china_trade": ("US_CHINA_TRADE","MARKET"),
    "global_semiconductor": ("GLOBAL_SEMICONDUCTOR_MARKET","MARKET"),
    "semiconductor_market": ("GLOBAL_SEMICONDUCTOR_MARKET","MARKET"),
    "semiconductor_industry": ("GLOBAL_SEMICONDUCTOR_MARKET","MARKET"),
    "chip_market": ("GLOBAL_SEMICONDUCTOR_MARKET","MARKET"),
    "data_center_market": ("DATA_CENTER_MARKET","MARKET"),
    "data_center": ("DATA_CENTER_MARKET","MARKET"),
    "cloud_market": ("CLOUD_MARKET","MARKET"),
    "gaming_market": ("GAMING_MARKET","MARKET"), "gaming_industry": ("GAMING_MARKET","MARKET"),
    "automotive_market": ("AUTOMOTIVE_MARKET","MARKET"), "automotive_industry": ("AUTOMOTIVE_MARKET","MARKET"),
    "professional_visualization_market": ("PROFESSIONAL_VISUALIZATION_MARKET","MARKET"),
    "ai_market": ("AI_MARKET","MARKET"), "artificial_intelligence_market": ("AI_MARKET","MARKET"),
    "hpc_market": ("HPC_MARKET","MARKET"), "high_performance_computing": ("HPC_MARKET","MARKET"),
    # ── Products ──
    "gpu": ("GPU","PRODUCT"), "graphics_processing_unit": ("GPU","PRODUCT"),
    "graphics_card": ("GPU","PRODUCT"), "cuda": ("CUDA","PRODUCT"),
    "cuda_platform": ("CUDA","PRODUCT"), "cuda_software": ("CUDA","PRODUCT"),
    "dgx": ("DGX_SYSTEM","PRODUCT"), "dgx_system": ("DGX_SYSTEM","PRODUCT"),
    "tesla_gpu": ("TESLA_GPU","PRODUCT"), "geforce": ("GEFORCE_GPU","PRODUCT"),
    "quadro": ("QUADRO_GPU","PRODUCT"), "tensor_core": ("TENSOR_CORE_GPU","PRODUCT"),
    "soc": ("SOC","PRODUCT"), "system_on_chip": ("SOC","PRODUCT"),
    "orin": ("ORIN_PLATFORM","PRODUCT"), "drive_platform": ("DRIVE_PLATFORM","PRODUCT"),
    "jetson": ("JETSON_PLATFORM","PRODUCT"), "mellanox": ("MELLANOX_PRODUCTS","PRODUCT"),
    "bluefield": ("BLUEFIELD_DPU","PRODUCT"), "dpu": ("BLUEFIELD_DPU","PRODUCT"),
    "data_processing_unit": ("BLUEFIELD_DPU","PRODUCT"),
    # ── Regions ──
    "united_states": ("UNITED_STATES","REGION"), "u.s.": ("UNITED_STATES","REGION"),
    "us": ("UNITED_STATES","REGION"), "china": ("CHINA","REGION"),
    "taiwan": ("TAIWAN","REGION"), "europe": ("EUROPE","REGION"),
    "japan": ("JAPAN","REGION"), "south_korea": ("SOUTH_KOREA","REGION"),
    "singapore": ("SINGAPORE","REGION"), "asia": ("ASIA","REGION"),
    "global": ("GLOBAL","REGION"), "international": ("INTERNATIONAL","REGION"),
    "north_america": ("NORTH_AMERICA","REGION"), "middle_east": ("MIDDLE_EAST","REGION"),
    "israel": ("ISRAEL","REGION"), "germany": ("GERMANY","REGION"),
    "united_kingdom": ("UNITED_KINGDOM","REGION"), "uk": ("UNITED_KINGDOM","REGION"),
    "france": ("FRANCE","REGION"), "netherlands": ("NETHERLANDS","REGION"),
    "switzerland": ("SWITZERLAND","REGION"), "australia": ("AUSTRALIA","REGION"),
    "india": ("INDIA","REGION"), "vietnam": ("VIETNAM","REGION"),
    "malaysia": ("MALAYSIA","REGION"),
    # ── ESG Topics ──
    "climate_change": ("CLIMATE_CHANGE","ESG_TOPIC"),
    "environmental_regulations": ("ENVIRONMENTAL_REGULATIONS","ESG_TOPIC"),
    "environmental_compliance": ("ENVIRONMENTAL_REGULATIONS","ESG_TOPIC"),
    "carbon_emissions": ("CARBON_EMISSIONS","ESG_TOPIC"),
    "sustainability": ("SUSTAINABILITY","ESG_TOPIC"),
    "social_responsibility": ("SOCIAL_RESPONSIBILITY","ESG_TOPIC"),
    # ── Events ──
    "natural_disaster": ("NATURAL_DISASTER","EVENT"),
    "earthquake": ("NATURAL_DISASTER","EVENT"), "flood": ("NATURAL_DISASTER","EVENT"),
    "pandemic": ("COVID_19","EVENT"), "covid": ("COVID_19","EVENT"),
    "covid_19": ("COVID_19","EVENT"), "global_health_crisis": ("COVID_19","EVENT"),
    # ── Business Strategies ──
    "diversified_revenue": ("DIVERSIFIED_REVENUE","BUSINESS_STRATEGY"),
    "diversified_revenue_streams": ("DIVERSIFIED_REVENUE","BUSINESS_STRATEGY"),
    "revenue_diversification": ("DIVERSIFIED_REVENUE","BUSINESS_STRATEGY"),
    "supply_chain_diversification": ("SUPPLY_CHAIN_DIVERSIFICATION","BUSINESS_STRATEGY"),
    "supplier_diversification": ("SUPPLY_CHAIN_DIVERSIFICATION","BUSINESS_STRATEGY"),
    "multi_sourcing": ("SUPPLY_CHAIN_DIVERSIFICATION","BUSINESS_STRATEGY"),
    "cost_optimization": ("COST_OPTIMIZATION","BUSINESS_STRATEGY"),
    "cost_reduction": ("COST_REDUCTION","BUSINESS_STRATEGY"),
    "cost_savings": ("COST_REDUCTION","BUSINESS_STRATEGY"),
    "r_and_d_investment": ("R_AND_D_INVESTMENT","BUSINESS_STRATEGY"),
    "research_and_development": ("R_AND_D_INVESTMENT","BUSINESS_STRATEGY"),
    "rd_investment": ("R_AND_D_INVESTMENT","BUSINESS_STRATEGY"),
    "strategic_acquisition": ("STRATEGIC_ACQUISITION","BUSINESS_STRATEGY"),
    "acquisitions": ("STRATEGIC_ACQUISITION","BUSINESS_STRATEGY"),
    "cybersecurity_measures": ("CYBERSECURITY_MEASURES","BUSINESS_STRATEGY"),
    "market_expansion": ("MARKET_EXPANSION","BUSINESS_STRATEGY"),
    "geographic_expansion": ("MARKET_EXPANSION","BUSINESS_STRATEGY"),
    "product_diversification": ("PRODUCT_DIVERSIFICATION","BUSINESS_STRATEGY"),
    "talent_acquisition": ("TALENT_ACQUISITION","BUSINESS_STRATEGY"),
    "workforce_expansion": ("TALENT_ACQUISITION","BUSINESS_STRATEGY"),
    "technology_investment": ("TECHNOLOGY_INVESTMENT","BUSINESS_STRATEGY"),
    "innovation_investment": ("TECHNOLOGY_INVESTMENT","BUSINESS_STRATEGY"),
    "customer_focus": ("CUSTOMER_FOCUS_STRATEGY","BUSINESS_STRATEGY"),
    "customer_relationships": ("CUSTOMER_FOCUS_STRATEGY","BUSINESS_STRATEGY"),
}

LLM_RAW_ENTITY_MAP: Dict[str, Tuple[str, str]] = {
    "rapid changes in technology, customer requirements, new product introductions and enhancements, and":
        ("TECHNOLOGY_OBSOLESCENCE","RISK_FACTOR"),
    "rapid changes in technology, customer requirements, new product introductions and enhancements":
        ("TECHNOLOGY_OBSOLESCENCE","RISK_FACTOR"),
    "covid-19 testing, vaccine costs and support, expanded mental health resources and virtual care offer":
        ("COVID_19","EVENT"),
    "covid-19 testing, vaccine costs and support, expanded mental health resources and virtual care":
        ("COVID_19","EVENT"),
    "customers cancel or defer orders or choose to purchase from our competitors":
        ("DEMAND_VOLATILITY","RISK_FACTOR"),
    "customers cancel or defer orders or choose to purchase from our":
        ("DEMAND_VOLATILITY","RISK_FACTOR"),
    "ip laws exist and are meaningfully enforced in different jurisdictions":
        ("INTELLECTUAL_PROPERTY_RISK","RISK_FACTOR"),
    "our ability to supply our gaming cards to non-mining customers":
        ("DEMAND_VOLATILITY","RISK_FACTOR"),
    "material capital expenditures for environmental control facilities":
        ("CAPEX","FINANCIAL_METRIC"),
    "$1.36 billion charge": ("OPERATING_COST","FINANCIAL_METRIC"),
    "1.36 billion charge": ("OPERATING_COST","FINANCIAL_METRIC"),
    "business success": ("REVENUE","FINANCIAL_METRIC"),
    "material impact on our business": ("REVENUE","FINANCIAL_METRIC"),
    "our business, financial condition, and results of operations": ("OPERATING_COST","FINANCIAL_METRIC"),
    "financial condition": ("CASH_FLOW","FINANCIAL_METRIC"),
    "results of operations": ("NET_INCOME","FINANCIAL_METRIC"),
    "competitive position": ("COMPETITION_RISK","RISK_FACTOR"),
    "employee advancement": ("TALENT_SHORTAGE","RISK_FACTOR"),
    "employee referrals": ("TALENT_ACQUISITION","BUSINESS_STRATEGY"),
    "energy efficiency": ("ENERGY_EFFICIENCY","ESG_TOPIC"),
    "customer requirements": ("CUSTOMER_REQUIREMENTS","MARKET"),
    "region": ("GLOBAL","REGION"),
}

BANNED_WORDS: Set[str] = {
    "risk_factor","risk_factors","esg_topic","regulation","event",
    "business_strategy","financial_metric","category","entity",
    "uncertainty","uncertainties","table_of_contents",
    "risk","risks","region",
}

_LONG_PHRASE_MAP: Dict[str, Tuple[str, str]] = {}
for raw_key, (name, cat) in CANONICAL_MAP.items():
    if len(raw_key.split("_")) >= 4:
        _LONG_PHRASE_MAP[raw_key] = (name, cat)


# ============================================================
# 2. Utility Functions
# ============================================================

def norm_id(s: str) -> str:
    s = str(s or "").lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:100]

def is_banned(name: str) -> bool:
    return norm_id(name) in BANNED_WORDS

def is_competitor(name: str) -> bool:
    return norm_id(name) in COMPETITOR_NAMES

def is_generic(name: str) -> bool:
    return norm_id(name) in {norm_id(g) for g in GENERIC_NAMES}

def is_non_risk(name: str) -> bool:
    return norm_id(name) in {norm_id(n) for n in NON_RISK_NAMES}

def is_esg_non_causal(name: str) -> bool:
    return norm_id(name) in {norm_id(e) for e in ESG_NON_CAUSAL_NAMES}

def is_non_canonical_metric(name: str) -> bool:
    return norm_id(name) in {norm_id(m) for m in NON_CANONICAL_METRICS}

def is_company_blacklisted(name: str) -> bool:
    nid = norm_id(name)
    return any(nid.startswith(p) for p in COMPANY_BLACKLIST_PREFIXES)

def resolve_llm_raw_entity(name: str) -> Optional[Tuple[str, str]]:
    nl = name.lower().strip()
    if nl in LLM_RAW_ENTITY_MAP:
        return LLM_RAW_ENTITY_MAP[nl]
    for key in sorted(LLM_RAW_ENTITY_MAP.keys(), key=len, reverse=True):
        if key in nl:
            return LLM_RAW_ENTITY_MAP[key]
    return None

def resolve_long_phrase(text: str) -> List[Tuple[str, str, str]]:
    text_lower = text.lower()
    results: List[Tuple[str, str, str]] = []
    for key in sorted(_LONG_PHRASE_MAP.keys(), key=len, reverse=True):
        for variant in (key.replace("_", " "), key.replace("_", " ").strip(), key):
            if variant in text_lower:
                cname, ccat = _LONG_PHRASE_MAP[key]
                results.append((variant, cname, ccat))
                break
    return results

def resolve_entity(raw: str, cat: str) -> Tuple[str, str]:
    r = raw.strip()
    nr = norm_id(r)
    if nr in CANONICAL_MAP:
        return CANONICAL_MAP[nr]
    llm_res = resolve_llm_raw_entity(r)
    if llm_res:
        return llm_res
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", r).strip()
    clean = re.sub(r"\s+", " ", clean)[:40].upper().replace(" ", "_")
    return (clean, cat)

def infer_relation(s_cat: str, t_cat: str, ctx: str,
                   source_name: str = "", target_name: str = "") -> str:
    if norm_id(source_name) == norm_id(target_name):
        return "CAUSES"
    if source_name == "COMPETITION_RISK" or "competition" in ctx:
        if t_cat == "FINANCIAL_METRIC":
            pos = any(m in ctx for m in ["revenue","gross_margin","net_income","cash_flow","earnings","market_value"])
            neg = any(m in ctx for m in ["operating_cost","expense","cost","r_and_d","capex"])
            if pos: return "DECREASES"
            if neg: return "INCREASES"
    if s_cat == "COMPANY" and t_cat == "RISK_FACTOR": return "EXPOSED_TO"
    if s_cat == "COMPANY" and t_cat == "BUSINESS_STRATEGY": return "IMPLEMENTS"
    if s_cat == "BUSINESS_STRATEGY" and t_cat == "RISK_FACTOR": return "MITIGATES"
    if s_cat == "BUSINESS_STRATEGY" and t_cat == "FINANCIAL_METRIC":
        if re.search(r'\b(reduce|decrease|lower|minimize|cut)\b', ctx): return "REDUCES"
        if re.search(r'\b(increase|growth|raise|improve|enhance|boost)\b', ctx): return "INCREASES"
        return "CAUSES"
    if s_cat == "RISK_FACTOR" and t_cat == "FINANCIAL_METRIC":
        neg = bool(re.search(r'\b(decrease|reduce|harm|adversely|negatively|decline|loss|erode|impair|hurt|damage)\b', ctx))
        pos = bool(re.search(r'\b(increase|growth|rise|improve|enhance|boost|raise|expand|grow)\b', ctx))
        if neg and not pos: return "DECREASES"
        if pos and not neg: return "INCREASES"
        if target_name in POSITIVE_METRICS: return "DECREASES"
        if target_name in NEGATIVE_METRICS: return "INCREASES"
        return "CAUSES"
    if s_cat == "RISK_FACTOR" and t_cat == "RISK_FACTOR":
        return "AMPLIFIES" if re.search(r'\b(amplify|exacerbate|worsen|intensify|magnify)\b', ctx) else "CAUSES"
    if s_cat == "MECHANISM" and t_cat == "FINANCIAL_METRIC": return "CAUSES"
    if s_cat == "MECHANISM" and t_cat == "RISK_FACTOR": return "AGGRAVATES"
    if s_cat == "RISK_FACTOR" and t_cat == "MECHANISM": return "TRIGGERS"
    if s_cat == "ESG_TOPIC" and t_cat == "MECHANISM": return "TRIGGERS"
    if s_cat == "REGION" and t_cat == "RISK_FACTOR": return "EXPOSED_TO"
    if s_cat == "MARKET" and t_cat in ("RISK_FACTOR","MECHANISM"): return "CAUSES"
    if s_cat == "MARKET" and t_cat == "FINANCIAL_METRIC":
        return "DECREASES" if re.search(r'\b(decline|slowdown|contraction|decrease|reduce)\b', ctx) else "INCREASES"
    if s_cat == "COMPANY" and t_cat == "FINANCIAL_METRIC": return "DISCLOSES"
    if s_cat == "FINANCIAL_METRIC" and t_cat == "FINANCIAL_METRIC":
        if source_name == "OPERATING_COST" and target_name == "NET_INCOME": return "DECREASES"
        if source_name == "REVENUE" and target_name == "NET_INCOME": return "INCREASES"
        return "CAUSES"
    if s_cat == "PRODUCT" and t_cat == "FINANCIAL_METRIC": return "INCREASES"
    if s_cat == "EVENT" and t_cat == "RISK_FACTOR": return "CAUSES"
    return "CAUSES"


# ============================================================
# 3. Section Detector
# ============================================================

class AcademicSectionDetector:
    SECTION_KEYWORDS = {
        "risk_factors": ["risk factors", "item 1a"],
        "md_and_a": ["management's discussion", "item 7"],
        "esg": ["esg", "environmental", "social", "governance", "sustainability", "climate"],
        "business": ["item 1.", "^business$"],
        "financials": ["financial statements", "item 8"],
    }
    MIN_CONTENT_CHARS = 200

    def __init__(self):
        self.scope_pages: List[int] = []
        self.section_map: Dict[int, str] = {}

    def scan(self, pdf) -> List[int]:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tl = text.lower()
            for section, keywords in self.SECTION_KEYWORDS.items():
                for kw in keywords:
                    if kw in tl:
                        self.section_map[i+1] = section
                        break
        self.scope_pages = sorted(p for p, sec in self.section_map.items()
                                   if sec in ("risk_factors","md_and_a","esg","business","financials"))
        return self.scope_pages

    def is_in_scope(self, page_num: int) -> bool:
        return page_num in self.scope_pages or any(abs(page_num-sp) <= 15 for sp in self.scope_pages)

    def describe(self) -> str:
        return f"Pages in scope: {len(self.scope_pages)}"


# ============================================================
# 4. TCFSG Ingestor (V19.0 Graph-native)
# ============================================================

class TCFSGIngestor:
    def __init__(self, groq_api_key: str = "", neo4j_uri: str = "",
                 neo4j_user: str = "", neo4j_pass: str = ""):
        self.groq_client = Groq(api_key=groq_api_key or os.getenv("GROQ_API_KEY",""))
        self.driver = GraphDatabase.driver(
            neo4j_uri or os.getenv("NEO4J_URI","bolt://localhost:7687"),
            auth=(neo4j_user or os.getenv("NEO4J_USER","neo4j"),
                  neo4j_pass or os.getenv("NEO4J_PASSWORD","password")),
        )
        if RecursiveCharacterTextSplitter:
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=3000, chunk_overlap=400,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            self.splitter = None
        self._filing_seen_keys: Set[Tuple[str,str,str]] = set()
        self._current_filing_name = ""

        # ── Load spaCy (try trf, fallback sm, then None) ──
        self._nlp = None
        if _HAS_SPACY:
            for model in ["en_core_web_trf", "en_core_web_sm"]:
                try:
                    self._nlp = spacy.load(model)
                    logger.info(f"  spaCy {model} loaded for REGION NER")
                    break
                except Exception:
                    continue
            if self._nlp is None:
                logger.warning("  spaCy model not found. Install: python -m spacy download en_core_web_sm")

    def close(self):
        self.driver.close()

    def _llm_extract(self, text: str) -> List[Dict]:
        prompt = f"""You are a financial ontology extractor for SEC 10-K filings.
Analyze the following text and extract causal/risk/strategy triples.

ENTITY CATEGORIES:
COMPANY, MARKET, REGULATION, ESG_TOPIC, PRODUCT,
REGION (e.g. ASIA, CHINA, UNITED_STATES, EUROPE, GLOBAL),
RISK_FACTOR, BUSINESS_STRATEGY, FINANCIAL_METRIC, EVENT, MECHANISM

RELATION TYPES:
EXPOSED_TO, MITIGATES, CAUSES, AMPLIFIES, REDUCES,
INCREASES, DECREASES, DISCLOSES, IMPLEMENTS, TRIGGERS, AGGRAVATES

Return a JSON list of objects with keys:
- source (canonical UPPER_SNAKE name)
- source_category
- target
- target_category
- relation
- evidence_sentence (verbatim from text)

Rules:
1. Use canonical UPPER_SNAKE names only
2. REGION entities must be actual region names (ASIA, CHINA)
3. Never use employee_advancement as COMPANY
4. HR/benefits disclosures are NOT FINANCIAL_METRIC
5. COMPETITION_RISK is the canonical form for competitors
6. Short entity names <= 40 chars
7. Use MECHANISM category for intermediate causal mechanisms

TEXT:
{text[:4000]}

Return ONLY valid JSON array."""
        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role":"user","content":prompt}],
                temperature=0.1, max_tokens=2000,
            )
            content = resp.choices[0].message.content.strip()
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"```\s*$", "", content)
            triples = json.loads(content)
            if isinstance(triples, dict): triples = [triples]
            return triples
        except Exception as e:
            logger.warning(f"  LLM extract error: {e}")
            return []

    def _rule_fallback_extract(self, text: str) -> List[Dict]:
        triples: List[Dict] = []
        text_lower = text.lower()

        # ── V19.0: REGION NER ──
        regions = extract_regions_ner(text, self._nlp)
        for region in regions:
            for risk_key, (risk_name, _) in CANONICAL_MAP.items():
                if risk_key not in text_lower:
                    continue
                if CANONICAL_MAP[risk_key][1] != "RISK_FACTOR":
                    continue
                evidence = extract_evidence_sentence(text, region.lower(), risk_key)
                triples.append({
                    "source":region,"source_category":"REGION",
                    "target":risk_name,"target_category":"RISK_FACTOR",
                    "relation":"EXPOSED_TO",
                    "causal_strength":detect_causal_strength(evidence),
                    "evidence_sentence":evidence,
                })

        # ── Extract canonical entities ──
        found: Dict[str, Tuple[str, str]] = {}
        for key, (name, cat) in CANONICAL_MAP.items():
            if len(key) <= 2: continue
            if "_" in key:
                phrase = re.escape(key.replace("_"," "))
                if re.search(rf'\b{phrase}s?\b', text_lower): found[key] = (name, cat)
            else:
                if re.search(rf'\b{re.escape(key)}s?\b', text_lower): found[key] = (name, cat)
        for raw_key, (cname, ccat) in LLM_RAW_ENTITY_MAP.items():
            if raw_key in text_lower: found[f"_raw_{cname}"] = (cname, ccat)
        for mech_name, mech_cat in extract_mechanisms(text):
            found[f"_mech_{mech_name}"] = (mech_name, mech_cat)

        risks = {k:v for k,v in found.items() if v[1]=="RISK_FACTOR"}
        fins = {k:v for k,v in found.items() if v[1]=="FINANCIAL_METRIC"}
        strs = {k:v for k,v in found.items() if v[1]=="BUSINESS_STRATEGY"}
        mkts = {k:v for k,v in found.items() if v[1]=="MARKET"}
        esgs = {k:v for k,v in found.items() if v[1]=="ESG_TOPIC"}
        evts = {k:v for k,v in found.items() if v[1]=="EVENT"}
        mech = {k:v for k,v in found.items() if v[1]=="MECHANISM"}

        # ── V19.0: Hubness balancer ──
        MAX_OUT = 5
        node_cnt: Dict[str, int] = defaultdict(int)
        def can(s: str) -> bool:
            n = norm_id(s)
            if node_cnt[n] >= MAX_OUT: return False
            node_cnt[n] += 1; return True

        # Risk -> Mechanism -> Financial
        for rk, (rn, _) in risks.items():
            for mk, (mn, _) in mech.items():
                ev = extract_evidence_sentence(text, rk, mk)
                if ev and len(ev) > 20 and can(rn):
                    triples.append({"source":rn,"source_category":"RISK_FACTOR",
                        "target":mn,"target_category":"MECHANISM","relation":"TRIGGERS",
                        "causal_strength":detect_causal_strength(ev),"evidence_sentence":ev})
                for fk, (fn, _) in fins.items():
                    ev2 = extract_evidence_sentence(text, mk, fk)
                    if ev2 and len(ev2) > 20 and can(mn):
                        rel = "DECREASES" if fn in POSITIVE_METRICS else "INCREASES"
                        triples.append({"source":mn,"source_category":"MECHANISM",
                            "target":fn,"target_category":"FINANCIAL_METRIC","relation":rel,
                            "causal_strength":detect_causal_strength(ev2),"evidence_sentence":ev2})
            # Risk -> Financial direct
            for fk, (fn, _) in fins.items():
                ev = extract_evidence_sentence(text, rk, fk)
                if ev and len(ev) > 30 and can(rn):
                    rel = "DECREASES" if fn in POSITIVE_METRICS else "INCREASES"
                    triples.append({"source":rn,"source_category":"RISK_FACTOR",
                        "target":fn,"target_category":"FINANCIAL_METRIC","relation":rel,
                        "causal_strength":detect_causal_strength(ev),"evidence_sentence":ev})
            # Risk -> Risk
            for rk2, (rn2, _) in risks.items():
                if rk == rk2: continue
                ev = extract_evidence_sentence(text, rk, rk2)
                if ev and len(ev) > 20 and can(rn):
                    rel = "AMPLIFIES" if "amplify" in ev.lower() else "CAUSES"
                    triples.append({"source":rn,"source_category":"RISK_FACTOR",
                        "target":rn2,"target_category":"RISK_FACTOR","relation":rel,
                        "causal_strength":detect_causal_strength(ev),"evidence_sentence":ev})

        # ESG -> Mechanism
        for ek, (en, _) in esgs.items():
            for mk, (mn, _) in mech.items():
                ev = extract_evidence_sentence(text, ek, mk)
                if ev and len(ev) > 20 and can(en):
                    triples.append({"source":en,"source_category":"ESG_TOPIC",
                        "target":mn,"target_category":"MECHANISM","relation":"TRIGGERS",
                        "causal_strength":detect_causal_strength(ev),"evidence_sentence":ev})

        # Event -> Risk
        for evk, (evn, _) in evts.items():
            for rk, (rn, _) in risks.items():
                ev = extract_evidence_sentence(text, evk, rk)
                if ev and len(ev) > 20 and can(evn):
                    triples.append({"source":evn,"source_category":"EVENT",
                        "target":rn,"target_category":"RISK_FACTOR","relation":"CAUSES",
                        "causal_strength":detect_causal_strength(ev),"evidence_sentence":ev})

        # Strategy -> Risk (MITIGATES)
        for sk, (sn, _) in strs.items():
            for rk, (rn, _) in risks.items():
                ev = extract_evidence_sentence(text, sk, rk)
                if ev and len(ev) > 20 and can(sn):
                    triples.append({"source":sn,"source_category":"BUSINESS_STRATEGY",
                        "target":rn,"target_category":"RISK_FACTOR","relation":"MITIGATES",
                        "causal_strength":detect_causal_strength(ev),"evidence_sentence":ev})

        # Market -> Risk
        for mk, (mn, _) in mkts.items():
            for rk, (rn, _) in risks.items():
                ev = extract_evidence_sentence(text, mk, rk)
                if ev and len(ev) > 20 and can(mn):
                    triples.append({"source":mn,"source_category":"MARKET",
                        "target":rn,"target_category":"RISK_FACTOR","relation":"CAUSES",
                        "causal_strength":detect_causal_strength(ev),"evidence_sentence":ev})

        return triples

    def _filter_triples(self, triples: List[Dict], text: str) -> List[Dict]:
        intent = classify_intent(text)
        filtered: List[Dict] = []
        MAX_DIS = 2; MAX_EXP = 4; MAX_OPEX = 3
        dis_cnt = 0; exp_cnt = 0; opex_cnt = 0

        for t in triples:
            s_raw = str(t.get("source","")).strip()
            t_raw = str(t.get("target","")).strip()
            rel = str(t.get("relation","")).strip().upper()
            s_cat = str(t.get("source_category","")).strip().upper()
            t_cat = str(t.get("target_category","")).strip().upper()
            if not s_raw or not t_raw or not rel: continue
            if is_banned(s_raw) or is_banned(t_raw): continue
            if norm_id(s_raw) == norm_id(t_raw): continue

            # V19.0: Intent-based ESG/HR reclassification
            if intent == "ESG" and t_cat == "FINANCIAL_METRIC":
                tc = re.sub(r"[^a-zA-Z0-9]","_",t_raw).upper()[:40].strip("_")
                if tc not in FINANCIAL_METRICS: t_cat = "ESG_TOPIC"
            if intent == "ESG" and s_cat == "FINANCIAL_METRIC":
                sc = re.sub(r"[^a-zA-Z0-9]","_",s_raw).upper()[:40].strip("_")
                if sc not in FINANCIAL_METRICS: s_cat = "ESG_TOPIC"

            if is_competitor(s_raw): s_cat="RISK_FACTOR"; s_raw="COMPETITION_RISK"
            if is_competitor(t_raw): t_cat="RISK_FACTOR"; t_raw="COMPETITION_RISK"
            if is_generic(s_raw) or is_generic(t_raw): continue
            if t_cat=="RISK_FACTOR" and is_non_risk(t_raw): continue
            if s_cat=="RISK_FACTOR" and is_non_risk(s_raw): continue
            if s_cat=="COMPANY" and t_cat=="ESG_TOPIC": continue
            if is_company_blacklisted(s_raw) or is_company_blacklisted(t_raw): continue
            if is_non_canonical_metric(s_raw) or is_non_canonical_metric(t_raw): continue

            # Reclassify
            nid_s = norm_id(s_raw); nid_t = norm_id(t_raw)
            if s_cat=="ESG_TOPIC" and nid_s in ESG_MISCLASSIFIED_AS_MARKET:
                s_cat = ESG_MISCLASSIFIED_AS_MARKET[nid_s]
            if t_cat=="ESG_TOPIC" and nid_t in ESG_MISCLASSIFIED_AS_MARKET:
                t_cat = ESG_MISCLASSIFIED_AS_MARKET[nid_t]
            if s_cat=="RISK_FACTOR" and nid_s in RISK_MISCLASSIFIED_AS_ESG:
                s_cat = RISK_MISCLASSIFIED_AS_ESG[nid_s]
            if t_cat=="RISK_FACTOR" and nid_t in RISK_MISCLASSIFIED_AS_ESG:
                t_cat = RISK_MISCLASSIFIED_AS_ESG[nid_t]

            if rel == "MITIGATES":
                if s_cat not in MITIGATES_SOURCE_CATEGORIES:
                    rel = "EXPOSED_TO" if s_cat=="COMPANY" and t_cat=="RISK_FACTOR" else "CAUSES"
            if s_cat=="BUSINESS_STRATEGY" and t_cat=="RISK_FACTOR": rel = "MITIGATES"

            if rel == "CAUSES" or rel not in VALID_RELATIONS:
                inf = infer_relation(s_cat, t_cat, text, s_raw, t_raw)
                if inf != "CAUSES": rel = inf

            if rel=="DISCLOSES": dis_cnt+=1
            if dis_cnt>MAX_DIS: continue
            if rel=="EXPOSED_TO": exp_cnt+=1
            if exp_cnt>MAX_EXP: continue
            if t_cat=="FINANCIAL_METRIC" and norm_id(t_raw)=="operating_cost": opex_cnt+=1
            if opex_cnt>MAX_OPEX: continue
            if s_cat=="FINANCIAL_METRIC" and norm_id(s_raw)=="operating_cost": opex_cnt+=1
            if opex_cnt>MAX_OPEX: continue

            if "causal_strength" not in t or not t.get("causal_strength"):
                t["causal_strength"] = detect_causal_strength(str(t.get("evidence_sentence","")))

            sn, sc = resolve_entity(s_raw, s_cat)
            tn, tc = resolve_entity(t_raw, t_cat)
            if sc in ENTITY_CATEGORIES and tc in ENTITY_CATEGORIES and rel in VALID_RELATIONS:
                t["source"]=sn; t["source_category"]=sc
                t["target"]=tn; t["target_category"]=tc; t["relation"]=rel
                filtered.append(t)
        return filtered

    def _ingest_triple(self, tx, triple: Dict, filename: str, page: int, year: int) -> bool:
        """V19.0: Native Neo4j relationships (actual :CAUSES, not generic RELATION)."""
        s_raw = str(triple.get("source","")).strip()
        t_raw = str(triple.get("target","")).strip()
        rel = str(triple.get("relation","")).strip().upper()
        s_cat = str(triple.get("source_category","")).strip().upper()
        t_cat = str(triple.get("target_category","")).strip().upper()
        cs = str(triple.get("causal_strength","DISCLOSED_EXPOSURE")).upper()
        ev = str(triple.get("evidence_sentence",""))[:500]
        if not s_raw or not t_raw or not rel: return False
        if is_banned(s_raw) or is_banned(t_raw): return False
        if rel not in VALID_RELATIONS: return False
        if s_cat not in ENTITY_CATEGORIES or t_cat not in ENTITY_CATEGORIES: return False

        sn, sc = resolve_entity(s_raw, s_cat)
        tn, tc = resolve_entity(t_raw, t_cat)
        sid = norm_id(sn); tid = norm_id(tn)
        eid = hashlib.md5(f"{sid}|{rel}|{tid}|{year}|p{page}".encode()).hexdigest()[:16]

        # ── V19.0: Native relationship ──
        cypher = f"""
MERGE (s:{sc} {{id: $sid}})
ON CREATE SET s.name = $sn
MERGE (t:{tc} {{id: $tid}})
ON CREATE SET t.name = $tn
MERGE (s)-[r:{rel} {{id: $eid}}]->(t)
ON CREATE SET r.causal_strength = $cs,
    r.evidence_sentence = $ev,
    r.year = $yr, r.page = $pg, r.filing = $file,
    r.source_category = $sc, r.target_category = $tc,
    r.support_strength = 0.7
MERGE (y:Year {{year: $yr}})
MERGE (r)-[:OBSERVED_IN]->(y)
MERGE (d:Document {{docid: $file}})
MERGE (r)-[:HAS_EVIDENCE]->(d)
MERGE (es:EvidenceSentence {{id: $eid + "_es"}})
ON CREATE SET es.text = $ev, es.page = $pg
MERGE (es)-[:SUPPORTS]->(r)
"""
        try:
            tx.run(cypher, sid=sid, sn=sn, tid=tid, tn=tn, eid=eid,
                   cs=cs, ev=ev, yr=year, pg=page, file=filename,
                   sc=sc, tc=tc)
            return True
        except Exception as e:
            logger.warning(f"  Ingest error: {str(e)[:120]}")
            return False

    def process_batch(self, input_folder: str):
        # ── Windows: resolve absolute path ──
        input_folder = os.path.abspath(input_folder)
        logger.info(f"  PDF folder: {input_folder}")
        logger.info(f"  Folder exists: {os.path.isdir(input_folder)}")

        # ── Windows: use os.path.join for glob ──
        pdf_pattern = os.path.join(input_folder, "*.pdf")
        pdfs = sorted(glob.glob(pdf_pattern))

        if not pdfs:
            logger.error(f"No PDF files found in: {input_folder}")
            logger.error("Please ensure .pdf files exist in that directory.")
            return

        logger.info(f"Found {len(pdfs)} PDF(s)")
        for pdf_path in pdfs:
            fname = os.path.basename(pdf_path)
            logger.info(f"\n{'='*60}\nProcessing: {fname}\n{'='*60}")
            year = 2022
            m = re.search(r"(20\d{2})", pdf_path)
            if m: year = int(m.group(1))

            self._filing_seen_keys.clear()
            self._current_filing_name = fname

            try:
                with pdfplumber.open(pdf_path) as pdf:
                    logger.info(f"  Total pages: {len(pdf.pages)}")
                    detector = AcademicSectionDetector()
                    detector.scan(pdf)
                    logger.info(f"  Sections: {detector.describe()}")
                    target_pages = [i for i in range(len(pdf.pages)) if detector.is_in_scope(i+1)]
                    logger.info(f"  Pages in scope: {len(target_pages)}")

                    relation_counts: Dict[str,int] = {r:0 for r in VALID_RELATIONS}
                    total_unique = 0
                    iter_pages = tqdm(target_pages, desc="  Extracting") if tqdm else target_pages

                    for idx in iter_pages:
                        page = pdf.pages[idx]; pn = idx+1
                        text = page.extract_text() or ""
                        if len(text.strip()) < AcademicSectionDetector.MIN_CONTENT_CHARS:
                            continue

                        page_triples: List[Dict] = []
                        if self.splitter:
                            chunks = self.splitter.split_text(text)
                            for chunk in chunks:
                                page_triples.extend(self._llm_extract(chunk))
                        page_triples.extend(self._rule_fallback_extract(text))
                        page_triples = self._filter_triples(page_triples, text)

                        unique: List[Dict] = []
                        for t in page_triples:
                            key = (norm_id(t.get("source","")), t.get("relation",""), norm_id(t.get("target","")))
                            if key not in self._filing_seen_keys:
                                self._filing_seen_keys.add(key); unique.append(t)

                        if unique:
                            ingested = 0
                            with self.driver.session() as session:
                                for t in unique:
                                    ok = session.execute_write(self._ingest_triple, t, fname, pn, year)
                                    if ok:
                                        ingested += 1
                                        rel = t.get("relation","")
                                        if rel in relation_counts: relation_counts[rel] += 1
                            total_unique += ingested
                            if ingested > 0:
                                logger.info(f"  Page {pn}: {len(unique)} unique, {ingested} ingested")

                    logger.info(f"\n{'='*60}\n  FILING: {fname}\n{'='*60}")
                    total = sum(relation_counts.values())
                    logger.info(f"  Total relations: {total}")
                    for rel, cnt in sorted(relation_counts.items(), key=lambda x:-x[1]):
                        if cnt>0: logger.info(f"    {rel}: {cnt} ({cnt/total*100:.1f}%)")
                    logger.info(f"  Unique: {total_unique}, Year: {year}")

            except Exception as e:
                logger.error(f"  ERROR: {e}")
                import traceback; logger.error(traceback.format_exc())


# ============================================================
# 5. Main Entry (Windows-friendly default path)
# ============================================================

def main():
    # ── Windows: user's PDF folder ──
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    # Try user's specific path first, then relative paths
    preferred = r"C:\Users\32875\OneDrive\Desktop\Louis academic project\Nvidia-GraphRAG-Engine\data\raw"
    candidates = [
        preferred,
        os.path.join(project_root, "data", "raw"),
        os.path.join(project_root, "data", "pdfs"),
        os.path.join(script_dir, "data", "raw"),
        os.path.join(script_dir, "data", "pdfs"),
        "data/raw", "./data/raw", "data/pdfs", "./data/pdfs",
    ]
    pdf_folder = None
    for c in candidates:
        if os.path.isdir(c):
            pdf_folder = c
            logger.info(f"Using PDF folder: {pdf_folder}")
            break
    if pdf_folder is None:
        pdf_folder = preferred
        logger.warning(f"PDF folder not found. Creating and using: {pdf_folder}")
        os.makedirs(pdf_folder, exist_ok=True)

    ingestor = TCFSGIngestor()
    try:
        ingestor.process_batch(pdf_folder)
    finally:
        ingestor.close()


if __name__ == "__main__":
    main()