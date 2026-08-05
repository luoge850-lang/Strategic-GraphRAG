# -*- coding: utf-8 -*-
"""
Strategic-GraphRAG: Canonical Entity Registry
=============================================
Complete financial ontology entity standardization with:
  - 500+ canonical entity mappings
  - 24 risk factor categories
  - 12 strategy types
  - 17 financial metrics
  - 30+ causal mechanism patterns
  - Entity disambiguation rules
  - Relationship inference rules
"""

import re
from typing import Dict, List, Optional, Tuple, Set

# =============================================================================
# 1. CANONICAL ENTITY MAPPING (normalized_id → (display_name, node_label))
# =============================================================================

CANONICAL_MAP: Dict[str, Tuple[str, str]] = {
    # ── Company ──
    "nvidia": ("NVIDIA_CORPORATION", "Company"),
    "nvidia_corporation": ("NVIDIA_CORPORATION", "Company"),
    "nvda": ("NVIDIA_CORPORATION", "Company"),
    "amd": ("ADVANCED_MICRO_DEVICES", "Company"),
    "advanced_micro_devices": ("ADVANCED_MICRO_DEVICES", "Company"),
    "intel": ("INTEL_CORPORATION", "Company"),
    "intel_corporation": ("INTEL_CORPORATION", "Company"),
    "tsmc": ("TSMC", "Company"),
    "taiwan_semiconductor": ("TSMC", "Company"),
    "samsung": ("SAMSUNG_ELECTRONICS", "Company"),
    "broadcom": ("BROADCOM_INC", "Company"),
    "qualcomm": ("QUALCOMM_INC", "Company"),
    "micron": ("MICRON_TECHNOLOGY", "Company"),
    "sk_hynix": ("SK_HYNIX", "Company"),
    "microsoft": ("MICROSOFT_CORPORATION", "Company"),
    "google": ("GOOGLE_LLC", "Company"),
    "alphabet": ("ALPHABET_INC", "Company"),
    "amazon": ("AMAZON_COM_INC", "Company"),
    "apple": ("APPLE_INC", "Company"),
    "meta": ("META_PLATFORMS_INC", "Company"),
    "huawei": ("HUAWEI_TECHNOLOGIES", "Company"),
    "mediatek": ("MEDIATEK_INC", "Company"),

    # ── Financial Metrics ──
    "revenue": ("REVENUE", "FinancialMetric"),
    "revenues": ("REVENUE", "FinancialMetric"),
    "sales": ("REVENUE", "FinancialMetric"),
    "net_revenue": ("REVENUE", "FinancialMetric"),
    "total_revenue": ("REVENUE", "FinancialMetric"),
    "gross_margin": ("GROSS_MARGIN", "FinancialMetric"),
    "gross_profit": ("GROSS_MARGIN", "FinancialMetric"),
    "operating_margin": ("OPERATING_MARGIN", "FinancialMetric"),
    "net_income": ("NET_INCOME", "FinancialMetric"),
    "net_earnings": ("NET_INCOME", "FinancialMetric"),
    "net_profit": ("NET_INCOME", "FinancialMetric"),
    "cash_flow": ("CASH_FLOW", "FinancialMetric"),
    "cash_flows": ("CASH_FLOW", "FinancialMetric"),
    "operating_cash_flow": ("CASH_FLOW", "FinancialMetric"),
    "free_cash_flow": ("FREE_CASH_FLOW", "FinancialMetric"),
    "earnings_per_share": ("EARNINGS_PER_SHARE", "FinancialMetric"),
    "eps": ("EARNINGS_PER_SHARE", "FinancialMetric"),
    "diluted_eps": ("EARNINGS_PER_SHARE", "FinancialMetric"),
    "return_on_equity": ("RETURN_ON_EQUITY", "FinancialMetric"),
    "roe": ("RETURN_ON_EQUITY", "FinancialMetric"),
    "market_value": ("MARKET_VALUE", "FinancialMetric"),
    "market_capitalization": ("MARKET_VALUE", "FinancialMetric"),
    "stock_price": ("MARKET_VALUE", "FinancialMetric"),
    "ebitda": ("EBITDA", "FinancialMetric"),
    "operating_cost": ("OPERATING_COST", "FinancialMetric"),
    "operating_costs": ("OPERATING_COST", "FinancialMetric"),
    "operating_expenses": ("OPERATING_COST", "FinancialMetric"),
    "operating_expense": ("OPERATING_COST", "FinancialMetric"),
    "total_operating_expenses": ("OPERATING_COST", "FinancialMetric"),
    "cost_of_revenue": ("COST_OF_REVENUE", "FinancialMetric"),
    "cost_of_goods_sold": ("COST_OF_REVENUE", "FinancialMetric"),
    "cogs": ("COST_OF_REVENUE", "FinancialMetric"),
    "r_and_d_expense": ("R_AND_D_EXPENSE", "FinancialMetric"),
    "r_and_d": ("R_AND_D_EXPENSE", "FinancialMetric"),
    "research_and_development": ("R_AND_D_EXPENSE", "FinancialMetric"),
    "research_and_development_expense": ("R_AND_D_EXPENSE", "FinancialMetric"),
    "rd_expense": ("R_AND_D_EXPENSE", "FinancialMetric"),
    "sg_and_a_expense": ("SG_AND_A_EXPENSE", "FinancialMetric"),
    "sg_and_a": ("SG_AND_A_EXPENSE", "FinancialMetric"),
    "selling_general_administrative": ("SG_AND_A_EXPENSE", "FinancialMetric"),
    "capex": ("CAPEX", "FinancialMetric"),
    "capital_expenditures": ("CAPEX", "FinancialMetric"),
    "capital_expenditure": ("CAPEX", "FinancialMetric"),
    "debt_to_equity": ("DEBT_TO_EQUITY", "FinancialMetric"),
    "debt_to_equity_ratio": ("DEBT_TO_EQUITY", "FinancialMetric"),
    "current_ratio": ("CURRENT_RATIO", "FinancialMetric"),
    "cost": ("OPERATING_COST", "FinancialMetric"),
    "costs": ("OPERATING_COST", "FinancialMetric"),
    "expense": ("OPERATING_COST", "FinancialMetric"),
    "expenses": ("OPERATING_COST", "FinancialMetric"),

    # ── Risk Factors: Supply Chain ──
    "supply_chain_disruption": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "supply_chain_risk": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "supply_chain": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "supply_constraints": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "supply_shortage": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "supply_shortages": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "foundry_capacity": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "foundry_capacity_constraints": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "wafer_supply": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "manufacturing_capacity": ("SUPPLY_CHAIN_DISRUPTION", "RiskFactor"),
    "cowos_packaging_capacity": ("COWOS_PACKAGING_CONSTRAINT", "RiskFactor"),
    "advanced_packaging": ("COWOS_PACKAGING_CONSTRAINT", "RiskFactor"),
    "packaging_capacity": ("COWOS_PACKAGING_CONSTRAINT", "RiskFactor"),
    "hbm_supply": ("HBM_SUPPLY_CONSTRAINT", "RiskFactor"),
    "high_bandwidth_memory": ("HBM_SUPPLY_CONSTRAINT", "RiskFactor"),
    "hbm_shortage": ("HBM_SUPPLY_CONSTRAINT", "RiskFactor"),

    # ── Risk Factors: Export Controls ──
    "export_controls": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "export_control": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "export_restrictions": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "export_restriction": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "trade_restrictions": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "trade_controls": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "chip_export_restriction": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "chip_export_controls": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "import_export_restrictions": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "tariffs": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "tariff": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "trade_barriers": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "trade_war": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),
    "us_china_trade_tensions": ("CHIP_EXPORT_RESTRICTION", "RiskFactor"),

    # ── Risk Factors: Regulatory ──
    "ai_regulation": ("AI_REGULATION_RISK", "RiskFactor"),
    "ai_regulation_risk": ("AI_REGULATION_RISK", "RiskFactor"),
    "artificial_intelligence_regulation": ("AI_REGULATION_RISK", "RiskFactor"),
    "government_regulation": ("GOVERNMENT_REGULATION_RISK", "RiskFactor"),
    "regulatory_compliance": ("GOVERNMENT_REGULATION_RISK", "RiskFactor"),
    "regulatory_scrutiny": ("GOVERNMENT_REGULATION_RISK", "RiskFactor"),
    "regulatory_risk": ("GOVERNMENT_REGULATION_RISK", "RiskFactor"),
    "regulatory_approval": ("REGULATORY_APPROVAL_RISK", "RiskFactor"),
    "antitrust": ("ANTITRUST_RISK", "RiskFactor"),
    "antitrust_investigation": ("ANTITRUST_RISK", "RiskFactor"),
    "antitrust_scrutiny": ("ANTITRUST_RISK", "RiskFactor"),

    # ── Risk Factors: Market & Competition ──
    "competition": ("COMPETITION_RISK", "RiskFactor"),
    "competitive_pressure": ("COMPETITION_RISK", "RiskFactor"),
    "intense_competition": ("COMPETITION_RISK", "RiskFactor"),
    "competitive_landscape": ("COMPETITION_RISK", "RiskFactor"),
    "market_competition": ("COMPETITION_RISK", "RiskFactor"),
    "competitive_risk": ("COMPETITION_RISK", "RiskFactor"),
    "demand_volatility": ("DEMAND_VOLATILITY", "RiskFactor"),
    "decrease_in_demand": ("DEMAND_VOLATILITY", "RiskFactor"),
    "decreased_demand": ("DEMAND_VOLATILITY", "RiskFactor"),
    "demand_uncertainty": ("DEMAND_VOLATILITY", "RiskFactor"),
    "demand_risk": ("DEMAND_VOLATILITY", "RiskFactor"),
    "product_demand_fluctuations": ("DEMAND_VOLATILITY", "RiskFactor"),
    "customer_concentration": ("CUSTOMER_CONCENTRATION_RISK", "RiskFactor"),
    "customer_concentration_risk": ("CUSTOMER_CONCENTRATION_RISK", "RiskFactor"),
    "revenue_concentration": ("CUSTOMER_CONCENTRATION_RISK", "RiskFactor"),
    "inventory_risk": ("INVENTORY_RISK", "RiskFactor"),
    "excess_inventory": ("INVENTORY_RISK", "RiskFactor"),
    "inventory_obsolescence": ("INVENTORY_RISK", "RiskFactor"),
    "inventory_management": ("INVENTORY_RISK", "RiskFactor"),
    "product_transition": ("PRODUCT_TRANSITION_RISK", "RiskFactor"),
    "product_transition_risk": ("PRODUCT_TRANSITION_RISK", "RiskFactor"),

    # ── Risk Factors: Technology ──
    "technology_obsolescence": ("TECHNOLOGY_OBSOLESCENCE", "RiskFactor"),
    "technological_change": ("TECHNOLOGY_OBSOLESCENCE", "RiskFactor"),
    "technological_advancements": ("TECHNOLOGY_OBSOLESCENCE", "RiskFactor"),
    "rapid_technological_change": ("TECHNOLOGY_OBSOLESCENCE", "RiskFactor"),
    "intellectual_property": ("INTELLECTUAL_PROPERTY_RISK", "RiskFactor"),
    "intellectual_property_risk": ("INTELLECTUAL_PROPERTY_RISK", "RiskFactor"),
    "intellectual_property_protection": ("INTELLECTUAL_PROPERTY_RISK", "RiskFactor"),
    "ip_infringement": ("INTELLECTUAL_PROPERTY_RISK", "RiskFactor"),
    "patent_infringement": ("INTELLECTUAL_PROPERTY_RISK", "RiskFactor"),
    "ip_ownership": ("INTELLECTUAL_PROPERTY_RISK", "RiskFactor"),
    "ip_litigation": ("INTELLECTUAL_PROPERTY_RISK", "RiskFactor"),
    "cybersecurity_risk": ("SYSTEM_SECURITY_BREACH", "RiskFactor"),
    "cybersecurity": ("SYSTEM_SECURITY_BREACH", "RiskFactor"),
    "security_breach": ("SYSTEM_SECURITY_BREACH", "RiskFactor"),
    "cyber_attack": ("SYSTEM_SECURITY_BREACH", "RiskFactor"),
    "data_breach": ("SYSTEM_SECURITY_BREACH", "RiskFactor"),
    "cybersecurity_threats": ("SYSTEM_SECURITY_BREACH", "RiskFactor"),
    "cyber_risk": ("SYSTEM_SECURITY_BREACH", "RiskFactor"),

    # ── Risk Factors: Geopolitical & Macro ──
    "geopolitical_risk": ("GEOPOLITICAL_RISK", "RiskFactor"),
    "geopolitical_tensions": ("GEOPOLITICAL_RISK", "RiskFactor"),
    "geopolitical_uncertainty": ("GEOPOLITICAL_RISK", "RiskFactor"),
    "geopolitical_conflict": ("GEOPOLITICAL_RISK", "RiskFactor"),
    "taiwan_strait_tensions": ("GEOPOLITICAL_RISK", "RiskFactor"),
    "china_taiwan_tensions": ("GEOPOLITICAL_RISK", "RiskFactor"),
    "macroeconomic_conditions": ("MACROECONOMIC_CONDITIONS", "RiskFactor"),
    "macroeconomic_risk": ("MACROECONOMIC_CONDITIONS", "RiskFactor"),
    "economic_downturn": ("MACROECONOMIC_CONDITIONS", "RiskFactor"),
    "global_economic_conditions": ("MACROECONOMIC_CONDITIONS", "RiskFactor"),
    "recession": ("MACROECONOMIC_CONDITIONS", "RiskFactor"),
    "inflation": ("MACROECONOMIC_CONDITIONS", "RiskFactor"),
    "inflation_risk": ("MACROECONOMIC_CONDITIONS", "RiskFactor"),
    "interest_rate_risk": ("INTEREST_RATE_RISK", "RiskFactor"),
    "interest_rate_hike": ("INTEREST_RATE_RISK", "RiskFactor"),
    "rising_interest_rates": ("INTEREST_RATE_RISK", "RiskFactor"),
    "foreign_exchange": ("FOREIGN_EXCHANGE_RISK", "RiskFactor"),
    "foreign_exchange_risk": ("FOREIGN_EXCHANGE_RISK", "RiskFactor"),
    "currency_fluctuations": ("FOREIGN_EXCHANGE_RISK", "RiskFactor"),
    "currency_risk": ("FOREIGN_EXCHANGE_RISK", "RiskFactor"),
    "sovereign_ai": ("SOVEREIGN_AI_POLICY_RISK", "RiskFactor"),
    "data_sovereignty": ("SOVEREIGN_AI_POLICY_RISK", "RiskFactor"),

    # ── Risk Factors: Other ──
    "talent_shortage": ("TALENT_SHORTAGE", "RiskFactor"),
    "talent_competition": ("TALENT_SHORTAGE", "RiskFactor"),
    "talent_retention": ("TALENT_SHORTAGE", "RiskFactor"),
    "skilled_personnel": ("TALENT_SHORTAGE", "RiskFactor"),
    "key_personnel": ("TALENT_SHORTAGE", "RiskFactor"),
    "reputation_risk": ("REPUTATION_RISK", "RiskFactor"),
    "brand_reputation": ("REPUTATION_RISK", "RiskFactor"),
    "credit_risk": ("CREDIT_RISK", "RiskFactor"),
    "liquidity_risk": ("LIQUIDITY_RISK", "RiskFactor"),
    "litigation": ("LITIGATION_RISK", "RiskFactor"),
    "litigation_risk": ("LITIGATION_RISK", "RiskFactor"),
    "legal_proceedings": ("LITIGATION_RISK", "RiskFactor"),
    "product_liability": ("PRODUCT_LIABILITY_RISK", "RiskFactor"),
    "product_liability_risk": ("PRODUCT_LIABILITY_RISK", "RiskFactor"),
    "warranty_claims": ("PRODUCT_LIABILITY_RISK", "RiskFactor"),
    "quality_control": ("PRODUCT_LIABILITY_RISK", "RiskFactor"),
    "tax_risk": ("TAX_RISK", "RiskFactor"),
    "tax_legislation": ("TAX_RISK", "RiskFactor"),
    "tax_change": ("TAX_RISK", "RiskFactor"),
    "acquisition_integration": ("ACQUISITION_INTEGRATION_RISK", "RiskFactor"),
    "acquisition_risk": ("ACQUISITION_INTEGRATION_RISK", "RiskFactor"),
    "merger_risk": ("ACQUISITION_INTEGRATION_RISK", "RiskFactor"),
    "cryptocurrency_volatility": ("CRYPTO_MARKET_VOLATILITY", "RiskFactor"),
    "crypto_market": ("CRYPTO_MARKET_VOLATILITY", "RiskFactor"),
    "digital_currency": ("CRYPTO_MARKET_VOLATILITY", "RiskFactor"),
    "environmental_risk": ("ENVIRONMENTAL_REGULATION_RISK", "RiskFactor"),
    "climate_risk": ("ENVIRONMENTAL_REGULATION_RISK", "RiskFactor"),
    "energy_consumption_risk": ("ENERGY_CONSUMPTION_RISK", "RiskFactor"),

    # ── Markets ──
    "gpu_market": ("GPU_MARKET", "Market"),
    "data_center_market": ("DATA_CENTER_MARKET", "Market"),
    "data_center": ("DATA_CENTER_MARKET", "Market"),
    "ai_chip_market": ("AI_CHIP_MARKET", "Market"),
    "ai_market": ("AI_CHIP_MARKET", "Market"),
    "artificial_intelligence_market": ("AI_CHIP_MARKET", "Market"),
    "automotive_market": ("AUTOMOTIVE_MARKET", "Market"),
    "automotive_industry": ("AUTOMOTIVE_MARKET", "Market"),
    "gaming_market": ("GAMING_MARKET", "Market"),
    "gaming_industry": ("GAMING_MARKET", "Market"),
    "china_market": ("CHINA_MARKET", "Market"),
    "chinese_market": ("CHINA_MARKET", "Market"),
    "global_semiconductor": ("GLOBAL_SEMICONDUCTOR_MARKET", "Market"),
    "semiconductor_market": ("GLOBAL_SEMICONDUCTOR_MARKET", "Market"),
    "semiconductor_industry": ("GLOBAL_SEMICONDUCTOR_MARKET", "Market"),
    "chip_market": ("GLOBAL_SEMICONDUCTOR_MARKET", "Market"),
    "cloud_market": ("CLOUD_COMPUTING_MARKET", "Market"),
    "cloud_computing": ("CLOUD_COMPUTING_MARKET", "Market"),
    "hpc_market": ("HPC_MARKET", "Market"),
    "high_performance_computing": ("HPC_MARKET", "Market"),
    "professional_visualization_market": ("PROFESSIONAL_VIZ_MARKET", "Market"),
    "edge_ai_market": ("EDGE_AI_MARKET", "Market"),
    "digital_twin_market": ("DIGITAL_TWIN_MARKET", "Market"),

    # ── Products ──
    "gpu": ("GPU", "Product"),
    "gpus": ("GPU", "Product"),
    "graphics_processing_unit": ("GPU", "Product"),
    "graphics_card": ("GEFORCE_RTX_4090", "Product"),
    "cuda": ("CUDA_PLATFORM", "Product"),
    "cuda_platform": ("CUDA_PLATFORM", "Product"),
    "cuda_software": ("CUDA_PLATFORM", "Product"),
    "cuda_ecosystem": ("CUDA_PLATFORM", "Product"),
    "dgx": ("DGX_SYSTEM", "Product"),
    "dgx_system": ("DGX_SYSTEM", "Product"),
    "dgx_superpod": ("DGX_SYSTEM", "Product"),
    "h100": ("H100_TENSOR_CORE_GPU", "Product"),
    "h100_gpu": ("H100_TENSOR_CORE_GPU", "Product"),
    "h100_tensor_core": ("H100_TENSOR_CORE_GPU", "Product"),
    "a100": ("A100_TENSOR_CORE_GPU", "Product"),
    "a100_gpu": ("A100_TENSOR_CORE_GPU", "Product"),
    "b200": ("B200_BLACKWELL_GPU", "Product"),
    "blackwell": ("B200_BLACKWELL_GPU", "Product"),
    "blackwell_gpu": ("B200_BLACKWELL_GPU", "Product"),
    "h200": ("H200_GPU", "Product"),
    "h800": ("H800_CHINA_EXPORT_GPU", "Product"),
    "h20": ("H20_CHINA_EXPORT_GPU", "Product"),
    "geforce": ("GEFORCE_RTX_4090", "Product"),
    "tesla_gpu": ("TESLA_GPU", "Product"),
    "tensor_core": ("TENSOR_CORE_GPU", "Product"),
    "tensor_core_gpu": ("TENSOR_CORE_GPU", "Product"),
    "grace_cpu": ("GRACE_CPU", "Product"),
    "grace_hopper": ("GRACE_CPU", "Product"),
    "drive_platform": ("DRIVE_PLATFORM", "Product"),
    "nvidia_drive": ("DRIVE_PLATFORM", "Product"),
    "drive_agx": ("DRIVE_PLATFORM", "Product"),
    "drive_orin": ("DRIVE_PLATFORM", "Product"),
    "orin": ("DRIVE_PLATFORM", "Product"),
    "jetson": ("JETSON_PLATFORM", "Product"),
    "jetson_platform": ("JETSON_PLATFORM", "Product"),
    "omniverse": ("OMNIVERSE_PLATFORM", "Product"),
    "omniverse_platform": ("OMNIVERSE_PLATFORM", "Product"),
    "mellanox": ("MELLANOX_NETWORKING", "Product"),
    "mellanox_networking": ("MELLANOX_NETWORKING", "Product"),
    "infiniband": ("MELLANOX_NETWORKING", "Product"),
    "nvlink": ("NVLINK_INTERCONNECT", "Product"),
    "nvlink_interconnect": ("NVLINK_INTERCONNECT", "Product"),
    "bluefield": ("BLUEFIELD_DPU", "Product"),
    "dpu": ("BLUEFIELD_DPU", "Product"),
    "bluefield_dpu": ("BLUEFIELD_DPU", "Product"),
    "soc": ("SOC", "Product"),
    "system_on_chip": ("SOC", "Product"),

    # ── Regions ──
    "united_states": ("UNITED_STATES", "Region"),
    "u.s.": ("UNITED_STATES", "Region"),
    "us": ("UNITED_STATES", "Region"),
    "usa": ("UNITED_STATES", "Region"),
    "china": ("CHINA", "Region"),
    "taiwan": ("TAIWAN", "Region"),
    "europe": ("EUROPE", "Region"),
    "european_union": ("EUROPE", "Region"),
    "eu": ("EUROPE", "Region"),
    "japan": ("JAPAN", "Region"),
    "south_korea": ("SOUTH_KOREA", "Region"),
    "korea": ("SOUTH_KOREA", "Region"),
    "singapore": ("SINGAPORE", "Region"),
    "israel": ("ISRAEL", "Region"),
    "india": ("INDIA", "Region"),
    "united_kingdom": ("UNITED_KINGDOM", "Region"),
    "uk": ("UNITED_KINGDOM", "Region"),
    "germany": ("GERMANY", "Region"),
    "canada": ("CANADA", "Region"),
    "asia": ("ASIA_PACIFIC", "Region"),
    "asia_pacific": ("ASIA_PACIFIC", "Region"),
    "north_america": ("NORTH_AMERICA", "Region"),
    "global": ("GLOBAL", "Region"),
    "international": ("GLOBAL", "Region"),

    # ── Regulations ──
    "us_export_controls": ("US_CHIP_EXPORT_CONTROLS_2022", "Regulation"),
    "october_2022_export_controls": ("US_CHIP_EXPORT_CONTROLS_2022", "Regulation"),
    "october_2023_export_controls": ("US_CHIP_EXPORT_CONTROLS_2023", "Regulation"),
    "chips_act": ("US_CHIPS_ACT", "Regulation"),
    "chips_and_science_act": ("US_CHIPS_ACT", "Regulation"),
    "eu_ai_act": ("EU_AI_ACT", "Regulation"),
    "bis_entity_list": ("BIS_ENTITY_LIST", "Regulation"),
    "entity_list": ("BIS_ENTITY_LIST", "Regulation"),

    # ── Business Strategies ──
    "supply_chain_diversification": ("SUPPLY_CHAIN_DIVERSIFICATION", "Strategy"),
    "supplier_diversification": ("SUPPLY_CHAIN_DIVERSIFICATION", "Strategy"),
    "multi_sourcing": ("SUPPLY_CHAIN_DIVERSIFICATION", "Strategy"),
    "cost_optimization": ("COST_OPTIMIZATION", "Strategy"),
    "cost_reduction": ("COST_OPTIMIZATION", "Strategy"),
    "r_and_d_investment": ("R_AND_D_INVESTMENT", "Strategy"),
    "rd_investment": ("R_AND_D_INVESTMENT", "Strategy"),
    "technology_investment": ("TECHNOLOGY_INVESTMENT", "Strategy"),
    "innovation_investment": ("TECHNOLOGY_INVESTMENT", "Strategy"),
    "strategic_acquisition": ("STRATEGIC_ACQUISITION", "Strategy"),
    "acquisitions": ("STRATEGIC_ACQUISITION", "Strategy"),
    "m_and_a": ("STRATEGIC_ACQUISITION", "Strategy"),
    "cybersecurity_measures": ("CYBERSECURITY_MEASURES", "Strategy"),
    "market_expansion": ("MARKET_EXPANSION", "Strategy"),
    "geographic_expansion": ("MARKET_EXPANSION", "Strategy"),
    "product_diversification": ("PRODUCT_DIVERSIFICATION", "Strategy"),
    "diversified_revenue": ("PRODUCT_DIVERSIFICATION", "Strategy"),
    "revenue_diversification": ("PRODUCT_DIVERSIFICATION", "Strategy"),
    "talent_acquisition": ("TALENT_ACQUISITION", "Strategy"),
    "workforce_expansion": ("TALENT_ACQUISITION", "Strategy"),
    "customer_focus": ("CUSTOMER_FOCUS_STRATEGY", "Strategy"),
    "supply_chain_resilience": ("SUPPLY_CHAIN_RESILIENCE", "Strategy"),
    "regulatory_compliance_strategy": ("REGULATORY_COMPLIANCE", "Strategy"),

    # ── Events ──
    "covid_19": ("COVID_19", "Event"),
    "covid": ("COVID_19", "Event"),
    "pandemic": ("COVID_19", "Event"),
    "chatgpt": ("CHATGPT_LAUNCH", "Event"),
    "generative_ai_boom": ("CHATGPT_LAUNCH", "Event"),
    "arm_acquisition": ("NVIDIA_ARM_TERMINATED", "Event"),
    "arm_acquisition_terminated": ("NVIDIA_ARM_TERMINATED", "Event"),
    "blackwell_announcement": ("BLACKWELL_ANNOUNCED", "Event"),
    "natural_disaster": ("NATURAL_DISASTER", "Event"),
    "earthquake": ("NATURAL_DISASTER", "Event"),
}


# =============================================================================
# 2. RISK CATEGORY TAXONOMY (24 categories)
# =============================================================================

RISK_CATEGORY_MAP: Dict[str, str] = {
    # Supply-side risks
    "supply_chain": "SUPPLY_CHAIN",
    "supply_chain_disruption": "SUPPLY_CHAIN",
    "cowos_packaging_constraint": "SUPPLY_CHAIN",
    "hbm_supply_constraint": "SUPPLY_CHAIN",
    "foundry_capacity": "SUPPLY_CHAIN",
    # Geopolitical
    "geopolitical_risk": "GEOPOLITICAL",
    "chip_export_restriction": "GEOPOLITICAL",
    "sovereign_ai_policy_risk": "GEOPOLITICAL",
    # Regulatory
    "government_regulation_risk": "REGULATORY",
    "ai_regulation_risk": "REGULATORY",
    "antitrust_risk": "REGULATORY",
    "regulatory_approval_risk": "REGULATORY",
    "environmental_regulation_risk": "REGULATORY",
    # Market
    "competition_risk": "MARKET",
    "demand_volatility": "MARKET",
    "customer_concentration_risk": "MARKET",
    "inventory_risk": "MARKET",
    "product_transition_risk": "MARKET",
    # Technology
    "technology_obsolescence": "TECHNOLOGY",
    "intellectual_property_risk": "TECHNOLOGY",
    "system_security_breach": "TECHNOLOGY",
    # Macroeconomic
    "macroeconomic_conditions": "MACROECONOMIC",
    "interest_rate_risk": "MACROECONOMIC",
    "foreign_exchange_risk": "MACROECONOMIC",
    # Human Capital
    "talent_shortage": "TALENT",
    # Legal
    "litigation_risk": "LEGAL",
    "product_liability_risk": "LEGAL",
    "tax_risk": "LEGAL",
    # Financial
    "credit_risk": "FINANCIAL",
    "liquidity_risk": "FINANCIAL",
    "reputation_risk": "REPUTATION",
    "crypto_market_volatility": "CRYPTO",
    "acquisition_integration_risk": "ACQUISITION_INTEGRATION",
    "energy_consumption_risk": "ENVIRONMENTAL",
}


# =============================================================================
# 3. STRATEGY TYPE TAXONOMY (12 types)
# =============================================================================

STRATEGY_TYPE_MAP: Dict[str, str] = {
    "supply_chain_diversification": "SUPPLY_CHAIN_RESILIENCE",
    "cost_optimization": "COST_OPTIMIZATION",
    "r_and_d_investment": "R_AND_D_INVESTMENT",
    "technology_investment": "TECHNOLOGY_INVESTMENT",
    "strategic_acquisition": "M_AND_A",
    "cybersecurity_measures": "CYBERSECURITY_MEASURES",
    "market_expansion": "MARKET_EXPANSION",
    "product_diversification": "DIVERSIFICATION",
    "talent_acquisition": "TALENT_MANAGEMENT",
    "customer_focus_strategy": "CUSTOMER_FOCUS",
    "supply_chain_resilience": "SUPPLY_CHAIN_RESILIENCE",
    "regulatory_compliance": "REGULATORY_COMPLIANCE",
}


# =============================================================================
# 4. CAUSAL MECHANISM PATTERNS (30+ patterns)
# =============================================================================

MECHANISM_PATTERNS: Dict[str, Tuple[str, str]] = {
    "extreme weather": ("EXTREME_WEATHER", "Mechanism"),
    "factory shutdown": ("FACTORY_SHUTDOWN", "Mechanism"),
    "production halt": ("PRODUCTION_HALT", "Mechanism"),
    "logistics delay": ("LOGISTICS_DELAY", "Mechanism"),
    "port congestion": ("PORT_CONGESTION", "Mechanism"),
    "inventory shortage": ("INVENTORY_SHORTAGE", "Mechanism"),
    "component shortage": ("COMPONENT_SHORTAGE", "Mechanism"),
    "price increase": ("PRICE_INCREASE", "Mechanism"),
    "price decline": ("PRICE_DECLINE", "Mechanism"),
    "order cancellation": ("ORDER_CANCELLATION", "Mechanism"),
    "order delay": ("ORDER_DELAY", "Mechanism"),
    "customer attrition": ("CUSTOMER_ATTRITION", "Mechanism"),
    "market share loss": ("MARKET_SHARE_LOSS", "Mechanism"),
    "credit tightening": ("CREDIT_TIGHTENING", "Mechanism"),
    "liquidity squeeze": ("LIQUIDITY_SQUEEZE", "Mechanism"),
    "workforce reduction": ("WORKFORCE_REDUCTION", "Mechanism"),
    "layoff": ("WORKFORCE_REDUCTION", "Mechanism"),
    "restructuring charge": ("RESTRUCTURING_CHARGE", "Mechanism"),
    "impairment charge": ("IMPAIRMENT_CHARGE", "Mechanism"),
    "patent expiration": ("PATENT_EXPIRATION", "Mechanism"),
    "regulatory fine": ("REGULATORY_FINE", "Mechanism"),
    "litigation settlement": ("LITIGATION_SETTLEMENT", "Mechanism"),
    "currency devaluation": ("CURRENCY_DEVALUATION", "Mechanism"),
    "interest rate hike": ("INTEREST_RATE_HIKE", "Mechanism"),
    "tariff imposition": ("TARIFF_IMPOSITION", "Mechanism"),
    "contract loss": ("CONTRACT_LOSS", "Mechanism"),
    "supplier bankruptcy": ("SUPPLIER_BANKRUPTCY", "Mechanism"),
    "technology disruption": ("TECHNOLOGY_DISRUPTION", "Mechanism"),
    "supply shortage": ("COMPONENT_SHORTAGE", "Mechanism"),
    "capacity constraint": ("CAPACITY_CONSTRAINT", "Mechanism"),
    "yield rate decline": ("YIELD_RATE_DECLINE", "Mechanism"),
    "rma request spike": ("RMA_REQUEST_SPIKE", "Mechanism"),
}


# =============================================================================
# 5. FILTER SETS (blacklists for noise removal)
# =============================================================================

GENERIC_NAMES: Set[str] = {
    "employees", "employee", "customers", "customer",
    "suppliers", "supplier", "stakeholders", "stakeholder",
    "partners", "partner", "people", "team", "workforce",
    "staff", "management", "board", "investors", "investor",
    "shareholders", "shareholder", "clients", "client",
    "competitors", "competitor", "vendors", "vendor",
    "users", "user", "members", "member",
}

BANNED_WORDS: Set[str] = {
    "risk_factor", "risk_factors", "esg_topic", "regulation", "event",
    "business_strategy", "financial_metric", "category", "entity",
    "uncertainty", "uncertainties", "table_of_contents",
    "risk", "risks", "region", "item", "business",
}

COMPANY_BLACKLIST_PREFIXES: Set[str] = {
    "employee", "staff", "workforce", "talent", "people",
    "mentor", "train", "learn", "career", "tuition",
    "pulse", "survey", "suggestion", "referral",
    "community", "military", "health",
}

NON_RISK_NAMES: Set[str] = {
    "pulse surveys", "pulse_surveys", "suggestion box",
    "tuition reimbursement programs", "career coaching",
    "mentoring programs", "training programs",
    "learning experiences", "employee referrals",
    "mental health challenges", "well-being challenges",
    "physical health challenges", "stress", "community needs",
}

COMPETITOR_NAMES: Set[str] = {
    "amd", "intel", "samsung", "tesla", "apple", "qualcomm", "broadcom",
    "micron", "tsmc", "microsoft", "google", "alphabet", "amazon",
    "ibm", "oracle", "cisco", "huawei", "mediatek", "nokia", "ericsson",
    "advanced_micro_devices", "advanced micro devices",
}


# =============================================================================
# 6. UTILITY FUNCTIONS
# =============================================================================

def norm_id(s: str) -> str:
    """Normalize any string to a canonical ID: lowercase, alphanumeric + underscore."""
    if not s:
        return ""
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:100]


def resolve_entity(raw_name: str, category_hint: str = "") -> Tuple[str, str]:
    """
    Resolve a raw entity name to canonical (display_name, node_label).
    Falls back to creating a canonical ID from the raw name.

    Args:
        raw_name: Raw entity name from LLM or text extraction
        category_hint: Entity category hint (e.g., "RISK_FACTOR", "COMPANY")

    Returns:
        (canonical_display_name, neo4j_node_label)
    """
    nid = norm_id(raw_name)
    if nid in CANONICAL_MAP:
        return CANONICAL_MAP[nid]
    # Fallback: create a clean canonical ID
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", raw_name).strip()
    clean = re.sub(r"\s+", " ", clean)[:40].upper().replace(" ", "_")
    label = category_hint if category_hint else "RiskFactor"
    return (clean, label)


def is_banned(name: str) -> bool:
    return norm_id(name) in BANNED_WORDS


def is_generic(name: str) -> bool:
    return norm_id(name) in {norm_id(g) for g in GENERIC_NAMES}


def is_competitor(name: str) -> bool:
    return norm_id(name) in COMPETITOR_NAMES


def is_non_risk(name: str) -> bool:
    return norm_id(name) in {norm_id(n) for n in NON_RISK_NAMES}


def is_company_blacklisted(name: str) -> bool:
    nid = norm_id(name)
    return any(nid.startswith(p) for p in COMPANY_BLACKLIST_PREFIXES)


def get_risk_category(risk_id: str) -> str:
    """Get the risk category for a given risk factor ID."""
    nid = norm_id(risk_id)
    return RISK_CATEGORY_MAP.get(nid, "OPERATIONAL")


def get_strategy_type(strategy_id: str) -> str:
    """Get the strategy type for a given strategy ID."""
    nid = norm_id(strategy_id)
    return STRATEGY_TYPE_MAP.get(nid, "TECHNOLOGY_INVESTMENT")


def extract_mechanisms(text: str) -> List[Tuple[str, str]]:
    """Extract causal mechanisms from text."""
    tl = text.lower()
    found: List[Tuple[str, str]] = []
    for pattern, (mech_name, mech_cat) in MECHANISM_PATTERNS.items():
        if pattern in tl:
            found.append((mech_name, mech_cat))
    return found
