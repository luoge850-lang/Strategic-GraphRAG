# 🌌 Strategic-GraphRAG: Neuro-Symbolic Reasoning for Financial Intelligence

**Author**: Louis Harrington  
**Institution/Target**: National University of Singapore (NUS) Application Project  
**Domain**: Quantitative Finance, Natural Language Processing, Knowledge Graphs  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-blue)
![LLM](https://img.shields.io/badge/LLM-Llama--3.3--70B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

![System Dashboard UI](data/ui_screenshot.png)

## 📑 Executive Summary

Traditional Vector-based Retrieval-Augmented Generation (RAG) models suffer from **"Relevance Collapse"** when tasked with long-range, multi-hop causal reasoning in highly unstructured financial documents (e.g., SEC 10-K filings). They rely on semantic similarity, which fails to connect disparate logical dots across dozens of pages.

**Strategic-GraphRAG** introduces a Neuro-Symbolic architecture designed specifically for high-stakes financial reasoning. By converting raw corporate filings into a **Topological Knowledge Graph** and leveraging `shortestPath` graph traversal combined with neural reranking (Cross-Encoder), this system successfully mitigates hallucination and bridges causal gaps (e.g., linking geopolitical sanctions directly to specific supply chain bottlenecks).

---

## 🏗️ System Architecture & Methodology

This project implements a rigorous, symmetric **Ablation Study** comparing a baseline Vector RAG against the proposed GraphRAG. The pipeline is divided into three core subsystems:

### 1. Dual-Track Data Ingestion
* **Control Group (Vector Space)**: PDF extraction, heuristic cleaning, and recursive chunking embedded into **ChromaDB** using `all-MiniLM-L6-v2`.
* **Experimental Group (Topological Space)**: Agentic triplet extraction (Llama-3.1-8B) with SEC-specific administrative noise filters, mapped idempotently into **Neo4j** via Cypher `MERGE` operations.

### 2. Neuro-Symbolic Retrieval Engine
* **Agentic Keyword Extraction**: Dynamically isolates core strategic entities from user queries.
* **Topological Traversal**: Uses Cypher `shortestPath([*1..3])` to resolve multi-hop dependencies.
* **Neural Reranking**: Applies `ms-marco-MiniLM-L-6-v2` (Cross-Encoder) to prune semantic noise and optimize the Signal-to-Noise Ratio (SNR) before LLM synthesis.

### 3. Autonomous Evaluation (LLM-as-a-Judge)
An automated evaluator (Llama-3.3-70B, Temperature=0.0) audits the Q-C-A (Question-Context-Answer) traces based on a modified RAGAS framework across 40 golden strategic queries.

---

## 📊 Quantitative Results (Ablation Study)

The empirical results over 40 complex financial reasoning queries (e.g., *Supply Chain, Geopolitics, Hardware Architecture*) demonstrate a massive paradigm shift in retrieval accuracy and reasoning depth.

| Architecture | Faithfulness (F) | Answer Relevance (R) | Context Precision (P) |
| :--- | :---: | :---: | :---: |
| Baseline (Vector RAG) | 5.00* | 1.00 | 1.00 |
| **Strategic-GraphRAG (Ours)** | **4.05** | **4.53** | **3.15** |

> *\*Note: The Baseline Vector RAG scored artificially high on Faithfulness because it consistently output "Cannot conclude based on the given documents," effectively preventing hallucinations but failing entirely at the reasoning task.*

---

## 💻 Tech Stack
* **Language Models**: Groq Cloud (Llama-3.3-70B for Synthesis/Judging, Llama-3.1-8B for Ingestion)
* **Graph Database**: Neo4j Aura Cloud & Cypher Query Language
* **Vector Database**: ChromaDB
* **Machine Learning**: Sentence-Transformers (Cross-Encoder)
* **Visual Analytics**: Streamlit, PyVis Network

---

## 🚀 Quick Start / Reproduction Guide

### Prerequisites
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
Configure your environment variables in a .env file (refer to .env.example).

Step-by-Step Execution
Phase 1: Data Infrastructure

Bash
python src/step1_build_vector_baseline.py  # Builds ChromaDB
python src/step2_graph_ingestion.py        # Ingests Triplets to Neo4j
Phase 2: Automated Ablation Study

Bash
python src/step6_batch_experiment_runner.py # Runs the 40-question benchmark
python src/step7_report_generator.py        # Outputs metrics & LaTeX table
Phase 3: Interactive Dashboard

Bash
streamlit run src/app_dashboard.py

---

##  📂 Repository Structure

```text
├── data/
│   ├── evaluation/          # Golden dataset and generated JSON results
│   └── pdfs/                # Raw SEC 10-K Filings
├── src/
│   ├── app_dashboard.py                 # Streamlit visual grounding console
│   ├── step1_build_vector_baseline.py   # Baseline vector ingestion
│   ├── step2_graph_ingestion.py         # Topological graph ingestion
│   ├── step3_vector_rag_engine.py       # Baseline retrieval logic
│   ├── step4_graphrag_query_engine.py   # Proposed shortest-path logic
│   ├── step5_academic_evaluator.py      # LLM-as-a-Judge module
│   ├── step6_batch_experiment_runner.py # Orchestrator script
│   └── step7_report_generator.py        # Statistical aggregation
├── .env.example
├── .gitignore
└── requirements.txt
Designed & Developed with a focus on academic rigor, idempotency, and resilience engineering.
