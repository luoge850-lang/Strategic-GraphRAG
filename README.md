# 🌌 Strategic-GraphRAG: Neuro-Symbolic Reasoning for Financial Intelligence

**Author**: Louis Harrington

**Target**: Application to National University of Singapore (AI)

**Domain**: Quantitative Finance, Natural Language Processing, Knowledge Graphs  

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Neo4j](https://img.shields.io/badge/Neo4j-AuraDB-blue)
![LLM](https://img.shields.io/badge/LLM-Llama--3.3--70B-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

<img width="2547" height="1260" alt="ui_screenshot" src="https://github.com/user-attachments/assets/aca5f282-ac13-48a9-9d6e-94312ffefe16" />
👉 Interactive dashboard:

* Ask a question
* See the reasoning answer 
* Highlighted nodes show how the answer is constructed

---
## ❗ Problem

Traditional Vector-based RAG fails on financial documents because:

* Relies on semantic similarity only
* Cannot connect long-range dependencies
* Breaks down on multi-hop reasoning

➡️ Result:
Either irrelevant answers or
“Cannot answer from context” (no reasoning)

---

## 🚀 What does this project do?

This system allows users to ask complex strategic questions about companies (e.g., “How do geopolitical risks affect NVIDIA’s supply chain?”) and visualizes the reasoning process through a knowledge graph.

👉 Unlike traditional RAG systems, it does not just retrieve similar text —
it reconstructs multi-hop causal relationships across hundreds of pages of financial reports.

---

## 📑 Executive Summary

This project demonstrates how Graph + LLM systems enable structured reasoning beyond traditional retrieval, leading to more reliable and interpretable AI.

Traditional vector-based Retrieval-Augmented Generation (RAG) struggles with long-range, multi-hop reasoning in highly unstructured financial documents such as SEC 10-K filings. Because it relies purely on semantic similarity, it often fails to connect causally related information scattered across dozens of pages, resulting in irrelevant answers or incomplete reasoning.

To address this limitation, Strategic-GraphRAG introduces a neuro-symbolic architecture for financial intelligence. The system transforms raw corporate filings into a topological knowledge graph, and replaces similarity-based retrieval with graph-based multi-hop reasoning (shortest-path traversal), further refined by neural reranking.

As a result, the model can trace and reconstruct complex causal chains, significantly improving answer relevance while reducing hallucination—for example, linking geopolitical events directly to downstream supply chain impacts.

---

## 🏗️ System Architecture & Methodology

This system is designed to evaluate whether graph-based reasoning can outperform traditional vector retrieval in complex financial analysis.

To ensure fairness, we implement a rigorous and symmetric ablation study, comparing a baseline Vector RAG with the proposed GraphRAG under identical conditions.

The pipeline consists of three core subsystems:

### 1. Dual-Track Data Ingestion
* **Vector Space (Baseline)**: Raw PDF filings are processed through extraction, heuristic cleaning, and recursive chunking, then embedded using `all-MiniLM-L6-v2` and stored in ChromaDB. 
* **Graph Space (Ours)**: Documents are transformed into a structured knowledge graph via LLM-based triplet extraction (Llama-3.1-8B).
Domain-specific noise filtering is applied to remove administrative artifacts, and entities/relations are stored in Neo4j using idempotent Cypher MERGE operations.

### 2. Neuro-Symbolic Retrieval Engine 
Instead of relying solely on semantic similarity, our method performs explicit multi-hop reasoning over graph structures, followed by neural refinement:

* **Entity Extraction**: Key strategic entities are dynamically identified from user queries.
* **Topological Reasoning**: Uses Cypher `shortestPath([*1..3])` to resolve multi-hop dependencies.
* **Neural Reranking**: A Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) filters irrelevant candidates and improves the signal-to-noise ratio (SNR) before final answer generation.

### 3. Autonomous Evaluation (LLM-as-a-Judge)
To ensure objective evaluation, we design an automated assessment pipeline:

* An evaluator model (Llama-3.3-70B, temperature=0.0) scores outputs based on Question–Context–Answer (Q-C-A) traces
* Evaluation follows a modified RAGAS-style framework
* The system is tested on 40 complex financial reasoning queries
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

1. Create virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate environment:
   * On Windows:
   ```bash
   venv\Scripts\activate
   ```
   * On macOS/Linux:
   ```bash
   source venv/bin/activate
   ```
3. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Configure your environment variables in a `.env` file (refer to `.env.example`).
   
> *\*Note:
>* Using a virtual environment helps avoid dependency conflicts 
>* All required packages are listed in `requirements.txt`
>* Tested with Python 3.x environment

### Step-by-Step Execution

**Phase 1: Data Infrastructure**
```bash
python src/step1_build_vector_baseline.py  # Builds ChromaDB
python src/step2_graph_ingestion.py        # Ingests Triplets to Neo4j
```

**Phase 2: Automated Ablation Study**

> *\*Note: step3 (Vector Engine), step4 (Graph Engine), and step5 (AI Judge) are decoupled, object-oriented modules. They are automatically managed by the Step 6 Orchestrator.
```bash
python src/step6_batch_experiment_runner.py # Runs the 40-question benchmark
python src/step7_report_generator.py        # Outputs metrics & LaTeX table
```

**Phase 3: Interactive Dashboard**
```bash
streamlit run src/app_dashboard.py
```
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
```
##  📌 Future Work

* Temporal GraphRAG (time-evolving knowledge graphs)
* Multi-company comparative analysis
* Improved reasoning path scoring

---

Designed & Developed with a focus on academic rigor, idempotency, and resilience engineering.

