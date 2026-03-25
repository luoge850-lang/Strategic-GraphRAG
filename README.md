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

This project implements a rigorous, symmetric **Ablation Study** comparing a baseline Vector RAG against the proposed GraphRAG. The pipeline is divided into three core subsystems, illustrated in the architecture flow below:

```mermaid
graph TD
    %% Define Node Styles
    classDef database fill:#f9f2f4,stroke:#d32f2f,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef llm fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    
    Data[Raw SEC 10-K Filings] --> Split(Recursive Text Chunking):::process
    
    subgraph Control Group: Vector Baseline
        Split --> V_Embed[Dense Embedding]:::process
        V_Embed --> V_DB[(ChromaDB)]:::database
        Query[Strategic Query] --> V_Search[Cosine Similarity Search]:::process
        V_DB --> V_Search
        V_Search --> V_Gen[Llama-3.3-70B Synthesis]:::llm
    end
    
    subgraph Experimental Group: Strategic-GraphRAG
        Split --> G_Extract[Llama-3.1-8B Triplet Extraction]:::llm
        G_Extract --> G_DB[(Neo4j Aura)]:::database
        Query --> G_Agent[Agentic Keyword Extraction]:::llm
        G_Agent --> G_Traverse[Cypher shortestPath Traversal]:::process
        G_DB --> G_Traverse
        G_Traverse --> G_Rerank[Cross-Encoder Reranking]:::process
        G_Rerank --> G_Gen[Llama-3.3-70B Grounded Synthesis]:::llm
    end
    
    V_Gen --> Eval[Academic Evaluator: LLM-as-a-Judge]:::llm
    G_Gen --> Eval
1. Dual-Track Data IngestionControl Group (Vector Space): PDF extraction, heuristic cleaning, and recursive chunking embedded into ChromaDB using all-MiniLM-L6-v2.Experimental Group (Topological Space): Agentic triplet extraction (Llama-3.1-8B) with SEC-specific administrative noise filters, mapped idempotently into Neo4j via Cypher MERGE operations.2. Neuro-Symbolic Retrieval EngineAgentic Keyword Extraction: Dynamically isolates core strategic entities from user queries.Topological Traversal: Uses Cypher shortestPath([*1..3]) to resolve multi-hop dependencies.Neural Reranking: Applies ms-marco-MiniLM-L-6-v2 (Cross-Encoder) to prune semantic noise and optimize the Signal-to-Noise Ratio (SNR) before LLM synthesis.3. Autonomous Evaluation (LLM-as-a-Judge)An automated evaluator (Llama-3.3-70B, Temperature=0.0) audits the Q-C-A (Question-Context-Answer) traces based on a modified RAGAS framework across 40 golden strategic queries.📊 Quantitative Results (Ablation Study)The empirical results over 40 complex financial reasoning queries (e.g., Supply Chain, Geopolitics, Hardware Architecture) demonstrate a massive paradigm shift in retrieval accuracy and reasoning depth.ArchitectureFaithfulness (F)Answer Relevance (R)Context Precision (P)Baseline (Vector RAG)5.00*1.001.00Strategic-GraphRAG (Ours)4.054.533.15*Note: The Baseline Vector RAG scored artificially high on Faithfulness because it consistently output "Cannot conclude based on the given documents," effectively preventing hallucinations but failing entirely at the reasoning task.💻 Tech StackLanguage Models: Groq Cloud (Llama-3.3-70B for Synthesis/Judging, Llama-3.1-8B for Ingestion)Graph Database: Neo4j Aura Cloud & Cypher Query LanguageVector Database: ChromaDBMachine Learning: Sentence-Transformers (Cross-Encoder)Visual Analytics: Streamlit, PyVis Network🚀 Quick Start / Reproduction GuidePrerequisitesClone the repository and install dependencies:Bashpip install -r requirements.txt
Configure your environment variables in a .env file (refer to .env.example).Step-by-Step ExecutionPhase 1: Data InfrastructureBashpython src/step1_build_vector_baseline.py  # Builds ChromaDB
python src/step2_graph_ingestion.py        # Ingests Triplets to Neo4j
Phase 2: Automated Ablation StudyBashpython src/step6_batch_experiment_runner.py # Runs the 40-question benchmark
python src/step7_report_generator.py        # Outputs metrics & LaTeX table
Phase 3: Interactive DashboardBashstreamlit run src/app_dashboard.py
📂 Repository StructurePlaintext├── data/
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
