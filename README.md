## 🏗️ System Architecture & Methodology

This project implements a rigorous, symmetric **Ablation Study** comparing a baseline Vector RAG against the proposed GraphRAG. The pipeline is divided into three core subsystems, illustrated in the architecture flow below:

```mermaid
graph TD
    classDef database fill:#f9f2f4,stroke:#d32f2f,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef llm fill:#fff3e0,stroke:#f57c00,stroke-width:2px;
    
    Data[Raw SEC 10-K Filings] --> Split(Recursive Text Chunking):::process
    
    subgraph CG [Control Group: Vector Baseline]
        Split --> V_Embed[Dense Embedding]:::process
        V_Embed --> V_DB[(ChromaDB)]:::database
        Query[Strategic Query] --> V_Search[Cosine Similarity Search]:::process
        V_DB --> V_Search
        V_Search --> V_Gen[Llama-3.3-70B Synthesis]:::llm
    end
    
    subgraph EG [Experimental Group: Strategic-GraphRAG]
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
