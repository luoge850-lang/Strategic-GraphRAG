"""
Module: step7_report_generator.py
Project: Strategic GraphRAG Analysis for SEC Filings (NUS Research)
-----------------------------------------------------------------------
Description:
    This module serves as the final statistical aggregation layer. It 
    parses the multi-dimensional scores generated during the ablation 
    study and computes the mean performance metrics (Faithfulness, 
    Relevance, and Precision) for both the Baseline and Proposed systems.

Key Features:
    - Automated Metric Aggregation (n-case analysis).
    - Statistical Mean Computation.
    - Standardized Academic Console Output.
    - (Optional) LaTeX Table Code Generation for direct paper inclusion.

Author: Louis Harrington
Date: 2026-03-24 (Optimized Version)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any

# ==========================================
# 1. Path Management & Configuration
# ==========================================
# Resolving absolute paths to ensure data integrity across environments
CURRENT_DIR: Path = Path(__file__).resolve().parent
FILE_PATH: Path = CURRENT_DIR.parent / "data" / "evaluation" / "ablation_results_final.json"

# Professional Academic Logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ReportGenerator")

def generate_academic_report() -> None:
    """
    Parses the finalized ablation results and generates a comprehensive 
    performance report across all strategic test cases.
    """
    if not FILE_PATH.exists():
        logger.error(f"IO Error: Results file not found at {FILE_PATH}. Run Step 6 first.")
        return

    # Load experimental artifacts
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        results: List[Dict[str, Any]] = json.load(f)

    # Initialize metric containers
    # f: Faithfulness, r: Answer Relevance, p: Context Precision
    metrics: Dict[str, Dict[str, List[float]]] = {
        "baseline": {"f": [], "r": [], "p": []},
        "graph": {"f": [], "r": [], "p": []}
    }

    # Aggregate scores from all strategic test cases
    for res in results:
        b_scores = res["baseline_vector_rag"]["scores"]
        g_scores = res["proposed_graph_rag"]["scores"]
        
        # Populate Baseline metrics
        metrics["baseline"]["f"].append(b_scores.get("faithfulness_score", 0))
        metrics["baseline"]["r"].append(b_scores.get("relevance_score", 0))
        metrics["baseline"]["p"].append(b_scores.get("context_precision_score", 0))
        
        # Populate Strategic-GraphRAG metrics
        metrics["graph"]["f"].append(g_scores.get("faithfulness_score", 0))
        metrics["graph"]["r"].append(g_scores.get("relevance_score", 0))
        metrics["graph"]["p"].append(g_scores.get("context_precision_score", 0))

    # Average computation lambda (Standard mean calculation)
    calc_avg = lambda x: sum(x) / len(x) if x else 0.0
    
    # --- [Console Report Rendering] ---
    print("\n" + "█" * 65)
    print(f"📊 FINAL ACADEMIC PERFORMANCE SUMMARY (Sample Size: n={len(results)})")
    print("█" * 65)
    
    # Header
    header = f"{'Metric (Scale 1-5)':<25} | {'Vector Baseline':<18} | {'Strategic-GraphRAG':<18}"
    print(header)
    print("-" * 65)
    
    # Rows
    f_row = (f"{'Faithfulness (F)':<25} | "
             f"{calc_avg(metrics['baseline']['f']):^18.2f} | "
             f"{calc_avg(metrics['graph']['f']):^18.2f}")
    
    r_row = (f"{'Answer Relevance (R)':<25} | "
             f"{calc_avg(metrics['baseline']['r']):^18.2f} | "
             f"{calc_avg(metrics['graph']['r']):^18.2f}")
    
    p_row = (f"{'Context Precision (P)':<25} | "
             f"{calc_avg(metrics['baseline']['p']):^18.2f} | "
             f"{calc_avg(metrics['graph']['p']):^18.2f}")
    
    print(f_row)
    print(r_row)
    print(p_row)
    print("=" * 65)
    
    # --- [LaTeX Table Snippet Generation] ---
    # This snippet can be copied directly into your Overleaf document
    print("\n[System] LaTeX Table Code for Paper Submission:")
    latex_code = f"""
\\begin{{table}}[h]
\\centering
\\caption{{Quantitative Evaluation on SEC 10-K Strategic Queries (n={len(results)})}}
\\label{{tab:ablation_results}}
\\begin{{tabular}}{{@{{}}lccc@{{}}}}
\\toprule
\\textbf{{Architecture}} & \\textbf{{Faithfulness}} & \\textbf{{Relevance}} & \\textbf{{Precision}} \\\\ \\midrule
Baseline (Vector RAG) & {calc_avg(metrics['baseline']['f']):.2f} & {calc_avg(metrics['baseline']['r']):.2f} & {calc_avg(metrics['baseline']['p']):.2f} \\\\
\\textbf{{GraphRAG (Ours)}} & \\textbf{{{calc_avg(metrics['graph']['f']):.2f}}} & \\textbf{{{calc_avg(metrics['graph']['r']):.2f}}} & \\textbf{{{calc_avg(metrics['graph']['p']):.2f}}} \\\\ \\bottomrule
\\end{{tabular}}
\\end{{table}}
    """
    print(latex_code)
    print("=" * 65)

if __name__ == "__main__":
    generate_academic_report()