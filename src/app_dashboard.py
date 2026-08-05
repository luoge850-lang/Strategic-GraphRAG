"""
Module: app_dashboard.py
Project: Strategic GraphRAG Analysis for SEC Filings (NUS Research)
-----------------------------------------------------------------------
Description:
    This module serves as the interactive frontend for the Strategic 
    GraphRAG system. It provides a real-time, bi-lingual (EN/ZH) 
    Semantic Resolution Console where users can query the AI and 
    visually explore the underlying reasoning paths.

Key Architectural Features:
    - Context-Aware Visual Grounding: Dynamically highlights topological 
      paths retrieved by the AI in high-alert red.
    - Hallucination Control Loop: Automatically clears visual highlights 
      if the LLM judges the retrieved evidence as insufficient ("Cannot conclude").
    - Agentic Keyword Extraction: Uses LLM to intelligently isolate entry 
      nodes for Cypher traversal.
    - Neural Reranking: Integrates Cross-Encoder to filter semantic noise 
      before rendering the local subgraph.

Author: Louis Harrington
Date: 2026-03-24 (Optimized Version)
"""

import os
import re
import math
import logging
from pathlib import Path
from typing import Tuple, Set, List

import certifi
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from groq import Groq
from neo4j import GraphDatabase, TrustCustomCAs
from pyvis.network import Network
from sentence_transformers import CrossEncoder

# ==========================================
# 1. System Initialization & Logging Setup
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger("DashboardUI")

ROOT_DIR = Path(__file__).resolve().parent.parent
HTML_OUTPUT_PATH = str(ROOT_DIR / "combat_graph.html")
ENV_PATH = ROOT_DIR / ".env"

# [防御性更新]：增强 .env 文件的寻址鲁棒性
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    load_dotenv()  # Fallback: 让系统在当前工作目录自动寻找 .env
    logger.warning(f"Root .env not found at {ENV_PATH}, attempting default load.")

os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

# ==========================================
# 2. UI Configuration & Text Dictionary
# ==========================================
st.set_page_config(page_title="Strategic Knowledge Graph", layout="wide", initial_sidebar_state="collapsed")

# Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "hl_nodes" not in st.session_state:
    st.session_state.hl_nodes = set()
if "hl_edges" not in st.session_state:
    st.session_state.hl_edges = set()
if "lang" not in st.session_state:
    st.session_state.lang = "EN"

UI_TEXT = {
    "ZH": {
        "nav_title": "NVIDIA 战略知识图谱分析系统",
        "nav_credit": "Designed & Developed by Louis Harrington",
        "chat_header": "语义解析控制台",
        "chat_placeholder": "输入战略解析指令...",
        "btn_lang": "Switch to English UI",
        "btn_reset": "重置拓扑视图",
        "stats": "系统就绪: {n} 核心实体 | {r} 战略引力线",
        "popup_title": "星云数据解析",
        "expander_label": "🔍 展开查看底部溯源网络数据",
        "system_prompt": (
            "You are a deterministic Graph Retrieval-Augmented Generation (GraphRAG) engine. "
            "Your sole directive is to objectively synthesize the retrieved topological triplets.\n\n"
            "[STRICT ACADEMIC PROTOCOLS]:\n"
            "1. NO ANTHROPOMORPHISM: Do not act as an analyst. Use passive voice and objective academic phrasing.\n"
            "2. EXPLICIT PROVENANCE: Every claim MUST be supported by a topological citation: (Provenance: [A] -[R]-> [B]).\n"
            "3. CAUSAL BOUNDARIES: Do not interpolate or hallucinate links between disconnected nodes.\n"
            "4. FALLBACK: If the subgraph is insufficient, output ONLY: 'Topological subgraph lacks sufficient causal density to evaluate this directive.'\n\n"
            "[Active Subgraph Data]:\n{context}"
        )
    },
    "EN": {
        "nav_title": "NVIDIA Strategic Knowledge Graph",
        "nav_credit": "Designed & Developed by Louis Harrington",
        "chat_header": "Semantic Resolution Console",
        "chat_placeholder": "Enter strategic directive...",
        "btn_lang": "切换至中文界面",
        "btn_reset": "Reset Topology",
        "stats": "System Ready: {n} Nodes | {r} Connections",
        "popup_title": "Nebula Data Trace",
        "expander_label": "View Retrieved Graph Evidence",
       "system_prompt": (
            "You are a deterministic Graph Retrieval-Augmented Generation (GraphRAG) engine. "
            "Your sole directive is to objectively synthesize the retrieved topological triplets.\n\n"
            "[STRICT ACADEMIC PROTOCOLS]:\n"
            "1. NO ANTHROPOMORPHISM: Do not act as an analyst. Use passive voice and objective academic phrasing.\n"
            "2. EXPLICIT PROVENANCE: Every claim MUST be supported by a topological citation: (Provenance: [A] -[R]-> [B]).\n"
            "3. CAUSAL BOUNDARIES: Do not interpolate or hallucinate links between disconnected nodes.\n"
            "4. FALLBACK: If the subgraph is insufficient, output ONLY: 'Topological subgraph lacks sufficient causal density to evaluate this directive.'\n\n"
            "[Active Subgraph Data]:\n{context}"
        )
    }
}

t = UI_TEXT[st.session_state.lang]

# Custom CSS for Cyber-Academic Aesthetics
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Rajdhani:wght@500;600&family=Noto+Sans+SC:wght@400;500;700&display=swap');
    .stApp, p, span, div, input {{ font-family: 'Rajdhani', 'Noto Sans SC', sans-serif !important; }}
    h1, h2, h3, .nav-brand {{ font-family: 'Orbitron', 'Noto Sans SC', sans-serif !important; }}
    .stApp {{ background-color: #030101 !important; color: #e0e0e0; overflow: hidden; }}
    .louis-navbar {{ position: fixed; top: 0; left: 0; width: 100%; height: 55px; background: rgba(10, 0, 0, 0.85); backdrop-filter: blur(20px); border-bottom: 1px solid rgba(150, 0, 0, 0.3); z-index: 999999; display: flex; align-items: center; justify-content: space-between; padding: 0 3rem; }}
    .nav-brand {{ font-size: 1.1rem; color: #ff1a1a; letter-spacing: 1px; font-weight: 700; }}
    .nav-credit {{ font-size: 0.85rem; color: #666; text-transform: uppercase; }}
    .block-container {{ padding-top: 5rem !important; padding-bottom: 0 !important; }}
    [data-testid="stChatMessage"] {{ background: rgba(15, 5, 5, 0.7) !important; border: 1px solid rgba(200, 0, 0, 0.1) !important; border-left: 3px solid #aa0000 !important; border-radius: 4px; }}
    .stats-bar, .stButton>button {{ height: 40px !important; background: #110505 !important; border: 1px solid #440000 !important; color: #ff4d4d !important; font-family: 'Orbitron', sans-serif !important; font-size: 0.8rem !important; }}
    .stButton>button:hover {{ border-color: #ff1a1a !important; color: white !important; }}
    footer, #MainMenu {{ visibility: hidden; }}
    .streamlit-expanderHeader svg {{ display: none !important; }}
    </style>
    <div class="louis-navbar"><div class="nav-brand">{t['nav_title']}</div><div class="nav-credit">{t['nav_credit']}</div></div>
""", unsafe_allow_html=True)

# ==========================================
# 3. Core Engine (Graph Operations & Viz)
# ==========================================
class StrategicDashboard:
    """Encapsulates backend retrieval and PyVis rendering logic for the UI."""
    
    def __init__(self) -> None:
        self.ai_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.db_driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI"),
            auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD")),
            encrypted=True, 
            trusted_certificates=TrustCustomCAs(certifi.where())
        )
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def get_stats(self) -> Tuple[int, int]:
        """Fetches global node and relationship counts for the status bar."""
        try:
            with self.db_driver.session() as session:
                res = session.run("MATCH (n) WITH count(n) as nodes MATCH ()-[r]->() RETURN nodes, count(r) as rels").single()
                if res:
                    return res["nodes"], res["rels"]
                return 0, 0
        except Exception as e: 
            # [防御性更新]：暴露出真实的连接错误，方便排查
            logger.error(f"Neo4j Subsystem Failure (Stats): {e}")
            return 0, 0

    def scout_highlight(self, prompt: str) -> Tuple[Set[str], Set[Tuple[str, str]], str]:
        """
        Executes Agentic Keyword Extraction, topological traversal, and neural reranking 
        to identify nodes/edges for UI highlighting.
        """
        # [Architecture Update]: LLM as an Agentic Keyword Extractor (Enhanced with Anti-Hub Node logic)
        # [Architecture Update]: LLM as an Agentic Keyword Extractor (Enhanced Anti-Hub & Quantitative Logic)
        try:
            kw_prompt = (
                f"Extract 2 to 4 HIGHLY SPECIFIC strategic entities from this query. "
                f"[STRICT RULES]: "
                f"1. NEVER extract generic hub words like 'RISKS', 'CHALLENGES', 'IMPACT', 'BUSINESS', or 'COMPANY'. "
                f"2. If the user asks about money, revenue, or percentages, explicitly extract terms like 'REVENUE' or the specific product/partner name. "
                f"Return ONLY a comma-separated list in uppercase (e.g., CLOUD SERVICE PROVIDERS, REVENUE, INTELLECTUAL PROPERTY). "
                f"No explanations, no markdown.\nQuery: {prompt}"
            )
            # ... 后续调用保持不变
            kw_res = self.ai_client.chat.completions.create(
                messages=[{"role": "user", "content": kw_prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.0
            ).choices[0].message.content
            words = [w.strip().upper() for w in kw_res.replace('"', '').split(',') if len(w.strip()) > 2]
        except Exception as e:
            logger.error(f"LLM Keyword Extraction failed: {e}")
            words = []
            
        # Fallback mechanism: Heuristic stopword filtering
        if not words:
            stopwords = {"NVIDIA", "NVIDIAS", "CORPORATION", "CORP", "COMPANY", "HOW", "DOES", "WHAT", "IS", "ARE", "THE", "A", "AN", "IN", "ON", "OF", "TO", "FOR", "AND", "OR", "WITH", "BY", "THIS", "THAT", "HELP", "IT", "ITS", "BEYOND", "SPECIFICALLY", "ABOUT", "CAN", "COULD", "WOULD", "SHOULD"}
            clean_prompt = re.sub(r"[^\w\s]", " ", prompt)
            words = [w.strip().upper() for w in clean_prompt.split() if len(w) > 2 and w.strip().upper() not in stopwords]
        
        if not words: return set(), set(), ""
            
        match_query = " OR ".join([f"toLower(n.id) CONTAINS toLower('{w}')" for w in words])
        cypher = f"""
            MATCH p=(n:Entity)-[*1..2]-(m:Entity) 
            WHERE ({match_query}) 
            UNWIND relationships(p) as r
            RETURN startNode(r).id as s, endNode(r).id as t, type(r) as rel, r.description as desc 
            LIMIT 150
        """
        
        raw_data = []
        with self.db_driver.session() as session:
            res = session.run(cypher)
            for r in res:
                s, t, rel, desc = r["s"], r["t"], r["rel"], r["desc"]
                triplet_str = f"[{s}] - {rel} -> [{t}] (Context: {desc})"
                raw_data.append({"s": s, "t": t, "text": triplet_str})
                
        # Deduplication
        unique_map = {item["text"]: item for item in raw_data}
        unique_triplets = list(unique_map.keys())
        
        if not unique_triplets: return set(), set(), ""
            
        # Neural Reranking for UI precision
        pairs = [[prompt, item] for item in unique_triplets]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, unique_triplets), reverse=True, key=lambda x: x[0])
        
        # Isolate Top 15 signals to prevent LLM confusion and UI clutter
        top_items = ranked[:15]
        
        hl_nodes = set()
        hl_edges = set()
        filtered_triplets = []
        
        for score, t_str in top_items:
            filtered_triplets.append(t_str)
            item_data = unique_map[t_str]
            hl_nodes.update([item_data["s"], item_data["t"]])
            hl_edges.add((item_data["s"], item_data["t"]))
        
        return hl_nodes, hl_edges, "\n".join(filtered_triplets)

    def render_graph(self, hl_nodes: Set[str], hl_edges: Set[Tuple[str, str]]) -> str:
        """Generates the interactive HTML representation of the knowledge graph."""
        net = Network(height="800px", width="100%", bgcolor="#030101", font_color="white", directed=True)
        net.barnes_hut(gravity=-3000, central_gravity=0.05, spring_length=150, damping=0.09, overlap=0)
        
        try:
            with self.db_driver.session() as session:
                records = list(session.run("MATCH (n:Entity)-[r]->(m:Entity) RETURN n.id as s, m.id as t, type(r) as rel, r.description as d, r.source as src, r.page as pg LIMIT 400"))
                
                degrees = {}
                for rec in records:
                    degrees[rec["s"]] = degrees.get(rec["s"], 0) + 1
                    degrees[rec["t"]] = degrees.get(rec["t"], 0) + 1

                has_hl = len(hl_nodes) > 0

                for rec in records:
                    s, t, rel, d, src, pg = rec["s"], rec["t"], rec["rel"], rec["d"], rec["src"], rec["pg"]
                    s_size = 28 if s in hl_nodes else min(45, 15 + math.sqrt(degrees[s]) * 2.5)
                    t_size = 28 if t in hl_nodes else min(45, 15 + math.sqrt(degrees[t]) * 2.5)
                    
                    # Dynamic coloring based on retrieved evidence state
                    s_color = "#ff1a1a" if s in hl_nodes else ("#222222" if has_hl else "#555555")
                    t_color = "#ff1a1a" if t in hl_nodes else ("#222222" if has_hl else "#555555")
                    e_color = "#ff1a1a" if (s, t) in hl_edges else ("rgba(255,255,255,0.02)" if has_hl else "rgba(200,200,200,0.15)")
                    
                    net.add_node(s, label=s, title=f"<b>{s}</b><hr>Source: {src}<br>Page: {pg}", color=s_color, size=s_size, font={"color": "#ffffff", "size": 14})
                    net.add_node(t, label=t, title=f"<b>{t}</b><hr>Source: {src}<br>Page: {pg}", color=t_color, size=t_size, font={"color": "#ffffff", "size": 14})
                    net.add_edge(s, t, label=" ", title=d, color=e_color, width=2.5 if (s, t) in hl_edges else 0.8)
        except Exception as e:
            logger.error(f"Neo4j Subsystem Failure (Render): {e}")
            
        net.save_graph(HTML_OUTPUT_PATH)
        
        # CSS Injection for seamless Streamlit integration
        with open(HTML_OUTPUT_PATH, "r", encoding="utf-8") as f:
            html = f.read().replace("<style type=\"text/css\">", "<style type=\"text/css\">\nbody { margin: 0; padding: 0; overflow: hidden; background-color: #030101; }\n#mynetwork { border: none !important; outline: none !important; }")
        with open(HTML_OUTPUT_PATH, "w", encoding="utf-8") as f: 
            f.write(html)
            
        return HTML_OUTPUT_PATH

# ==========================================
# 4. Main Application Layout & Execution
# ==========================================
engine = StrategicDashboard()
total_n, total_r = engine.get_stats()

col1, col2 = st.columns([0.28, 0.72])

# --- Chat Interface (Left Column) ---
with col1:
    st.markdown(f"### {t['chat_header']}")
    chat_container = st.container(height=680)
    
    # Render chat history
    for msg in st.session_state.messages:
        with chat_container.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "🧑‍💻"):
            st.markdown(msg["content"])
            if msg.get("context"):
                with st.expander(t["expander_label"]):
                    st.code(msg["context"])

    # User Input Processing
    if prompt := st.chat_input(t["chat_placeholder"]):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)
            
        # Execute Graph Retrieval
        hl_n, hl_e, graph_context = engine.scout_highlight(prompt)
        st.session_state.hl_nodes = hl_n
        st.session_state.hl_edges = hl_e
        
        # Construct rolling context window (Last 4 messages)
        api_messages = [{"role": "system", "content": t["system_prompt"].format(context=graph_context)}]
        for msg in st.session_state.messages[-4:]:
            api_messages.append({"role": msg["role"], "content": msg["content"]})
            
        # Execute LLM Generation
        with chat_container.chat_message("assistant", avatar="🤖"):
            try:
                res = engine.ai_client.chat.completions.create(
                    messages=api_messages, 
                    model="llama-3.3-70b-versatile",
                    temperature=0.0 
                ).choices[0].message.content
                
                # ==========================================
                # [终极物理拦截与防幻觉控制循环]
                # ==========================================
                fallback_en = "Insufficient strategic context in the current graph database to conclude."
                fallback_zh = "图谱数据库中暂无支撑该结论的战略上下文。"
                
                # 1. 检查是否触发了兜底指令
                if fallback_en in res or fallback_zh in res:
                    # 物理覆盖大模型生成的冗余模板内容，强制截断
                    res = f"### ⚠️Retrieval Aborted\n\n{fallback_en if fallback_en in res else fallback_zh}"
                    
                    # 清理前端图谱高亮与底部的 Context 展示
                    st.session_state.hl_nodes.clear()
                    st.session_state.hl_edges.clear()
                    graph_context = "" 
                    
                # 2. 兼容旧版的控制词
                elif "Cannot conclude" in res or "cannot conclude" in res.lower() or "无法得出结论" in res:
                    st.session_state.hl_nodes.clear()
                    st.session_state.hl_edges.clear()
                    graph_context = "" 
                
                st.markdown(res)
                
                if graph_context:
                    with st.expander(t["expander_label"]):
                        st.code(graph_context)
                
                st.session_state.messages.append({"role": "assistant", "content": res, "context": graph_context})
                st.rerun()
            except Exception as e:
                st.error(f"**[System Error]** Inference Engine Failure: {e}")

# --- Graph Visualization (Right Column) ---
with col2:
    h1, h2, h3 = st.columns([0.45, 0.25, 0.3])
    h1.markdown(f"<div class='stats-bar' style='display:flex; align-items:center; padding-left:15px;'>{t['stats'].format(n=total_n, r=total_r)}</div>", unsafe_allow_html=True)
    
    if h2.button(t["btn_lang"]):
        st.session_state.lang = "EN" if st.session_state.lang == "ZH" else "ZH"
        st.rerun()
    if h3.button(t["btn_reset"]):
        st.session_state.hl_nodes.clear()
        st.session_state.hl_edges.clear()
        st.session_state.messages.clear() 
        st.rerun()

    # Render and inject PyVis Network with custom JavaScript Tooltips
    html_path = engine.render_graph(st.session_state.hl_nodes, st.session_state.hl_edges)
    with open(html_path, "r", encoding="utf-8") as f:
        html_data = f.read()
        popup_js = f"""
        <style>#tracePopup {{ display: none; position: absolute; top: 20px; right: 20px; width: 300px; background: rgba(8, 2, 2, 0.85); backdrop-filter: blur(20px); border: 1px solid rgba(255, 26, 26, 0.4); border-top: 4px solid #ff1a1a; padding: 18px; color: #eee; z-index: 1000; font-size: 13px; border-radius: 8px; box-shadow: 0 15px 35px rgba(0,0,0,0.9); }} #tracePopup h4 {{ color: #ff1a1a; margin: 0 0 12px 0; font-family: 'Orbitron'; font-size: 16px; text-transform: uppercase; }}</style>
        <div id="tracePopup"><h4 id="traceTitle">{t['popup_title']}</h4><div id="traceContent"></div></div>
        <script>setTimeout(() => {{ network.on("click", function (params) {{ let popup = document.getElementById('tracePopup'); if (params.nodes.length > 0) {{ let nodeId = params.nodes[0]; let node = nodes.get(nodeId); document.getElementById('traceContent').innerHTML = node.title; popup.style.display = 'block'; }} else {{ popup.style.display = 'none'; }} }}); }}, 1500);</script>
        """
        components.html(html_data.replace("</body>", f"{popup_js}</body>"), height=820, scrolling=False)