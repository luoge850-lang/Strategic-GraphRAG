# Strategic-GraphRAG：项目理解、技术选型建议与发展时间线

> 面向本科毕业设计 + AI 硕士申请的研究型 GraphRAG 系统策略文档
> 2026-07-21

---

## 目录

1. [项目定位与学术价值](#1-项目定位与学术价值)
2. [LLM API 选型分析](#2-llm-api-选型分析)
3. [当前状态评估](#3-当前状态评估)
4. [发展路线图](#4-发展路线图)
5. [实验设计与评估框架](#5-实验设计与评估框架)
6. [论文投稿策略](#6-论文投稿策略)
7. [风险与应对](#7-风险与应对)

---

## 1. 项目定位与学术价值

### 1.1 核心创新点

你的项目不是"又一个 LLM + 知识图谱的问答系统"。它的学术区分度在于：

| 维度 | 传统 Vector RAG | 普通 GraphRAG | **你的 Strategic-GraphRAG** |
|------|----------------|---------------|---------------------------|
| 检索方式 | 向量相似度 | 实体一跳邻居 | **因果多跳路径遍历** |
| 关系语义 | 无 | `RELATED_TO` | **21 种严格金融语义（CAUSES, MITIGATES, AMPLIFIES…）** |
| 时间建模 | 无 | 弱 | **6 层时序结构（Year → Event → Risk → Metric）** |
| 证据溯源 | chunk 引用 | 无 | **Document → Page → Sentence 三级溯源** |
| 推理能力 | 语义匹配 | 路径匹配 | **因果链评分 + Cross-Encoder 重排序** |
| LLM 角色 | 主要推理者 | 协同推理 | **受约束的自然语言生成器（知识图谱做推理）** |

**一句话定位**：一个神经符号融合的时序因果知识图谱框架，将 LLM 从"推理者"降级为"生成器"，让结构化知识图谱完成真正的因果推理。

### 1.2 对应的文献空白

从 2025-2026 年的最新研究来看，你的项目直接对应以下几个学术空白：

1. **FinReflectKG-HalluBench (Kumar et al., 2026)**：明确指出 GraphRAG 在金融领域的幻觉问题，并建立了 SEC 10-K 基准数据集（755 QA pairs）。你的项目天然可以在这个基准上评估。

2. **Agentic GraphRAG (Capozzi & Helbing, 2026, ETH)**：提出了 agentic 多层级评估协议，但聚焦于瑞士商业公报。你的项目在 SEC 10-K 金融风险领域构成了互补。

3. **Cross-Entity Financial Sentiment (2026)**：证明了 GraphRAG 在关系型查询上比 Vector RAG 提升 +16.1% 的回答相关性。但他们的图谱缺乏因果语义和时间维度——这正是你的优势。

4. **GraphRAG for Financial Narrative Summarization (FinNLP 2025)**：发现 naive GraphRAG 反而不如 Vector RAG，需要领域优化。你的项目从设计之初就针对金融领域做了本体优化。

**关键洞察**：你的项目不是重复造轮子，而是在 4 篇最新顶会/预印本论文的交集中找到了一个尚未被充分探索的方向——**时序因果知识图谱 + 金融风险推理 + 证据溯源**。

---

## 2. LLM API 选型分析

### 2.1 项目对 LLM 的实际需求

你的项目在不同阶段对 LLM 的需求完全不同：

| 阶段 | 任务 | 调用频率 | Token 量 | 关键要求 |
|------|------|---------|---------|---------|
| **PDF→KG 管线** | 三元组提取、实体标准化 | 批量（每页 1-2 次） | 高（每页 ~3K tokens） | 结构化输出、稳定、便宜 |
| **意图分类** | Query → Intent + Anchors | 每次查询 1 次 | 低（~200 tokens） | 低延迟 |
| **锚点提取** | Query → 实体关键词 | 每次查询 1 次 | 低（~100 tokens） | 低延迟 |
| **报告合成** | 因果路径 → 分析报告 | 每次查询 1 次 | 中（~2K tokens） | 推理质量、引用准确性 |
| **评估实验** | 批量运行对比实验 | 100-500 次查询 | 中 | 一致性、可复现性 |

### 2.2 候选方案对比

#### 方案 A：Groq（当前）

| 维度 | 评分 | 说明 |
|------|------|------|
| 免费额度 | ⭐⭐⭐⭐⭐ | 14,400 req/day，500K tokens/day |
| 推理速度 | ⭐⭐⭐⭐⭐ | ~500-700 tok/s，业界最快 |
| 模型选择 | ⭐⭐⭐ | 仅 Llama 系列（3.3 70B 等） |
| 结构化输出 | ⭐⭐⭐ | 一般，非原生支持 |
| 学术引用 | ⭐⭐ | "使用了 Llama 3.3 通过 Groq"——不如直接用知名 API |
| 中文能力 | ⭐⭐ | Llama 中文较弱（你的部分 PDF 是英文的，影响不大） |

**结论**：适合开发阶段的速度需求，但对学术论文不够"体面"——审稿人可能质疑"为什么用 Groq 而不是直接调用模型"。

#### 方案 B：DeepSeek API（★ 强烈推荐）

| 维度 | 评分 | 说明 |
|------|------|------|
| 价格 | ⭐⭐⭐⭐⭐ | V4-Flash: ¥0.14/M 输入, ¥0.28/M 输出。**处理整个 10-K 管线约 ¥2-5** |
| 推理质量 | ⭐⭐⭐⭐ | V4-Flash 对标 GPT-4o-mini，足够三元组提取 |
| 结构化输出 | ⭐⭐⭐⭐ | JSON mode 原生支持 |
| 学术引用 | ⭐⭐⭐⭐⭐ | "DeepSeek V4-Flash" 是 2026 年顶会最常见的模型之一 |
| 中英双语 | ⭐⭐⭐⭐⭐ | 天生双语，中文论文写作直接可用 |
| 免费额度 | ⭐⭐⭐ | 新注册有小额试用金（~¥7），之后需付费（但极便宜） |

**关键优势**：
- 一篇完整的消融实验（500 次查询）仅需约 **¥5-15**
- V4-Flash 支持 1M 上下文，可直接输入完整 SEC 章节而不需要过度分块
- 学术认可度高——2026 年 ACL/EMNLP/AAAI 投稿大量使用 DeepSeek
- **你已经有了 DeepSeek API key 的使用基础**

#### 方案 C：Google Gemini (AI Studio 免费)

| 维度 | 评分 | 说明 |
|------|------|------|
| 免费额度 | ⭐⭐⭐⭐⭐ | 完全免费，无需信用卡 |
| 上下文窗口 | ⭐⭐⭐⭐⭐ | 1M tokens（可直接输入整个 10-K 章节） |
| 提取准确度 | ⭐⭐⭐⭐⭐ | 2026 年 7 月实验：46 个领域数据提取中位 concordance 0.76，超越 GPT-5 Instant (0.63) |
| 学术引用 | ⭐⭐⭐⭐ | "Gemini 2.5 Flash" 是体面的引用 |
| 数据隐私 | ⚠️ | **免费层的数据可能被用于训练**（EU/EEA 地区除外） |
| 速率限制 | ⭐⭐ | ~250 req/day，批量实验可能不够 |

**结论**：适合开发+调试阶段，但不适合包含敏感数据的大规模实验。

#### 方案 D：OpenRouter（多模型聚合）

| 维度 | 评分 | 说明 |
|------|------|------|
| 模型选择 | ⭐⭐⭐⭐⭐ | 20+ 免费模型，一键切换 |
| 学术价值 | ⭐⭐⭐⭐⭐ | 可以对比不同模型在同一 KG 上的表现，直接成为实验章节 |
| 速率 | ⭐⭐ | 50 req/day 免费，付费后 1000/day |
| 价格 | ⭐⭐⭐⭐ | $10 充值即可满足全部实验需求 |

**关键优势**：通过同一 API 对比 3+ 模型，可以直接写成实验的 Model Ablation 章节。

### 2.3 最终推荐：双引擎策略

```
开发/调试阶段：    Gemini 2.5 Flash (免费)     ← 零成本开发
生产/实验阶段：    DeepSeek V4-Flash (付费)     ← ~¥20-50 覆盖全部实验
消融实验：        OpenRouter (多模型对比)       ← 证明方法不依赖特定模型
```

**为什么不用 Groq 了**：
- DeepSeek V4-Flash 的价格（¥0.14/M tokens）远低于 Groq 的推理成本
- DeepSeek 在学术界的认可度远超"通过 Groq 调用 Llama"
- DeepSeek 的中英双语能力对中文论文写作有直接帮助
- Gemini 免费层可以完全替代 Groq 的开发调试角色

**具体配置**：
```bash
# .env 建议结构
# 主力模型 — 批量提取 + 报告合成
DEEPSEEK_API_KEY=sk-your-deepseek-key
DEEPSEEK_BASE_URL=https://api.deepseek.com

# 开发调试 — 免费 1M 上下文
GEMINI_API_KEY=your-gemini-key

# 实验对比 — 多模型切换
OPENROUTER_API_KEY=your-openrouter-key
```

---

## 3. 当前状态评估

### 3.1 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| Neo4j Schema (6 层) | ✅ 完成 | 13 种节点标签、21 种关系、约束/索引/全文检索 |
| 金融本体 | ✅ 完成 | 500+ 实体映射、24 种风险类别、12 种策略类型 |
| PDF→KG 管线 | ✅ 可用（需修复） | LLM + 规则双重提取，已验证 109 条三元组 |
| GraphRAG 引擎 | ✅ 可用 | 因果多跳路径搜索、路径评分、证据溯源 |
| Vector RAG 基线 | ⚠️ 未验证 | 代码存在，未测试 |
| FastAPI 服务 | ⚠️ 未验证 | 代码存在，未启动 |
| 前端 Dashboard | ❌ 未开始 | Sprint 1 计划中 |

### 3.2 已验证的推理示例

```
Query: "How do export controls impact NVIDIA revenue in China?"

Path 1 (score 0.53): CHINA_MARKET → INVENTORY_RISK → REVENUE
  [Evidence] "We incurred a $4.5 billion charge in Q1 FY2026 for H20
   excess inventory... export controls reduced demand." (2025-10-Q, p.39)

Path 4 (score 0.51): CHINA_MARKET → SUPPLY_CHAIN_DISRUPTION
  → PRODUCT_TRANSITION_RISK → OPERATING_COST (3-hop causal chain!)
```

### 3.3 待解决的技术债务

1. **Ingestor Cypher bug 已修复**（`OBSERVED_IN` 和 `SUPPORTS` 的关系变量冲突）
2. **LLM 提取需要更换 API**（从 Groq → DeepSeek/Gemini）
3. **Section Detector 在非 SEC 文档上表现差**（需要泛化）
4. **缺少系统性错误处理**（管线某一步失败会丢失整页数据）

---

## 4. 发展路线图

### 阶段总览

```
7月下旬 ─── 8月中旬 ─── 9月初 ─── 9月下旬 ─── 10月中旬 ─── 11月
[重构完成]  [KG填充]    [实验]     [前端]      [论文撰写]    [投稿]
   │           │           │           │            │            │
   ✅        现在         2周         3周          4周          投稿截止
```

### Sprint A：知识图谱完整构建（7月下旬 – 8月中旬，~2 周）

**目标**：用 DeepSeek V4-Flash 处理全部 12 个 PDF，构建完整的金融因果知识图谱

| 任务 | 时间 | 产出 |
|------|------|------|
| A1. 集成 DeepSeek API 替代 Groq | 1 天 | 新的 LLM 提取模块 |
| A2. 用 DeepSeek V4-Flash 批量处理全部 PDF | 2 天 | 完整 KG（预计 800-1500 条因果三元组） |
| A3. 人工审核 + 错误分析 | 2 天 | 三元组质量报告 |
| A4. 运行 Vector RAG 基线并构建 ChromaDB | 1 天 | 可对比的 Vector 基线 |
| A5. 构建 Golden Dataset（手动标注 30-50 个 QA pairs） | 3 天 | 评估基准 |
| A6. 图统计 + 质量指标仪表板 | 1 天 | 论文中的图统计表 |

**关键决策**：
- Golden Dataset 参考 FinReflectKG-HalluBench 的标注协议：每条 QA 需要包含 (question, ground_truth_answer, supporting_triples, supporting_chunks)
- 建议覆盖 5 类问题：直接事实查询、因果链推理、时间演化、跨文档综合、对比分析

### Sprint B：实验与评估（8月中旬 – 9月初，~3 周）

**目标**：完成完整的对比实验，产生论文级的实验结果

| 任务 | 时间 | 产出 |
|------|------|------|
| B1. 设计实验矩阵 | 1 天 | 实验设计文档 |
| B2. 实现评估指标（RAGAS + 自定义金融指标） | 2 天 | 评估框架代码 |
| B3. 运行 Baseline 实验组 | 2 天 | Vector RAG 结果 |
| B4. 运行 GraphRAG 实验组 | 2 天 | GraphRAG 结果 |
| B5. 运行 Temporal GraphRAG 实验组 | 2 天 | 带时间约束的 GraphRAG 结果 |
| B6. Ablation Study（移除/保留各层） | 3 天 | 消融实验结果 |
| B7. Model Ablation（不同 LLM 对比） | 2 天 | 多模型鲁棒性验证 |
| B8. 统计显著性检验 + 结果可视化 | 2 天 | 论文图表 |

**实验矩阵设计**：

| 方法 | 检索方式 | 因果建模 | 时间建模 | 证据溯源 |
|------|---------|---------|---------|---------|
| Vector RAG (Baseline) | 向量相似度 | ❌ | ❌ | chunk 引用 |
| GraphRAG (Standard) | 一阶邻居遍历 | ❌ | ❌ | ❌ |
| Causal GraphRAG | 因果多跳遍历 | ✅ | ❌ | ❌ |
| **Temporal Causal GraphRAG (Ours)** | 因果多跳 + 时间约束 | ✅ | ✅ | ✅ (完整三层) |

**评估指标（6 个维度）**：

```yaml
准确性:
  - Answer Correctness (RAGAS): LLM-judge 评分 1-5
  - Factual Consistency (AlignScore): 回答与证据的一致性
  - Entity Recall: 回答中正确实体的比例

推理质量:
  - Causal Path Relevance: 检索路径与问题的因果相关性
  - Temporal Coherence: 时间顺序的正确性
  - Hallucination Rate: 基于 FinReflectKG 协议的幻觉检测

引用完整性:
  - Citation Precision: 引用是否正确支持对应主张
  - Citation Recall: 所有事实主张是否都有引用
  - Evidence Strength: 引用的证据质量（页级→句级）

效率:
  - Retrieval Latency: Graph query vs Vector search
  - End-to-end Latency: 总响应时间
```

### Sprint C：前端 Dashboard（9月初 – 9月下旬，~3 周）

**目标**：构建可交互的可视化 Demo，用于论文展示和面试演示

| 任务 | 时间 | 产出 |
|------|------|------|
| C1. React/Next.js 项目初始化 | 1 天 | 前端项目骨架 |
| C2. 问答界面（Question Input + Answer Display） | 2 天 | 核心交互 |
| C3. 知识图谱可视化（React Flow/D3.js） | 4 天 | 因果路径图 |
| C4. 证据溯源面板（文档页码高亮） | 3 天 | 可信性展示 |
| C5. 实验对比页面（Baseline vs Ours） | 2 天 | 实验 Dashboard |
| C6. 集成 FastAPI 后端 | 2 天 | 端到端 Demo |

**关键设计要求**：
- 用户输入问题 → 展示检索到的因果路径（图谱高亮）→ 展示证据原文（页码标注）→ 展示 LLM 合成报告
- 同时展示 Vector RAG 的对比结果（分屏）
- 暗色金融终端风格（配合金融主题）

### Sprint D：论文撰写（9月下旬 – 10月中旬，~4 周）

**目标**：完成一篇可投稿的学术论文

#### 论文结构（建议 8 页，ACL/EMNLP 格式）

```
1. Introduction (1 page)
   - 金融 RAG 的问题：幻觉、无因果、无时间、无证据
   - 我们的解决方案：时序因果知识图谱 + 神经符号融合
   - 贡献三点：本体设计、多跳推理引擎、全面实验验证

2. Related Work (1 page)
   - GraphRAG 综述（Edge et al., 2024 → Agentic GraphRAG, 2026）
   - 金融 NLP + KG（FinReflectKG, FinDoc-RAG, Cross-Entity Sentiment）
   - 神经符号推理（Neuro-Symbolic AI in Finance）
   - 我们与现有工作的区别

3. Methodology (2.5 pages)
   3.1 六层金融知识图谱本体（Layer architecture + schema design）
   3.2 数据工程管线（PDF → Section → Chunk → Extract → Ingest）
   3.3 因果多跳推理引擎（Path search + Scoring + Evidence collection）
   3.4 LLM 合成与引用生成（Report generation with citations）

4. Experimental Setup (1 page)
   4.1 数据集（12 SEC filings, 50 annotated QA pairs）
   4.2 基线方法（Vector RAG, Standard GraphRAG, Causal GraphRAG）
   4.3 评估指标（6 dimensions, 9 metrics）
   4.4 实现细节（DeepSeek V4-Flash, Neo4j 5.27, Python 3.14）

5. Results & Analysis (1.5 pages)
   5.1 主实验表（所有方法 × 所有指标）
   5.2 消融实验（移除时间/因果/证据层）
   5.3 模型鲁棒性（DeepSeek vs Gemini vs Llama）
   5.4 Case Study（2-3 个复杂推理案例）

6. Conclusion (0.5 page)
   - 总结 + 局限 + 未来工作

Appendix: 完整的本体定义表、Prompt 模板、评估数据集示例
```

#### 目标会议/期刊（按截止日期排序）

| 会议/期刊 | 级别 | 预计截止 | 匹配度 | 策略 |
|----------|------|---------|--------|------|
| **FinNLP @ COLING 2026** | Workshop (B 类附属) | ~10 月 | ⭐⭐⭐⭐⭐ | **首选**——金融 NLP 专业 Workshop |
| **AAAI 2027** | CCF-A | ~8 月（太紧） | ⭐⭐⭐ | 需要更多实验 |
| **EMNLP 2026** | CCF-B | ~6 月（已过） | ⭐⭐⭐⭐ | 已错过 |
| **NeurIPS 2026** | CCF-A | ~5 月（已过） | ⭐⭐⭐ | 已错过 |
| **ACM Web Conference 2027** | CCF-A | ~10-11 月 | ⭐⭐⭐⭐ | 知识图谱 Track |
| **IEEE Access / Neurocomputing** | SCI 期刊 | 滚动 | ⭐⭐⭐⭐ | 期刊备选，审稿 3-6 个月 |
| **Applied Soft Computing** | SCI 一区 | 滚动 | ⭐⭐⭐ | 需要强调方法论创新 |
| **arXiv 预印本** | 无评审 | 随时 | — | **立即上传**——建立优先权 + 申请时可引用 |

**投稿策略**：
1. **8 月底**：上传 arXiv 预印本（建立优先权 + 申请材料）
2. **10 月**：投稿 FinNLP Workshop（最匹配的 venue）
3. **同步**：准备期刊投稿（如 Workshop 不中，延长后投期刊）
4. **11-12 月**：硕士申请提交（论文可以是 under review 状态）

---

## 5. 实验设计与评估框架

### 5.1 Golden Dataset 构建协议

参考 **FinReflectKG-HalluBench** 的标注方法：

**QA 类型分布（目标 50 条）**：

| 类型 | 数量 | 示例 |
|------|------|------|
| 直接事实查询 | 10 | "What was NVIDIA's total revenue in FY2025?" |
| 因果链推理 | 15 | "How did export controls affect NVIDIA's data center revenue?" |
| 时间演化分析 | 10 | "How did supply chain risk evolve from 2022 to 2025?" |
| 跨文档综合 | 10 | "Compare the risk factors disclosed in 2022 vs 2025 10-K" |
| 对比分析 | 5 | "Which market segment is most affected by regulatory risk?" |

**标注协议**：
```
每条 QA 包含：
  - question: str
  - answer: str (ground truth, manually written)
  - evidence_triples: [(source, relation, target), …] (from KG)
  - evidence_chunks: [chunk_text, …] (from original PDF)
  - difficulty: EASY | MEDIUM | HARD
  - requires_temporal: bool
  - requires_causal: bool
  - requires_cross_document: bool
```

### 5.2 消融实验设计

| 实验组 | 因果层 | 时间层 | 证据层 | 目的 |
|--------|--------|--------|--------|------|
| Full System (Ours) | ✅ | ✅ | ✅ | 完整系统 |
| – Causal Layer | ❌ | ✅ | ✅ | 验证因果关系的贡献 |
| – Temporal Layer | ✅ | ❌ | ✅ | 验证时间建模的贡献 |
| – Evidence Layer | ✅ | ✅ | ❌ | 验证证据溯源的贡献 |
| – Causal – Evidence | ❌ | ✅ | ❌ | 退化到 Standard GraphRAG |
| Vector RAG Only | ❌ | ❌ | ❌ | 基线 |

### 5.3 模型鲁棒性实验

通过 OpenRouter 在同一 KG 上对比：

- DeepSeek V4-Flash（主力）
- Gemini 2.5 Flash（免费验证）
- Llama 3.3 70B（Groq）
- Qwen 3（开源对比）

证明方法不依赖某个特定 LLM。

---

## 6. 论文投稿策略

### 6.1 故事线（论文叙事弧）

> **Problem**: 金融领域 LLM 应用面临四个核心挑战——语义幻觉、因果缺失、时间盲视、证据断裂。
>
> **Existing Solutions**: Vector RAG 无法建模因果关系；GraphRAG 引入了结构但缺乏因果语义和时间维度。
>
> **Our Approach**: 提出一个六层时序因果金融知识图谱，通过以下关键设计解决上述问题：
> 1. 严格的金融本体（21 种因果语义关系，500+ 标准化实体）
> 2. 因果多跳路径搜索（因果强度×路径完整性×时间一致性×证据质量）
> 3. 完整的三级证据溯源（Document → Page → Sentence）
>
> **Results**: 在 12 份 SEC 10-K/Q 文件上构建的图谱（X 节点、Y 关系）上：
> - 幻觉率降低 X%
> - 因果推理准确率提升 Y%
> - 在所有复杂查询类型上显著优于 Vector RAG 和 Standard GraphRAG
>
> **Takeaway**: 神经符号融合是金融 AI 的下一个方向——让 LLM 做生成，让知识图谱做推理。

### 6.2 硕士申请材料整合

这个项目可以直接支撑以下申请材料：

| 材料 | 使用方式 |
|------|---------|
| **Personal Statement** | 核心项目经历——展示"研究 × 工程"双能力 |
| **CV/Resume** | Research Project 条目 + GitHub 链接 |
| **Writing Sample** | 论文本身（提交 arXiv 后即可作为 writing sample） |
| **推荐信素材** | 提供给导师的具体技术细节 + 你的贡献 |
| **面试 Demo** | 前端 Dashboard 的可交互演示 |
| **Portfolio** | GitHub README + 论文 + 可视化截图 |

---

## 7. 风险与应对

| 风险 | 概率 | 影响 | 应对 |
|------|------|------|------|
| KG 质量不够好（规则提取噪音大） | 中 | 高 | 切换到 DeepSeek V4-Flash 全量 LLM 提取；人工审核 100 条 |
| 实验效果不显著 | 中 | 中 | 增加 Case Study 定性分析；优化 Path Scoring 权重 |
| 前端开发耗时超出预期 | 中 | 低 | 先用 Streamlit 快速原型；论文不依赖前端 |
| 论文被拒 | 高 | 中 | 先 arXiv 占坑；投多个 venue；期刊做备选 |
| 时间不足 | 中 | 高 | **砍掉前端 Sprint C**——论文 > 工程 Demo |
| 审稿人质疑"为什么不用 GPT-4" | 低 | 中 | Model Ablation 实验证明方法不依赖特定模型 |

---

## 附录：即时行动清单

### 本周（7 月 21-27 日）

- [ ] **注册 DeepSeek API** (platform.deepseek.com) → 获取 API key
- [ ] **更新 .env** 添加 `DEEPSEEK_API_KEY`
- [ ] **修改 `extractor.py`** 支持 DeepSeek endpoint（OpenAI 兼容，只需改 base_url）
- [ ] **批量处理全部 12 个 PDF** 用 DeepSeek V4-Flash
- [ ] **运行 Vector RAG 基线** 建立 Vector DB + 测试基线查询

### 下周（7 月 28 日 – 8 月 3 日）

- [ ] **设计 Golden Dataset**（50 QA pairs）
- [ ] **手动标注** 10 条 QA（建立标注协议）
- [ ] **编写评估脚本**（RAGAS + 自定义指标）

### 8 月中旬

- [ ] **完成全部实验**
- [ ] **arXiv 预印本** 提交草稿
- [ ] **开始前端原型**

---

> **最后的建议**：你做的不是一个玩具项目。这是一个在 2026 年顶会论文中都算有竞争力的研究设计。不要为了"完整"而牺牲深度——宁可让 KG 只覆盖 10 份文件但因果链质量极高，也不要 100 份文件但充满噪音。审稿人和招生委员会看重的是 **问题的清晰性、方法的严谨性、实验的可信度**，而不是数据量或 UI 多漂亮。
>
> 优先保证：**论文 > 实验 > KG 质量 > 前端 Demo > 代码规范**。
