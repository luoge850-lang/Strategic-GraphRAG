# LLM API 客观选型分析：免费开发阶段

> 不带滤镜，纯数据对比。目标：先用免费 API 完成开发，后续实验阶段再切换付费。

---

## 一句话结论

**开发阶段用 Gemini 2.5 Flash 做提取 + Groq 做批量管线。两者完全免费，无需信用卡。实验阶段迁移到 DeepSeek V4-Flash（~¥20 完成全部实验）。**

---

## 1. 免费 API 硬数据对比（2026 年 7 月）

|  | **Gemini 2.5 Flash** | **Groq (Llama 3.3 70B)** | **OpenRouter 免费** | **DeepSeek V4-Flash** |
|---|---|---|---|---|
| **费用** | 完全免费 | 完全免费 | 50 req/天免费 | ❌ 无永久免费层 |
| **信用卡** | 不需要 | 不需要 | 不需要 | 需要充值 |
| **上下文窗口** | **1M tokens** | 128K | 最高 1M（看模型） | 1M |
| **速率** | ~250 req/天, 10 RPM | **14,400 req/天**, 30 RPM | 50 req/天 | 付费不限 |
| **速度** | ~150 tok/s | **~700 tok/s** | 看模型 | ~200 tok/s |
| **数据训练** | ⚠️ 免费层可能用于训练 | ✅ 不用你的数据训练 | ✅ 不用你的数据训练 | ✅ 不用你的数据训练 |
| **提取准确度** | **0.76 (最高)** | 未见基准测试 | 看模型 | 0.72+ (估计) |
| **结构化输出** | ✅ 原生 JSON mode | ❌ 需 Prompt 约束 | 看模型 | ✅ 原生 JSON mode |
| **学术引用** | "Gemini 2.5 Flash" ← 可引用 | "Llama 3.3 via Groq" ← 间接 | 看模型 | "DeepSeek V4-Flash" ← 可引用 |
| **中文能力** | ⭐⭐⭐⭐ | ⭐⭐ | 看模型 | ⭐⭐⭐⭐⭐ |
| **API 兼容性** | OpenAI 兼容 (`/v1beta/openai/`) | 原生 OpenAI 兼容 | OpenAI 兼容 | OpenAI 兼容 |

---

## 2. 按任务拆解：哪个免费 API 最适合

### 任务 1：三元组提取（PDF→KG，批量处理 12 个文件）

**核心需求**：大上下文（SEC 章节很长）、结构化 JSON 输出、稳定可靠

| API | 优势 | 劣势 |
|-----|------|------|
| **Gemini 2.5 Flash** ✅ | 1M 上下文 → 一次可以放整章 Item 1A（30-40 页）；原生 JSON mode；提取准确度业界最高 | 250 req/天 ≈ 处理 2-3 个 PDF/天 |
| Groq | 14,400 req/天 → 一天能处理全部 12 个 PDF；极快 | 128K 上下文 → 必须切成小块（chunk）；无原生 JSON mode，格式不稳定 |

**客观判断**：**Gemini 2.5 Flash 更适合提取质量**。1M 上下文意味着你可以把整个 "Item 1A Risk Factors"（通常 20-40 页）放进一个 prompt，让模型看到完整上下文再做提取。Groq 虽然快，但 128K 窗口意味着必须把文档切成 3000-token 的块，模型看不到跨段落/跨页面的因果链。

**但 Gemini 的 250 req/天是瓶颈**。12 个 PDF 可能需要 3-5 天才能处理完。

**折中方案**：日常开发用 Gemini（质量优先），需要大量快速迭代时用 Groq（速度优先）。

### 任务 2：意图分类 + 实体提取（每次查询，低 token）

**核心需求**：低延迟、稳定

**结论**：**两个都行，差别不大**。这种 200-token 的任务，Groq 更快（700 tok/s vs 150 tok/s），但差异在用户体验上几乎不可感知。

### 任务 3：LLM 报告合成（GraphRAG 最后一步）

**核心需求**：理解复杂的因果路径、生成准确引用、不幻觉

| API | 优势 | 劣势 |
|-----|------|------|
| **Gemini 2.5 Flash** ✅ | 1M 上下文 → 可以把所有检索到的路径+证据一次性放进去 | 150 tok/s 稍慢 |
| Groq | 极快生成 | 128K 窗口可能不够放多条完整路径 |

**客观判断**：**Gemini 2.5 Flash**。当检索到 10 条因果路径、每条有 3-5 个证据句子时，总 context 可能超过 10K tokens。Gemini 的 1M 窗口绰绰有余，Groq 的 128K 也够——但 Gemini 的原生 JSON mode 对引用格式的约束更强，减少幻觉。

---

## 3. 数据隐私：一个被忽略但重要的问题

| API | 免费层数据政策 | 对项目的影响 |
|-----|--------------|------------|
| **Gemini 2.5 Flash** | ⚠️ 免费层数据可能用于训练（EU/EEA 除外） | SEC 10-K 是**公开文件**——你的数据是公开的，Google 本来就可以爬取。所以这个风险对你的项目**实际为零**。 |
| **Groq** | ✅ 不用于训练 | 更安心，但无实际差别（因为你的数据是公开 SEC 文件） |
| **OpenRouter** | ✅ 不用于训练 | 同上 |
| **Mistral** | ⚠️ **必须同意用于训练**才能免费用 | 对你来说 OK（公开数据） |

**客观判断**：对于 SEC 公开文件，所有 API 的数据隐私问题都不构成实际风险。如果你处理的是未公开的内部财务数据，那 Gemini 和 Mistral 都不合适——但你处理的是公开的 10-K/Q，所以**这个维度不是决策因素**。

---

## 4. 真实成本估算：如果从免费切换到付费

好的实验设计需要在 **同一 API** 上跑完所有实验（保证可比性）。免费 API 可能在你跑实验时遇到速率限制。

| 实验规模 | DeepSeek V4-Flash 成本 | Gemini 2.5 Flash 付费层成本 | Groq 付费层成本 |
|---------|----------------------|--------------------------|--------------|
| 批量处理 12 个 PDF (~500 次提取) | **¥2-5** | ~$0.50 | ~$2 |
| 50 条 QA 评估 (×4 方法对比) = 200 次查询 | **¥3-8** | ~$0.80 | ~$3 |
| 模型鲁棒性实验 (×3 模型) = 150 次查询 | **¥2-5** | ~$0.60 | ~$2 |
| **总计** | **~¥10-20** | **~$2-5** | **~$7-12** |

> DeepSeek 虽然"不免费"，但整个实验周期的总花费约 ¥10-20（两杯奶茶）。Gemini 付费层也极便宜。**在免费和付费之间纠结的经济意义接近于零。**

---

## 5. 最终推荐：三阶段策略

### 阶段 1：开发 & 调试（现在 – 8 月初）→ 纯免费

```
主要：Gemini 2.5 Flash (AI Studio, 免费, 无需信用卡)
  - 三元组提取（利用 1M 上下文 + JSON mode）
  - 意图分类 + 锚点提取
  - LLM 报告合成

辅助：Groq (Llama 3.3 70B, 免费)
  - 快速迭代需要批量处理时（14,400 req/天）
  - Gemini 速率不够时做补充
```

**现在需要做的事**：
```bash
# .env 添加
GEMINI_API_KEY=你的Gemini key (去 aistudio.google.com 创建，免费)
GROQ_API_KEY=你的Groq key (去 console.groq.com 创建，免费)
```

### 阶段 2：正式实验（8 月中旬）→ 近乎免费

```
切换到：DeepSeek V4-Flash (充值 ¥20 足够全部实验)
保留 Gemini 免费层做快速验证
```

**为什么要切换**：
1. 实验需要一致性——所有数据必须用同一个模型跑
2. DeepSeek 的学术引用比 "Gemini free tier" 更正式
3. ¥20 的经济成本几乎为零

### 阶段 3：模型鲁棒性实验（8 月下旬）

```
通过 OpenRouter 统一调用 3+ 模型（充值 $10）：
  - DeepSeek V4-Flash
  - Gemini 2.5 Flash
  - Llama 3.3 70B
  - Qwen 3 (可选)
```

**目的**：证明你的方法不依赖某个特定模型，这是审稿人爱看的维度。

---

## 6. 对 Groq 的客观评价（不说好坏，只说事实）

**事实 1**：Groq 不生产模型，只提供推理基础设施。你调用的是 Llama 3.3 70B，不是 "Groq 的模型"。

**事实 2**：Llama 3.3 70B 是一个通用模型，没有针对结构化提取或金融领域优化。

**事实 3**：在你的项目里，80% 的 LLM 调用是结构化 JSON 提取。Gemini 和 DeepSeek 都原生支持 JSON mode（通过 response_format 参数约束输出必须是合法 JSON）。Groq/Llama 不支持——你必须靠 prompt engineering 让模型输出 JSON，失败率更高。

**事实 4**：在 2026 年 7 月发布的最新提取基准测试中，Gemini 2.5 Flash 以 0.76 mean concordance 排名第一，超越了 GPT-5 Instant (0.63)。Llama 3.3 70B 未出现在该测试中。

**事实 5**：如果论文中写 "We use Llama 3.3 70B via Groq for extraction"，审稿人的反应是"为什么用 Groq？"。如果写 "We use DeepSeek V4-Flash / Gemini 2.5 Flash"，审稿人的反应是"好的，这个模型我知道"。

**客观结论**：Groq 在速度上有绝对优势，但在结构化提取质量、学术引用规范性和 JSON mode 支持上处于劣势。**作为开发阶段的辅助工具完全合适，但不适合作为论文中的主力模型。**

---

## 7. 立即行动（不写代码，只配置）

### 步骤 1：获取免费 API Key（5 分钟）

1. **Gemini**：访问 https://aistudio.google.com → 创建 API key → 复制
2. **Groq**（保留备用）：访问 https://console.groq.com/keys → 创建 API key → 复制
3. **DeepSeek**（先注册，不用充值）：访问 https://platform.deepseek.com → 注册账号

### 步骤 2：更新 .env

```bash
# 开发阶段主力 — 免费
GEMINI_API_KEY=<YOUR_GEMINI_KEY>     # 本地配置，不提交真实密钥
GEMINI_MODEL=gemini-2.5-flash

# 辅助批量处理 — 免费
GROQ_API_KEY=<YOUR_GROQ_KEY>         # 本地配置，不提交真实密钥
GROQ_MODEL=llama-3.3-70b-versatile

# 实验阶段 — 后续充值 ¥20
DEEPSEEK_API_KEY=<YOUR_DEEPSEEK_KEY>      # 本地配置，不提交真实密钥

# Neo4j（已完成）
NEO4J_URI=neo4j+s://<your-instance>.databases.neo4j.io
NEO4J_USERNAME=<your-neo4j-username>
NEO4J_PASSWORD=<REDACTED_USE_LOCAL_ENV>
NEO4J_DATABASE=<your-database-name>
```

### 步骤 3：我可以帮你做的事

你说"先不写代码"，准备好了告诉我。这些是我的待办：

- 修改 `extractor.py`：添加 Gemini API 支持（OpenAI 兼容，改 3 行）
- 添加 `LLMProvider` 抽象层：统一 Gemini / DeepSeek / Groq 接口
- 批量处理全部 12 个 PDF
- 运行 Vector RAG 基线
