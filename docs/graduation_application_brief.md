# Strategic-GraphRAG：毕业设计与申请材料定位

## 一句话定位

Strategic-GraphRAG 是一个面向 NVIDIA SEC 10-K 财报的证据约束型 GraphRAG
原型：它把 PDF 中可定位的文本证据抽取为带关系类型的知识图谱，再结合
向量检索、定向图路径和时序模型，生成可回溯到财报页码的回答。

它目前最适合被表述为“可复现的研究型工程原型”，而不是生产系统、投资
顾问或已经完成语义准确率验证的金融模型。

## 适合作为毕业设计的研究问题

建议把论文问题收敛为一个可验证的问题：

> 在固定的 SEC 10-K 财报语料上，带有关系类型、严格证据约束和页码溯源的
> GraphRAG，是否比纯向量检索更适合回答因果、多跳和跨年度财务问题？

论文的自变量是检索模式：`vector`、`graph`、`hybrid`、`hybrid_temporal`；
因变量是证据召回、Precision@K、Recall@K、MRR、nDCG、答案忠实性、相关性、
完整性、引用正确性和拒答准确率。所有模式必须在相同问题集、相同语料和
相同 top-k 下比较。

## 当前可核实成果

- 语料冻结为 NVIDIA 2023、2024、2025 三份 10-K，共 395 页。
- 当前活动图谱有 381 条严格业务边，其中 2025 份重建后为 126 条。
- 每条活动业务边都连接 `VERBATIM` EvidenceClaim、Sentence、页码、chunk
  和来源/目标实体。
- 2025 份重建已完成 170 次 LLM 调用，全部成功，126/126 条写入。
- 最新结构审计和严格链路审计通过，当前审计报告位于 `reports/`。
- 四种检索模式已经实现，FastAPI 和 React/Vite Demo 可以运行。
- Neo4j 写入前增加了连接存活检查和自动重连，避免长时间 PDF 抽取后提交
  阶段使用失效的 Aura 路由连接。
- 自动 Silver 检索回归集已经建立为 37 条问题，四种模式均已运行；它用于
  工程回归，不冒充独立人工金标准。
- 同一 2025 PDF 的抽取复现已实际执行：固定 temperature=0.0 仍出现
  126 与 131 条 accepted claims 的差异，因此 exact external-LLM
  repeatability 仍被明确标记为未满足。

## 目前不能写成论文结论的内容

- 30 条抽取样本仍是候选/AI 辅助诊断，不能称为独立人工金标准。
- 其中两个旧 claim ID 在 2025 替换后已经失效或发生变化，必须重新映射。
- 当前的 27/30 relation/evidence 结果只能作为 precision-like 诊断，不能
  直接当作 Recall 或 F1。
- Golden QA 还没有全部完成人工复核，因此四种检索模式目前只有自动 Silver
  回归指标，没有独立人工金标准下的正式答案质量结论。
- 2025 图谱替换后的 `TemporalFact` 和 `TemporalChange` 等派生模型已经重新
  物化并通过结构检查；未来任何图谱重建后都必须重新物化、捕获快照并审计，
  不能沿用旧的时序数量或时序结果。
- 财报中的 `CAUSES` 表示公司披露的关系，不等同于经过计量识别的因果效应，
  不能据此宣称投资收益、概率或反事实结论。

## 论文建议结构

1. **绪论**：财报问答中的证据可追溯性、关系方向和跨年度比较问题。
2. **相关工作**：Vector RAG、GraphRAG、金融信息抽取、时序知识图谱。
3. **系统设计**：PDF 解析、SEC section 检测、混合抽取、本体、Neo4j、向量
   索引、查询路由、证据验证和前端展示。
4. **数据与标注协议**：三份 10-K、抽取样本、Golden QA、人工复核和一致性
   计算方式。
5. **实验**：四种检索模式的统一评估，以及消融实验（去掉图路径、时序或
   证据约束）。
6. **错误分析**：实体规范化、关系方向、跨页证据、拒答和引用错误。
7. **结论与限制**：研究结论只针对冻结语料和协议，不外推到投资预测。

## 申请材料中的安全表述

可以写：

> Built a provenance-constrained GraphRAG prototype for NVIDIA SEC 10-K
> filings, combining hybrid relation extraction, Neo4j evidence paths,
> filing-scoped vector retrieval, temporal disclosure modeling, and a
> FastAPI/React evidence dashboard.

如果需要量化，可以写已经核实的工程事实：395 页语料、381 条当前严格
EvidenceClaim、四种检索模式、170 次 LLM 调用全部成功的 2025 重建，以及
结构审计通过。不要把这些数字写成语义准确率，也不要写成生产部署、金融
预测或已证明 GraphRAG 优于 Vector RAG。

## 接下来按这个顺序推进

1. **已完成（当前冻结状态）**：保存 Neo4j 快照并重新物化 disclosure links、
   financial observations、TemporalFact 和 TemporalChange；未来重建时重复这
   一流程，并在替换前后重新捕获快照和审计。
2. 基于当前 claim ID 生成新的抽取标注候选集，保留旧样本为历史对照，不覆盖。
3. 建立至少 30 条人工 Golden QA：每条只需核对问题、证据、是否可回答、标准
   答案和拒答条件；不需要判断公司经营好坏。
4. 先将 Silver 回归固定为开发质量门；若要写论文结果，再对独立 Golden QA
   问题集运行四种 retrieval baseline，报告逐题结果、宏平均和 bootstrap 区间；
   缓存命中和冷启动延迟分开统计。
5. 在主实验完成后再加入只读 Agent，让 Agent 负责问题分解和调用检索工具，
   不允许 Agent 写图谱或写标注；额外评估工具调用次数、延迟、成本和拒答。

## Demo 验收方式

进入项目目录后运行：

```powershell
.\scripts\start_demo.ps1
```

需要控制重启时运行：

```powershell
.\scripts\start_demo.ps1 -Restart
```

然后打开 `http://127.0.0.1:8000/`。如果页面显示 `Failed to fetch`，先检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health/live
Invoke-RestMethod http://127.0.0.1:8000/health/ready
```

`live` 表示本地 API 已启动；`ready` 还会检查 Neo4j、向量库和 LLM。Demo 不
设置开机自启，关闭电脑后需要再次运行启动脚本，这是刻意保留的临时部署策略。
