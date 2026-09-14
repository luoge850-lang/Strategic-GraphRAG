# Strategic-GraphRAG 当前研究与交付策略

> 本文件是当前策略入口，事实以 `README.md`、
> `docs/research_evaluation_protocol.md` 和
> `reports/research_readiness_current.json` 为准。历史策略草案已保存在
> `archive/cleanup-2026-09-14/STRATEGY_legacy_2026-07-21.md`，不再作为实验依据。

## 项目目标

在固定的 NVIDIA 2023、2024、2025 年 SEC 10-K 语料上，构建一个能够返回
关系方向、跨年度约束和 PDF 页码证据的 GraphRAG 原型，并比较四种检索模式：

- `vector`：仅向量检索；
- `graph`：实体和关系路径检索；
- `hybrid`：向量锚点加图路径融合；
- `hybrid_temporal`：在混合检索上加入时序事实和跨年度约束。

毕业论文的核心问题应保持为：在同一语料和同一问题集上，带有关系类型、
严格证据约束和时序信息的图检索，是否比纯向量检索更适合金融因果、多跳和
跨年度问题。项目不应把财报披露关系描述成经过计量识别的因果效应，也不应
宣称投资预测能力。

## 当前已核实基线

- 语料固定为三份 10-K，共 395 页；活动图谱有 381 条严格业务
  `EvidenceClaim`。
- 每条活动业务边均有 `VERBATIM` 证据、来源文件、页码、section、chunk、
  实体和关系元数据；图谱结构审计和严格链审计通过。
- 向量集合为 `nvidia_sec_filings_active`，当前有 1,686 个 chunks。
- FastAPI/React Demo 已有固定启动脚本；健康检查同时验证本地 API、Neo4j、
  向量库和 LLM。
- 自动 Silver 检索集有 37 条问题，四种模式均已运行，结果保存在
  `reports/retrieval_benchmark_silver_2026-09-09.json`。
- Silver 结果只用于工程回归和消融：其期望证据来自当前图谱，不能当作独立
  人工金标准，也不能证明抽取语义 Recall 或论文优越性。

## 当前明确限制

`reports/research_readiness_current.json` 采用 fail-closed 规则，当前仍为
`NOT_READY`，原因是：

1. `evaluation/golden_qa_human_v1.jsonl` 尚未完成独立人工复核；因此答案级
   faithfulness、relevance、completeness、citation correctness 和 abstention
   不能写成正式论文指标。
2. 同一 PDF、同一模型、同一 prompt、temperature 0.0 的第二次外部 LLM 抽取
   得到 131 条，而冻结写入运行得到 126 条。temperature 0 不能保证外部服务
   完全确定性；论文实验应使用版本化响应缓存或冻结抽取产物，并明确记录。

这两个限制不是图谱结构审计失败：前者是金标准缺失，后者是外部模型复现性
不足。当前结果适合作为高质量工程原型和毕业设计基础，正式语义结论仍需补齐
独立标注或明确把研究范围收窄为“证据约束检索工程”。

## 推进顺序

1. 保持三份 PDF、Neo4j 快照、claim ID 和当前向量集合冻结；任何重建必须先
   输出新报告、通过连接安全门，再替换单一 filing。
2. 用自动 Silver 作为每次代码修改后的回归门，观察四种模式是否出现错误、
   证据页错配、跨年度遗漏或拒答失效。
3. 为抽取过程增加版本化 response cache 或保存完整的冻结抽取 JSON；在此
   之前不宣称外部 LLM exact repeatability。
4. 如果要写正式论文实验，再进行独立人工关系 inventory 和 30--50 条
   Golden QA；人工只判断证据和问题是否被支持，不需要判断股票涨跌。
5. 四种基线稳定后再加入只读 Agent。Agent 只能分解问题、选择检索工具和
   汇总证据，不得写 Neo4j、改 label 或绕过 citation guard。
6. 最后再做容器化、认证、CORS、限流、超时、成本和日志治理；当前本地 Demo
   不开机自启，运行 `scripts/start_demo.ps1` 即可恢复。

## 交付验收

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q strategic_graphrag scripts tests
.\scripts\audit_research_readiness.py
.\scripts\start_demo.ps1 -Restart
```

然后确认：

- `/health/live` 为 `alive`；
- `/health/ready` 为 `ready`，且 Neo4j、向量库和 LLM 均为可用；
- Demo 查询返回证据页和 `grounding.status=VERIFIED`；
- 研究 readiness 仍明确显示未完成的人工金标准和复现性限制，而不是被自动
  Silver 结果掩盖。
