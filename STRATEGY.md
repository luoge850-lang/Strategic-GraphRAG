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
- 自动 Silver 检索集有 37 条问题，四种模式均已运行，当前冻结结果保存在
  `reports/retrieval_benchmark_silver_2026-09-18_ranking_v2_metrics.json`，
  原始四模式追踪保存在同日期的 `ranking_v2.json`，配置说明见
  `docs/reproducibility_freeze_2026-09-18.md`。
- Silver 结果只用于工程回归和消融：其期望证据来自当前图谱，不能当作独立
  人工金标准，也不能证明抽取语义 Recall 或论文优越性。
- Golden QA 同时保留候选证据级视图和问题级去重视图。相同问题如果在不同
  候选证据上出现冲突标签，问题级评测会保留冲突元数据并合并人工认可的证据，
  不会把候选行误当成相互独立的测试问题。
- 当前三份 PDF、图谱 inventory、claim ID 版本、向量集合和运行时配置的派生
  清单见 `reports/corpus_manifest_2026-09-19.json`；
  `archive/cleanup-2026-09-18/historical-reports/2026-08-14_corpus_manifest.json`
  是历史清单，不应覆盖当前事实。
- 当前 canonical state、依赖锁和保留/归档决策见
  `docs/canonical_project_state_2026-09-19.md` 与
  `requirements-lock-2026-09-19.txt`。60 条表格质量标注队列是未标注候选，
  不能直接当作 Gold；FinanceBench、FinQA、TAT-QA 仅完成独立注册和 schema
  校验，未与 NVIDIA 语料混合打分。

## 当前明确限制

`reports/research_readiness_current.json` 采用 fail-closed 规则，当前仍为
`NOT_READY`，原因是：

1. `evaluation/golden_qa_human_v2.jsonl` 已完成 30/30 条单人复核，可以作为
   工程 Golden QA；但它不是双人独立标注和仲裁后的论文级金标准。因此答案级
   faithfulness、relevance、completeness、citation correctness 和 abstention
   目前只能作为受限工程指标，不能写成普适准确率。
2. 同一 PDF、同一模型、同一 prompt、temperature 0.0 的第二次外部 LLM 抽取
   得到 131 条，而冻结写入运行得到 126 条。temperature 0 不能保证外部服务
   完全确定性。版本化 response cache 的 record/replay 已通过（126 条、170
   个唯一键、replay 零网络调用），但这只能证明冻结响应的确定性回放，不能
   把它写成 fresh external-LLM exact repeatability。

这不是图谱结构审计失败：当前唯一 blocking gate 是外部模型 fresh
repeatability，不是 Neo4j provenance 或 Golden QA 文件缺失。当前结果适合作为
高质量工程原型和毕业设计基础；正式语义结论仍需增加独立复核者，或明确把研究
范围收窄为“证据约束检索工程”。

## 推进顺序

1. 保持三份 PDF、Neo4j 快照、claim ID 和当前向量集合冻结；任何重建必须先
   输出新报告、通过连接安全门，再替换单一 filing。
2. 新的结构语义门已写入未来抽取路径，但本轮没有写 Neo4j。2025 冻结缓存
   dry-run 从 126 降为 112 条严格候选；在没有 2023/2024 等价缓存前不做部分
   图谱替换，避免把不可重复的重建结果误当成改进。
3. 用自动 Silver 作为每次代码修改后的回归门，观察四种模式是否出现错误、
   证据页错配、跨年度遗漏或拒答失效。
4. 维护版本化 response cache，并把 fresh external-LLM repeatability 与 cache
   replay 分开报告；在此基础上不宣称外部 LLM exact repeatability 已满足。
5. 当前 30 条 Golden QA 已可用于工程回归，但正式论文前应增加第二名独立
   复核者，对全部分歧做仲裁，并保留 reviewer、理由和页码证据。旧 v1 工作集
   不能混入。人工只判断证据和问题是否被支持，不需要判断股票涨跌。
6. 四种基线稳定后再加入只读 Agent。Agent 只能分解问题、选择检索工具和
   汇总证据，不得写 Neo4j、改 label 或绕过 citation guard。
7. 最后再做容器化、认证、CORS、限流、超时、成本和日志治理；当前本地 Demo
   启动脚本会等待依赖恢复；SchemaManager 对 Aura 暂停/唤醒造成的失效连接
   自动重连一次。明确关系和两个端点的问题会自动走 Graph-only，避免向量
   噪声进入回答上下文。
   不开机自启，运行 `scripts/start_demo.ps1` 即可恢复。
   日常使用可直接双击项目根目录的 `open_demo.cmd`；它会先等待
   `/health/ready`，再打开浏览器，因此不会再出现页面已经打开但图谱尚未连接的
   假故障。
8. 运行时评估单独使用 `scripts/benchmark_runtime_performance.py`，分别记录
   cache miss、cache hit/fill、P50/P95/P99、阶段耗时、错误率和并发吞吐。
   `use_cache=false` 不是进程级 cold start；进程级冷启动必须通过受控重启
   单独测量。所有 synthesis 结果必须和 retrieval-only 结果分开报告。
9. 本轮只做不改变数据的 P1 工程优化：证据排序继续执行“语义路径合并、保留
   evidence variants、直接性/端点/谓词门控、ANSWER_CRITICAL 优先”；Graph/
   Hybrid 的 PPR 改为进程内有界 TTL 缓存，时序事实融合先按 claim ID 定位
   `EvidenceClaim`，并把 PPR 耗时从 anchor resolution 中单独计时。优化后的
   首次查询语义不变，缓存命中只改善重复查询延迟；任何图谱或 PDF 变化仍需
   重新冻结和重新审计。

## 交付验收

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q strategic_graphrag scripts tests
.\scripts\audit_research_readiness.py
.\.venv\Scripts\python.exe scripts/audit_graph_semantic_consistency.py
.\scripts\start_demo.ps1 -Restart
```

然后确认：

- `/health/live` 为 `alive`；
- `/health/ready` 为 `ready`，且 Neo4j、向量库和 LLM 均为可用；
- Demo 查询返回证据页和 `grounding.status=VERIFIED`；
- 研究 readiness 仍明确显示 fresh 外部模型复现性限制；Golden QA 的单人
  工程状态和其论文级限制均已单独记录，而不是被自动 Silver 结果掩盖。
