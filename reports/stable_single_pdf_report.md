# Strategic-GraphRAG 单 PDF 稳定候选版报告

生成日期：2026-08-12
数据范围：`data/pdfs/2025-10-K.pdf`，共 130 页
LLM：DeepSeek `deepseek-v4-flash`
状态：`SINGLE_PDF_STABLE_CANDIDATE`

## 结论

当前版本已经完成单 PDF 的 PDF→知识图谱→混合检索→证据约束回答→自动审计闭环，可以作为求职展示、毕业设计工程基线和后续科研实验起点。

它还不是生产部署版本，也不是论文最终结果。当前只有一个 FY2025 10-K，不能证明跨年度趋势；Golden QA 是自动生成的回归候选集，不能冒充人工确认的学术金标准。

## 图谱与证据

| 指标 | 结果 |
|---|---:|
| PDF 总页数 | 130 |
| 可解析页数 | 130 |
| 目标抽取页数 | 37 |
| 含严格 EvidenceClaim 的页数 | 25 |
| 页面覆盖率 | 19.23% |
| 严格 EvidenceClaim | 68 |
| 严格业务关系 | 68 |
| 无证据业务关系 | 0 |
| PDF 引用错位 | 0 |
| 缺少字符范围 | 0 |
| 缺少时间元数据 | 0 |
| Neo4j 全文索引 | 2 个，ONLINE |

覆盖率不是 PDF 可解析率。130 页都能读取，但系统只对 SEC 目标章节和高价值财务/风险内容建立严格业务关系；没有满足证据契约的页面不会被强行写入业务图谱。

## 服务与前端验收

- `http://127.0.0.1:8000/` 可打开。
- FastAPI `/health`：Neo4j connected，DeepSeek configured。
- `/graph/statistics`：54 个可视化节点、68 条严格关系。
- `/graph/subgraph`：返回 200，画布可以加载。
- React/Vite 生产构建通过。
- 浏览器交互测试通过：输入出口管制问题后，页面显示分析报告、EvidenceClaim、页码、因果路径和证据链。

统计接口已从多次 Neo4j 远程查询合并为一次聚合查询。实测冷启动约 4.6 秒，后续约 0.32 秒。

## 五问标准回归

标准问题审计结果：

- 5/5 HTTP 200。
- 3 个可回答问题返回 `VERIFIED`。
- 3 个回答中的 EvidenceClaim 均能在 Neo4j 找到，并与 PDF 页文本匹配。
- 2 个不可可靠回答的问题被正确拒答：一个缺少 mitigation 路径，一个需要 FY2023→FY2025 的跨年度证据。
- 5/5 通过严格引用审计。
- 最终审计延迟：P50 12,228.91 ms，P95 15,086.39 ms；包含 DeepSeek 合成。

## 38 条自动回归集

当前数据集由 25 条 single-hop、8 条 multi-hop、5 条 temporal abstention 组成，共 38 条。

本次最终运行结果：

| 指标 | 结果 |
|---|---:|
| Evidence Recall | 0.5152 |
| Evidence Precision | 0.1606 |
| Structural Grounding Proxy | 0.8788 |
| Abstention Accuracy | 1.0000 |
| P50 延迟 | 7,550.28 ms |
| P95 延迟 | 12,782.98 ms |
| LLM Judge Faithfulness | 未启用 |
| LLM Judge Answer Relevance | 未启用 |

这些指标只代表当前自动回归候选集的一次运行。Evidence Precision 仍偏低，说明检索返回的候选路径过多，需要下一阶段进行路径去重、查询意图约束和更严格的 top-k 选择。结构代理指标不能等同于语义 Faithfulness。

## 已完成的工程优化

1. 修复 PDF chunk 截断，使用段落/句子感知切块并保留重叠区。
2. 默认对目标页面执行 LLM 验证，而不是只处理少数风险页。
3. 为 EvidenceClaim 保存页码、原文、字符起止位置、时间范围、模态和关系极性。
4. 清理旧版本 34 条无证据业务关系。
5. 加入向量检索、Neo4j 图检索、全文实体锚点和因果路径融合。
6. 加入阶段延迟统计和自动 PDF 引用对齐审计。
7. 创建并验证 Neo4j `entity_fulltext`、`evidence_fulltext` 两个在线索引。
8. 为评估结果和图谱快照生成 SHA-256 provenance。

## 已知限制

- 只有一个 PDF，不能证明真正的跨年份时序推理。
- 38 条 QA 是 `AUTO_GENERATED_REGRESSION_CANDIDATE`，需要人工去重、补充真实业务问题和负例后，才能称为 Golden Dataset。
- 本轮没有启用 LLM Judge，因此没有论文级 Faithfulness 和 Answer Relevance 数值。
- DeepSeek 报告合成的冷启动和网络延迟仍较高。
- 前端主 JS bundle 约 895 KB，后续应做代码分割。
- `API_AUTH_ENABLED=false` 只适合本地开发；公网部署前必须开启认证、TLS、限流和密钥管理。

## 关联文件

- [final_kg_audit.json](final_kg_audit.json)
- [final_page_coverage.json](final_page_coverage.json)
- [standard_query_audit.json](standard_query_audit.json)
- [golden_qa_v2_results.json](golden_qa_v2_results.json)
- [stable_single_pdf_provenance.json](stable_single_pdf_provenance.json)
