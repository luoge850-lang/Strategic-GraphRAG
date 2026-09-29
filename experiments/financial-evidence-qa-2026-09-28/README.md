# NVIDIA 财务证据问答 — 可复算实验材料

该目录保存 2026-09-28 冻结协议 v4 的离线检索运行记录和 AI/PDF 诊断材料。此版本是**开发集、单公司三年报、非穷尽页标签**的研究实验候选，不是 Gold 测试或生产验收。

## 文件

- `protocol_v4.md`：冻结协议全文。
- `financial_qa_dev_source_review_20260924_v3.jsonl` 与 `.manifest.json`：20 个语义家族、26 个问句形式的 AI/PDF 来源诊断标签。人工主审、复核和裁决都未运行。
- `financial_retrieval_raw_20260928-dev20-final-v4.jsonl` 与同名 `.manifest.json`：156 条原始运行记录（6 种方法 × 26 个问句形式）、候选证据、状态、时延和构建身份。`run_manifest.json` 是便于浏览的同内容副本。
- `financial_qa_summary_20260928-dev20-final-v4.json` / `.png`：从原始 JSONL 重算的摘要与六面板图。
- `table_quality_ai_visual_diagnostic_2026-09-24_v2.jsonl`：60 条表格候选的 AI/PDF 视觉诊断，不是人工准确率。
- `graph_audit_summary.json`：当前本地 Neo4j 只读结构/来源审计的路径脱敏摘要。
- `live_browser_trials.json`：真实本地前端/服务试运行问题及通过边界；不代表已通过 PDF 点击验收。
- `model_dispatch_record.json`：仅记录可访问的多执行者调度请求与限制；实际模型身份/强度未验证。
- `cleanup_and_security.json`：清理扫描范围、被拦截的删除动作及恢复方式。
- `repeat_20260929/`：9 月 29 日通过统一验收脚本产生的同协议复测原始记录、清单、摘要和图表。它是可披露的重复运行，不是独立测试或调参后成绩；首轮和复测时延分开报告。
- `SHA256SUMS.json`：本目录所含文件的 SHA-256 与字节数。

## 重算摘要和图

仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock-2026-09-19.txt
.\.venv\Scripts\python.exe scripts/summarize_financial_candidate_run.py `
  --raw experiments\financial-evidence-qa-2026-09-28\financial_retrieval_raw_20260928-dev20-final-v4.jsonl `
  --dataset experiments\financial-evidence-qa-2026-09-28\financial_qa_dev_source_review_20260924_v3.jsonl `
  --table-audit experiments\financial-evidence-qa-2026-09-28\table_quality_ai_visual_diagnostic_2026-09-24_v2.jsonl `
  --output reports\evaluation\recomputed_summary.json `
  --chart reports\evaluation\recomputed_summary.png
```

这一步从已保存原始结果重新计算指标，不调用 LLM、Neo4j 或网络，也不重跑检索。矩阵重跑需要原始 PDF 和同身份的隔离候选构建包；为避免传播原始申报 PDF、数据库副本与本地数据，本目录不包含它们。所有计数、分母、统计限制和在线演示结果见 [交付记录](../../docs/financial_qa_candidate_release_2026-09-28.md)。

## 标签与解读

标签角色为 AI/PDF 源文件诊断。显式标注的支持页并不穷尽相关语料，未判断页不是不相关页。候选生成流程可能造成选择偏差，题包已经用于开发排错，不得称作独立测试。不要把检索支持页命中当成答案正确率、引用支持准确率或正确拒答率。
