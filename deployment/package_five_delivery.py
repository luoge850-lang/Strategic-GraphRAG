"""Write reviewable delivery tables; never change runtime or frozen inputs."""
import csv,json,hashlib,subprocess,sys,platform,zipfile,importlib.metadata
from pathlib import Path
from deployment.acceptance_20261015 import ROOT,OUT,save,hashes
from deployment.score_five_acceptance import metric

def write(name,text):
    with (OUT/name).open('x',encoding='utf-8') as f:f.write(text)
def main():
    import pymupdf
    package=OUT/'five-package-round2';identity=json.loads((package/'build_identity.json').read_text())
    engine=json.loads((OUT/'engineering-results.json').read_text())
    auto=json.loads((OUT/'validation-scored.json').read_text())
    audit=json.loads((OUT/'post-freeze-semantic-audit.json').read_text())
    retrieval=json.loads((OUT/'retrieval-summary.json').read_text())
    graph_review=json.loads((OUT/'graph-source-judgments.json').read_text())
    extraction=json.loads((OUT/'independent-extraction-checklist.json').read_text())
    graph=json.loads((package/'graph.json').read_text())
    sources=[];downloads={r['fiscal_year']:r for r in json.loads((OUT/'new-source-downloads.json').read_text())}
    sec={2023:('2023-02-24','2023-01-29','000104581023000017','0001045810-23-000017'),2024:('2024-02-21','2024-01-28','000104581024000029','0001045810-24-000029'),2025:('2025-02-26','2025-01-26','000104581025000023','0001045810-25-000023')}
    for name,record in identity['pdfs'].items():
        tsmc=name.startswith('TSMC');year=int(name[5:9] if tsmc else name[:4]);company='TSMC' if tsmc else 'NVIDIA_CORPORATION'
        path=OUT/'sources'/name if tsmc else ROOT/'data'/('pdfs' if year==2025 else 'pdfs_other')/name
        with pymupdf.open(path) as doc:pages=len(doc)
        sources.append(dict(filename=name,company_id=company,fiscal_year=year,document_type='Taiwan annual report; NOT Form 20-F' if tsmc else 'Original retained PDF containing NVIDIA 10-K',
            fiscal_year_end=f'{year}-12-31' if tsmc else sec[year][1],currency='TWD' if tsmc else 'USD',scale='thousands in accepted financial-performance table' if tsmc else 'millions in sampled financial tables',
            sha256=record['sha256'],bytes=record['bytes'],physical_pages=pages,path=str(path),
            submission_date=None if tsmc else sec[year][0],
            submission_date_status='Taiwan annual report publication/submission not verified; not an SEC 20-F submission date' if tsmc else 'Corresponding 10-K date independently checked on SEC index; PDF original download provenance unavailable',
            original_url=downloads[year]['url'] if tsmc else None,
            official_disclosure_reference=downloads[year]['official_url'] if tsmc else f'https://www.sec.gov/Archives/edgar/data/1045810/{sec[year][2]}/{sec[year][3]}-index.htm',
            permission='Public copyrighted disclosure; local research copy. No redistribution license asserted.',
            provenance_limit=downloads[year]['mirror_identity'] if tsmc else 'Retained original PDF byte identity preserved; SEC HTML reference is not asserted as original PDF download URL.'))
    save('five-corpus-identity-table.json',dict(build_id=identity['build_id'],primary_documents=sources,supplementary_relationship_pdfs=[],
        differences='NVIDIA: US GAAP, January fiscal year ends, Form 10-K. TSMC: Taiwan annual reports rather than 20-F, December year ends, TWD and separate convenience USD presentations. No currency conversion performed.',
        incomplete_fields='TSMC submission dates and old NVIDIA PDF download URLs remain unverified; registry completeness not accepted.'))
    problems=[]
    roots=['Pretax wording fell through generic income/profit alias to NET_INCOME.','Pretax profit alias absent; generic profit selected NET_INCOME.',
       'divided by not recognized as an operation; first scalar observation was accepted.',
       'Neo4j omits null currency property; required nullable constructor argument caused ratio records to be dropped. Round1 also introduced reverse operand parsing for an implicit disclosed percentage.',
       'Quarter must not be replaced by an annual observation; original refusal still displayed VERIFIED grounding.',
       'AMD not resolved; default NVIDIA metric retrieval produced a wrong-company PASS.']
    old=json.loads((OUT/'development-before.json').read_text())
    first=json.loads((OUT/'development-round1-raw.json').read_text())
    second=json.loads((OUT/'development-round2-raw.json').read_text())
    for i,row in enumerate(old):
        problems.append(dict(question=row['question'],root_cause=roots[i],before=row['result'],round1=first[i]['result'],round2=second[i]['result'],
           review_grade='EXPOSED_DEVELOPMENT_SOURCE_DIAGNOSIS',fix_status='two rounds exhausted; no further code changes',
           conditions='Exact same request body (graph, generation off, cache off); different isolated build/corpus identity recorded. Not comparable performance timing.'))
    save('problem-plan-facts-answer-citation-status-root-fix-retest.json',problems)
    known_bad=[]
    for edge in graph['edges']:
        reason=None
        if edge['source_filing']=='2023-10-K.pdf' and edge['page']==65 and edge['relation']=='REPORTS_METRIC':reason='Mellanox April 2020 purchase-price allocation wrongly typed as NVIDIA FY2023 consolidated balance.'
        if edge['source_filing']=='2024-10-K.pdf' and edge['page']==39 and edge['target']=='PRETAX_INCOME' and edge.get('metric_unit')=='USD millions':reason='Percentage-of-revenue row read as USD millions due another table currency context on same page.'
        if (edge['source_filing'],edge['page']) in {('2023-10-K.pdf',85),('2024-10-K.pdf',78),('2025-10-K.pdf',78)} and edge['target']=='OPERATING_INCOME':reason='Business-segment columns paired as fiscal-year columns and labelled consolidated.'
        if reason:known_bad.append(dict(edge=edge,root_cause=reason,source_review='post-freeze-conflict-source-pages.json; independent original page context',repair_status='NOT_REPAIRED_AFTER_FREEZE'))
    save('known-financial-graph-errors.json',dict(rows=known_bad,known_bad_records=len(known_bad),total_observations=355,
       scope='Targeted source-confirmed diagnostics separate from the random sample. This is a known-error lower bound, not an estimated overall error rate.'))
    versions={}
    for name in ['PyMuPDF','pdfplumber','neo4j','chromadb','onnxruntime','tokenizers','numpy','fastapi','uvicorn','psutil','pydantic']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    save('runtime-identity.json',dict(python=sys.version,platform=platform.platform(),cpu='Intel Core i7-13620H (Win32_Processor)',logical_cpus=16,
        physical_ram_bytes=json.loads((OUT/'initial-inventory.json').read_text())['hardware']['ram'],packages=versions,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_tree_sha256=identity['source_tree_sha256'],
        frontend_dist_hashes=hashes(ROOT/'frontend/dist'),database=dict(uri='bolt://127.0.0.1:17693',database='neo4j',state_directory=str(OUT/'neo4j-round2'),credentials='local ignored private file; not exported'),
        vector_manifest=json.loads((package/'vector_index_manifest.json').read_text())))
    with (OUT/'per-question-results.csv').open('x',encoding='utf-8-sig',newline='') as f:
        fields=['item_id','question','family_id','type','company','document','http_status','calculation_status','core_semantic_automatic','core_semantic_ai_audit','numeric_and_unit','joint_automatic','false_refusal','review_grade'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        review={x['item_id']:x for x in audit['per_question']}
        for row in auto['per_question']:
            l=row['label'];s=row['score'];writer.writerow(dict(item_id=l['item_id'],question=l['question'],family_id=l['family_id'],type=l['operation'],company=l.get('company','MULTI_OR_UNSPECIFIED'),document=l['source_filing'],http_status=row['result']['http_status'],calculation_status=s['status'],core_semantic_automatic=s['core_semantic'],core_semantic_ai_audit=review[l['item_id']]['ai_source_audit']['core_semantic'],numeric_and_unit=s['numeric_and_unit'],joint_automatic=s['joint'],false_refusal=s['false_refusal'],review_grade=s['review_grade']))
    gates=[dict(gate='New validation core semantic >=90%',result='FAIL',measurement='Frozen automatic 45/52 (86.54%); separate AI source audit 43/52 (82.69%).'),
      dict(gate='Answerable false refusal <=10%',result='FAIL',measurement='7/42 (16.67%). Same-table comparability enum not recognized.'),
      dict(gate='Severe substituted fact/operation PASS =0',result='PASS_FINITE_VALIDATION',measurement='0/52 scalar PASS substitutions; two wrong partial responses separately fail semantic audit. Not universal reliability.'),
      dict(gate='Relation source support >=95%',result='PASS_RETAINED_CENSUS_ONLY',measurement='3/3 retained company relations; broad unreviewed relations quarantined. General relation extraction not accepted.'),
      dict(gate='Sampled financial necessary-field joint >=95%',result='PASS_SAMPLE_ONLY',measurement=f'40/40 stratified sample; 315 unreviewed. {len(known_bad)} other source-confirmed erroneous records prevent a corpus-wide correctness conclusion.'),
      dict(gate='Judged citation support >=95%',result='PASS_FINITE_JUDGMENTS',measurement='35/35 answered validation questions. 7 answerable refusals have no citation judgment; 10 safety questions not applicable.'),
      dict(gate='Successful numeric answers without source =0',result='PASS_FINITE_VALIDATION',measurement='0/35 successful numeric answers.'),
      dict(gate='Comparable cross-company numeric example',result='FAIL',measurement='TSMC margin observation missing; no completed comparable numerical case.'),
      dict(gate='Browser click opens correct physical page automatically',result='NOT_FULLY_VERIFIED',measurement='Clicked links; popup not exposed by in-app browser. Exact href opened separately; actual PDF pages rendered and screenshotted.'),
      dict(gate='Native cancellation/timeout and crash recovery',result='NOT_MEASURED',measurement='Only external timeout/cancel and pre-third-file injected failure/retry tested.'),
      dict(gate='Full regression suite',result='FAIL',measurement='273 passed, 1 failed, 7 subtests passed. Targeted suite 73/73; frontend build succeeded.'),
      dict(gate='Large-scale engineering',result='NOT_MEASURED',measurement='Five documents only; large-scale capability not accepted.'),
      dict(gate='Human Gold / real LLM generation',result='NOT_MEASURED',measurement='0 human Gold; no configured LLM provider; no mock generation results.')]
    save('acceptance-gates.json',dict(overall='未通过验收',classification='可复算的受限研究原型',gates=gates))
    write('tracked-source-changes.diff',subprocess.check_output(['git','diff','--no-ext-diff','--binary'],cwd=ROOT).decode('utf-8'))
    names=['strategic_graphrag/engine/cross_company.py','strategic_graphrag/pipeline/company_disclosures.py','strategic_graphrag/pipeline/consolidated_ntd_table.py','tests/test_acceptance_semantic_regressions.py']
    names += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'deployment').glob('*five*.py')]
    names += ['deployment/acceptance_20261015.py','deployment/download_acceptance_corpus.py','deployment/record_graph_development_review.py','deployment/finish_source_reviews.py']
    with zipfile.ZipFile(OUT/'source-additions.zip','x',zipfile.ZIP_DEFLATED) as z:
        for name in sorted(set(names)):z.write(ROOT/name,name)
    make_report(sources,identity,engine,auto,audit,retrieval,graph_review,extraction,gates,len(known_bad))

def make_report(sources,identity,engine,auto,audit,retrieval,graph_review,extraction,gates,bad_count):
    def pct(value):return f'{100*value:.2f}%'
    source_rows='\n'.join(f"| {x['company_id']} | {x['filename']} | {x['physical_pages']} | {x['currency']} | `{x['sha256']}` |" for x in sources)
    gate_rows='\n'.join(f"| {x['gate']} | {x['result']} | {x['measurement']} |" for x in gates)
    retrieval_rows='\n'.join(f"| {name} | {v['known_positive_row_coverage']['numerator']}/{v['known_positive_row_coverage']['denominator']} | {v['mean_known_positive_reciprocal_rank']:.4f} | {v['latency_ms']['median']:.1f} / {v['latency_ms']['p95']:.1f} |" for name,v in retrieval['methods'].items())
    type_rows='\n'.join(f"| {k} | {v['core_semantic']['numerator']}/{v['core_semantic']['denominator']} | {v['numeric_and_unit']['numerator']}/{v['numeric_and_unit']['denominator']} | {v['false_refusal']['numerator']}/{v['false_refusal']['denominator']} |" for k,v in auto['strata']['type'].items())
    doc_rows='\n'.join(f"| {k} | {v['core_semantic']['numerator']}/{v['core_semantic']['denominator']} | {v['citation_support']['numerator']}/{v['citation_support']['denominator']} | {v['citation_support']['unjudged']} |" for k,v in auto['strata']['document'].items())
    r=engine['resources'];ci=audit['core_semantic']['wilson95']
    write('验收报告.md',f'''# Strategic-GraphRAG 实现与验收交付（2026-10-15）

**未通过验收。** 本交付是可复算的受限研究原型；不能称可靠系统、投资决策工具或大规模生产平台。两轮开发修复已经用尽；52 道冻结验证只运行一次，之后未改生产代码、参考答案或阈值。

## 版本、保护范围与入口

- 原工作区 HEAD `9270c3bff269e67e0eea80788be7139379390ea7`；交付工作树基底 `80dc9cc958711e6e7ed5da8c67a96a1f3ba1de59`。修改保留为未提交补丁和新增文件，未上传 GitHub。
- 实际路径为 `<USER_HOME>\\.codex\\worktrees\\public-demo-delivery\\Nvidia-GraphRAG-Engine`；任务中 `32875.codex` 路径不存在。
- 最终构建 `{identity['build_id']}`；源码指纹 `{identity['source_tree_sha256']}`。
- 最终真实 Neo4j `bolt://127.0.0.1:17693`、独立 state `neo4j-round2`；Chroma 集合 `isolated_{identity['build_id']}`，1861 块。HTTP/UI [最终候选](http://127.0.0.1:8003)。原服务 8000 仍属旧候选，不能用于新候选验收。
- 旧不可变包全文件前后哈希一致；原 72 项哈希和 3 份摘要精确重算再次 PASS，旧质量门槛仍 FAIL。新包前后哈希与当前生产源码指纹一致。原 PDF、原配置、凭证与用户材料保留。正式发布指针未修改。
- 第一轮是五篇真实新解析、抽取和向量化；第二轮重用经过哈希检查的解析/财务抽取缓存，新抽取明确公司关系并重建向量集合。第二轮耗时不能冒充重新解析五篇的耗时。

## 五份主语料

| 公司统一 ID | 文件 | PDF 物理页 | 币种 | SHA256 |
|---|---|---:|---|---|
{source_rows}

共 815 个 PDF 物理页；双页排版的一张物理页不能当作两张 PDF 页。主语料只有以上五份，无其他关系 PDF 偷换计数。TSMC 选择依据是 NVIDIA 原始年报直接披露的代工关系，不是名称共现。

新增两份是台湾公司年报，**不是 SEC 20-F**。与 NVIDIA 10-K 的报告制度、会计框架、12 月/1 月财年结点及 TWD/USD 不同；未做汇率换算。TSMC 官网下载两次 403 已记录，实际下载自公开镜像，不能宣称与未下载成功的官网 PDF 字节相同。每份文件的链接、日期状态、权限与哈希见 [身份表](five-corpus-identity-table.json)。NVIDIA 对应 10-K 提交日期见身份表内 SEC 官方索引；旧 PDF 的原下载 URL 无记录。TSMC 提交日期未核实；登记完整性尚未验收。

## 修复、失败与复测

[完整逐题问题链](problem-plan-facts-answer-citation-status-root-fix-retest.json) 保存问题、计划、候选事实、答案、引用、状态、根因和两轮复测。

1. 税前利润别名落入净利润：第二轮相同请求返回 84,026 USD millions；不再返回 72,880 净利润。
2. `divided by` 被当作事实：第二轮读取两个有身份的操作数，返回约 0.558480 ratio。零分母有定向回归。
3. Neo4j 不存 null 属性，丢弃比例记录：适配器仅为可空 currency/scale 恢复 None。第一轮新增比例处理曾把 R&D 占收入算反并 PASS，原始失败保留；第二轮恢复披露比例 19.6 percent。
4. AMD 未识别默认取 NVIDIA：第二轮拒绝用 NVIDIA 替代。季度请求拒绝年度替代，拒答状态为 ABSTAINED / NOT_APPLICABLE，相关年度候选不作为已验证答案展示。
5. 总额/分部、收购资产/合并余额等仍有缺陷，未全部修复，见下方图谱诊断。不得把别名和季度回归当作这些范围已全面验收。

## 冻结验证与审核等级

52 题、17 个语义家族：42 道财务题来自 TSMC 两份原始 PDF 的合并财务表，10 道边界/跨公司题。TSMC revenue 家族和已曝光 NVIDIA 题为开发数据，七个其他 TSMC 指标家族的所有年份、改写和运算整体留出。没有凑够 60 题。

**全部参考是 AI/PDF 核查，不是人工 Gold；人工审核数为 0。** NVIDIA 没有新增未曝光财务验证家族；不能从本集推出 NVIDIA 全题型准确率。未来用这些失败修复时，必须将对应家族转开发并另建验证。

| 指标 | 分子/分母 | 未判断 | 不适用 | 结果与限定 |
|---|---:|---:|---:|---|
| 核心语义，冻结自动评分 | 45/52 | 0 | 0 | 86.54%，FAIL |
| 核心语义，独立 AI 原文复审 | 43/52 | 0 | 0 | 82.69%，FAIL；描述性 Wilson 95% [{pct(ci[0])}, {pct(ci[1])}] |
| 数值与单位 | 35/42 | 0 | 10 | 83.33%，含未作答失败 |
| 财务题完整事实联合正确 | 35/42 | 0 | 10 | 83.33% |
| 已回答题引用事实支持 | 35/35 | 7 | 10 | 100%，描述性 Wilson 下限 90.11% |
| 已回答题已核查物理页定位 | 35/35 | 7 | 10 | 原文位置判断，不等于浏览器自动跳转 |
| 自动拒绝标量结果 | 10/10 | 0 | 42 | 不等于语义拒答均正确 |
| AI 语义复审正确拒答 | 8/10 | 0 | 42 | 80%；两题仍返回不合题意的局部事实 |
| 可答题错误拒答 | 7/42 | 0 | 10 | 16.67%，超过 10% |
| 请求成功 | 52/52 | 0 | 0 | 超时 0/52，执行故障 0/52 |
| 严重事实/操作替换的标量 PASS | 0/52 | 0 | 0 | 仅本冻结集；两道错误 partial 回答仍算语义失败 |
| 无来源的成功数值答案 | 0/35 | 0 | 17 | 本集观察为 0 |

自动评分脚本发现不了两道 partial 回答的语义问题：客户收入题返回供应依赖；NVIDIA gross profit vs TSMC gross margin 题的局部值换成 NVIDIA gross margin。[独立复审](post-freeze-semantic-audit.json) 单列这些盲点，没有覆盖原自动 45/52 成绩或更改冻结标签。区间只描述这个目的性有限样本；同家族、同页题目相关，不证明总体可靠。自动家族聚类区间保存在原评分文件。

七道差值题的源值均已加载，失败原因是 `SAME_TABLE` 不被计算器认作已确认可比，状态为 `observation_comparability_not_confirmed`。保留失败，未在验证后调参。

| 题型（冻结自动） | 核心语义 | 数值单位 | 错误拒答 |
|---|---:|---:|---:|
{type_rows}

| 文件（冻结自动） | 核心语义 | 引用支持已判断 | 引用未判断/不适用合计 |
|---|---:|---:|---:|
{doc_rows}

公司层面：TSMC 财务题 35/42；MULTI_OR_UNSPECIFIED 边界题自动 10/10、AI 复审 8/10。NVIDIA 仅有曝光开发回归和浏览器案例，不给未曝光准确率。全部分层、未判断字段与逐题请求/完整响应见 [评分原件](validation-scored.json)、[CSV](per-question-results.csv) 和 [原始结果](validation-round2-raw.json)。

## 图谱、抽取及冲突

- 第一轮 2024 非数值规则候选的 18 条开发诊断中，关系方向/直接支持 12/18，完整语义 5/18，均非随机总体估计。条件风险曾标 CONFIRMED_CAUSAL，增加成本曾写成 DECREASES。第二轮保留原候选，但广义规则关系进入待审，不能把减少接受量称作召回提高。
- 最终保留 3 条公司关系原文支持 3/3、方向与条件性 3/3（两份 NVIDIA 代工披露加一份 TSMC 条件风险）。这是当前窄关系集合普查；不足以验收通用关系抽取。
- 真实 Neo4j 按每份文件 8 条抽样，共 40/40 财务必要字段联合通过，未抽样 315；描述性 Wilson 95% 下限约 91.24%，审核等级 AI/PDF。不能证明全图达到 95%。
- **另有 {bad_count} 条源文已确认错误的财务记录**：[清单](known-financial-graph-errors.json)。包括 Mellanox 2020 收购分配被当 FY2023 合并余额，百分比被标 USD millions，以及分部列被当成年份和合并 operating income。随机样本没有抽中这些错误，不得以 40/40 抹去它们。
- 90 组多来源事实仍分别保留；10 组不同数值冲突未被静默覆盖，但其中存在上述抽取错误，不能一律称重述或合法修订。
- 从原始 PDF 独立建立应抽清单：[32 个 TSMC 表内期间事实](independent-extraction-checklist.json) 抽取 32/32；NVIDIA 2025 p9 与 TSMC 2024 p70 的 9 项供应/风险断言仅 2/9 抽到。两者都是选页召回，不是全语料召回。仅查候选的 40 条质量检查没有被当作召回率。

## 跨公司能力与浏览器

已实测公司隔离事实、保留 USD/TWD 和财年结点的并列展示、明确披露的代工依赖，以及 TSMC 自然灾害条件风险到 NVIDIA 晶圆供应的**条件性解释**。逐条给出依赖与风险原文；未声称风险导致了实际 NVIDIA 损失。缺失环节包括客户专属交付、替代产能、库存与实际营收影响，需要客户交付及经营影响披露，不能用多跳路径或模型常识补齐。

可比数值跨公司案例未完成：TSMC gross margin 没有可加载观察，浏览器显示部分回答并拒绝计算。反向关系、错误公司、断链、未披露客户收入、未来时间与未披露损失均保留原始结果；两道 partial 语义失败如上。

浏览器目录保存输入、答案、状态、来源、构建身份的 DOM 快照和截图。税前利润、比值、TSMC 收入成功；不可比、条件路径与缺证拒答均实测。原 PDF 物理页 75 / 64 / 70 实际显示。`target=_blank` 引用点击后新窗口未被内置浏览器暴露，我打开了已观察到的确切 href；**自动点击跳转全链不记 PASS**。页面还有 `7 of 0` 等计数显示及泛化营销文案问题，冻结后未修改。

## 检索比较（先关闭生成）

同一最终构建、42 个财务问题、50 候选/臂、最终 10 个证据块预算、固定随机顺序；真实 ONNX/Chroma 查询，252 次请求。49 个已标注事实需求包含差值题的双操作数。这里判断的是已知原文行与数值支持；未判断检索结果没有被当作负例。

| 方法 | 已知正例行覆盖 | 已知正例 MRR 下界 | 延迟中位/P95 ms |
|---|---:|---:|---:|
{retrieval_rows}

广义图扩展单独加入没有净收益；加图与时间条件在这个只有两个 TSMC 参考页的窄样本中提升明显。保留简单基线和固定图/时间诊断，不据此证明普遍 GraphRAG 优越性，也不继续增加复杂度。完整 Recall 与 nDCG 因未穷尽相关性标注均未测量。原摘要的未判断 hit 计数有事实数/命中数混用，修正计数单列在 [hit 审核](retrieval-hit-audit.json)，未覆盖原结果。[原始检索结果](retrieval-raw.jsonl) 可复算。查询 embedding 调用 211；build embedding 调用/token 与费用未计量。真实生成没有可用配置，未用 mock 补结果；规则回答是前述独立验收。

## 五篇录入与工程边界

机器为 i7-13620H、16 逻辑 CPU，RAM 和依赖版本见 runtime-identity.json。冻结文档作业并发 1、内存预算 8 GiB；没有硬限额执行，不冒充已验证内存上限。

- 五篇实际新解析/抽取/向量化总日志区间 {r['batch_seconds']:.3f}s，815 物理页；逐阶段页/秒、耗时和候选数在 five-package/journal.jsonl。
- 独立单篇 94 页总区间 {r['single_seconds']:.3f}s；单篇 338 块，五篇 1861 块。五篇向量阶段 {r['batch_vector_seconds']:.3f}s，{r['batch_chunks_per_second']:.2f} 块/s；这些是本机一次运行，不是规模外推。
- 五篇阶段采样最大 RSS {r['batch_sampled_max_rss_bytes']} bytes，真实峰值未持续测量；输入 {r['total_input_bytes']} bytes，最终候选目录 {r['candidate_disk_bytes']} bytes。
- 第三篇在解析前注入失败，后续两篇继续，之后第三篇重试完成；没有真实模型失败，也不是崩溃中途恢复测试。新进程能读回五份完成 checkpoint。
- 对非空 Neo4j 重复导入原子拒绝，节点数未增加；这证明安全拒绝重放，不等于幂等成功录入。90 组多来源证据保留，冲突未覆盖。
- PDF / accepted facts / Neo4j / Chroma 身份核对无差异；错误 build 的 Neo4j 与 Chroma 查询命中均 0；新旧实例/集合分离。
- 错误 PDF 产生明确异常。取消、超时只测试外部进程 harness，原生取消/超时状态 API 与中途 crash-resume **未验收**。长表/双页表真实解析与合并范围核查有记录，但不是全部复杂表格泛化证据。
- LLM 网络调用 0、LLM token 0；embedding build token、费用、连续内存峰值均未计量，保存 null，未填零。**仅能称五篇批量录入验证，大规模能力未验收。**

## 门槛结论

| 门槛 | 结果 | 证据/边界 |
|---|---|---|
{gate_rows}

## 最短复现与交付文件

在交付工作树用原环境 Python 执行（不重跑冻结验证、不覆盖结果）：

```powershell
& '<USER_HOME>\\OneDrive\\Desktop\\Nvidia-GraphRAG-Engine\\.venv\\Scripts\\python.exe' -m deployment.verify_five_delivery
```

该命令只校验交付哈希、生产源码身份并在内存重算已冻结分数。预期完整性 PASS、验收 FAIL，退出码 2。运行中的前端入口为 http://127.0.0.1:8003 。需要重新启动已有最终数据库上的 HTTP 时：

```powershell
& '<USER_HOME>\\OneDrive\\Desktop\\Nvidia-GraphRAG-Engine\\.venv\\Scripts\\python.exe' -m deployment.serve_five_acceptance --round round2 --port 8003
```

端口已被本次服务占用时直接使用现有服务，不启动重复实例。Neo4j 私有凭证留在被忽略的本地 state 文件，不在源码包内。

- tracked-source-changes.diff：相对于 80dc9cc 的已跟踪源码补丁；source-additions.zip：新增实现、测试与验收脚本（不含凭证、数据库或 PDF）。
- five-package-round2：最终 accepted facts、图、向量快照与构建身份；package-round2-frozen-hashes.json：运行前全包哈希。
- validation-labels.json、validation-freeze.json、final-scoring-freeze.json、validation-round2-raw.json、validation-scored.json：冻结输入、评分与每题原始证据。
- post-freeze-semantic-audit.json、graph-review-packet.json、graph-source-judgments.json、known-financial-graph-errors.json、independent-extraction-checklist.json：审核等级、已知错误及抽取边界。
- engineering-results.json、five-package/journal.jsonl、single-package/journal.jsonl：实录、失败、重试与资源边界。
- browser/：浏览器截图、DOM 和点击限制；pytest-final.log：未掩盖的全套测试失败。

最终分类：**未通过验收；可交付可复算的研究原型，不交付可靠性或生产规模承诺。**
''')
if __name__=='__main__':main()
