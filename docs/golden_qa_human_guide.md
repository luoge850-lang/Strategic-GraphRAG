# Human Golden QA 人工审阅指南

这份指南面向不熟悉金融和 NVIDIA 业务的人工审阅者。你的任务不是判断
一家公司经营得好不好，也不是补充自己的金融知识，而是检查问题、答案和
证据文本之间是否成立。

## 空白抽取集和 Golden QA 的区别

`data/evaluation/golden_qa_v2.jsonl` 是自动生成的回归候选集，状态为
`AUTO_GENERATED_REGRESSION_CANDIDATE`。它可以帮助发现检索退化，但不是人工
金标准，候选答案、`answerable` 和证据 ID 都可能有错。

`evaluation/golden_qa_human_v1.jsonl` 是从候选复制出的独立空白审阅集。脚本
会把候选字段改名为 `candidate_*` 供追溯，并将人工填写字段留空，状态设为
`HUMAN_REVIEW_PENDING`。生成这个模板不等于完成审阅，也不能把它直接当作
Golden QA 使用。

最终只有人工核对完成、状态改为 `HUMAN_REVIEWED` 的记录，才能进入 Golden
QA。候选字段是参考和审计线索，不是人工结论；不要直接复制候选答案或
`candidate_answerable`。

## 每行如何填写

每行先确认 `id` 和 `question`，再根据题目给出的来源与证据文本填写以下字段。
人工字段使用最终 QA schema 的顶层名称；候选参考字段都带有 `candidate_`
前缀。

- `reference_answer`：只写证据文本能够支持的答案。可以简洁改写，但不能
  添加证据中没有的数字、原因、时间或结论。若问题不可回答，写明缺少什么
  证据，或写一个简短的拒答说明。
- `gold_evidence_ids`：填写真正支持答案的证据 ID，通常来自候选的
  `candidate_evidence_claim_ids`。逐个核对 ID 和对应文本，不要凭空创造 ID。
  不相关或不能支撑结论的 ID 不要放入这里。
- `gold_pages`：填写真正支持答案的财报页码。只有在核对过证据 ID 对应的原文
  后才填写；如果问题不可回答，通常保持为空。
- `relevant_evidence_grades`：为审阅过的相关证据填写整数评分，例如
  `{"claim_v2_...": 2}`。评分只描述证据对本题的支持程度：`2` 表示直接且
  足够支持，`1` 表示相关但只能部分或间接支持，`0` 表示不相关、矛盾或
  不能支持本题。通常只有评分为 `2` 且确实支撑答案的 ID 才进入
  `gold_evidence_ids`。
- `answerable`：证据文本足够回答且问题清楚时填 `true`；证据不足、证据
  互相冲突而无法解决，或问题要求来源中不存在的信息时填 `false`。不要用
  自己知道的金融常识替来源补全答案。
- `requires_abstention`：当 `answerable` 为 `false` 时通常填 `true`，表示
  系统应该明确说无法根据现有证据回答；可回答的问题通常填 `false`。若两者
  看起来不一致，请在 `review_notes` 说明原因。
- `reviewer`：填写实际完成该行最终核对的审阅者标识，不要填写模型名称。
- `review_notes`：记录歧义、证据冲突、无法读取的文本、需要仲裁的地方，或
  任何与该行判定有关的简短说明。
- `review_status`：在所有人工字段核对完并通过复核后才改为
  `HUMAN_REVIEWED`。模板刚生成时必须保持 `HUMAN_REVIEW_PENDING`。

## 不懂金融时的判断方法

把题目当成“根据指定材料找答案”的阅读理解题：

1. 先读问题，识别它要找的是关系、数字、年份、变化，还是是否存在证据。
2. 再打开候选证据 ID 对应的原文，检查主体、对象、关系、时间和限定词是否
   一致。`may`、`could`、`approximately` 等限定词不能被改写成确定事实。
3. 只有原文足够支持时才填写肯定答案。原文没有提到、只支持一半、或题目
   与证据不匹配时，应选择不可回答并要求系统 abstain。
4. 不要因为答案听起来符合常识就判定正确；也不要因为不认识财务术语就
   判定错误。看不懂的术语可以原样保留，关键是文本是否支持题目要求。

## 独立复核要求

人工金标准应由至少两名审阅者独立完成，或由一名审阅者完成后进行一次不
看第一份结论的独立复核；分歧交给仲裁者处理。独立审阅时不要先看另一位
审阅者的 `reference_answer`、`answerable` 或评分，也不要把候选的
`candidate_expected_answer` 当成标准答案。仲裁后只保留统一的最终字段，并
在 `review_notes` 记录重要分歧。

完成前逐行确认：问题可复述、答案有原文依据、证据 ID 可定位、证据评分有
理由、`answerable` 与 `requires_abstention` 相符、`reviewer` 已填写。最后
确认所有完成行的 `review_status` 都是 `HUMAN_REVIEWED`，并且至少有 30 行；
否则审计会保持 fail-closed，不会把候选集当作人工 Golden QA。
