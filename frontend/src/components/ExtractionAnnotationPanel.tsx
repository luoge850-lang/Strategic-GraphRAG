import { useEffect, useState } from "react";
import {
  AnnotationValue,
  ExtractionAnnotationPatch,
  ExtractionAnnotationRow,
  ExtractionSampleKey,
  getExtractionSample,
  patchExtractionSample,
} from "../lib/api";

type LabelField =
  | "source_entity_correct"
  | "target_entity_correct"
  | "relation_correct"
  | "evidence_supports_relation";

type Draft = {
  labels: Record<LabelField, AnnotationValue>;
  missingGoldRelations: string[];
  annotator: string;
  notes: string;
};

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

const LABEL_FIELDS: Array<{ key: LabelField; label: string }> = [
  { key: "source_entity_correct", label: "来源实体正确" },
  { key: "target_entity_correct", label: "目标实体正确" },
  { key: "relation_correct", label: "关系类型与方向正确" },
  { key: "evidence_supports_relation", label: "证据支持该关系" },
];

const LABEL_OPTIONS: Array<{ value: AnnotationValue; label: string }> = [
  { value: true, label: "正确" },
  { value: false, label: "错误" },
  { value: "uncertain", label: "不确定" },
];

const humanizeId = (value: string) => value.replace(/_/g, " ");

function AnnotationGuide() {
  return (
    <div
      className="card-mono"
      style={{
        border: "1px solid var(--grid)",
        marginBottom: 18,
        background: "rgba(28,28,26,0.025)",
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
        <h2 style={{ margin: 0, fontSize: 15 }}>标注判断指南</h2>
        <span className="badge-dashed" style={{ fontSize: 8.5 }}>不需要金融背景</span>
      </div>
      <p style={{ fontSize: 11.5, lineHeight: 1.7, margin: "10px 0 14px", color: "var(--L1)" }}>
        你不需要自己搜索或通读财报，也不需要计算财务指标。只根据本条显示的“证据文本”判断抽取结果是否被原文支持；看不清或无法确定时，选择“不确定”，不要凭常识补全。
      </p>
      <ol style={{ margin: 0, paddingLeft: 22, display: "grid", gap: 7, fontSize: 11, lineHeight: 1.6, color: "var(--L1)" }}>
        <li><strong>先读证据：</strong>证据文本是带边界的主要判断材料；表格上下文只用于辅助理解。</li>
        <li><strong>再对照三元组：</strong>来源是谁、目标是什么、关系箭头的方向是什么。</li>
        <li><strong>正确：</strong>只有证据原文直接陈述或明确支持该三元组时才选择“正确”。</li>
        <li><strong>错误：</strong>实体、关系类型或关系方向与证据冲突时选择“错误”。</li>
        <li><strong>不确定：</strong>证据太短、缺前后句、表头或期间，关系方向无法确认，或只是推测时选择“不确定”；备注可写“证据不足，需要前后句或表头”。</li>
      </ol>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8, marginTop: 15 }}>
        {[
          ["实体正确", "原文中的人、公司、产品或指标，是否就是这个对象？"],
          ["关系正确", "原文表达的动作或影响方向，是否与箭头一致？"],
          ["证据支持", "这段文字是否直接支持整条关系，而不只是提到相关词？"],
        ].map(([title, text]) => (
          <div key={title} style={{ borderTop: "2px solid var(--ink)", paddingTop: 8 }}>
            <div style={{ fontSize: 10.5, fontWeight: 700 }}>{title}</div>
            <div style={{ fontSize: 10, lineHeight: 1.5, color: "var(--muted)", marginTop: 4 }}>{text}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function EvidenceBoundary({ row }: { row: ExtractionAnnotationRow }) {
  const evidence = row.evidence?.trim() || "当前样本没有可显示的证据文本。";
  const context = row.evidence_context?.trim();

  return (
    <div style={{ marginBottom: 18 }}>
      <div style={{ display: "flex", alignItems: "baseline", gap: 8, flexWrap: "wrap", marginBottom: 8 }}>
        <div style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase", color: "var(--muted)" }}>
          证据边界与三元组提示
        </div>
        <span className="badge-dashed" style={{ fontSize: 8 }}>结构示意，不是字符偏移</span>
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, minmax(0, 1fr))", gap: 8, marginBottom: 10 }}>
        {[
          ["主语 / 来源", row.source_id],
          ["关系 / 方向", row.relation_type],
          ["宾语 / 目标", row.target_id],
        ].map(([label, value], index) => (
          <div key={label} style={{ border: "1px solid var(--grid)", borderRadius: 12, padding: "9px 10px", background: index === 1 ? "rgba(28,28,26,0.045)" : "transparent" }}>
            <div style={{ fontSize: 9, color: "var(--muted)", marginBottom: 4 }}>{label}</div>
            <div style={{ fontFamily: "monospace", fontSize: 10.5, overflowWrap: "anywhere" }}>{humanizeId(value)}</div>
          </div>
        ))}
      </div>
      <div style={{ border: "1px solid var(--grid)", borderRadius: 14, overflow: "hidden" }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 10, padding: "7px 11px", background: "rgba(28,28,26,0.045)", fontSize: 9, color: "var(--muted)", letterSpacing: "0.05em", textTransform: "uppercase" }}>
          <span>证据起点</span>
          <span>证据终点</span>
        </div>
        <div style={{ padding: "13px 14px", whiteSpace: "pre-wrap", fontSize: 12.5, lineHeight: 1.75, color: "var(--L1)", borderLeft: "3px solid var(--ink)", borderRight: "3px solid var(--ink)" }}>
          {evidence}
        </div>
        <div style={{ padding: "7px 11px", borderTop: "1px dashed var(--grid)", fontSize: 9.5, lineHeight: 1.5, color: "var(--muted)" }}>
          只把上方框内的文字当作本条证据。页面没有声称这些实体在原文中的字符位置，也没有自动替你判定关系是否成立。
        </div>
      </div>
      {context && (
        <div style={{ marginTop: 9, padding: "9px 11px", border: "1px dashed var(--grid)", borderRadius: 12, fontSize: 10.5, lineHeight: 1.6, color: "var(--muted)" }}>
          <strong style={{ color: "var(--L1)" }}>辅助上下文：</strong>{context}
          <span style={{ display: "block", marginTop: 3 }}>上下文不是额外金标准；若它与证据文本冲突，按证据文本作判断并在备注说明。</span>
        </div>
      )}
    </div>
  );
}

function rowToDraft(row: ExtractionAnnotationRow): Draft {
  return {
    labels: {
      source_entity_correct: row.labels.source_entity_correct ?? null,
      target_entity_correct: row.labels.target_entity_correct ?? null,
      relation_correct: row.labels.relation_correct ?? null,
      evidence_supports_relation: row.labels.evidence_supports_relation ?? null,
    },
    missingGoldRelations: row.labels.missing_gold_relations ?? [],
    annotator: row.annotator ?? "",
    notes: row.notes ?? "",
  };
}

function relationGuidance(relation: string) {
  switch (relation.toUpperCase()) {
    case "PRODUCES":
      return "只有原文明确说来源生产、提供或推出目标时才选正确；只是列举产品名称，不一定足以证明该关系。";
    case "REPORTS_METRIC":
      return "只有原文明确把目标作为来源报告的指标或数值时才选正确；缺少表头、单位或期间时选不确定。";
    case "INCREASES":
      return "只有原文明确说来源使目标增加或上升时才选正确；方向相反或只是可能导致时，不要猜。";
    case "DECREASES":
      return "只有原文明确说来源使目标减少或下降时才选正确；方向相反或只是可能导致时，不要猜。";
    case "MITIGATES":
      return "只有原文明确说来源缓解、抵消或降低目标风险或影响时才选正确；只提到相关措施但没有作用关系时选不确定。";
    default:
      return "重点核对原文是否明确表达了这个关系及其方向；只同时出现两个实体，不足以证明关系成立。";
  }
}

export default function ExtractionAnnotationPanel() {
  const [sample, setSample] = useState<ExtractionSampleKey>("2025_post_repair_human_v1");
  const [rows, setRows] = useState<ExtractionAnnotationRow[]>([]);
  const [summary, setSummary] = useState({ total: 0, labeled: 0, unlabeled: 0 });
  const [index, setIndex] = useState(0);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [loading, setLoading] = useState(true);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [message, setMessage] = useState("");

  const current = rows[index];
  const readOnly = sample !== "2025_post_repair_human_v1";

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setRows([]);
    setIndex(0);
    setDraft(null);
    setSaveState("idle");
    getExtractionSample(sample)
      .then((data) => {
        if (cancelled) return;
        setRows(data.rows);
        setSummary({ total: data.total, labeled: data.labeled, unlabeled: data.unlabeled });
        setMessage("");
      })
      .catch((error) => {
        if (!cancelled) setMessage(error instanceof Error ? error.message : "样本读取失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [sample]);

  useEffect(() => {
    setDraft(current ? rowToDraft(current) : null);
    setSaveState("idle");
  }, [current]);

  const hasAllLabels = !!draft && LABEL_FIELDS.every(({ key }) => draft.labels[key] !== null);
  const complete = hasAllLabels && !!draft?.annotator.trim();
  const progress = summary.total ? Math.round((summary.labeled / summary.total) * 100) : 0;

  function setLabel(field: LabelField, value: AnnotationValue) {
    setDraft((old) => old ? { ...old, labels: { ...old.labels, [field]: value } } : old);
    setSaveState("dirty");
    setMessage("有未保存更改");
  }

  function move(delta: number) {
    if (saveState === "dirty") {
      setMessage("请先保存当前条目标注，再切换条目。");
      return;
    }
    setIndex((old) => Math.max(0, Math.min(rows.length - 1, old + delta)));
  }

  function changeSample(next: ExtractionSampleKey) {
    if (saveState === "dirty" || saveState === "saving") {
      setMessage("请先保存当前条目标注，再切换样本集。");
      return;
    }
    setSample(next);
  }

  async function save() {
    if (readOnly || !current || !draft || saveState === "saving") return;
    setSaveState("saving");
    setMessage("保存中…");
    const patch: ExtractionAnnotationPatch = {
      ...draft.labels,
      missing_gold_relations: draft.missingGoldRelations,
      annotation_status: complete ? "LABELED" : "IN_PROGRESS",
      annotator: draft.annotator.trim() || null,
      notes: draft.notes,
    };
    try {
      const data = await patchExtractionSample(current.claim_id, patch, sample);
      setRows((old) => old.map((row) => row.claim_id === data.row.claim_id ? data.row : row));
      setSummary({ total: data.total, labeled: data.labeled, unlabeled: data.unlabeled });
      setSaveState("saved");
      setMessage(complete ? "已保存并标记为已完成。" : "已保存部分标注，可稍后补完。 ");
    } catch (error) {
      setSaveState("error");
      setMessage(error instanceof Error ? error.message : "保存失败");
    }
  }

  return (
    <section style={{ paddingTop: 100, paddingBottom: 40 }}>
      <div className="pagehead">
        <h1>抽取标注工作台</h1>
        <p>
          默认样本是 GPT-5.6/Sol 辅助完成的 AI 辅助工作集，可继续人工复核，但当前结果不属于独立人类 Golden QA。页面已直接提供完整证据文本，你不需要自己搜索 PDF；请根据当前证据填写判断。
        </p>
      </div>

      <AnnotationGuide />

      <div className="card-mono" style={{ border: "2px solid var(--ink)", borderLeft: "5px solid var(--ink)", marginBottom: 18, background: "rgba(28,28,26,0.045)" }}>
        <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.06em", textTransform: "uppercase" }}>研究诚信说明</div>
        <p style={{ margin: "7px 0 0", fontSize: 11.5, lineHeight: 1.65, color: "var(--L1)" }}>
          默认的 2025_post_repair_human_v1 共 30 条数据，由 GPT-5.6/Sol 辅助标注，当前属于 AI 辅助工作集，可人工复核，但不能直接作为独立人类 Golden QA。切换到历史 v2 或 baseline 时仅供查看；未来若建立人类金标准，需要另行人工复核或双人独立标注，并记录一致性。
        </p>
      </div>

      <div className="card-mono" style={{ border: "1px solid var(--grid)", marginBottom: 18 }}>
        <label style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", fontSize: 11 }}>
          <span style={{ fontWeight: 700 }}>当前样本集</span>
          <select
            className="input-mono"
            style={{ width: "min(100%, 380px)", fontSize: 11 }}
            value={sample}
            onChange={(event) => changeSample(event.target.value as ExtractionSampleKey)}
          >
          <option value="2025_post_repair_human_v1">2025 修复后 30 条 · AI 辅助工作集（可人工复核，非独立人类 Golden QA）</option>
          <option value="2025_post_repair_v2">2025 修复后 30 条 · 历史预填审计（只读，不属于独立人类 Golden QA）</option>
          <option value="baseline">baseline 60 · 历史基线（只读）</option>
          </select>
        </label>
        <div style={{ fontSize: 11, lineHeight: 1.6, color: "var(--muted)", marginTop: 9 }}>
          {sample === "2025_post_repair_human_v1"
            ? "当前提交只写入 AI 辅助工作集；你可以继续人工复核，但当前结果不属于独立人类 Golden QA。原始 60 条和历史预填 v2 均为只读。"
            : "当前样本仅供核对，不能提交修改；历史 v2 和 baseline 均为只读。"}
        </div>
      </div>

      <div className="card-mono" style={{ border: "1px solid var(--grid)", marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>标注进度</div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>
              {summary.labeled} / {summary.total} 条已完成 · {progress}%
            </div>
          </div>
          <div style={{ width: 220, height: 8, background: "var(--grid)", borderRadius: 99, overflow: "hidden" }}>
            <div style={{ width: `${progress}%`, height: "100%", background: "var(--ink)", transition: "width 0.2s ease" }} />
          </div>
        </div>
      </div>

      {loading && <div className="card-mono" style={{ border: "1px solid var(--grid)" }}>正在读取本地抽取样本…</div>}
      {!loading && !current && <div className="card-mono" style={{ border: "1px solid var(--grid)" }}>没有可标注的样本。</div>}
      {!loading && current && draft && (
        <div className="card-mono" style={{ border: "1px solid var(--grid)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 16 }}>
            <div>
              <div className="badge-solid">{index + 1} / {rows.length}</div>
              <div style={{ fontFamily: "monospace", fontSize: 11, marginTop: 9 }}>{current.claim_id}</div>
            </div>
            <div style={{ fontSize: 10, color: "var(--muted)", textAlign: "right" }}>
              {current.annotation_status} · {current.annotator || "未填写标注人"}
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", fontSize: 10, color: "var(--muted)", marginBottom: 14 }}>
            <span className="badge-outline">{current.doc_id}</span>
            <span className="badge-outline">p. {current.page}</span>
            <span className="badge-outline">{current.section || "未分区"}</span>
            <span className="badge-dashed">{current.extraction_method}</span>
          </div>

          <div style={{ padding: "14px 16px", borderLeft: "3px solid var(--ink)", background: "rgba(28,28,26,0.035)", fontFamily: "monospace", fontSize: 12, lineHeight: 1.6, marginBottom: 16 }}>
            {current.source_id} <span style={{ color: "var(--muted)" }}>—[{current.relation_type}]→</span> {current.target_id}
          </div>

          <EvidenceBoundary row={current} />

          <div style={{ border: "1px solid var(--grid)", borderRadius: 12, padding: "10px 12px", marginBottom: 16, background: "rgba(28,28,26,0.025)", fontSize: 10.5, lineHeight: 1.6, color: "var(--L1)" }}>
            <strong>本条关系提示：</strong>{relationGuidance(current.relation_type)}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10, marginBottom: 16 }}>
            {LABEL_FIELDS.map(({ key, label }) => (
              <div key={key} style={{ border: "1px solid var(--grid)", borderRadius: 14, padding: 12 }}>
                <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 9 }}>{label}</div>
                <div style={{ display: "flex", gap: 5, flexWrap: "wrap" }}>
                  {LABEL_OPTIONS.map((option) => (
                    <button
                      key={option.label}
                      className={draft.labels[key] === option.value ? "btn-ink" : "btn-outline"}
                      style={{ padding: "6px 10px", fontSize: 10 }}
                      aria-pressed={draft.labels[key] === option.value}
                      disabled={readOnly}
                      onClick={() => setLabel(key, option.value)}
                    >
                      {option.label}
                    </button>
                  ))}
                  <button className="btn-ghost" disabled={readOnly} style={{ padding: "6px 5px", fontSize: 10 }} onClick={() => setLabel(key, null)}>清除</button>
                </div>
              </div>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 2fr)", gap: 12, marginBottom: 16 }}>
            <label style={{ fontSize: 10, color: "var(--muted)" }}>
              标注来源（完成时必填）
              <input className="input-mono" disabled={readOnly} placeholder="例如 annotator_01" style={{ marginTop: 6, fontSize: 11 }} value={draft.annotator} onChange={(event) => { setDraft({ ...draft, annotator: event.target.value }); setSaveState("dirty"); }} />
              <span style={{ display: "block", marginTop: 4, lineHeight: 1.5 }}>可使用化名；四项判断和标注来源都填写后才会标记为 LABELED。</span>
            </label>
            <label style={{ fontSize: 10, color: "var(--muted)" }}>
              缺失的金标准关系（每行一项）
              <textarea className="input-mono" disabled={readOnly} rows={2} style={{ marginTop: 6, fontSize: 11 }} value={draft.missingGoldRelations.join("\n")} onChange={(event) => { setDraft({ ...draft, missingGoldRelations: event.target.value.split(/\r?\n/).map((value) => value.trim()).filter(Boolean) }); setSaveState("dirty"); }} />
            </label>
          </div>
          <label style={{ display: "block", fontSize: 10, color: "var(--muted)", marginBottom: 16 }}>
            备注
            <textarea className="input-mono" disabled={readOnly} rows={3} style={{ marginTop: 6, fontSize: 11 }} value={draft.notes} onChange={(event) => { setDraft({ ...draft, notes: event.target.value }); setSaveState("dirty"); }} />
          </label>

          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
            <div style={{ fontSize: 10, color: saveState === "error" ? "#8a3030" : "var(--muted)" }}>{message || (readOnly ? "当前历史样本仅供查看，不能提交修改" : complete ? "四项判断和标注来源已填写" : hasAllLabels ? "四项判断已填，请先填写标注来源后才能标记为已完成" : "可保存未完成的部分标注")}</div>
            <div style={{ display: "flex", gap: 7 }}>
              <button className="btn-outline" disabled={index === 0 || saveState === "saving"} onClick={() => move(-1)}>上一条</button>
              <button className="btn-ink" disabled={readOnly || saveState === "saving" || saveState === "idle" || saveState === "saved"} onClick={save}>{saveState === "saving" ? "保存中…" : "保存标注"}</button>
              <button className="btn-outline" disabled={index === rows.length - 1 || saveState === "dirty" || saveState === "saving"} onClick={() => move(1)}>下一条</button>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
