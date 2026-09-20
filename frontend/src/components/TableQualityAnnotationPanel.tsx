import { useEffect, useMemo, useState } from "react";
import {
  TableQualityGold,
  TableQualityPatch,
  TableQualityRow,
  getTableQuality,
  patchTableQuality,
} from "../lib/api";

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

const FIELD_DEFS: Array<{ key: keyof TableQualityGold; label: string; hint: string }> = [
  { key: "company_id", label: "公司", hint: "通常是 NVIDIA Corporation；以原文主体为准。" },
  { key: "fiscal_year", label: "财务年度", hint: "看表头年份，不要只看文件名。" },
  { key: "metric_id", label: "指标/行名", hint: "照表格行名填写，可保留原文含义。" },
  { key: "value", label: "数值", hint: "保留负号、括号和小数；不要自行计算。" },
  { key: "unit", label: "单位", hint: "例如 USD millions；必须和表头或脚注一致。" },
  { key: "source_filing", label: "来源文件", hint: "选择实际核对的 10-K 文件。" },
  { key: "page", label: "PDF 页码", hint: "填写证据所在的 PDF 页码。" },
  { key: "table_name", label: "表名/章节", hint: "没有清楚表名时填章节或写 UNKNOWN。" },
  { key: "row_label", label: "表格行名", hint: "直接抄表格左侧行名。" },
  { key: "column_label", label: "表格列名", hint: "直接抄表头列名/年份。" },
];

function draftFromRow(row: TableQualityRow): TableQualityGold {
  return { ...(row.gold || {}) };
}

function textValue(value: unknown): string {
  return value === undefined || value === null ? "" : String(value);
}

export default function TableQualityAnnotationPanel() {
  const [rows, setRows] = useState<TableQualityRow[]>([]);
  const [summary, setSummary] = useState({ total: 0, reviewed: 0, in_progress: 0, pending: 0 });
  const [index, setIndex] = useState(0);
  const [gold, setGold] = useState<TableQualityGold>({});
  const [reviewer, setReviewer] = useState("");
  const [notes, setNotes] = useState("");
  const [loading, setLoading] = useState(true);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [message, setMessage] = useState("");

  const current = rows[index];
  const isSupported = gold.cell_supported === true;
  const isUnsupported = gold.cell_supported === false;
  const coreFieldsComplete = FIELD_DEFS.every(({ key }) => textValue(gold[key]).trim().length > 0);
  const canComplete = !!reviewer.trim() && (isUnsupported || (isSupported && coreFieldsComplete && !!textValue(gold.evidence_text).trim()));
  const progress = summary.total ? Math.round((summary.reviewed / summary.total) * 100) : 0;

  const sourceUrl = useMemo(() => {
    if (!current) return "";
    return `/evaluation/table-quality/source/${encodeURIComponent(current.source_filing)}#page=${current.page}`;
  }, [current]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getTableQuality()
      .then((data) => {
        if (cancelled) return;
        setRows(data.rows);
        setSummary({ total: data.total, reviewed: data.reviewed, in_progress: data.in_progress, pending: data.pending });
      })
      .catch((error) => { if (!cancelled) setMessage(error instanceof Error ? error.message : "表格标注读取失败"); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!current) return;
    setGold(draftFromRow(current));
    setReviewer(current.reviewer || "");
    setNotes(current.review_notes || "");
    setSaveState("idle");
  }, [current]);

  function updateGold(key: keyof TableQualityGold, value: string) {
    setGold((old) => ({ ...old, [key]: value }));
    setSaveState("dirty");
  }

  function setSupported(value: boolean | "uncertain") {
    setGold((old) => ({ ...old, cell_supported: value }));
    setSaveState("dirty");
  }

  function move(delta: number) {
    if (saveState === "dirty" || saveState === "saving") {
      setMessage("请先保存当前条目，再切换条目。");
      return;
    }
    setIndex((old) => Math.max(0, Math.min(rows.length - 1, old + delta)));
    setMessage("");
  }

  async function save() {
    if (!current || saveState === "saving") return;
    setSaveState("saving");
    setMessage("保存中…");
    const patch: TableQualityPatch = {
      gold,
      reviewer: reviewer.trim(),
      review_notes: notes,
      review_status: canComplete ? "HUMAN_REVIEWED" : "IN_PROGRESS",
    };
    try {
      const data = await patchTableQuality(current.queue_id, patch);
      setRows((old) => old.map((row) => row.queue_id === data.row.queue_id ? data.row : row));
      setSummary({ total: data.total, reviewed: data.reviewed, in_progress: data.in_progress, pending: data.pending });
      setSaveState("saved");
      setMessage(canComplete ? "已保存为 HUMAN_REVIEWED。" : "已保存草稿；还没有达到独立 Gold 完成条件。");
    } catch (error) {
      setSaveState("error");
      setMessage(error instanceof Error ? error.message : "表格标注保存失败");
    }
  }

  return (
    <section style={{ paddingTop: 100, paddingBottom: 40 }}>
      <div className="pagehead">
        <h1>表格质量 · 独立 Gold 标注</h1>
        <p>你不需要懂金融。只需打开原始 PDF 对照表格，判断这一行/单元格是否真的被证据支持，并把看到的内容抄入 Gold 字段。</p>
      </div>

      <div className="card-mono" style={{ border: "2px solid var(--ink)", borderLeft: "5px solid var(--ink)", marginBottom: 18, background: "rgba(28,28,26,0.045)" }}>
        <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.06em" }}>你每条只做四步</div>
        <div style={{ marginTop: 8, fontSize: 11.5, lineHeight: 1.75 }}>
          ①打开原始 PDF 对应页；②核对表头、行名、年份、数值和单位；③选择“单元格被支持”或“未被支持”；④填写审阅者和备注后保存。不要根据系统预测猜答案；系统预测只放在下方可展开区域，不能复制到 Gold。
        </div>
        <div style={{ marginTop: 8, fontSize: 10.5, color: "var(--muted)", lineHeight: 1.6 }}>
          “未被支持”不是说财务事实一定错误，而是说本条候选证据没有在该页明确支持这个单元格。看不清时先保存草稿并备注“需要复核”，不要猜测。
        </div>
      </div>

      <div className="card-mono" style={{ border: "1px solid var(--grid)", marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <div><div style={{ fontSize: 11, fontWeight: 700 }}>独立 Gold 进度</div><div style={{ marginTop: 4, fontSize: 12, color: "var(--muted)" }}>{summary.reviewed} / {summary.total} 条已完成 · {progress}% · 草稿 {summary.in_progress}</div></div>
          <div style={{ width: 220, height: 8, background: "var(--grid)", borderRadius: 99, overflow: "hidden" }}><div style={{ width: `${progress}%`, height: "100%", background: "var(--ink)" }} /></div>
        </div>
      </div>

      {message && <div className="card-mono" style={{ border: "1px solid var(--grid)", marginBottom: 18, fontSize: 11 }}>{message}</div>}
      {loading && <div className="card-mono">正在读取表格标注队列…</div>}
      {!loading && !current && <div className="card-mono">没有可标注的候选。</div>}

      {!loading && current && (
        <div className="card-mono" style={{ border: "1px solid var(--grid)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 14 }}>
            <div><span className="badge-solid">{index + 1} / {rows.length}</span><div style={{ fontFamily: "monospace", fontSize: 11, marginTop: 8 }}>{current.queue_id}</div></div>
            <div style={{ fontSize: 10, color: "var(--muted)" }}>{current.review_status} · {current.reviewer || "未填写审阅者"}</div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", fontSize: 10, color: "var(--muted)", marginBottom: 14 }}>
            <span className="badge-outline">候选文件：{current.source_filing}</span><span className="badge-outline">候选页：{current.page}</span><span className="badge-dashed">候选指标：{current.metric_id}</span>
          </div>

          <div style={{ padding: "13px 15px", borderLeft: "3px solid var(--ink)", background: "rgba(28,28,26,0.035)", fontSize: 12, lineHeight: 1.7, marginBottom: 14 }}>
            <strong>原文候选证据：</strong><br />{current.evidence_sentence || "没有候选证据句，请打开 PDF 核对。"}
          </div>
          <a href={sourceUrl} target="_blank" rel="noreferrer" className="btn-outline" style={{ display: "inline-block", textDecoration: "none", marginBottom: 14, fontSize: 10 }}>打开原始 PDF · 第 {current.page} 页</a>

          <details style={{ border: "1px dashed var(--grid)", borderRadius: 10, padding: "9px 11px", marginBottom: 16 }}>
            <summary style={{ cursor: "pointer", fontSize: 10.5, fontWeight: 700 }}>系统候选值（默认不看；独立标注完成后再用于误差分析）</summary>
            <div style={{ marginTop: 8, color: "var(--muted)", fontSize: 10, lineHeight: 1.6 }}>下列内容是程序预测，不是 Gold。独立标注时请只看 PDF 和原文证据。</div>
            <pre style={{ whiteSpace: "pre-wrap", fontSize: 10, lineHeight: 1.55, marginBottom: 0 }}>{JSON.stringify({ company_id: current.company_id, fiscal_year: current.fiscal_year, metric_id: current.metric_id, value: current.value, unit: current.unit, row_label: current.row_label, column_label: current.column_label, table_name: current.table_name }, null, 2)}</pre>
          </details>

          <div style={{ border: "1px solid var(--grid)", borderRadius: 12, padding: "12px 13px", marginBottom: 14 }}>
            <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 8 }}>第一判断：该单元格是否被该页原文支持？</div>
            <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
              <button className={isSupported ? "btn-ink" : "btn-outline"} onClick={() => setSupported(true)}>被支持</button>
              <button className={isUnsupported ? "btn-ink" : "btn-outline"} onClick={() => setSupported(false)}>未被支持</button>
              <button className={gold.cell_supported === "uncertain" ? "btn-ink" : "btn-outline"} onClick={() => setSupported("uncertain")}>看不清，先存草稿</button>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10, marginBottom: 14 }}>
            {FIELD_DEFS.map(({ key, label, hint }) => (
              <label key={String(key)} style={{ fontSize: 10, color: "var(--muted)" }}>{label}
                <input className="input-mono" style={{ marginTop: 5, fontSize: 11 }} value={textValue(gold[key])} onChange={(event) => updateGold(key, event.target.value)} placeholder={hint} />
                <span style={{ display: "block", marginTop: 3, lineHeight: 1.4 }}>{hint}</span>
              </label>
            ))}
          </div>

          <label style={{ display: "block", fontSize: 10, color: "var(--muted)", marginBottom: 12 }}>原文证据句/表格单元格（必须来自 PDF）
            <textarea className="input-mono" rows={3} style={{ marginTop: 5, fontSize: 11, lineHeight: 1.6 }} value={textValue(gold.evidence_text)} onChange={(event) => updateGold("evidence_text", event.target.value)} placeholder="抄录能直接支持该单元格的行、列、表头或脚注内容" />
          </label>
          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 2fr)", gap: 12, marginBottom: 15 }}>
            <label style={{ fontSize: 10, color: "var(--muted)" }}>独立审阅者<input className="input-mono" style={{ marginTop: 5, fontSize: 11 }} value={reviewer} onChange={(event) => { setReviewer(event.target.value); setSaveState("dirty"); }} placeholder="例如 reviewer_b" /></label>
            <label style={{ fontSize: 10, color: "var(--muted)" }}>备注<textarea className="input-mono" rows={2} style={{ marginTop: 5, fontSize: 11 }} value={notes} onChange={(event) => { setNotes(event.target.value); setSaveState("dirty"); }} placeholder="例如：表头跨两行；括号表示负数；页码已核对" /></label>
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", gap: 8, flexWrap: "wrap" }}>
            <div style={{ display: "flex", gap: 7 }}><button className="btn-outline" disabled={index === 0 || saveState === "saving"} onClick={() => move(-1)}>上一条</button><button className="btn-outline" disabled={index === rows.length - 1 || saveState === "dirty" || saveState === "saving"} onClick={() => move(1)}>下一条</button></div>
            <button className="btn-ink" disabled={saveState === "saving" || saveState === "idle" || saveState === "saved"} onClick={save}>{saveState === "saving" ? "保存中…" : canComplete ? "保存并完成 Gold" : "保存草稿"}</button>
          </div>
          {!canComplete && <div style={{ marginTop: 9, fontSize: 10, color: "var(--muted)" }}>完成条件：选择“被支持”并填写全部 Gold 字段，或选择“未被支持”；两种情况都必须填写独立审阅者。看不清只能保存草稿。</div>}
        </div>
      )}
    </section>
  );
}
