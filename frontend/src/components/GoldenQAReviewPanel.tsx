import { useEffect, useMemo, useState } from "react";
import {
  GoldenQAPatch,
  GoldenQARow,
  getGoldenQA,
  patchGoldenQA,
} from "../lib/api";

type SaveState = "idle" | "dirty" | "saving" | "saved" | "error";

interface Draft {
  reference_answer: string;
  gold_evidence_ids: string[];
  gold_pages: number[];
  relevant_evidence_grades: Record<string, number>;
  answerable: boolean | null;
  requires_abstention: boolean | null;
  reviewer: string;
  review_notes: string;
}

const pageInput = (pages: number[]) => pages.join(", ");

function parsePages(value: string): number[] {
  return Array.from(
    new Set(
      value
        .split(",")
        .map((part) => Number(part.trim()))
        .filter((part) => Number.isInteger(part) && part > 0),
    ),
  );
}

function parseIds(value: string): string[] {
  return Array.from(new Set(value.split(",").map((part) => part.trim()).filter(Boolean)));
}

function draftFromRow(row: GoldenQARow): Draft {
  return {
    reference_answer: row.reference_answer || "",
    gold_evidence_ids: row.gold_evidence_ids || [],
    gold_pages: row.gold_pages || [],
    relevant_evidence_grades: row.relevant_evidence_grades || {},
    answerable: row.answerable ?? null,
    requires_abstention: row.requires_abstention ?? null,
    reviewer: row.reviewer || "",
    review_notes: row.review_notes || "",
  };
}

function CandidateEvidence({ row }: { row: GoldenQARow }) {
  const ids = row.candidate_evidence_claim_ids || [];
  const facts = row.candidate_atomic_facts || [];
  const text = facts.length ? facts : row.candidate_expected_answer ? [row.candidate_expected_answer] : [];
  return (
    <details style={{ border: "1px dashed var(--grid)", borderRadius: 12, padding: "10px 12px", marginBottom: 16 }}>
      <summary style={{ cursor: "pointer", fontSize: 11, fontWeight: 700 }}>
        候选参考（不是标准答案） · {ids.length} 个候选证据
      </summary>
      <div style={{ marginTop: 10, fontSize: 10.5, color: "var(--muted)", lineHeight: 1.65 }}>
        这些内容由程序生成，只用于帮助定位原文。不要直接复制候选答案；最终标准答案和证据必须由你核对。
      </div>
      {ids.map((id, index) => (
        <div key={id} style={{ marginTop: 10, padding: "9px 10px", background: "rgba(28,28,26,0.035)", borderLeft: "3px solid var(--grid)" }}>
          <div style={{ fontFamily: "monospace", fontSize: 10 }}>{id}</div>
          <div style={{ marginTop: 5, whiteSpace: "pre-wrap", lineHeight: 1.6 }}>{text[index] || text[0] || "候选文件未提供文本；请根据证据 ID 回到财报核对。"}</div>
        </div>
      ))}
      {!ids.length && <div style={{ marginTop: 10, whiteSpace: "pre-wrap" }}>{text.join("\n") || "没有候选证据，请将本题作为不可回答候选并核对原始财报。"}</div>}
      {row.candidate_expected_answer && (
        <div style={{ marginTop: 10, whiteSpace: "pre-wrap", color: "var(--L1)" }}>
          <strong>候选答案：</strong>{row.candidate_expected_answer}
        </div>
      )}
    </details>
  );
}

export default function GoldenQAReviewPanel() {
  const [rows, setRows] = useState<GoldenQARow[]>([]);
  const [summary, setSummary] = useState({ total: 0, reviewed: 0, pending: 0 });
  const [index, setIndex] = useState(0);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [loading, setLoading] = useState(true);
  const [saveState, setSaveState] = useState<SaveState>("idle");
  const [message, setMessage] = useState("");

  const current = rows[index];
  const candidateIds = useMemo(() => current?.candidate_evidence_claim_ids || [], [current]);
  const validForReview = !!draft
    && !!draft.reference_answer.trim()
    && !!draft.reviewer.trim()
    && typeof draft.answerable === "boolean"
    && typeof draft.requires_abstention === "boolean"
    && (draft.answerable === false || (draft.gold_evidence_ids.length > 0 && draft.gold_pages.length > 0))
    && (draft.answerable === true ? draft.requires_abstention === false : draft.requires_abstention === true)
    && (draft.answerable === false || draft.gold_evidence_ids.every((id) => [0, 1, 2].includes(draft.relevant_evidence_grades[id])));

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    getGoldenQA()
      .then((data) => {
        if (cancelled) return;
        setRows(data.rows);
        setSummary({ total: data.total, reviewed: data.reviewed, pending: data.pending });
        setMessage("");
      })
      .catch((error) => {
        if (!cancelled) setMessage(error instanceof Error ? error.message : "Golden QA 读取失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    setDraft(current ? draftFromRow(current) : null);
    setSaveState("idle");
  }, [current]);

  function update(next: Partial<Draft>) {
    setDraft((old) => old ? { ...old, ...next } : old);
    setSaveState("dirty");
    setMessage("有未保存更改");
  }

  function chooseAnswerable(value: boolean) {
    update({ answerable: value, requires_abstention: !value });
  }

  function toggleEvidence(id: string) {
    if (!draft) return;
    const exists = draft.gold_evidence_ids.includes(id);
    update({ gold_evidence_ids: exists ? draft.gold_evidence_ids.filter((item) => item !== id) : [...draft.gold_evidence_ids, id] });
  }

  function move(delta: number) {
    if (saveState === "dirty" || saveState === "saving") {
      setMessage("请先保存当前条目，再切换问题。");
      return;
    }
    setIndex((old) => Math.max(0, Math.min(rows.length - 1, old + delta)));
  }

  async function save() {
    if (!current || !draft || saveState === "saving") return;
    setSaveState("saving");
    setMessage("保存中…");
    const patch: GoldenQAPatch = {
      ...draft,
      review_status: validForReview ? "HUMAN_REVIEWED" : "IN_PROGRESS",
    };
    try {
      const data = await patchGoldenQA(current.id, patch);
      setRows((old) => old.map((row) => row.id === data.row.id ? data.row : row));
      setSummary({ total: data.total, reviewed: data.reviewed, pending: data.pending });
      setSaveState("saved");
      setMessage(validForReview ? "已保存并标记为 HUMAN_REVIEWED。" : "已保存草稿，尚未达到 Golden QA 完成条件。");
    } catch (error) {
      setSaveState("error");
      setMessage(error instanceof Error ? error.message : "Golden QA 保存失败");
    }
  }

  const progress = summary.total ? Math.round((summary.reviewed / summary.total) * 100) : 0;

  return (
    <section style={{ paddingTop: 100, paddingBottom: 40 }}>
      <div className="pagehead">
        <h1>Golden QA 人工审核</h1>
        <p>把每道题当成财报阅读理解：只判断问题、答案和证据是否匹配，不需要判断投资价值，也不需要金融专业背景。</p>
      </div>

      <div className="card-mono" style={{ border: "2px solid var(--ink)", borderLeft: "5px solid var(--ink)", marginBottom: 18, background: "rgba(28,28,26,0.045)" }}>
        <div style={{ fontSize: 11, fontWeight: 800, letterSpacing: "0.06em", textTransform: "uppercase" }}>判断标准</div>
        <div style={{ marginTop: 8, fontSize: 11.5, lineHeight: 1.7 }}>
          原文明确支持问题时选“可回答”；缺少年份、数字、前后文，或问题要求财报没有的信息时选“不可回答”，并要求系统拒答。候选答案只是提示，不能直接当作标准答案；限定词如 may/could 不能改成确定事实。
        </div>
      </div>

      <div className="card-mono" style={{ border: "1px solid var(--grid)", marginBottom: 18 }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 16, alignItems: "center", flexWrap: "wrap" }}>
          <div>
            <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>人工审核进度</div>
            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 4 }}>{summary.reviewed} / {summary.total} 条已完成 · {progress}%</div>
          </div>
          <div style={{ width: 220, height: 8, background: "var(--grid)", borderRadius: 99, overflow: "hidden" }}><div style={{ width: `${progress}%`, height: "100%", background: "var(--ink)" }} /></div>
        </div>
      </div>

      {message && <div className="card-mono" style={{ border: "1px solid var(--grid)", marginBottom: 18, fontSize: 11 }}>{message}</div>}
      {loading && <div className="card-mono" style={{ border: "1px solid var(--grid)" }}>正在读取 Golden QA…</div>}
      {!loading && !current && <div className="card-mono" style={{ border: "1px solid var(--grid)" }}>没有可审核的问题。</div>}

      {!loading && current && draft && (
        <div className="card-mono" style={{ border: "1px solid var(--grid)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginBottom: 15 }}>
            <div><span className="badge-solid">{index + 1} / {rows.length}</span><div style={{ fontFamily: "monospace", fontSize: 11, marginTop: 9 }}>{current.id}</div></div>
            <div style={{ fontSize: 10, color: "var(--muted)" }}>{current.review_status}</div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", fontSize: 10, color: "var(--muted)", marginBottom: 14 }}>
            <span className="badge-outline">{current.candidate_source_filing || "来源财报待确认"}</span>
            {(current.candidate_pages || []).map((page) => <span className="badge-outline" key={page}>候选 p. {page}</span>)}
            <span className="badge-dashed">{current.candidate_question_type || "QA"}</span>
          </div>

          <div style={{ padding: "15px 16px", borderLeft: "3px solid var(--ink)", background: "rgba(28,28,26,0.035)", fontSize: 14, lineHeight: 1.7, marginBottom: 15 }}>{current.question}</div>

          <CandidateEvidence row={current} />

          <div style={{ display: "grid", gridTemplateColumns: "repeat(2, minmax(0, 1fr))", gap: 10, marginBottom: 15 }}>
            <button className={draft.answerable === true ? "btn-ink" : "btn-outline"} aria-pressed={draft.answerable === true} onClick={() => chooseAnswerable(true)} style={{ padding: "10px 12px", textAlign: "left" }}><strong>可回答</strong><span style={{ display: "block", marginTop: 4, fontSize: 10, opacity: 0.75 }}>证据直接支持问题</span></button>
            <button className={draft.answerable === false ? "btn-ink" : "btn-outline"} aria-pressed={draft.answerable === false} onClick={() => chooseAnswerable(false)} style={{ padding: "10px 12px", textAlign: "left" }}><strong>不可回答 / 应拒答</strong><span style={{ display: "block", marginTop: 4, fontSize: 10, opacity: 0.75 }}>证据不足或问题超出材料</span></button>
          </div>

          <label style={{ display: "block", fontSize: 10, color: "var(--muted)", marginBottom: 12 }}>标准答案（不要复制候选答案）
            <textarea className="input-mono" rows={4} style={{ marginTop: 6, fontSize: 11, lineHeight: 1.6 }} placeholder={draft.answerable === false ? "根据当前证据无法回答。" : "用证据支持的事实简洁回答，并保留年份、单位和限定词。"} value={draft.reference_answer} onChange={(event) => update({ reference_answer: event.target.value })} />
          </label>

          <div style={{ border: "1px solid var(--grid)", borderRadius: 12, padding: "12px 13px", marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 6 }}>标准证据</div>
            <div style={{ fontSize: 10, color: "var(--muted)", lineHeight: 1.5, marginBottom: 9 }}>勾选真正支持答案的 EvidenceClaim ID。不可回答的问题通常不填证据 ID。</div>
            {candidateIds.map((id) => <label key={id} style={{ display: "flex", alignItems: "center", gap: 8, fontFamily: "monospace", fontSize: 10, marginTop: 7 }}><input type="checkbox" checked={draft.gold_evidence_ids.includes(id)} onChange={() => toggleEvidence(id)} />{id}</label>)}
            <input className="input-mono" style={{ marginTop: 9, fontSize: 10 }} placeholder="也可以手动填写 ID，逗号分隔" value={draft.gold_evidence_ids.filter((id) => !candidateIds.includes(id)).join(", ")} onChange={(event) => update({ gold_evidence_ids: [...candidateIds.filter((id) => draft.gold_evidence_ids.includes(id)), ...parseIds(event.target.value)] })} />
            <input className="input-mono" style={{ marginTop: 9, fontSize: 10 }} placeholder="标准页码，例如 80, 81" value={pageInput(draft.gold_pages)} onChange={(event) => update({ gold_pages: parsePages(event.target.value) })} />
          </div>

          {draft.answerable !== false && draft.gold_evidence_ids.length > 0 && <div style={{ border: "1px solid var(--grid)", borderRadius: 12, padding: "12px 13px", marginBottom: 12 }}>
            <div style={{ fontSize: 11, fontWeight: 700 }}>证据支持等级</div>
            <div style={{ fontSize: 10, color: "var(--muted)", margin: "5px 0 8px" }}>2=直接完整支持，1=部分支持，0=不相关或矛盾。</div>
            {draft.gold_evidence_ids.map((id) => <label key={id} style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8, fontFamily: "monospace", fontSize: 10, marginTop: 7 }}>{id}<select className="input-mono" style={{ width: 120, fontSize: 10 }} value={draft.relevant_evidence_grades[id] ?? ""} onChange={(event) => update({ relevant_evidence_grades: { ...draft.relevant_evidence_grades, [id]: Number(event.target.value) } })}><option value="">选择等级</option><option value="2">2 · 直接</option><option value="1">1 · 部分</option><option value="0">0 · 不支持</option></select></label>)}
          </div>}

          <div style={{ display: "grid", gridTemplateColumns: "minmax(0, 1fr) minmax(0, 2fr)", gap: 12, marginBottom: 15 }}>
            <label style={{ fontSize: 10, color: "var(--muted)" }}>审阅者<input className="input-mono" style={{ marginTop: 6, fontSize: 11 }} placeholder="例如 reviewer_01" value={draft.reviewer} onChange={(event) => update({ reviewer: event.target.value })} /></label>
            <label style={{ fontSize: 10, color: "var(--muted)" }}>备注<textarea className="input-mono" rows={2} style={{ marginTop: 6, fontSize: 11 }} placeholder="记录歧义、限定词或需要复核的地方" value={draft.review_notes} onChange={(event) => update({ review_notes: event.target.value })} /></label>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", justifyContent: "space-between" }}>
            <div style={{ display: "flex", gap: 8 }}><button className="btn-outline" onClick={() => move(-1)} disabled={index === 0}>上一条</button><button className="btn-outline" onClick={() => move(1)} disabled={index === rows.length - 1}>下一条</button></div>
            <button className="btn-ink" onClick={save} disabled={saveState === "saving"}>{validForReview ? "保存并完成" : "保存草稿"}</button>
          </div>
          {draft.answerable === null && <div style={{ marginTop: 10, fontSize: 10, color: "var(--muted)" }}>请选择“可回答”或“不可回答 / 应拒答”。</div>}
          {draft.answerable === true && !validForReview && <div style={{ marginTop: 10, fontSize: 10, color: "var(--muted)" }}>完成前还需要：标准答案、标准证据 ID、页码、所有证据等级和审阅者。</div>}
        </div>
      )}
    </section>
  );
}
