import { useState, useEffect, useCallback, useRef, Component } from "react";
import { motion, AnimatePresence } from "motion/react";
import {
  postQuery,
  getStats,
  getSubgraph,
  QueryResult,
  GraphStats,
  Subgraph,
  CausalPath,
  GNode,
} from "./lib/api";
import { fmtLabel, nodeLabelStyle } from "./lib/graph";
import GraphCanvas from "./components/GraphCanvas";
import NodeTooltip from "./components/NodeTooltip";

/* ═══════════════════════════════════════════════════════
   Error Boundary
   ═══════════════════════════════════════════════════════ */
interface EBState {
  hasError: boolean;
  error: Error | null;
}
class ErrorBoundary extends Component<
  { children: React.ReactNode },
  EBState
> {
  state: EBState = { hasError: false, error: null };
  static getDerivedStateFromError(error: Error): EBState {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, info: any) {
    console.error("App crash:", error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            minHeight: "100vh",
            background: "var(--paper)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontFamily: "Inter, sans-serif",
          }}
        >
          <div className="card-mono" style={{ maxWidth: 520 }}>
            <h2>Application Error</h2>
            <div className="sub">{this.state.error?.message}</div>
            <pre
              style={{
                fontSize: 9.5,
                color: "var(--muted)",
                background: "rgba(28,28,26,0.03)",
                borderRadius: 12,
                padding: 12,
                overflow: "auto",
                maxHeight: 160,
                lineHeight: 1.6,
              }}
            >
              {this.state.error?.stack?.split("\n").slice(0, 6).join("\n")}
            </pre>
            <button
              className="btn-ink"
              onClick={() =>
                this.setState({ hasError: false, error: null })
              }
              style={{ marginTop: 12 }}
            >
              Retry
            </button>
            <div className="src">
              If this persists, check the browser console
            </div>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ═══════════════════════════════════════════════════════
   Example prompts
   ═══════════════════════════════════════════════════════ */

const PROMPTS = [
  "How do US export controls impact NVIDIA revenue?",
  "How does NVIDIA mitigate supply chain risks?",
  "What risks does NVIDIA face in the China market?",
  "How do supply chain disruptions affect NVIDIA margins?",
];

/* ═══════════════════════════════════════════════════════
   Collapsible Card — expand/collapse secondary content
   ═══════════════════════════════════════════════════════ */

function CollapsibleCard({
  title,
  badge,
  defaultOpen,
  children,
}: {
  title: string;
  badge?: string;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  return (
    <div className="card-mono" style={{ border: "1px solid var(--grid)", padding: "20px 24px" }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          display: "flex", alignItems: "center", justifyContent: "space-between",
          width: "100%", background: "none", border: "none", cursor: "pointer",
          padding: 0, fontFamily: "Inter, sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <h2 style={{ margin: 0, fontSize: 14 }}>{title}</h2>
          {badge && (
            <span className="badge-dashed" style={{ fontSize: 8.5, padding: "2px 8px" }}>
              {badge}
            </span>
          )}
        </div>
        <span style={{
          fontSize: 12, color: "var(--muted)",
          transform: open ? "rotate(180deg)" : "rotate(0deg)",
          transition: "transform 0.25s cubic-bezier(0.2,0.7,0.3,1)",
        }}>
          ▾
        </span>
      </button>
      {open && (
        <motion.div
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: "auto", opacity: 1 }}
          transition={{ duration: 0.35, ease: [0.2, 0.7, 0.3, 1] }}
          style={{ marginTop: 14, overflow: "hidden" }}
        >
          {children}
        </motion.div>
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════
   Stats Bar — editorial bento
   ═══════════════════════════════════════════════════════ */

function StatsBar({
  stats,
  result,
}: {
  stats: GraphStats | null;
  result: QueryResult | null;
}) {
  if (!stats) return null;
  const graphNodes = stats.graph_nodes ?? stats.total_nodes;
  const graphRelationships =
    stats.graph_relationships ?? stats.total_relationships;
  const items = [
    [graphNodes.toLocaleString(), "Graph Nodes", "#1C1C1A"],
    [graphRelationships.toLocaleString(), "Graph Edges", "#4A4944"],
    [
      String(Object.keys(stats.by_label || {}).length),
      "Entity Types",
      "#8F8E88",
    ],
    [
      result ? String(result.metadata.total_candidates) : "—",
      "Candidates",
      "#6A6963",
    ],
  ];
  return (
    <div className="grid2" style={{ gridTemplateColumns: "repeat(4,1fr)" }}>
      {items.map(([v, l, c], i) => (
        <div key={l} className="reveal" style={{ animationDelay: `${i * 60}ms` }}>
          <div
            style={{
              background: "var(--paper)",
              borderRadius: 20,
              padding: "22px 24px 18px",
              border: "1px solid var(--grid)",
            }}
          >
            <span
              style={{
                fontSize: 28,
                fontWeight: 800,
                color: c,
                fontFamily: "'Inter', sans-serif",
                letterSpacing: "-0.02em",
                lineHeight: 1,
              }}
            >
              {v}
            </span>
            <div
              style={{
                fontSize: 9.5,
                fontWeight: 500,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--muted)",
                marginTop: 6,
              }}
            >
              {l}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════
   Path Card — mono editorial
   ═══════════════════════════════════════════════════════ */

function PathCard({
  p,
  i,
  selected,
  onClick,
}: {
  p: CausalPath;
  i: number;
  selected: boolean;
  onClick: () => void;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 6 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: i * 0.04, duration: 0.5, ease: [0.2, 0.7, 0.3, 1] }}
      onClick={onClick}
      style={{
        padding: "14px 16px",
        borderRadius: 16,
        cursor: "pointer",
        background: selected
          ? "rgba(28,28,26,0.03)"
          : "transparent",
        border: selected
          ? "1.5px solid var(--L2)"
          : "1px solid var(--grid)",
        transition: "all 0.3s cubic-bezier(0.2,0.7,0.3,1)",
      }}
      onMouseEnter={(e) => {
        if (!selected)
          (e.currentTarget as HTMLElement).style.borderColor = "var(--L4)";
      }}
      onMouseLeave={(e) => {
        if (!selected)
          (e.currentTarget as HTMLElement).style.borderColor = "var(--grid)";
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
          marginBottom: 6,
        }}
      >
        <span style={{ fontSize: 9.5, fontWeight: 600, color: "var(--muted)", letterSpacing: "0.04em" }}>
          PATH {i + 1} · {p.total_hops} HOPS
        </span>
        <span
          style={{
            fontSize: 10,
            fontWeight: 700,
            fontFamily: "'Inter', sans-serif",
            color: "var(--ink)",
          }}
        >
          {p.score.toFixed(3)}
        </span>
      </div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 4, alignItems: "center" }}>
        {p.nodes.map((n, j) => (
          <span key={j} style={{ display: "flex", alignItems: "center", gap: 3 }}>
            <span className={nodeLabelStyle(p.node_labels[j] || "")} style={{ fontSize: 8.5 }}>
              {fmtLabel(n).slice(0, 24)}
            </span>
            {j < p.nodes.length - 1 && (
              <span
                style={{
                  fontSize: 7.5,
                  color: "var(--faint)",
                  fontWeight: 500,
                  padding: "0 1px",
                }}
              >
                {p.relationships[j]}
              </span>
            )}
          </span>
        ))}
      </div>
      <div style={{ display: "flex", gap: 10, marginTop: 6, fontSize: 9, color: "var(--muted)" }}>
        {p.pages.filter((pg) => pg > 0).length > 0 && (
          <span>pp. {[...new Set(p.pages.filter((pg) => pg > 0))].slice(0, 3).join(", ")}</span>
        )}
        {p.causal_strengths.filter((s) => s.includes("DIRECT")).length > 0 && (
          <span style={{ color: "var(--L1)", fontWeight: 600 }}>
            {p.causal_strengths.filter((s) => s.includes("DIRECT")).length} direct
          </span>
        )}
      </div>
    </motion.div>
  );
}

/* ═══════════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════════ */

export default function App() {
  const [q, setQ] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<QueryResult | null>(null);
  const [stats, setStats] = useState<GraphStats | null>(null);
  const [error, setError] = useState("");
  const [subgraph, setSubgraph] = useState<Subgraph | null>(null);
  const [hlNodes, setHlNodes] = useState<Set<string>>(new Set());
  const [hlEdges, setHlEdges] = useState<Set<string>>(new Set());
  const [selectedPath, setSelectedPath] = useState(0);
  const [hoveredNode, setHoveredNode] = useState<GNode | null>(null);
  const [selectedNode, setSelectedNode] = useState<GNode | null>(null);

  const graphContainerRef = useRef<HTMLDivElement>(null);
  const [graphDim, setGraphDim] = useState({ w: 900, h: 560 });

  /* ── Load data ── */
  useEffect(() => {
    getStats()
      .then(setStats)
      .catch(() => {});
    getSubgraph(undefined, 500)
      .then((d) => {
        console.log(
          "Graph:",
          d.nodes?.length,
          "nodes,",
          d.edges?.length,
          "edges"
        );
        setSubgraph(d);
      })
      .catch((e) => console.error("Subgraph error:", e));
  }, []);

  /* ── Measure graph container ── */
  useEffect(() => {
    const el = graphContainerRef.current;
    if (!el) return;
    const upd = () => {
      const r = el.getBoundingClientRect();
      setGraphDim({ w: r.width || 900, h: r.height || 560 });
    };
    upd();
    const ro = new ResizeObserver(upd);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /* ── Submit query ── */
  const submit = useCallback(async () => {
    if (!q.trim() || loading) return;
    setLoading(true);
    setError("");
    try {
      const r = await postQuery(q.trim());
      setResult(r);
      setSelectedPath(0);
      const hn = new Set<string>();
      const he = new Set<string>();
      r.paths.forEach((p) => {
        p.nodes.forEach((n) => hn.add(n));
        for (let i = 0; i < p.nodes.length - 1; i++) {
          he.add(`${p.nodes[i]}|${p.nodes[i + 1]}`);
          he.add(`${p.nodes[i + 1]}|${p.nodes[i]}`);
        }
      });
      setHlNodes(hn);
      setHlEdges(he);
      getStats()
        .then(setStats)
        .catch(() => {});
    } catch (e) {
      setError(e instanceof Error ? e.message : "Query failed");
    } finally {
      setLoading(false);
    }
  }, [q, loading]);

  const clearResults = useCallback(() => {
    setResult(null);
    setHlNodes(new Set());
    setHlEdges(new Set());
    setQ("");
    setError("");
  }, []);

  return (
    <ErrorBoundary>
      <div
        style={{
          minHeight: "100vh",
          background: "var(--paper)",
          fontFamily: "'Inter', sans-serif",
          paddingBottom: 60,
        }}
      >
        {/* ── Nav ── */}
        <nav
          style={{
            position: "fixed",
            top: 0,
            left: 0,
            right: 0,
            zIndex: 50,
            display: "flex",
            justifyContent: "center",
            paddingTop: 14,
            pointerEvents: "none",
          }}
        >
          <div
            className="nav-mono"
            style={{
              pointerEvents: "auto",
              display: "flex",
              alignItems: "center",
              gap: 14,
            }}
          >
            <span
              style={{
                width: 22,
                height: 22,
                borderRadius: 8,
                background: "var(--ink)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                flexShrink: 0,
              }}
            >
              <span
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: "var(--paper)",
                }}
              />
            </span>
            <span
              style={{
                fontSize: 13,
                fontWeight: 600,
                letterSpacing: "-0.01em",
                color: "var(--ink)",
              }}
            >
              Strategic-GraphRAG
            </span>
            <span
              style={{
                width: 1,
                height: 14,
                background: "var(--grid)",
                display: "none",
              }}
              className="nav-sep"
            />
            <span
              style={{
                fontSize: 9.5,
                fontWeight: 500,
                color: "var(--muted)",
                letterSpacing: "0.02em",
                display: "none",
              }}
              className="nav-sub"
            >
              Financial Causal Intelligence
            </span>
          </div>
        </nav>

        <main style={{ maxWidth: 1440, margin: "0 auto", padding: "0 36px" }}>
          {/* ── HERO ── */}
          <section
            className="reveal"
            style={{
              textAlign: "center",
              maxWidth: 640,
              margin: "0 auto",
              paddingTop: 100,
              paddingBottom: 40,
            }}
          >
            <h1
              style={{
                fontSize: 36,
                fontWeight: 800,
                letterSpacing: "-0.03em",
                lineHeight: 1.08,
                marginBottom: 12,
                color: "var(--ink)",
              }}
            >
              NVIDIA Financial Risk
              <br />
              <span style={{ color: "var(--L2)" }}>Causal Intelligence</span>
            </h1>
            <p
              style={{
                fontSize: 13,
                color: "var(--muted)",
                lineHeight: 1.6,
                maxWidth: 480,
                margin: "0 auto",
              }}
            >
              Temporal causal knowledge graph with evidence provenance.
              Every relationship traced to SEC filings, every claim verifiable.
            </p>
          </section>

          {/* ── STATS ── */}
          <section style={{ marginBottom: 36 }}>
            <StatsBar stats={stats} result={result} />
          </section>

          {/* ── GRAPH ── */}
          <section style={{ marginBottom: 36 }}>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "baseline",
                marginBottom: 12,
                padding: "0 4px",
              }}
            >
              <div>
                <h2
                  style={{
                    fontSize: 16.5,
                    fontWeight: 700,
                    letterSpacing: "-0.02em",
                    color: "var(--ink)",
                    margin: 0,
                  }}
                >
                  The knowledge graph, in ink
                </h2>
                <div className="sub" style={{ marginBottom: 0, marginTop: 2 }}>
                  {simRef(subgraph)} entities mapped from SEC filings · each
                  node is a financial fact
                  {result && (
                    <span style={{ color: "var(--L1)", fontWeight: 600 }}>
                      {" "}
                      · {result.metadata.total_candidates} candidates
                    </span>
                  )}
                </div>
              </div>
              <div
                style={{ display: "flex", alignItems: "center", gap: 12, fontSize: 9.5, color: "var(--muted)" }}
              >
                {result && (
                  <>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <span
                        style={{
                          width: 7,
                          height: 7,
                          borderRadius: "50%",
                          border: "2px solid var(--ink)",
                          flexShrink: 0,
                        }}
                      />
                      Causal path
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: 4 }}>
                      <span
                        style={{
                          width: 5,
                          height: 5,
                          borderRadius: "50%",
                          background: "var(--L4)",
                          flexShrink: 0,
                        }}
                      />
                      Graph node
                    </span>
                  </>
                )}
              </div>
            </div>

            <div className="card-graph reveal-card" style={{ animationDelay: "80ms" }}>
              <div
                ref={graphContainerRef}
                className="card-graph-inner"
                style={{ height: 560, position: "relative" }}
              >
                <GraphCanvas
                  subgraph={subgraph}
                  hlNodes={hlNodes}
                  hlEdges={hlEdges}
                  dim={graphDim}
                  onHoverNode={setHoveredNode}
                  onSelectNode={setSelectedNode}
                  selectedNode={selectedNode}
                />
                <NodeTooltip node={hoveredNode || selectedNode} />
              </div>
            </div>
            <div className="src" style={{ marginTop: 8, paddingLeft: 4 }}>
              KNOWLEDGE GRAPH · SEC FILINGS · NVIDIA 2025 10-K · {subgraph?.nodes?.length || 0} NODES
            </div>
          </section>

          {/* ── QUERY ── */}
          <section style={{ marginBottom: 36 }}>
            <div className="card-mono" style={{ border: "1px solid var(--grid)" }}>
              <h2>Ask a financial risk question</h2>
              <div className="sub">
                Natural language queries resolved against the causal graph ·
                every answer backed by evidence
              </div>

              <div style={{ position: "relative", marginBottom: 12 }}>
                <textarea
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      submit();
                    }
                  }}
                  placeholder="How does NVIDIA mitigate supply chain risks?"
                  rows={2}
                  className="input-mono"
                  disabled={loading}
                  style={{ paddingRight: 110 }}
                />
                <button
                  onClick={submit}
                  disabled={!q.trim() || loading}
                  className="btn-ink"
                  style={{
                    position: "absolute",
                    right: 8,
                    bottom: 8,
                    padding: "8px 18px",
                    fontSize: 11.5,
                  }}
                >
                  {loading ? "Analyzing…" : "Analyze →"}
                </button>
              </div>

              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {PROMPTS.map((p) => (
                  <button
                    key={p}
                    onClick={() => setQ(p)}
                    disabled={loading}
                    className="btn-ghost"
                    style={{ fontSize: 10.5 }}
                  >
                    {p}
                  </button>
                ))}
              </div>

              {result && (
                <button
                  onClick={clearResults}
                  className="btn-ghost"
                  style={{ marginTop: 8, fontSize: 10 }}
                >
                  Clear results
                </button>
              )}
            </div>

            {error && (
              <div
                style={{
                  marginTop: 12,
                  padding: "12px 16px",
                  borderRadius: 14,
                  background: "rgba(28,28,26,0.03)",
                  border: "1px solid var(--grid)",
                  fontSize: 11.5,
                  color: "var(--L1)",
                }}
              >
                {error}
              </div>
            )}

            {loading && (
              <div className="card-mono" style={{ marginTop: 14, border: "1px solid var(--grid)" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, opacity: 0.5 }}>
                  <div style={{ height: 8, width: "40%", borderRadius: 4, background: "var(--grid)" }} />
                  <div style={{ height: 80, borderRadius: 12, background: "var(--grid)" }} />
                </div>
              </div>
            )}
          </section>

          {/* ── RESULTS ── */}
          <AnimatePresence>
            {result && !loading && (
              <motion.div
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 12 }}
                transition={{ duration: 0.5, ease: [0.2, 0.7, 0.3, 1] }}
                style={{ display: "flex", flexDirection: "column", gap: 20 }}
              >
                {/* ══════════════════════════════════════════════
                    PRIMARY: Synthesis Report — the official answer
                    ══════════════════════════════════════════════ */}
                {result.answer && (
                  <motion.div
                    className="card-mono reveal-card"
                    style={{ border: "1.5px solid var(--L3)", background: "var(--paper)" }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
                      <span style={{
                        display: "inline-block",
                        fontSize: 9,
                        fontWeight: 700,
                        letterSpacing: "0.1em",
                        textTransform: "uppercase",
                        color: "var(--paper)",
                        background: "var(--ink)",
                        padding: "4px 12px",
                        borderRadius: 99,
                      }}>
                        Analysis
                      </span>
                      <span style={{ fontSize: 10, color: "var(--muted)", fontWeight: 500 }}>
                        {result.intent_display} · {result.metadata.total_candidates} candidates · avg {result.metadata.avg_score.toFixed(2)}
                      </span>
                    </div>

                    <div style={{
                      fontSize: 13,
                      lineHeight: 1.8,
                      color: "var(--L0)",
                      maxWidth: 860,
                    }}>
                      {result.answer.split("\n").map((line, i) => {
                        if (line.startsWith("## "))
                          return (
                            <h3 key={i} style={{
                              fontSize: 17, fontWeight: 700, color: "var(--ink)",
                              marginTop: 20, marginBottom: 8,
                              letterSpacing: "-0.01em",
                            }}>
                              {line.replace("## ", "")}
                            </h3>
                          );
                        if (line.startsWith("> "))
                          return (
                            <blockquote key={i} style={{
                              borderLeft: "3px solid var(--L3)", paddingLeft: 16, margin: "10px 0",
                              fontStyle: "italic", color: "var(--L2)", fontSize: 12.5,
                            }}>
                              {line.replace("> ", "")}
                            </blockquote>
                          );
                        if (line.startsWith("- "))
                          return (
                            <li key={i} style={{ color: "var(--L1)", marginLeft: 18, fontSize: 12.5 }}>
                              {line.replace("- ", "")}
                            </li>
                          );
                        if (line.trim())
                          return (
                            <p key={i} style={{ marginBottom: 6 }}>
                              {line}
                            </p>
                          );
                        return <br key={i} />;
                      })}
                    </div>
                    <div className="src" style={{ marginTop: 16 }}>
                      SYNTHESIS REPORT · {result.intent_display} · NVIDIA 2025 10-K
                    </div>
                  </motion.div>
                )}

                {/* ══════════════════════════════════════════════
                    SECONDARY: Paths + Evidence + Logic (collapsible grid)
                    ══════════════════════════════════════════════ */}
                {result.structured_report?.claims && result.structured_report.claims.length > 0 && (
                  <div className="card-mono reveal-card" style={{ border: "1px solid var(--grid)", background: "var(--paper)" }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 10 }}>
                      <span style={{ fontSize: 10, fontWeight: 700, letterSpacing: "0.08em", textTransform: "uppercase" }}>
                        EvidenceClaim citations
                      </span>
                      <span style={{ fontSize: 9, color: "var(--muted)" }}>
                        {result.structured_report.status || "STRUCTURED"} · {result.structured_report.claims.length} claims
                      </span>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                      {result.structured_report.claims.slice(0, 8).map((claim, i) => (
                        <div key={`${claim.evidence_claim_ids.join("-")}-${i}`} style={{ padding: "8px 10px", borderLeft: "2px solid var(--L3)", background: "rgba(28,28,26,0.02)" }}>
                          <div style={{ fontSize: 11, lineHeight: 1.5 }}>{claim.statement}</div>
                          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 4, fontSize: 9, color: "var(--muted)" }}>
                            <span>Claim {claim.evidence_claim_ids.join(", ") || "?"}</span>
                            <span>p.{claim.pages.join(", ") || "?"}</span>
                            <span>{claim.support_level || "LIMITED"}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="grid2">
                  {/* Left column: Causal Paths */}
                  <CollapsibleCard title="Causal paths" badge={`${result.paths.length} of ${result.metadata.total_candidates}`} defaultOpen={true}>
                    <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 360, overflowY: "auto" }}>
                      {result.paths.map((p, i) => (
                        <PathCard
                          key={p.path_id}
                          p={p}
                          i={i}
                          selected={selectedPath === i}
                          onClick={() => setSelectedPath(i)}
                        />
                      ))}
                    </div>
                  </CollapsibleCard>

                  {/* Right column: Evidence + Logic */}
                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <CollapsibleCard title={`Evidence chain · path ${selectedPath + 1}`} badge={`${result.paths[selectedPath]?.total_hops || 0} hops`} defaultOpen={true}>
                      {result.paths[selectedPath] && (
                        <div style={{ display: "flex", flexDirection: "column", gap: 6, maxHeight: 320, overflowY: "auto" }}>
                          {result.paths[selectedPath].evidence.map((ev, i) =>
                            ev && ev.length > 15 && (
                              <motion.div
                                key={i}
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: i * 0.03 }}
                                style={{
                                  padding: "10px 12px", borderRadius: 12,
                                  border: "1px solid var(--grid)",
                                  background: "rgba(28,28,26,0.01)",
                                }}
                              >
                                <span style={{ fontSize: 9, fontWeight: 700, color: "var(--L2)", marginRight: 6 }}>
                                  [{i + 1}]
                                </span>
                                <span style={{ fontSize: 10.5, color: "var(--L1)", lineHeight: 1.5 }}>
                                  {ev.length > 280 ? ev.slice(0, 280) + "…" : ev}
                                </span>
                                <div style={{ display: "flex", gap: 12, marginTop: 4, fontSize: 9, color: "var(--muted)", fontWeight: 500 }}>
                                  <span>p.{result.paths[selectedPath].pages[i] || "?"}</span>
                                  <span>{result.paths[selectedPath].causal_strengths[i] || ""}</span>
                                  <span>Y{result.paths[selectedPath].years[i] || "?"}</span>
                                  <span>Claim {result.paths[selectedPath].evidence_ids?.[i] || "?"}</span>
                                </div>
                              </motion.div>
                            )
                          )}
                        </div>
                      )}
                    </CollapsibleCard>

                    <CollapsibleCard title="Retrieval logic" badge={result.intent_display} defaultOpen={true}>
                      <div style={{ display: "flex", flexDirection: "column", gap: 5, fontSize: 10 }}>
                        {[
                          ["Intent", result.intent_display],
                          ["Candidates", String(result.metadata.total_candidates)],
                          ["Avg score", result.metadata.avg_score.toFixed(3)],
                          ["Anchors", result.metadata.anchors_used.slice(0, 5).join(", ")],
                        ].map(([k, v]) => (
                          <div key={k} style={{
                            display: "flex", justifyContent: "space-between",
                            padding: "5px 0", borderBottom: "1px solid var(--grid)",
                          }}>
                            <span style={{ color: "var(--muted)" }}>{k}</span>
                            <span style={{ fontWeight: 500, textAlign: "right", maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v}</span>
                          </div>
                        ))}
                      </div>
                    </CollapsibleCard>
                  </div>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </main>

        {/* ── Footer ── */}
        <footer
          style={{
            borderTop: "1px solid var(--grid)",
            marginTop: 60,
            paddingTop: 24,
            paddingBottom: 32,
            textAlign: "center",
          }}
        >
          <p
            style={{
              fontSize: 9.5,
              fontWeight: 500,
              letterSpacing: "0.08em",
              color: "var(--faint)",
              textTransform: "uppercase",
            }}
          >
            Strategic-GraphRAG v2 · Single-PDF Stable Candidate · Temporal Causal Knowledge Graph for
            Financial Risk Inference
          </p>
        </footer>
      </div>
    </ErrorBoundary>
  );
}

/* helper */
function simRef(sg: Subgraph | null) {
  if (!sg) return "—";
  return `${sg.nodes.length} nodes, ${sg.edges.length} edges`;
}
