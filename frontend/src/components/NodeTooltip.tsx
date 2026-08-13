import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { getEvidence, EvidenceItem, GNode, FilingScope } from "../lib/api";
import { fmtLabel, nodeColor } from "../lib/graph";

interface Props {
  node: GNode | null;
  scope: FilingScope;
}

export default function NodeTooltip({ node, scope }: Props) {
  const [evidence, setEvidence] = useState<EvidenceItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const requestId = useRef(0);

  useEffect(() => {
    if (!node) {
      requestId.current += 1;
      setEvidence([]);
      setError("");
      return;
    }
    const currentRequest = ++requestId.current;
    setEvidence([]);
    setError("");
    setLoading(true);
    getEvidence(node.id, 5, scope)
      .then((r) => {
        if (currentRequest === requestId.current) setEvidence(r.evidence || []);
      })
      .catch(() => {
        if (currentRequest === requestId.current) setError("Evidence request failed");
      })
      .finally(() => {
        if (currentRequest === requestId.current) setLoading(false);
      });
  }, [node?.id, scope]);

  if (!node) return null;

  const color = nodeColor(node.labels || []);
  const label = (node.labels || [""])[0] || "";
  const pages = [...new Set(evidence.map((e) => e.page).filter(Boolean))];
  const sections = [
    ...new Set(evidence.map((e) => e.section).filter(Boolean)),
  ];
  const relations = [
    ...new Set(evidence.map((e) => e.relation).filter(Boolean)),
  ];

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: 4 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0 }}
        transition={{ duration: 0.25, ease: [0.2, 0.7, 0.3, 1] }}
        className="tooltip-mono"
        style={{
          position: "fixed",
          zIndex: 100,
          pointerEvents: "none",
          width: 290,
          right: 20,
          top: 120,
        }}
      >
        {/* Header — matches Streamlit popup: entity name + source provenance */}
        <div style={{ marginBottom: 10 }}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginBottom: 6,
            }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: "50%",
                backgroundColor: color,
                flexShrink: 0,
              }}
            />
            <span
              className="badge-outline"
              style={{ fontSize: 8, padding: "2px 7px" }}
            >
              {label}
            </span>
          </div>
          <p
            style={{
              fontSize: 13,
              fontWeight: 700,
              color: "var(--ink)",
              marginBottom: 2,
            }}
          >
            {fmtLabel(node.name || node.id)}
          </p>
          <p
            style={{
              fontSize: 9.5,
              color: "var(--muted)",
              fontFamily: "'Inter', sans-serif",
            }}
          >
            ID: {node.id}
          </p>

          {/* Source provenance — matches Streamlit Source: {src} Page: {pg} */}
          {!loading && evidence.length > 0 && (
            <div
              style={{
                marginTop: 8,
                paddingTop: 8,
                borderTop: "1px solid var(--grid)",
              }}
            >
              {sections.length > 0 && (
                <p
                  style={{
                    fontSize: 9,
                    fontWeight: 600,
                    color: "var(--L2)",
                    marginBottom: 3,
                    fontFamily: "'Inter', sans-serif",
                  }}
                >
                  Source: {sections[0]}
                </p>
              )}
              {pages.length > 0 && (
                <p
                  style={{
                    fontSize: 9,
                    color: "var(--muted)",
                    fontFamily: "'Inter', sans-serif",
                  }}
                >
                  Page: {pages.join(", ")}
                </p>
              )}
              {relations.length > 0 && (
                <div style={{ display: "flex", flexWrap: "wrap", gap: 4, marginTop: 4 }}>
                  {relations.map((r) => (
                    <span
                      key={r}
                      className="badge-dashed"
                      style={{ fontSize: 8, padding: "2px 6px" }}
                    >
                      {r}
                    </span>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>

        {/* Evidence excerpts */}
        {!loading && evidence.length > 0 && (
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 6,
              maxHeight: 200,
              overflowY: "auto",
            }}
          >
            <p
              style={{
                fontSize: 8.5,
                fontWeight: 600,
                letterSpacing: "0.08em",
                textTransform: "uppercase",
                color: "var(--faint)",
              }}
            >
              Evidence Snippets
            </p>
            {evidence.slice(0, 3).map((ev, i) => (
              <div
                key={i}
                style={{
                  fontSize: 9.5,
                  lineHeight: 1.5,
                  color: "var(--L2)",
                  background: "rgba(28,28,26,0.02)",
                  borderRadius: 10,
                  padding: "7px 9px",
                  border: "1px solid var(--grid)",
                }}
              >
                <p style={{ fontStyle: "italic", margin: 0 }}>
                  "{ev.evidence.length > 200
                    ? ev.evidence.slice(0, 200) + "…"
                    : ev.evidence}"
                </p>
                {ev.connected_to && (
                  <p
                    style={{
                      fontSize: 8.5,
                      color: "var(--muted)",
                      margin: "4px 0 0",
                    }}
                  >
                    → {ev.connected_to}
                  </p>
                )}
                {(ev.evidence_id || ev.fiscal_year) && (
                  <p
                    style={{
                      fontSize: 8.5,
                      color: "var(--muted)",
                      margin: "4px 0 0",
                    }}
                  >
                    {ev.evidence_id ? `Claim ${ev.evidence_id}` : ""}
                    {ev.fiscal_year ? ` · FY${ev.fiscal_year}` : ""}
                  </p>
                )}
                {ev.metric_value && (
                  <p
                    style={{
                      fontSize: 8.5,
                      color: "var(--L1)",
                      margin: "4px 0 0",
                      fontWeight: 600,
                    }}
                  >
                    Disclosed value: {ev.metric_value}
                    {ev.metric_unit ? ` · ${ev.metric_unit}` : ""}
                    {ev.metric_period ? ` · FY ${ev.metric_period}` : ""}
                  </p>
                )}
              </div>
            ))}
          </div>
        )}

        {loading && (
          <p
            style={{
              fontSize: 9.5,
              color: "var(--faint)",
              fontStyle: "italic",
              marginTop: 4,
            }}
          >
            Loading evidence…
          </p>
        )}

        {!loading && evidence.length === 0 && (
          <p
            style={{
              fontSize: 9.5,
              color: "var(--faint)",
              fontStyle: "italic",
              marginTop: 4,
            }}
          >
            {error || "No verified evidence on file"}
          </p>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
