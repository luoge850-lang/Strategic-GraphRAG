import { useEffect, useRef, useCallback, useState } from "react";
import { Network } from "vis-network";
import { DataSet } from "vis-data";
import "vis-network/styles/vis-network.css";
import { nodeColor, fmtLabel } from "../lib/graph";
import type { Subgraph, GNode, GEdge } from "../lib/api";

/* ── vis-network type helpers (avoid @types/vis conflicts) ── */
interface VisNode {
  id: string;
  label: string;
  title: string;
  color: any;
  size: number;
  font: any;
  borderWidth: number;
  shape: string;
}
interface VisEdge {
  id: number;
  from: string;
  to: string;
  label: string;
  title: string;
  color: any;
  width: number;
  font: any;
  arrows: any;
  smooth: any;
}

interface Props {
  subgraph: Subgraph | null;
  hlNodes: Set<string>;
  hlEdges: Set<string>;
  dim: { w: number; h: number };
  onHoverNode: (n: GNode | null) => void;
  onSelectNode: (n: GNode | null) => void;
  selectedNode: GNode | null;
}

/* ── Degree computation ── */
function computeDegrees(nodes: GNode[], edges: GEdge[]): Map<string, number> {
  const deg = new Map<string, number>();
  nodes.forEach((n) => deg.set(n.id, 0));
  edges.forEach((e) => {
    const s = typeof e.source === "string" ? e.source : (e.source as any).id || e.source;
    const t = typeof e.target === "string" ? e.target : (e.target as any).id || e.target;
    deg.set(s, (deg.get(s) || 0) + 1);
    deg.set(t, (deg.get(t) || 0) + 1);
  });
  return deg;
}

export default function GraphCanvas({
  subgraph,
  hlNodes,
  hlEdges,
  dim,
  onHoverNode,
  onSelectNode,
}: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const networkRef = useRef<Network | null>(null);
  const nodesMapRef = useRef<Map<string, GNode>>(new Map());
  const [search, setSearch] = useState("");
  const [searchResults, setSearchResults] = useState<GNode[]>([]);
  const [statusMsg, setStatusMsg] = useState("Loading...");

  /* ── Build vis-network ── */
  useEffect(() => {
    const container = containerRef.current;
    if (!container || !subgraph?.nodes?.length) return;

    // Clear previous
    if (networkRef.current) {
      networkRef.current.destroy();
      networkRef.current = null;
    }

    const gNodes = subgraph.nodes.filter((n) => n && n.id);
    const gEdges = subgraph.edges.filter((e) => e && e.source && e.target);
    if (!gNodes.length) { setStatusMsg("No valid nodes"); return; }

    const degMap = computeDegrees(gNodes, gEdges);
    const nodeMap = new Map<string, GNode>();
    gNodes.forEach((n) => nodeMap.set(n.id, n));
    nodesMapRef.current = nodeMap;

    // Normalize highlight sets to lowercase for case-insensitive matching
    // (Query returns UPPER_CASE but subgraph stores lower_case ids)
    const toLower = (s: string) => (s || "").toLowerCase().replace(/\s+/g, "_");
    const hlNodesLower = new Set<string>();
    hlNodes.forEach((n) => { hlNodesLower.add(toLower(n)); });
    const hlEdgesLower = new Set<string>();
    hlEdges.forEach((e) => { hlEdgesLower.add(toLower(e)); });

    const hasHL = hlNodesLower.size > 0;

    // Convert to vis format with null-safe values
    const visNodes: VisNode[] = gNodes.map((n) => {
      const id = n.id || `node_${Math.random()}`;
      const d = degMap.get(id) || 0;
      const isHL = hlNodesLower.has(id.toLowerCase()) || hlNodesLower.has((n.name || "").toLowerCase());
      const size = isHL ? 28 : Math.max(10, Math.min(40, 12 + Math.sqrt(d) * 2.8));
      const c = nodeColor(n.labels || []);
      const labelText = fmtLabel(n.name || id);

      return {
        id,
        label: labelText,
        title: `<b>${labelText}</b><br>Type: ${(n.labels || []).join(", ")}<br>ID: ${id}<br>Connections: ${d}`,
        color: {
          background: hasHL && !isHL ? "rgba(180,175,169,0.35)" : c,
          border: isHL ? "#1C1C1A" : "rgba(28,28,26,0.15)",
          highlight: { background: "#1C1C1A", border: "#1C1C1A" },
          hover: { background: isHL ? "#1C1C1A" : "#4A4944", border: "#1C1C1A" },
        },
        size,
        font: {
          color: hasHL && !isHL ? "rgba(143,142,136,0.4)" : "#6A6963",
          size: isHL ? 12 : Math.max(8, Math.min(11, 8 + d * 0.15)),
          face: "Inter, sans-serif",
          strokeWidth: 0,
        },
        borderWidth: isHL ? 2.5 : 0.8,
        shape: "dot",
      };
    });

    const visEdges: VisEdge[] = [];
    gEdges.forEach((e, i) => {
      const s = typeof e.source === "string" ? e.source : (e.source as any)?.id;
      const t = typeof e.target === "string" ? e.target : (e.target as any)?.id;
      if (!s || !t) return; // skip edges with missing source/target
      const ek1 = `${s}|${t}`.toLowerCase();
      const ek2 = `${t}|${s}`.toLowerCase();
      const isHL = hlEdgesLower.has(ek1) || hlEdgesLower.has(ek2);

      visEdges.push({
        id: i,
        from: s,
        to: t,
        label: e.type || "",
        title: e.type || "",
        color: {
          color: isHL ? "#1C1C1A" : "rgba(222,221,214,0.5)",
          highlight: "#1C1C1A",
          hover: "#4A4944",
        },
        width: isHL ? 2.5 : 0.6,
        font: { color: "rgba(143,142,136,0.5)", size: 7, face: "Inter, sans-serif", strokeWidth: 0 },
        arrows: { to: { enabled: true, scaleFactor: 0.5 } },
        smooth: { enabled: true, type: "continuous", roundness: 0.5 },
      });
    });

    const data = {
      nodes: new DataSet(visNodes as any),
      edges: new DataSet(visEdges as any),
    };

    const options = {
      physics: {
        solver: "barnesHut",
        barnesHut: {
          gravitationalConstant: -2800,
          centralGravity: 0.08,
          springLength: 140,
          springConstant: 0.001,
          damping: 0.09,
        },
        stabilization: { enabled: true, iterations: 200, updateInterval: 25 },
      },
      interaction: {
        hover: true,
        tooltipDelay: 150,
        zoomView: true,
        dragView: true,
        navigationButtons: false,
      },
      edges: { smooth: { enabled: true, type: "continuous", roundness: 0.5 } },
      layout: { improvedLayout: true },
    };

    try {
      const network = new Network(container, data as any, options as any);
      networkRef.current = network;
      setStatusMsg(`${visNodes.length} nodes · ${visEdges.length} edges`);

      network.on("click", (params: any) => {
        if (params?.nodes?.length > 0) {
          const nodeId = params.nodes[0];
          const gNode = nodeMap.get(String(nodeId));
          if (gNode) onSelectNode(gNode);
        } else {
          onSelectNode(null);
        }
      });

      network.on("hoverNode", (params: any) => {
        const nodeId = params?.node;
        if (nodeId != null) {
          const gNode = nodeMap.get(String(nodeId));
          onHoverNode(gNode || null);
        }
      });
      network.on("blurNode", () => onHoverNode(null));

      network.once("stabilizationIterationsDone", () => {
        setStatusMsg(`${visNodes.length} nodes · ${visEdges.length} edges · ready`);
      });

      return () => {
        try { network.destroy(); } catch (_) { /* ok */ }
        networkRef.current = null;
      };
    } catch (err) {
      console.error("vis-network init error:", err);
      setStatusMsg("Visualization init failed");
    }
  }, [subgraph, hlNodes, hlEdges, onHoverNode, onSelectNode]);

  /* ── Focus node on search select ── */
  const focusNode = useCallback(
    (n: GNode) => {
      const net = networkRef.current;
      if (!net) return;
      try {
        net.focus(n.id, { scale: 1.8, animation: { duration: 500, easingFunction: "easeInOutQuad" } });
        net.selectNodes([n.id]);
      } catch (_) { /* ok */ }
      onSelectNode(n);
      setSearch("");
      setSearchResults([]);
    },
    [onSelectNode]
  );

  /* ── Search ── */
  const handleSearch = useCallback(
    (val: string) => {
      setSearch(val);
      if (!val.trim() || !subgraph?.nodes) {
        setSearchResults([]);
        return;
      }
      const q = val.toLowerCase();
      const results = subgraph.nodes.filter(
        (n) =>
          n.id.toLowerCase().includes(q) ||
          n.name.toLowerCase().includes(q) ||
          (n.labels || []).some((l) => l.toLowerCase().includes(q))
      );
      setSearchResults(results.slice(0, 8));
    },
    [subgraph]
  );

  /* ── Loading states ── */
  if (!subgraph) {
    return (
      <div className="fade-in" style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "var(--paper)" }}>
        <p style={{ fontSize: 12, color: "var(--muted)", fontFamily: "Inter, sans-serif" }}>Connecting to Neo4j…</p>
      </div>
    );
  }

  if (!subgraph.nodes?.length) {
    return (
      <div className="fade-in" style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%", background: "var(--paper)" }}>
        <div style={{ textAlign: "center" }}>
          <p style={{ fontSize: 12, color: "var(--muted)", fontFamily: "Inter, sans-serif" }}>No graph data</p>
          <p style={{ fontSize: 10, color: "var(--faint)", marginTop: 4 }}>Run the pipeline to populate entities</p>
        </div>
      </div>
    );
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%", background: "var(--paper)" }}>
      {/* Search input */}
      <div style={{ position: "absolute", top: 12, left: 12, zIndex: 20, display: "flex", flexDirection: "column", gap: 4 }}>
        <div style={{ position: "relative" }}>
          <input
            type="text"
            value={search}
            onChange={(e) => handleSearch(e.target.value)}
            placeholder="Search entities…"
            className="input-search"
            style={{ width: 210 }}
          />
          {search && (
            <button
              onClick={() => { setSearch(""); setSearchResults([]); }}
              style={{ position: "absolute", right: 8, top: "50%", transform: "translateY(-50%)", background: "none", border: "none", cursor: "pointer", color: "var(--muted)", fontSize: 12 }}
            >×</button>
          )}
        </div>
        {searchResults.length > 0 && (
          <div style={{ background: "rgba(240,239,235,0.96)", border: "1px solid var(--grid)", borderRadius: 14, backdropFilter: "blur(8px)", maxHeight: 220, overflowY: "auto" }}>
            {searchResults.map((n) => (
              <button
                key={n.id}
                onMouseDown={(e) => { e.stopPropagation(); focusNode(n); }}
                style={{ width: "100%", textAlign: "left", padding: "8px 14px", background: "none", border: "none", cursor: "pointer", display: "flex", alignItems: "center", gap: 8, fontFamily: "Inter, sans-serif" }}
                onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = "rgba(28,28,26,0.04)"; }}
                onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = "none"; }}
              >
                <span style={{ width: 8, height: 8, borderRadius: "50%", backgroundColor: nodeColor(n.labels || []), flexShrink: 0 }} />
                <span style={{ fontSize: 11.5, color: "var(--ink)" }}>{fmtLabel(n.name || n.id)}</span>
                <span style={{ fontSize: 9, color: "var(--muted)", marginLeft: "auto", letterSpacing: "0.04em", fontWeight: 500 }}>{(n.labels || [])[0] || ""}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      {/* vis-network container */}
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />

      {/* Status bar */}
      <div style={{ position: "absolute", bottom: 10, left: 12, zIndex: 20, fontSize: 9, fontFamily: "Inter, sans-serif", color: "var(--faint)", letterSpacing: "0.04em", background: "rgba(240,239,235,0.85)", padding: "4px 10px", borderRadius: 8, backdropFilter: "blur(4px)", pointerEvents: "none" }}>
        {statusMsg} · scroll to zoom · drag to pan · click to select
      </div>
    </div>
  );
}
