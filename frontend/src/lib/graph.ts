import {
  forceSimulation,
  forceLink,
  forceManyBody,
  forceCenter,
  forceCollide,
} from "d3-force";
import type { GNode, GEdge } from "./api";

/* ═══════════════════════════════════════════════════════
   Lieflat Charts · Mono Ladder for Entity Types
   L0 (blackest) → most structurally important
   L6 (lightest) → auxiliary/context entities
   ═══════════════════════════════════════════════════════ */

const NODE_R = 5.5;
const NODE_R_HL = 8;
const LABEL_CUTOFF = 4;
const MAX_LABEL_LEN = 18;

export type SimNode = GNode & {
  x: number;
  y: number;
  degree: number;
  fx?: number;
  fy?: number;
};
export type SimEdge = {
  source: string;
  target: string;
  type: string;
  idx: number;
};
export type SimResult = {
  nodes: SimNode[];
  edges: SimEdge[];
};

/* ── Mono Ladder Colors ── */
const LADDER: Record<string, { fill: string; stroke: string }> = {
  company:    { fill: "#1C1C1A", stroke: "#1C1C1A" },  // L0 — ink
  risk:       { fill: "#4A4944", stroke: "#4A4944" },  // L1
  strategy:   { fill: "#6A6963", stroke: "#6A6963" },  // L2
  regulation: { fill: "#8F8E88", stroke: "#8F8E88" },  // L3
  mechanism:  { fill: "#4A4944", stroke: "#4A4944" },  // L1 (structural, same as risk)
  product:    { fill: "#B0AFA9", stroke: "#B0AFA9" },  // L4
  market:     { fill: "#C6C5BF", stroke: "#C6C5BF" },  // L5
  metric:     { fill: "#D8D7D1", stroke: "#C6C5BF" },  // L6
  region:     { fill: "#D8D7D1", stroke: "#D8D7D1" },
  event:      { fill: "#6A6963", stroke: "#6A6963" },
  year:       { fill: "#D8D7D1", stroke: "#D8D7D1" },
  document:   { fill: "#D8D7D1", stroke: "#D8D7D1" },
};

export function nodeColor(labels: string[]): string {
  const l = (labels[0] || "").toLowerCase();
  for (const [k, v] of Object.entries(LADDER))
    if (l.includes(k)) return v.fill;
  return "#C6C5BF"; // default faint
}

export function nodeStroke(labels: string[]): string {
  const l = (labels[0] || "").toLowerCase();
  for (const [k, v] of Object.entries(LADDER))
    if (l.includes(k)) return v.stroke;
  return "#C6C5BF";
}

/* Label styling for UI badges */
export function nodeLabelStyle(label: string): string {
  const l = (label || "").toLowerCase();
  if (l.includes("company")) return "badge-solid";
  if (l.includes("risk")) return "badge-outline";
  if (l.includes("strategy")) return "badge-outline";
  if (l.includes("regulation")) return "badge-dashed";
  if (l.includes("product")) return "badge-dashed";
  if (l.includes("market")) return "badge-dashed";
  if (l.includes("metric")) return "badge-dashed";
  if (l.includes("mechanism")) return "badge-outline";
  return "badge-dashed";
}

export function fmtLabel(s: string) {
  return (s || "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .slice(0, MAX_LABEL_LEN);
}

/* ═══════════════════════════════════════════════════════
   Force Simulation
   ═══════════════════════════════════════════════════════ */

export function runSim(
  nodes: GNode[],
  edges: GEdge[],
  w: number,
  h: number
): SimResult {
  const deg = new Map<string, number>();
  nodes.forEach((n) => deg.set(n.id, 0));
  edges.forEach((e) => {
    const s = typeof e.source === "string" ? e.source : (e.source as any).id;
    const t = typeof e.target === "string" ? e.target : (e.target as any).id;
    deg.set(s, (deg.get(s) || 0) + 1);
    deg.set(t, (deg.get(t) || 0) + 1);
  });

  const simNodes: SimNode[] = nodes.map((n) => ({
    ...n,
    x: w / 2 + (Math.random() - 0.5) * 60,
    y: h / 2 + (Math.random() - 0.5) * 60,
    degree: deg.get(n.id) || 0,
  }));

  const simEdges: SimEdge[] = edges.map((e, i) => {
    const s = typeof e.source === "string" ? e.source : (e.source as any).id;
    const t = typeof e.target === "string" ? e.target : (e.target as any).id;
    return { source: s, target: t, type: e.type, idx: i };
  });

  const sim = forceSimulation(simNodes as any)
    .force(
      "link",
      forceLink(simEdges)
        .id((d: any) => d.id)
        .distance(50)
        .strength(0.35)
    )
    .force(
      "charge",
      forceManyBody().strength((d: any) => -100 - (d.degree || 0) * 3)
    )
    .force("center", forceCenter(w / 2, h / 2))
    .force(
      "collide",
      forceCollide((d: any) => NODE_R + 3 + Math.min((d.degree || 0) * 0.5, 7))
    )
    .stop();

  for (let i = 0; i < 300; i++) sim.tick();

  simNodes.forEach((n) => {
    if (n.x < 16) n.x = 16;
    if (n.x > w - 16) n.x = w - 16;
    if (n.y < 16) n.y = 16;
    if (n.y > h - 16) n.y = h - 16;
  });

  return { nodes: simNodes, edges: simEdges };
}

/* ═══════════════════════════════════════════════════════
   Canvas Rendering — Mono Style
   No glow. No shadow. Texture from line weight + opacity.
   ═══════════════════════════════════════════════════════ */

export interface RenderCtx {
  ctx: CanvasRenderingContext2D;
  w: number;
  h: number;
  transform: { x: number; y: number; k: number };
  sim: SimResult;
  hlNodes: Set<string>;
  hlEdges: Set<string>;
  hoveredId: string | null;
}

function buildLookup(nodes: SimNode[]): Map<string, SimNode> {
  const m = new Map<string, SimNode>();
  nodes.forEach((n) => m.set(n.id, n));
  return m;
}

export function renderGraph(rc: RenderCtx) {
  const { ctx, w, h, transform, sim, hlNodes, hlEdges, hoveredId } = rc;
  const { nodes, edges } = sim;
  const hasHL = hlNodes.size > 0;
  const lookup = buildLookup(nodes);

  // Paper background
  ctx.fillStyle = "#F0EFEB";
  ctx.fillRect(0, 0, w, h);

  ctx.save();
  ctx.translate(transform.x, transform.y);
  ctx.scale(transform.k, transform.k);

  const dimBG = hasHL;

  // ── Edges ──
  for (let i = 0; i < edges.length; i++) {
    const e = edges[i];
    const s = lookup.get(e.source as string);
    const t = lookup.get(e.target as string);
    if (!s || !t) continue;
    const ek = `${s.id}|${t.id}`;
    const isHL = hlEdges.has(ek) || hoveredId === s.id || hoveredId === t.id;
    if (dimBG && !isHL) continue;
    ctx.beginPath();
    ctx.moveTo(s.x, s.y);
    ctx.lineTo(t.x, t.y);
    if (isHL) {
      ctx.strokeStyle = "#1C1C1A";
      ctx.lineWidth = 1.6;
    } else {
      ctx.strokeStyle = "#DEDDD6";
      ctx.lineWidth = 0.45;
    }
    ctx.stroke();
  }

  // ── Hovered-node edges (drawn over grid) ──
  if (hoveredId && !hasHL) {
    for (let i = 0; i < edges.length; i++) {
      const e = edges[i];
      const s = lookup.get(e.source as string);
      const t = lookup.get(e.target as string);
      if (!s || !t) continue;
      if (s.id !== hoveredId && t.id !== hoveredId) continue;
      ctx.beginPath();
      ctx.moveTo(s.x, s.y);
      ctx.lineTo(t.x, t.y);
      ctx.strokeStyle = "#4A4944";
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // ── Nodes ──
  for (let i = 0; i < nodes.length; i++) {
    const n = nodes[i];
    const isHL =
      hlNodes.has(n.id) || hlNodes.has(n.name) || hoveredId === n.id;
    const fill = nodeColor(n.labels || []);
    const stroke = nodeStroke(n.labels || []);

    if (dimBG && !isHL) {
      // Dim background — faint dots
      ctx.beginPath();
      ctx.arc(n.x, n.y, NODE_R * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = "#DEDDD6";
      ctx.fill();
      continue;
    }

    const r = isHL ? NODE_R_HL : NODE_R;

    // Hover ring (subtle, no glow)
    if (hoveredId === n.id) {
      ctx.beginPath();
      ctx.setLineDash([2, 2]);
      ctx.arc(n.x, n.y, r + 5, 0, Math.PI * 2);
      ctx.strokeStyle = "#8F8E88";
      ctx.lineWidth = 0.8;
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Highlight ring
    if (hlNodes.has(n.id) || hlNodes.has(n.name)) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, r + 4, 0, Math.PI * 2);
      ctx.strokeStyle = "#1C1C1A";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Node circle
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, Math.PI * 2);
    ctx.fillStyle = isHL ? "#1C1C1A" : fill;
    ctx.globalAlpha = dimBG ? 0.4 : isHL ? 1 : 0.82;
    ctx.fill();
    ctx.globalAlpha = 1;

    // Thin stroke on each node for definition
    ctx.strokeStyle = "#F0EFEB";
    ctx.lineWidth = 0.6;
    ctx.stroke();

    // Label
    if (n.degree >= LABEL_CUTOFF || isHL || transform.k > 1.5) {
      const lbl = fmtLabel(n.name || n.id);
      const fs = isHL ? 8.5 : 7;
      const weight = isHL ? "600" : "400";
      ctx.font = `${weight} ${fs}px "Inter", sans-serif`;
      ctx.fillStyle = isHL ? "#1C1C1A" : transform.k > 1.5 ? "#6A6963" : "#8F8E88";
      ctx.textAlign = "center";
      ctx.fillText(lbl, n.x, n.y + r + 9);
    }
  }

  ctx.restore();
}

/* ── Hit test ── */
export function hitTest(
  sim: SimResult,
  mx: number,
  my: number,
  transform: { x: number; y: number; k: number }
): SimNode | null {
  const { nodes } = sim;
  const hitR = NODE_R + 6;
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    const sx = n.x * transform.k + transform.x;
    const sy = n.y * transform.k + transform.y;
    const dx = mx - sx;
    const dy = my - sy;
    if (dx * dx + dy * dy < hitR * hitR * transform.k) return n;
  }
  return null;
}
