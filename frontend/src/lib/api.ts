// Vite dev uses the /api proxy; the FastAPI-hosted production build is
// same-origin and exposes these routes at the root.
const B = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "/api" : "");

export interface CausalPath {
  path_id: string;
  nodes: string[];
  node_labels: string[];
  relationships: string[];
  causal_strengths: string[];
  evidence: string[];
  pages: number[];
  years: number[];
  evidence_ids?: string[];
  filings?: string[];
  total_hops: number;
  score: number;
  score_breakdown: Record<string, number>;
}

export interface QueryResult {
  query: string;
  intent: string;
  intent_display: string;
  answer: string;
  structured_report?: {
    format?: string;
    status?: string;
    executive_summary?: string;
    claims?: Array<{
      statement: string;
      evidence_claim_ids: string[];
      pages: number[];
      fiscal_years: number[];
      support_level?: string;
    }>;
    evidence_quality?: string;
    limitations?: string;
  };
  paths: CausalPath[];
  evidence_sentences: string[];
  metadata: {
    total_candidates: number;
    top_paths: number;
    anchors_used: string[];
    avg_score: number;
  };
}

export interface GraphStats {
  total_nodes: number;
  total_relationships: number;
  graph_nodes?: number;
  graph_relationships?: number;
  by_label: Record<string, number>;
  by_relationship: Record<string, number>;
}

export interface GNode {
  id: string;
  name: string;
  labels: string[];
}

export interface GEdge {
  source: string;
  target: string;
  type: string;
  evidence_id?: string;
}

export interface Subgraph {
  nodes: GNode[];
  edges: GEdge[];
}

export interface EvidenceItem {
  evidence: string;
  page: number;
  section: string;
  relation: string;
  evidence_id?: string;
  fiscal_year?: number;
  connected_to: string | null;
}

export interface EvidenceResult {
  entity_id: string;
  evidence: EvidenceItem[];
}

/* ── API ── */
export async function postQuery(
  q: string,
  max = 10,
  yearStart?: number,
  yearEnd?: number,
): Promise<QueryResult> {
  const body: Record<string, unknown> = { question: q, max_paths: max };
  if (yearStart !== undefined) body.year_start = yearStart;
  if (yearEnd !== undefined) body.year_end = yearEnd;
  const r = await fetch(`${B}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`Query ${r.status}`);
  return r.json();
}

export async function getStats(): Promise<GraphStats> {
  const r = await fetch(`${B}/graph/statistics`);
  if (!r.ok) throw new Error(`Stats ${r.status}`);
  return r.json();
}

export async function getSubgraph(
  entity?: string,
  limit = 120
): Promise<Subgraph> {
  const p = new URLSearchParams();
  if (entity) p.set("entity", entity);
  p.set("limit", String(limit));
  const r = await fetch(`${B}/graph/subgraph?${p}`);
  if (!r.ok) throw new Error(`Subgraph ${r.status}`);
  return r.json();
}

export async function getEvidence(
  entityId: string,
  limit = 5
): Promise<EvidenceResult> {
  const r = await fetch(`${B}/evidence/${entityId}?limit=${limit}`);
  if (!r.ok) throw new Error(`Evidence ${r.status}`);
  return r.json();
}

export interface TemporalEvent {
  target: string | null;
  relation: string;
  strength: string | null;
  year: number;
  evidence: string | null;
  page: number | null;
  filing: string | null;
  evidence_id: string | null;
}

export async function getTemporalEvolution(
  riskId: string,
  limit = 20,
): Promise<TemporalEvent[]> {
  const r = await fetch(`${B}/graph/temporal/${encodeURIComponent(riskId)}?limit=${limit}`);
  if (!r.ok) throw new Error(`Temporal ${r.status}`);
  return r.json();
}
