// Vite dev uses the /api proxy; the FastAPI-hosted production build is
// same-origin and exposes these routes at the root.
const B = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "/api" : "");

async function fetchWithTimeout(input: RequestInfo | URL, init: RequestInit = {}, timeoutMs = 15000) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(input, { ...init, signal: controller.signal });
  } finally {
    window.clearTimeout(timer);
  }
}

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
  metric_value?: string;
  metric_unit?: string;
  metric_period?: string;
  metric_values_json?: string;
}

export interface EvidenceResult {
  entity_id: string;
  evidence: EvidenceItem[];
}

export type FilingScope =
  | "all"
  | "2023-10-K.pdf"
  | "2024-10-K.pdf"
  | "2025-10-K.pdf";

function applyScope(params: URLSearchParams, scope?: FilingScope) {
  if (scope === "all") params.set("cross_filing", "true");
  else if (scope) params.set("source_filing", scope);
}

/* ── API ── */
export async function postQuery(
  q: string,
  max = 10,
  yearStart?: number,
  yearEnd?: number,
  scope: FilingScope = "all",
): Promise<QueryResult> {
  const body: Record<string, unknown> = { question: q, max_paths: max };
  if (yearStart !== undefined) body.year_start = yearStart;
  if (yearEnd !== undefined) body.year_end = yearEnd;
  body.cross_filing = scope === "all";
  if (scope !== "all") body.source_filing = scope;
  const r = await fetchWithTimeout(`${B}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }, 120000);
  if (!r.ok) throw new Error(`Query ${r.status}`);
  return r.json();
}

export async function getStats(scope: FilingScope = "all"): Promise<GraphStats> {
  const p = new URLSearchParams();
  applyScope(p, scope);
  const r = await fetchWithTimeout(`${B}/graph/statistics?${p}`);
  if (!r.ok) throw new Error(`Stats ${r.status}`);
  return r.json();
}

export async function getSubgraph(
  entity?: string,
  limit = 120,
  scope: FilingScope = "all",
): Promise<Subgraph> {
  const p = new URLSearchParams();
  if (entity) p.set("entity", entity);
  p.set("limit", String(limit));
  applyScope(p, scope);
  const r = await fetchWithTimeout(`${B}/graph/subgraph?${p}`);
  if (!r.ok) throw new Error(`Subgraph ${r.status}`);
  return r.json();
}

export async function getEvidence(
  entityId: string,
  limit = 5,
  scope: FilingScope = "all",
): Promise<EvidenceResult> {
  const p = new URLSearchParams({ limit: String(limit) });
  applyScope(p, scope);
  const r = await fetchWithTimeout(`${B}/evidence/${encodeURIComponent(entityId)}?${p}`);
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
  const r = await fetchWithTimeout(`${B}/graph/temporal/${encodeURIComponent(riskId)}?limit=${limit}`);
  if (!r.ok) throw new Error(`Temporal ${r.status}`);
  return r.json();
}
