import type { FilingScope, QueryResult } from "./api";

const KEY = "strategic-graphrag-query-session-v1";
type QuerySession = { question: string; scope: FilingScope; result: QueryResult | null };
const scopes = new Set(["all", "2023-10-K.pdf", "2024-10-K.pdf", "2025-10-K.pdf", "TSMC-2023-Annual-Report.pdf", "TSMC-2024-Annual-Report.pdf"]);

export function readQuerySession(): QuerySession {
  try {
    const value = JSON.parse(sessionStorage.getItem(KEY) || "null");
    if (value && typeof value.question === "string" && scopes.has(value.scope)) {
      const result = value.result;
      return { question: value.question, scope: value.scope, result: result && typeof result.query === "string" && Array.isArray(result.paths) && result.metadata?.build_id ? result : null };
    }
  } catch { /* Storage may be disabled or contain an obsolete session. */ }
  return { question: "", scope: "all", result: null };
}

export function writeQuerySession(value: QuerySession): void {
  try { sessionStorage.setItem(KEY, JSON.stringify(value)); }
  catch { /* A quota/storage failure must not turn a successful query into an error. */ }
}
