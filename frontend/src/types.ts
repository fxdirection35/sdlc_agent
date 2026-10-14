/**
 * Wire types: the mirror of src/sdlc_agent/service/models.py.
 *
 * Both sides are hand-maintained on purpose. A code generator would couple
 * the frontend build to the Python package; a hand mirror keeps the two
 * buildable and deployable independently, and drift is caught the moment a
 * field the UI renders stops arriving - renderers read exact field names.
 */

export type PauseKind = "gate" | "budget";

export interface PendingDecision {
  kind: PauseKind;
  gate: string;
  prompt: string;
  spent_usd: number;
  budget_usd: number | null;
  allowed_verdicts: string[];
  blocked_stage: string;
}

export interface RunSummary {
  run_id: string;
  status: string;
  pending_kind: string;
  pending_gate: string;
  spent_usd: number;
  budget_usd: number;
  title: string;
  requester: string;
  updated_at: string;
}

export interface RequirementOut {
  title: string;
  description: string;
  requester: string;
  source_repo: string;
}

export interface AuditEventOut {
  stage: string;
  event: string;
  detail: string;
  at: string;
}

export interface RunResponse {
  run_id: string;
  status: string;
  pending: PendingDecision | null;
  spent_usd: number;
  budget_usd: number;
  requirement: RequirementOut;
  artifacts: Record<string, string>;
  audit: AuditEventOut[];
}

export interface ReviewView {
  pending: PendingDecision;
  artifact_kind: string;
  artifact: Record<string, unknown>;
}

export interface RunCreateRequest {
  title: string;
  description: string;
  requester: string;
  source_repo: string;
}

export interface ReviewDecisionRequest {
  verdict: string;
  reviewer?: string;
  comment?: string;
}
