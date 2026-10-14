import type {
  ReviewDecisionRequest,
  ReviewView,
  RunCreateRequest,
  RunResponse,
  RunSummary,
} from "./types";

/**
 * The typed client for the FastAPI service. Same-origin only: in dev, Vite
 * proxies /api to the service; in a built deployment the service serves the
 * built files itself. VITE_API_BASE exists for the odd case of a split
 * deployment, not as an everyday knob.
 */
const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
  }
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    // The service's error body is {"detail": ...}; keep whatever it says,
    // because for 400/409 the detail IS the actionable message.
    const body = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new ApiError(response.status, body?.detail ?? `${response.status} ${path}`);
  }
  return (await response.json()) as T;
}

export function listRuns(): Promise<RunSummary[]> {
  return call<RunSummary[]>("/api/runs");
}

export function createRun(request: RunCreateRequest): Promise<RunResponse> {
  return call<RunResponse>("/api/runs", { method: "POST", body: JSON.stringify(request) });
}

export function getRun(runId: string): Promise<RunResponse> {
  return call<RunResponse>(`/api/runs/${runId}`);
}

export function getReview(runId: string): Promise<ReviewView> {
  return call<ReviewView>(`/api/runs/${runId}/review`);
}

export function postReview(
  runId: string,
  request: ReviewDecisionRequest,
): Promise<RunResponse> {
  return call<RunResponse>(`/api/runs/${runId}/review`, {
    method: "POST",
    body: JSON.stringify(request),
  });
}

export function postBudget(runId: string, grantUsd: number): Promise<RunResponse> {
  return call<RunResponse>(`/api/runs/${runId}/budget`, {
    method: "POST",
    body: JSON.stringify({ grant_usd: grantUsd }),
  });
}
