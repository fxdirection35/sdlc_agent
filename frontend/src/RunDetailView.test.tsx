import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getReview, getRun } from "./api";
import { RunDetailView } from "./RunDetailView";
import type { RunResponse } from "./types";

vi.mock("./api", () => ({
  getRun: vi.fn(),
  getReview: vi.fn(),
  postReview: vi.fn(),
  postBudget: vi.fn(),
}));

function makeRun(overrides: Partial<RunResponse> = {}): RunResponse {
  return {
    run_id: "run-7",
    status: "in_progress",
    pending: {
      kind: "gate",
      gate: "prd_review",
      prompt: "Review the prd and decide.",
      spent_usd: 0.4,
      budget_usd: 5,
      allowed_verdicts: ["approved", "rejected", "needs_human"],
      blocked_stage: "",
    },
    spent_usd: 0.4,
    budget_usd: 5,
    requirement: {
      title: "Add CSV export",
      description: "Users cannot get data out.",
      requester: "alice",
      source_repo: "https://github.com/example/reports.git",
    },
    artifacts: { prd: "11111111-2222-3333-4444-555555555555" },
    audit: [
      { stage: "pm", event: "produced", detail: "prd ready", at: "2026-10-14T10:00:00+00:00" },
    ],
    ...overrides,
  };
}

beforeEach(() => {
  vi.mocked(getRun).mockReset().mockResolvedValue(makeRun());
  // RunDetailView renders a ReviewPanel for the gate-pending run; that panel
  // fetches the artifact under review.
  vi.mocked(getReview).mockReset().mockResolvedValue({
    pending: makeRun().pending!,
    artifact_kind: "prd",
    artifact: { goals: ["Export any filtered report as CSV"] },
  });
});

describe("RunDetailView", () => {
  it("shows the requirement, the pending gate and the audit trail", async () => {
    render(<RunDetailView runId="run-7" onBack={vi.fn()} />);

    expect(await screen.findByText("Add CSV export")).toBeTruthy();
    expect(screen.getByText("等待PRD 评审")).toBeTruthy();
    expect(screen.getByText(/prd ready/)).toBeTruthy();
    expect(screen.getByText(/11111111/)).toBeTruthy(); // artifact id, short form
  });

  it("shows the budget panel when the run waits on money", async () => {
    vi.mocked(getRun).mockResolvedValue(
      makeRun({
        pending: {
          kind: "budget",
          gate: "budget_review",
          prompt: "Raise the ceiling or abort.",
          spent_usd: 6.2,
          budget_usd: null,
          allowed_verdicts: [],
          blocked_stage: "developer",
        },
      }),
    );

    render(<RunDetailView runId="run-7" onBack={vi.fn()} />);

    expect(await screen.findByRole("button", { name: "追加并继续" })).toBeTruthy();
  });

  it("shows 404 for a run this service never had", async () => {
    vi.mocked(getRun).mockRejectedValue(new Error("unknown run: run-7"));

    render(<RunDetailView runId="run-7" onBack={vi.fn()} />);

    await waitFor(() => expect(screen.getByText(/unknown run/)).toBeTruthy());
  });
});
