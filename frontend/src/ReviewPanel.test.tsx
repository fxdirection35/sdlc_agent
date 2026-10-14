import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { getReview, postReview } from "./api";
import { ReviewPanel } from "./ReviewPanel";
import type { PendingDecision, RunResponse } from "./types";

vi.mock("./api", () => ({
  getReview: vi.fn(),
  postReview: vi.fn(),
}));

const PENDING: PendingDecision = {
  kind: "gate",
  gate: "prd_review",
  prompt: "Review the prd and decide.",
  spent_usd: 0.4,
  budget_usd: 5,
  allowed_verdicts: ["approved", "rejected", "needs_human"],
  blocked_stage: "",
};

const decided = vi.fn();

beforeEach(() => {
  vi.mocked(getReview).mockReset().mockResolvedValue({
    pending: PENDING,
    artifact_kind: "prd",
    artifact: { goals: ["Export any filtered report as CSV"], non_goals: ["Excel"] },
  });
  vi.mocked(postReview).mockReset();
  decided.mockReset();
  render(<ReviewPanel runId="run-1" pending={PENDING} onDecided={decided} />);
});

describe("ReviewPanel", () => {
  it("renders the artifact under review as JSON", async () => {
    expect(
      await screen.findByText(/Export any filtered report as CSV/),
    ).toBeTruthy();
  });

  it("sends the verdict with reviewer and comment", async () => {
    vi.mocked(postReview).mockResolvedValue({ run_id: "run-1", status: "in_progress" } as RunResponse);

    fireEvent.change(await screen.findByLabelText("评审人", { exact: false }), {
      target: { value: "alice" },
    });
    fireEvent.change(screen.getByLabelText("意见", { exact: false }), {
      target: { value: "goals are testable" },
    });
    fireEvent.click(screen.getByRole("button", { name: "通过" }));

    await waitFor(() =>
      expect(postReview).toHaveBeenCalledWith("run-1", {
        verdict: "approved",
        reviewer: "alice",
        comment: "goals are testable",
      }),
    );
    expect(decided).toHaveBeenCalled();
  });

  it("sends a bare verdict when the optional fields are blank", async () => {
    vi.mocked(postReview).mockResolvedValue({ run_id: "run-1" } as RunResponse);

    fireEvent.click(await screen.findByRole("button", { name: "驳回修改" }));

    await waitFor(() =>
      expect(postReview).toHaveBeenCalledWith("run-1", { verdict: "rejected" }),
    );
  });

  it("shows the server's refusal and stays answerable", async () => {
    vi.mocked(postReview).mockRejectedValue(
      new Error("gate 'prd_review' got verdict 'maybe'; expected one of: ..."),
    );

    fireEvent.click(await screen.findByRole("button", { name: "通过" }));

    await waitFor(() =>
      expect(screen.getByText(/expected one of/)).toBeTruthy(),
    );
    expect(decided).not.toHaveBeenCalled();
    // The panel is still up: the buttons remain clickable for a corrected try.
    expect(screen.getByRole("button", { name: "通过" })).toBeTruthy();
  });
});
