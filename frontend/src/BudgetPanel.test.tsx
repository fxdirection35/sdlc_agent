import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { postBudget } from "./api";
import { BudgetPanel } from "./BudgetPanel";
import type { PendingDecision, RunResponse } from "./types";

vi.mock("./api", () => ({
  postBudget: vi.fn(),
}));

const PENDING: PendingDecision = {
  kind: "budget",
  gate: "budget_review",
  prompt: "Run exceeded its budget. Raise the ceiling or abort.",
  spent_usd: 6.1,
  budget_usd: null,
  allowed_verdicts: [],
  blocked_stage: "developer",
};

const decided = vi.fn();

function renderPanel() {
  decided.mockReset();
  vi.mocked(postBudget).mockReset();
  render(<BudgetPanel runId="run-9" pending={PENDING} onDecided={decided} />);
}

describe("BudgetPanel", () => {
  it("names the blocked stage and the spend", () => {
    renderPanel();

    expect(screen.getByText(/developer/)).toBeTruthy();
    expect(screen.getByText("$6.10")).toBeTruthy();
  });

  it("grants the entered amount and continues", async () => {
    renderPanel();
    vi.mocked(postBudget).mockResolvedValue({ run_id: "run-9" } as RunResponse);

    fireEvent.change(screen.getByLabelText(/追加预算/, { exact: false }), {
      target: { value: "2.5" },
    });
    fireEvent.click(screen.getByRole("button", { name: "追加并继续" }));

    await waitFor(() => expect(postBudget).toHaveBeenCalledWith("run-9", 2.5));
    expect(decided).toHaveBeenCalled();
  });

  it("refusing grants zero, which ends the run", async () => {
    renderPanel();
    vi.mocked(postBudget).mockResolvedValue({ run_id: "run-9" } as RunResponse);

    fireEvent.click(screen.getByRole("button", { name: "不追加，结束运行" }));

    await waitFor(() => expect(postBudget).toHaveBeenCalledWith("run-9", 0));
  });
});
