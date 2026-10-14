import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

vi.mock("./api", () => ({
  listRuns: vi.fn(),
  createRun: vi.fn(),
  getRun: vi.fn(),
  getReview: vi.fn(),
  postReview: vi.fn(),
  postBudget: vi.fn(),
}));

import { getRun, listRuns } from "./api";

beforeEach(() => {
  window.location.hash = "";
  vi.mocked(listRuns).mockReset().mockResolvedValue([]);
});

describe("App routing", () => {
  it("lands on the runs list", async () => {
    render(<App />);

    expect(await screen.findByText("全部运行")).toBeTruthy();
  });

  it("navigates to a run by hash and back home again", async () => {
    vi.mocked(getRun).mockResolvedValue({
      run_id: "abc123",
      status: "completed",
      pending: null,
      spent_usd: 1,
      budget_usd: 5,
      requirement: {
        title: "A finished run",
        description: "d",
        requester: "alice",
        source_repo: "s",
      },
      artifacts: {},
      audit: [],
    });

    render(<App />);
    window.location.hash = "#/runs/abc123";
    fireEvent(window, new HashChangeEvent("hashchange"));

    expect(await screen.findByText("A finished run")).toBeTruthy();
    expect(screen.getByText("已完成")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "← 返回列表" }));
    await waitFor(() => expect(screen.getByText("全部运行")).toBeTruthy());
  });
});
