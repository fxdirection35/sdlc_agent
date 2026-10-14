import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { listRuns } from "./api";
import { RunsView } from "./RunsView";
import type { RunSummary } from "./types";

vi.mock("./api", () => ({
  createRun: vi.fn(),
  listRuns: vi.fn(),
}));

const opened = vi.fn();

const SUMMARIES: RunSummary[] = [
  {
    run_id: "aa11",
    status: "in_progress",
    pending_kind: "gate",
    pending_gate: "prd_review",
    spent_usd: 0.5,
    budget_usd: 5,
    title: "Newer ask",
    requester: "bob",
    updated_at: "2026-10-14T10:00:00+00:00",
  },
  {
    run_id: "bb22",
    status: "completed",
    pending_kind: "",
    pending_gate: "",
    spent_usd: 3.2,
    budget_usd: 5,
    title: "Older ask",
    requester: "alice",
    updated_at: "2026-10-14T09:00:00+00:00",
  },
];

beforeEach(() => {
  vi.mocked(listRuns).mockReset().mockResolvedValue(SUMMARIES);
  opened.mockReset();
});

describe("RunsView", () => {
  it("renders one row per run with its waiting state", async () => {
    render(<RunsView onOpen={opened} />);

    expect(await screen.findByText("Newer ask")).toBeTruthy();
    expect(screen.getByText("Older ask")).toBeTruthy();
    expect(screen.getByText("等待PRD 评审")).toBeTruthy();
    expect(screen.getByText("已完成")).toBeTruthy();
    expect(screen.getByText("$0.50 / $5.00")).toBeTruthy();
  });

  it("shows the empty state when there are no runs", async () => {
    vi.mocked(listRuns).mockResolvedValue([]);
    render(<RunsView onOpen={opened} />);

    expect(await screen.findByText("还没有运行。发起第一个需求。")).toBeTruthy();
  });

  it("opens a run when its row is clicked", async () => {
    render(<RunsView onOpen={opened} />);

    fireEvent.click(await screen.findByText("Newer ask"));

    expect(opened).toHaveBeenCalledWith("aa11");
  });
});
