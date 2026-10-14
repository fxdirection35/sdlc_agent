import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { createRun } from "./api";
import { NewRunForm } from "./NewRunForm";

vi.mock("./api", () => ({
  createRun: vi.fn(),
}));

const created = vi.fn();

function fill(label: string, value: string) {
  fireEvent.change(screen.getByLabelText(label, { exact: false }), { target: { value } });
}

beforeEach(() => {
  vi.mocked(createRun).mockReset();
  created.mockReset();
  render(<NewRunForm onCreated={created} />);
});

async function fillAll() {
  fill("标题", "Add CSV export");
  fill("需求描述", "Users cannot get data out.");
  fill("提出人", "alice");
  fill("目标仓库", "https://github.com/example/reports.git");
}

describe("NewRunForm", () => {
  it("submits the four fields and hands back the new run id", async () => {
    vi.mocked(createRun).mockResolvedValue({ run_id: "run-1" } as Awaited<ReturnType<typeof createRun>>);

    await fillAll();
    fireEvent.click(screen.getByRole("button", { name: "创建并启动" }));

    await waitFor(() => expect(created).toHaveBeenCalledWith("run-1"));
    expect(createRun).toHaveBeenCalledWith({
      title: "Add CSV export",
      description: "Users cannot get data out.",
      requester: "alice",
      source_repo: "https://github.com/example/reports.git",
    });
  });

  it("shows the server's message instead of navigating when creation fails", async () => {
    vi.mocked(createRun).mockRejectedValue(new Error("title: field required"));

    await fillAll();
    fireEvent.click(screen.getByRole("button", { name: "创建并启动" }));

    await waitFor(() => expect(screen.getByText("title: field required")).toBeTruthy());
    expect(created).not.toHaveBeenCalled();
  });
});
