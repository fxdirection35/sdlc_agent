import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, createRun, listRuns, postBudget, postReview } from "./api";

function stubFetch(status: number, body: unknown) {
  const mock = vi.fn().mockResolvedValue(
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", mock);
  return mock;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api client", () => {
  it("lists runs with a GET", async () => {
    const fetchMock = stubFetch(200, []);
    const runs = await listRuns();
    expect(runs).toEqual([]);
    expect(fetchMock).toHaveBeenCalledWith("/api/runs", expect.anything());
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(init.method).toBeUndefined();
  });

  it("creates a run with a POST and a JSON body", async () => {
    const fetchMock = stubFetch(201, { run_id: "abc" });
    await createRun({ title: "t", description: "d", requester: "r", source_repo: "s" });
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      title: "t",
      description: "d",
      requester: "r",
      source_repo: "s",
    });
  });

  it("sends a review with its optional fields", async () => {
    const fetchMock = stubFetch(200, { run_id: "abc" });
    await postReview("abc", { verdict: "rejected", reviewer: "bob", comment: "thin goals" });
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/runs/abc/review");
    expect(JSON.parse(init.body as string)).toEqual({
      verdict: "rejected",
      reviewer: "bob",
      comment: "thin goals",
    });
  });

  it("sends the budget grant as grant_usd", async () => {
    const fetchMock = stubFetch(200, { run_id: "abc" });
    await postBudget("abc", 2.5);
    const init = fetchMock.mock.calls[0]![1] as RequestInit;
    expect(fetchMock.mock.calls[0]![0]).toBe("/api/runs/abc/budget");
    expect(JSON.parse(init.body as string)).toEqual({ grant_usd: 2.5 });
  });

  it("raises ApiError carrying the server's detail", async () => {
    stubFetch(409, { detail: "run abc is not waiting for a decision" });
    const error = await listRuns().catch((cause: unknown) => cause);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(409);
    expect((error as ApiError).message).toContain("not waiting");
  });

  it("survives a non-JSON error body", async () => {
    stubFetch(502, "bad gateway");
    const error = await listRuns().catch((cause: unknown) => cause);
    expect((error as ApiError).message).toContain("502");
  });
});
