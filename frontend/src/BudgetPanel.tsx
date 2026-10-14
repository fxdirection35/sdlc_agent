import { useState } from "react";
import { postBudget } from "./api";
import type { PendingDecision, RunResponse } from "./types";

/**
 * The budget gate: spent vs ceiling, the stage that got blocked, and the one
 * number a human controls - how much more to grant. Zero is a decision too
 * ("no more money") and ends the run honestly as budget_exceeded.
 */
export function BudgetPanel({
  runId,
  pending,
  onDecided,
}: {
  runId: string;
  pending: PendingDecision;
  onDecided: (run: RunResponse) => void;
}) {
  const [grant, setGrant] = useState("1.00");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function decide(grantUsd: number) {
    setBusy(true);
    setError("");
    try {
      onDecided(await postBudget(runId, grantUsd));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setBusy(false);
    }
  }

  const stage = pending.blocked_stage || "未知阶段";
  return (
    <div className="card budget">
      <h2>{pending.prompt}</h2>
      <p>
        已花费 <strong>${pending.spent_usd.toFixed(2)}</strong>，卡在阶段{" "}
        <code>{stage}</code>。
      </p>
      <div className="decision">
        <label className="inline">
          追加预算（美元）
          <input
            type="number"
            min="0"
            step="0.5"
            value={grant}
            onChange={(event) => setGrant(event.target.value)}
          />
        </label>
        <div className="verdicts">
          <button
            className="primary"
            disabled={busy}
            onClick={() => decide(Number.parseFloat(grant) || 0)}
          >
            追加并继续
          </button>
          <button className="danger" disabled={busy} onClick={() => decide(0)}>
            不追加，结束运行
          </button>
        </div>
        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}
