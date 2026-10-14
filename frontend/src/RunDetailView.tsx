import { useCallback, useEffect, useState } from "react";
import { getRun } from "./api";
import type { RunResponse } from "./types";
import { AuditTimeline } from "./AuditTimeline";
import { BudgetPanel } from "./BudgetPanel";
import { ReviewPanel } from "./ReviewPanel";
import { StatusBadge, WaitingAt } from "./StatusBadge";
import { useInterval } from "./useInterval";

const POLL_MS = 2500;
const KIND_LABELS: Record<string, string> = {
  prd: "PRD",
  design: "设计",
  code_change: "代码变更",
  test_plan: "测试计划",
  test_run: "测试执行",
};

function moneyBar(spent: number, budget: number) {
  const ratio = budget > 0 ? Math.min(spent / budget, 1) : 1;
  return (
    <div className="money">
      <div className="bar">
        <div className="fill" style={{ width: `${ratio * 100}%` }} />
      </div>
      <span>
        ${spent.toFixed(2)} / ${budget.toFixed(2)}
      </span>
    </div>
  );
}

/** One run: status, the decision it waits on, and how it got here. */
export function RunDetailView({ runId, onBack }: { runId: string; onBack: () => void }) {
  const [run, setRun] = useState<RunResponse | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(() => {
    getRun(runId)
      .then(setRun)
      .catch((cause: unknown) =>
        setError(cause instanceof Error ? cause.message : String(cause)),
      );
  }, [runId]);

  useEffect(refresh, [refresh]);
  // Poll only while something can still change: a pending decision waits for
  // a human, in_progress means model work between checkpoints. Terminal
  // statuses stop the polling - a finished run does not change.
  useInterval(refresh, run && (run.pending || run.status === "in_progress") ? POLL_MS : null);

  if (error) return <p className="error">{error}</p>;
  if (!run) return <p className="dim">加载中…</p>;

  return (
    <section>
      <p>
        <button className="link" onClick={onBack}>
          ← 返回列表
        </button>
      </p>
      <div className="card header">
        <h2>{run.requirement.title}</h2>
        <p className="dim">
          {run.requirement.requester} · {run.requirement.source_repo}
        </p>
        <p>{run.requirement.description}</p>
        <div className="meta">
          <StatusBadge status={run.status} />
          {run.pending && <WaitingAt kind={run.pending.kind} gate={run.pending.gate} />}
          {moneyBar(run.spent_usd, run.budget_usd)}
          <span className="dim">#{run.run_id.slice(0, 8)}</span>
        </div>
      </div>

      {run.pending?.kind === "gate" && (
        <ReviewPanel runId={runId} pending={run.pending} onDecided={setRun} />
      )}
      {run.pending?.kind === "budget" && (
        <BudgetPanel runId={runId} pending={run.pending} onDecided={setRun} />
      )}

      {Object.keys(run.artifacts).length > 0 && (
        <div className="card">
          <h2>产物</h2>
          <ul className="artifacts">
            {Object.entries(run.artifacts).map(([kind, id]) => (
              <li key={kind}>
                {KIND_LABELS[kind] ?? kind} <span className="dim">{id.slice(0, 8)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <AuditTimeline audit={run.audit} />
    </section>
  );
}
