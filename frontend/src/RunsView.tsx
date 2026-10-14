import { useCallback, useEffect, useState } from "react";
import { listRuns } from "./api";
import type { RunSummary } from "./types";
import { NewRunForm } from "./NewRunForm";
import { StatusBadge, WaitingAt } from "./StatusBadge";
import { useInterval } from "./useInterval";

const POLL_MS = 2500;

function money(usd: number): string {
  return `$${usd.toFixed(2)}`;
}

/** Home: the team's runs, newest first, plus the form that starts a new one. */
export function RunsView({ onOpen }: { onOpen: (runId: string) => void }) {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState("");

  const refresh = useCallback(() => {
    listRuns()
      .then(setRuns)
      .catch((cause: unknown) =>
        setError(cause instanceof Error ? cause.message : String(cause)),
      );
  }, []);

  useEffect(refresh, [refresh]);
  useInterval(refresh, POLL_MS);

  return (
    <section>
      <NewRunForm onCreated={onOpen} />
      <div className="card">
        <h2>全部运行</h2>
        {error && <p className="error">{error}</p>}
        {runs === null && <p className="dim">加载中…</p>}
        {runs !== null && runs.length === 0 && <p className="dim">还没有运行。发起第一个需求。</p>}
        {runs !== null && runs.length > 0 && (
          <table>
            <thead>
              <tr>
                <th>标题</th>
                <th>状态</th>
                <th>等待</th>
                <th>花费 / 预算</th>
                <th>提出人</th>
                <th>更新</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id} onClick={() => onOpen(run.run_id)} className="row">
                  <td className="title">{run.title}</td>
                  <td>
                    <StatusBadge status={run.status} />
                  </td>
                  <td>
                    <WaitingAt kind={run.pending_kind} gate={run.pending_gate} />
                  </td>
                  <td className="num">
                    {money(run.spent_usd)} / {money(run.budget_usd)}
                  </td>
                  <td>{run.requester}</td>
                  <td className="dim">{new Date(run.updated_at || 0).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}
