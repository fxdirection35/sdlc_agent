import type { PauseKind } from "./types";

const STATUS_LABELS: Record<string, string> = {
  in_progress: "进行中",
  completed: "已完成",
  rejected: "已终止",
  escalated: "已升级人工",
  budget_exceeded: "预算耗尽",
};

const STATUS_CLASSES: Record<string, string> = {
  in_progress: "badge running",
  completed: "badge ok",
  escalated: "badge warn",
  budget_exceeded: "badge warn",
  rejected: "badge warn",
};

/** The run status chip; unknown statuses still render their raw value. */
export function StatusBadge({ status }: { status: string }) {
  return (
    <span className={STATUS_CLASSES[status] ?? "badge"}>
      {STATUS_LABELS[status] ?? status}
    </span>
  );
}

const GATE_LABELS: Record<string, string> = {
  prd_review: "PRD 评审",
  design_review: "设计评审",
  merge_review: "合并评审",
  budget_review: "预算决定",
};

/** What a run waits on, or 「运行中」 when nothing (model work in flight). */
export function WaitingAt({ kind, gate }: { kind: string; gate: string }) {
  if (!kind) return <span className="dim">运行中…</span>;
  const label = GATE_LABELS[gate] ?? gate;
  return (
    <span className={kind === ("budget" satisfies PauseKind) ? "waiting budget" : "waiting gate"}>
      等待{label}
    </span>
  );
}
