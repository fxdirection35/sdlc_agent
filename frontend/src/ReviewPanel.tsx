import { useEffect, useState } from "react";
import { getReview, postReview } from "./api";
import type { PendingDecision, ReviewDecisionRequest, RunResponse } from "./types";

const VERDICT_LABELS: Record<string, string> = {
  approved: "通过",
  rejected: "驳回修改",
  needs_human: "转人工",
};

/**
 * A gate decision in context: the artifact under discussion, rendered as
 * formatted JSON - the same document the graph validated - plus the verdict
 * buttons. reviewer/comment are optional; the service omits blanks.
 */
export function ReviewPanel({
  runId,
  pending,
  onDecided,
}: {
  runId: string;
  pending: PendingDecision;
  onDecided: (run: RunResponse) => void;
}) {
  const [artifactJson, setArtifactJson] = useState("");
  const [reviewer, setReviewer] = useState("");
  const [comment, setComment] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    getReview(runId)
      .then((view) => setArtifactJson(JSON.stringify(view.artifact, null, 2)))
      .catch((cause: unknown) =>
        setError(cause instanceof Error ? cause.message : String(cause)),
      );
  }, [runId]);

  async function decide(verdict: string) {
    setBusy(true);
    setError("");
    // Blank optional fields are omitted, not sent empty: the graph defaults a
    // missing reviewer to "unspecified", and an empty string would overwrite
    // that default with nothing.
    const payload: ReviewDecisionRequest = { verdict };
    if (reviewer) payload.reviewer = reviewer;
    if (comment) payload.comment = comment;
    try {
      onDecided(await postReview(runId, payload));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setBusy(false);
    }
  }

  return (
    <div className="card review">
      <h2>{pending.prompt}</h2>
      <pre className="artifact">{artifactJson || "加载中…"}</pre>
      <div className="decision">
        <label className="inline">
          评审人
          <input
            value={reviewer}
            onChange={(event) => setReviewer(event.target.value)}
            placeholder="留空则记录为 unspecified"
          />
        </label>
        <label>
          意见
          <textarea
            value={comment}
            onChange={(event) => setComment(event.target.value)}
            rows={2}
            placeholder="驳回时必填更有用：这条意见会原样交给上一角色"
          />
        </label>
        <div className="verdicts">
          {pending.allowed_verdicts.map((verdict) => (
            <button
              key={verdict}
              className={verdict === "approved" ? "primary" : verdict === "rejected" ? "danger" : ""}
              disabled={busy}
              onClick={() => decide(verdict)}
            >
              {VERDICT_LABELS[verdict] ?? verdict}
            </button>
          ))}
        </div>
        {error && <p className="error">{error}</p>}
      </div>
    </div>
  );
}
