import { useState } from "react";
import { createRun } from "./api";
import type { RunCreateRequest } from "./types";

/** The entry form: the four fields a Requirement demands, nothing optional. */
export function NewRunForm({ onCreated }: { onCreated: (runId: string) => void }) {
  const [form, setForm] = useState<RunCreateRequest>({
    title: "",
    description: "",
    requester: "",
    source_repo: "",
  });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const set =
    (field: keyof RunCreateRequest) =>
    (event: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
      setForm((current) => ({ ...current, [field]: event.target.value }));
    };

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const run = await createRun(form);
      onCreated(run.run_id);
    } catch (cause) {
      // Validation (422) and service errors both land here; the message is
      // the server's, which says exactly which field failed.
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="card new-run" onSubmit={submit}>
      <h2>发起需求</h2>
      <label>
        标题
        <input
          value={form.title}
          onChange={set("title")}
          required
          maxLength={200}
          placeholder="一句话说明要做什么"
        />
      </label>
      <label>
        需求描述
        <textarea
          value={form.description}
          onChange={set("description")}
          required
          rows={4}
          placeholder="按你的原话描述；后续所有角色都以此为准"
        />
      </label>
      <label>
        提出人
        <input value={form.requester} onChange={set("requester")} required placeholder="你的名字" />
      </label>
      <label>
        目标仓库
        <input
          value={form.source_repo}
          onChange={set("source_repo")}
          required
          placeholder="https://github.com/org/repo.git"
        />
      </label>
      {error && <p className="error">{error}</p>}
      <button type="submit" disabled={busy}>
        {busy ? "创建中…" : "创建并启动"}
      </button>
    </form>
  );
}
