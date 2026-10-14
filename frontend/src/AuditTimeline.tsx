import type { AuditEventOut } from "./types";

/** The run's audit trail, chronological - the story of how it got here. */
export function AuditTimeline({ audit }: { audit: AuditEventOut[] }) {
  return (
    <div className="card audit">
      <h2>审计轨迹</h2>
      {audit.length === 0 && <p className="dim">还没有事件。</p>}
      <ol>
        {audit.map((event, index) => (
          <li key={index}>
            <span className="stage">{event.stage}</span>
            <span className="event">{event.event}</span>
            {event.detail && <span className="detail">{event.detail}</span>}
            <span className="dim time">{new Date(event.at).toLocaleTimeString()}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}
