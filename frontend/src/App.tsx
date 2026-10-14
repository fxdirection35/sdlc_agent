import { useEffect, useState } from "react";
import { RunDetailView } from "./RunDetailView";
import { RunsView } from "./RunsView";

/**
 * Hash routing by hand: two pages do not justify a router dependency, and a
 * run's URL (#/runs/<id>) stays shareable - a reviewer can paste it to a
 * teammate, which is exactly how gates get answered in a small team.
 */
function readHash(): { page: "home" } | { page: "run"; runId: string } {
  const match = window.location.hash.match(/^#\/runs\/([0-9a-f]+)$/);
  return match ? { page: "run", runId: match[1]! } : { page: "home" };
}

export default function App() {
  const [route, setRoute] = useState(readHash);

  useEffect(() => {
    const onChange = () => setRoute(readHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return (
    <main>
      <h1>
        SDLC Agent <span className="dim">· 需求到合并的人工门禁</span>
      </h1>
      {route.page === "home" ? (
        <RunsView onOpen={(runId) => (window.location.hash = `#/runs/${runId}`)} />
      ) : (
        <RunDetailView
          runId={route.runId}
          onBack={() => (window.location.hash = "")}
        />
      )}
    </main>
  );
}
