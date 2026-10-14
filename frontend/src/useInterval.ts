import { useEffect, useRef } from "react";

/**
 * Run ``callback`` every ``ms``; pass ``null`` to pause (a finished run stops
 * polling without unmounting). The latest callback is always the one that
 * fires, so a stale closure can never resurrect polling that was paused.
 */
export function useInterval(callback: () => void, ms: number | null): void {
  const ref = useRef(callback);
  useEffect(() => {
    ref.current = callback;
  }, [callback]);
  useEffect(() => {
    if (ms === null) return;
    const timer = setInterval(() => ref.current(), ms);
    return () => clearInterval(timer);
  }, [ms]);
}
