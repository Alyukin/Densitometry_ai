import { useEffect, useRef } from "react";

/** Calls `fn` every `intervalMs` while `enabled` is true. Pauses when the tab is hidden. */
export function usePolling(fn: () => void | Promise<void>, intervalMs: number, enabled: boolean) {
  const fnRef = useRef(fn);
  useEffect(() => {
    fnRef.current = fn;
  });

  useEffect(() => {
    if (!enabled) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;

    const tick = async () => {
      if (stopped) return;
      if (document.visibilityState === "visible") {
        try {
          await fnRef.current();
        } catch {
          /* errors are surfaced by the callee */
        }
      }
      if (!stopped) timer = setTimeout(tick, intervalMs);
    };
    timer = setTimeout(tick, intervalMs);
    return () => {
      stopped = true;
      clearTimeout(timer);
    };
  }, [intervalMs, enabled]);
}
