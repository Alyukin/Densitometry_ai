import { createContext, useCallback, useContext, useMemo, useRef, useState, type ReactNode } from "react";
import { IconAlert, IconCheck, IconClose, IconInfo } from "./Icons";

type Kind = "success" | "error" | "info";
interface Toast {
  id: number;
  kind: Kind;
  text: string;
}

interface ToastApi {
  success: (text: string) => void;
  error: (text: string) => void;
  info: (text: string) => void;
}

const Ctx = createContext<ToastApi | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const seq = useRef(0);

  const dismiss = useCallback((id: number) => setItems((xs) => xs.filter((t) => t.id !== id)), []);

  const push = useCallback(
    (kind: Kind, text: string) => {
      const id = ++seq.current;
      setItems((xs) => [...xs.slice(-3), { id, kind, text }]);
      setTimeout(() => dismiss(id), kind === "error" ? 7000 : 4000);
    },
    [dismiss],
  );

  const api = useMemo<ToastApi>(
    () => ({
      success: (t) => push("success", t),
      error: (t) => push("error", t),
      info: (t) => push("info", t),
    }),
    [push],
  );

  return (
    <Ctx.Provider value={api}>
      {children}
      <div className="toasts" role="status" aria-live="polite">
        {items.map((t) => (
          <div key={t.id} className={`toast toast--${t.kind}`}>
            <span className="toast__icon">
              {t.kind === "success" ? <IconCheck /> : t.kind === "error" ? <IconAlert /> : <IconInfo />}
            </span>
            <span className="toast__text">{t.text}</span>
            <button className="icon-btn icon-btn--sm" onClick={() => dismiss(t.id)} aria-label="Закрыть">
              <IconClose size={14} />
            </button>
          </div>
        ))}
      </div>
    </Ctx.Provider>
  );
}

export function useToast(): ToastApi {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useToast must be used inside ToastProvider");
  return ctx;
}
