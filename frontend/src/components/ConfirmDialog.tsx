import { useEffect, useRef } from "react";

interface Props {
  title: string;
  text: string;
  confirmLabel: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
}

export function ConfirmDialog({ title, text, confirmLabel, danger, onConfirm, onCancel }: Props) {
  const btn = useRef<HTMLButtonElement>(null);
  useEffect(() => {
    btn.current?.focus();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onCancel();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onCancel]);

  return (
    <div className="modal-backdrop" onClick={onCancel}>
      <div className="modal" role="dialog" aria-modal="true" aria-labelledby="confirm-title" onClick={(e) => e.stopPropagation()}>
        <h3 id="confirm-title">{title}</h3>
        <p>{text}</p>
        <div className="modal__actions">
          <button className="btn btn--ghost" onClick={onCancel}>
            Отмена
          </button>
          <button ref={btn} className={`btn ${danger ? "btn--danger" : "btn--primary"}`} onClick={onConfirm}>
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
