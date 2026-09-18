import type { StudyStatus } from "../api/types";
import { QUALITY_LABELS, STATUS_LABELS, label } from "../utils/format";

export function StatusBadge({ status, progress }: { status: StudyStatus; progress?: number }) {
  return (
    <span className={`badge badge--${status}`}>
      {(status === "processing" || status === "queued") && <span className="pulse" aria-hidden />}
      {STATUS_LABELS[status]}
      {status === "processing" && progress != null && <span className="badge__pct">{progress}%</span>}
    </span>
  );
}

export function QualityBadge({ quality }: { quality: string | null | undefined }) {
  if (!quality) return <span className="muted">—</span>;
  const tone = quality === "acceptable" ? "ok" : quality === "unacceptable" ? "bad" : "neutral";
  return <span className={`badge badge--q-${tone}`}>{label(QUALITY_LABELS, quality)}</span>;
}

export function ProgressBar({ value, status }: { value: number; status: StudyStatus }) {
  return (
    <div
      className={`progress progress--${status}`}
      role="progressbar"
      aria-valuenow={value}
      aria-valuemin={0}
      aria-valuemax={100}
    >
      <div className="progress__fill" style={{ width: `${Math.max(0, Math.min(100, value))}%` }} />
    </div>
  );
}

export function MockBadge() {
  return (
    <span className="badge badge--mock" title="Результат сформирован mock-процессором">
      MOCK
    </span>
  );
}
