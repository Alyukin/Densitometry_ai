import type { StudySummary } from "../api/types";
import { isActive } from "../utils/format";

export function StatsBar({ studies }: { studies: StudySummary[] }) {
  const total = studies.length;
  const active = studies.filter((s) => isActive(s.status)).length;
  const done = studies.filter((s) => s.status === "completed").length;
  const withIssues = studies.filter((s) => s.summary?.overall_quality === "unacceptable").length;
  const failed = studies.filter((s) => s.status === "failed").length;

  const items = [
    { label: "Исследований", value: total, tone: "" },
    { label: "В обработке", value: active, tone: active ? "info" : "" },
    { label: "Обработано", value: done, tone: "" },
    { label: "С нарушениями", value: withIssues, tone: withIssues ? "warn" : "" },
    { label: "Ошибки", value: failed, tone: failed ? "bad" : "" },
  ];

  return (
    <div className="stats">
      {items.map((i) => (
        <div key={i.label} className={`stat ${i.tone ? `stat--${i.tone}` : ""}`}>
          <div className="stat__value">{i.value}</div>
          <div className="stat__label">{i.label}</div>
        </div>
      ))}
    </div>
  );
}
