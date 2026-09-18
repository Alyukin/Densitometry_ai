import type { Health } from "../api/types";
import { API_BASE } from "../api/client";
import { IconExternal, Logo } from "./Icons";

export function Header({ health, healthError }: { health: Health | null; healthError: boolean }) {
  const state = healthError ? "down" : !health ? "unknown" : health.status === "ok" ? "up" : "degraded";
  const text =
    state === "down"
      ? "API недоступен"
      : state === "unknown"
        ? "Проверка API…"
        : `API ${health!.status === "ok" ? "работает" : "частично доступен"} · ${health!.processor}${
            health!.processor_version ? ` ${health!.processor_version}` : ""
          }`;

  return (
    <header className="topbar">
      <div className="topbar__inner">
        <div className="brand">
          <Logo />
          <div>
            <div className="brand__name">Densitometry AI</div>
            <div className="brand__sub">Контроль качества DXA-исследований</div>
          </div>
        </div>
        <div className="topbar__right">
          <span className={`api-pill api-pill--${state}`} title={health ? `v${health.version}` : undefined}>
            <span className="dot" />
            <span className="api-pill__text">{text}</span>
          </span>
          <a className="btn btn--ghost btn--sm" href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">
            <span>
              API<span className="hide-sm"> docs</span>
            </span>
            <IconExternal size={14} />
          </a>
        </div>
      </div>
    </header>
  );
}
