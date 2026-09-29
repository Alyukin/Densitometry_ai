import type { Health } from "../api/types";
import { API_BASE } from "../api/client";
import { IconExternal, Logo } from "./Icons";

export function Header({ health, healthError }: { health: Health | null; healthError: boolean }) {
  const state = healthError ? "down" : !health ? "unknown" : health.status === "ok" ? "up" : "degraded";
  const text = {
    up: "API работает",
    degraded: "API работает частично",
    down: "API не работает",
    unknown: "Проверка API…",
  }[state];

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
          <span className={`api-pill api-pill--${state}`}>
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
