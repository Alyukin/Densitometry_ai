import type { ExportFormat, StudyStatus, StudySummary } from "../api/types";
import { REGION_SHORT, STATUS_LABELS, formatDate, isActive, label, pluralRu } from "../utils/format";
import { ProgressBar, QualityBadge, StatusBadge } from "./Badges";
import { IconDownload, IconPlay, IconRefresh, IconTrash } from "./Icons";

export type Filter = "all" | StudyStatus;

interface Props {
  studies: StudySummary[];
  loading: boolean;
  filter: Filter;
  onFilter: (f: Filter) => void;
  selected: Set<string>;
  onSelectedChange: (next: Set<string>) => void;
  activeId: string | null;
  onOpen: (id: string) => void;
  onProcess: (id: string) => void;
  onDelete: (s: StudySummary) => void;
  onDownload: (id: string, format: ExportFormat) => void;
  onBatchProcess: (ids: string[] | null) => void;
  onBatchDownload: (format: ExportFormat, ids: string[] | null) => void;
  onRefresh: () => void;
  busy: Set<string>;
}

const FILTERS: Filter[] = ["all", "uploaded", "queued", "processing", "completed", "failed"];

export function StudiesTable(p: Props) {
  const counts = FILTERS.reduce<Record<string, number>>((acc, f) => {
    acc[f] = f === "all" ? p.studies.length : p.studies.filter((s) => s.status === f).length;
    return acc;
  }, {});
  const visible = p.filter === "all" ? p.studies : p.studies.filter((s) => s.status === p.filter);
  const selectedVisible = visible.filter((s) => p.selected.has(s.id));
  const allChecked = visible.length > 0 && selectedVisible.length === visible.length;
  const selectedIds = [...p.selected];
  const pendingCount = p.studies.filter((s) => s.status === "uploaded" || s.status === "failed").length;
  const hasResults = p.studies.some((s) => (s.summary?.total ?? 0) > 0 && !isActive(s.status));
  const selectedWithResults = p.studies.filter(
    (s) => p.selected.has(s.id) && (s.summary?.total ?? 0) > 0 && !isActive(s.status),
  );

  const toggleAll = () => {
    const next = new Set(p.selected);
    if (allChecked) visible.forEach((s) => next.delete(s.id));
    else visible.forEach((s) => next.add(s.id));
    p.onSelectedChange(next);
  };

  const toggle = (id: string) => {
    const next = new Set(p.selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    p.onSelectedChange(next);
  };

  return (
    <section className="card studies">
      <div className="card__head card__head--wrap">
        <div>
          <h2>Исследования</h2>
          <p className="muted">Выберите исследования для пакетной обработки или выгрузки результатов.</p>
        </div>
        <div className="toolbar">
          <button className="icon-btn" onClick={p.onRefresh} title="Обновить" aria-label="Обновить">
            <IconRefresh size={16} />
          </button>
          {selectedIds.length > 0 ? (
            <>
              <button className="btn btn--primary btn--sm" onClick={() => p.onBatchProcess(selectedIds)}>
                <IconPlay size={14} /> Обработать выбранные ({selectedIds.length})
              </button>
              <div className="btn-group">
                <button
                  className="btn btn--ghost btn--sm"
                  disabled={!selectedWithResults.length}
                  onClick={() => p.onBatchDownload("csv", selectedWithResults.map((s) => s.id))}
                >
                  <IconDownload size={14} /> CSV
                </button>
                <button
                  className="btn btn--ghost btn--sm"
                  disabled={!selectedWithResults.length}
                  onClick={() => p.onBatchDownload("xlsx", selectedWithResults.map((s) => s.id))}
                >
                  XLSX
                </button>
              </div>
              <button className="link" onClick={() => p.onSelectedChange(new Set())}>
                Снять выбор
              </button>
            </>
          ) : (
            <>
              <button
                className="btn btn--primary btn--sm"
                disabled={!pendingCount}
                onClick={() => p.onBatchProcess(null)}
                title="Обработать все исследования со статусом «Загружено» или «Ошибка»"
              >
                <IconPlay size={14} /> Обработать все новые{pendingCount ? ` (${pendingCount})` : ""}
              </button>
              <div className="btn-group">
                <button
                  className="btn btn--ghost btn--sm"
                  disabled={!hasResults}
                  onClick={() => p.onBatchDownload("csv", null)}
                  title="Сводный файл по всем обработанным исследованиям"
                >
                  <IconDownload size={14} /> Все CSV
                </button>
                <button className="btn btn--ghost btn--sm" disabled={!hasResults} onClick={() => p.onBatchDownload("xlsx", null)}>
                  XLSX
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      <div className="tabs" role="tablist">
        {FILTERS.map((f) => (
          <button
            key={f}
            role="tab"
            aria-selected={p.filter === f}
            className={`tab ${p.filter === f ? "tab--active" : ""}`}
            onClick={() => p.onFilter(f)}
          >
            {f === "all" ? "Все" : STATUS_LABELS[f]}
            <span className="tab__count">{counts[f]}</span>
          </button>
        ))}
      </div>

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th className="col-check">
                <input type="checkbox" checked={allChecked} onChange={toggleAll} aria-label="Выбрать все" />
              </th>
              <th>Исследование</th>
              <th className="num">Изобр.</th>
              <th>Область</th>
              <th>Статус</th>
              <th>Качество</th>
              <th className="col-actions">Действия</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((s) => {
              const active = isActive(s.status);
              const hasRes = (s.summary?.total ?? 0) > 0 && !active;
              const isBusy = p.busy.has(s.id);
              return (
                <tr
                  key={s.id}
                  className={`${p.activeId === s.id ? "row--active" : ""} ${p.selected.has(s.id) ? "row--selected" : ""}`}
                  onClick={() => p.onOpen(s.id)}
                >
                  <td className="col-check" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="checkbox"
                      checked={p.selected.has(s.id)}
                      onChange={() => toggle(s.id)}
                      aria-label={`Выбрать ${s.name}`}
                    />
                  </td>
                  <td>
                    <div
                      className="cell-title"
                      title={`${s.source_path}\nStudy UID: ${s.study_instance_uid ?? "—"}`}
                    >
                      {s.name}
                    </div>
                    <div className="cell-sub nowrap">
                      {[s.modality, formatDate(s.created_at)].filter(Boolean).join(" · ")}
                    </div>
                  </td>
                  <td className="num">{s.image_count}</td>
                  <td>
                    {s.summary?.regions.length ? (
                      <div className="chips chips--nowrap">
                        {s.summary.regions.map((r) => (
                          <span key={r} className="chip">
                            {label(REGION_SHORT, r)}
                          </span>
                        ))}
                      </div>
                    ) : (
                      <span className="muted">—</span>
                    )}
                  </td>
                  <td>
                    <div className="status-cell">
                      <StatusBadge status={s.status} progress={s.progress} />
                      {active && <ProgressBar value={s.progress} status={s.status} />}
                      {s.status === "completed" && s.summary && s.summary.errors > 0 && (
                        <span className="cell-sub warn-text">
                          {s.summary.errors} {pluralRu(s.summary.errors, "ошибка", "ошибки", "ошибок")}
                        </span>
                      )}
                      {s.status === "completed" && s.summary && s.summary.non_standard > 0 && (
                        <span className="cell-sub warn-text">{s.summary.non_standard} вне задачи</span>
                      )}
                    </div>
                  </td>
                  <td>
                    <div className="quality-cell">
                      <QualityBadge quality={s.summary?.overall_quality} />
                    </div>
                  </td>
                  <td className="col-actions" onClick={(e) => e.stopPropagation()}>
                    <div className="row-actions">
                      <button
                        className="icon-btn icon-btn--accent"
                        disabled={active || isBusy}
                        onClick={() => p.onProcess(s.id)}
                        title={s.status === "uploaded" ? "Запустить обработку" : "Обработать повторно"}
                        aria-label={s.status === "uploaded" ? "Запустить обработку" : "Обработать повторно"}
                      >
                        {s.status === "uploaded" ? <IconPlay size={15} /> : <IconRefresh size={15} />}
                      </button>
                      <div className="btn-group">
                        <button
                          className="btn btn--xs btn--ghost"
                          disabled={!hasRes}
                          onClick={() => p.onDownload(s.id, "csv")}
                          title="Скачать CSV"
                        >
                          CSV
                        </button>
                        <button
                          className="btn btn--xs btn--ghost"
                          disabled={!hasRes}
                          onClick={() => p.onDownload(s.id, "xlsx")}
                          title="Скачать XLSX"
                        >
                          XLSX
                        </button>
                      </div>
                      <button
                        className="icon-btn icon-btn--danger"
                        disabled={s.status === "processing" || isBusy}
                        onClick={() => p.onDelete(s)}
                        title="Удалить"
                        aria-label="Удалить"
                      >
                        <IconTrash size={15} />
                      </button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {!visible.length && (
          <div className="empty">
            {p.loading
              ? "Загрузка…"
              : p.studies.length
                ? "Нет исследований с таким статусом"
                : "Пока нет загруженных исследований. Загрузите DICOM-файлы выше."}
          </div>
        )}
      </div>
    </section>
  );
}
