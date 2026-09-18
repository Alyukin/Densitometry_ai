import { useCallback, useEffect, useState } from "react";
import { ApiError, api } from "../api/client";
import type { CheckDetail, ExportFormat, ResultRow, StudyDetail, StudyResult, StudySummary } from "../api/types";
import {
  REGION_LABELS,
  VIOLATION_LABELS,
  formatBytes,
  formatDate,
  formatSeconds,
  isActive,
  label,
  splitViolations,
} from "../utils/format";
import { MockBadge, ProgressBar, QualityBadge, StatusBadge } from "./Badges";
import { IconAlert, IconCheck, IconClose, IconDownload, IconPlay, IconRefresh, IconTrash } from "./Icons";

interface Props {
  studyId: string;
  summary: StudySummary | undefined;
  onClose: () => void;
  onProcess: (id: string) => void;
  onDownload: (id: string, format: ExportFormat) => void;
  onDelete: (s: StudySummary) => void;
}

const ROTATION_LABELS: Record<string, string> = {
  correct: "корректно",
  over_rotation: "переротация",
  under_rotation: "недоротация",
};
const MARGIN_LABELS: Record<string, string> = { top_cm: "верх", bottom_cm: "низ", left_cm: "лево", right_cm: "право" };

function formatCheckValue(c: CheckDetail): string | null {
  const v = c.value;
  if (v == null) return null;
  if (typeof v === "number") return `${v}${c.unit === "deg" ? "°" : c.unit ? ` ${c.unit}` : ""}`;
  if (typeof v === "string") return ROTATION_LABELS[v] ?? v;
  if (typeof v === "object") {
    return Object.entries(v as Record<string, unknown>)
      .map(([k, val]) => `${MARGIN_LABELS[k] ?? k} ${val}`)
      .join(" · ")
      .concat(c.unit ? ` ${c.unit === "cm" ? "см" : c.unit}` : "");
  }
  return String(v);
}

function ResultCard({ row, preview }: { row: ResultRow; preview: string | null }) {
  const violations = splitViolations(row.violation_type);
  const ok = row.processing_status === "success";
  const checks = row.details?.checks ?? [];
  return (
    <article className={`result ${ok ? "" : "result--error"}`}>
      <div className="result__media">
        {preview ? (
          <img src={api.assetUrl(preview)} alt={row.original_filename ?? "preview"} loading="lazy" />
        ) : (
          <div className="result__noimg">нет изображения</div>
        )}
      </div>
      <div className="result__body">
        <div className="result__head">
          <div>
            <div className="result__title">{ok ? label(REGION_LABELS, row.anatomical_region) : "Ошибка обработки"}</div>
            <div className="cell-sub mono" title={row.image_uid ?? ""}>
              {row.original_filename ?? row.image_uid}
            </div>
          </div>
          {ok ? <QualityBadge quality={row.quality_class} /> : <span className="badge badge--failed">{row.processing_status === "timeout" ? "Таймаут" : "Ошибка"}</span>}
        </div>

        {!ok && row.error_message && (
          <div className="alert alert--error">
            <IconAlert size={14} /> {row.error_message}
          </div>
        )}

        {violations.length > 0 && (
          <div className="chips">
            {violations.map((v) => (
              <span key={v} className="chip chip--bad" title={v}>
                {label(VIOLATION_LABELS, v)}
              </span>
            ))}
          </div>
        )}

        {checks.length > 0 && (
          <ul className="checks">
            {checks.map((c) => {
              const val = formatCheckValue(c);
              return (
                <li key={c.code} className={c.passed ? "check--ok" : "check--bad"}>
                  <span className="check__icon">{c.passed ? <IconCheck size={13} /> : <IconClose size={13} />}</span>
                  <span className="check__title">{c.title}</span>
                  {val && <span className="check__val">{val}</span>}
                </li>
              );
            })}
          </ul>
        )}

        <div className="result__meta">
          {row.confidence != null && <span>Уверенность: {(row.confidence * 100).toFixed(0)}%</span>}
          <span>Время: {formatSeconds(row.time_of_processing)}</span>
        </div>
      </div>
    </article>
  );
}

export function StudyDrawer({ studyId, summary, onClose, onProcess, onDownload, onDelete }: Props) {
  const [detail, setDetail] = useState<StudyDetail | null>(null);
  const [result, setResult] = useState<StudyResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const status = summary?.status ?? detail?.status;
  const updatedAt = summary?.updated_at;

  const load = useCallback(async () => {
    try {
      const d = await api.getStudy(studyId);
      setDetail(d);
      setError(null);
      if (!isActive(d.status) && (d.summary?.total ?? 0) > 0) {
        setResult(await api.getResult(studyId));
      } else {
        setResult(null);
      }
    } catch (e) {
      setError((e as ApiError).message);
    }
  }, [studyId]);

  useEffect(() => {
    setDetail(null);
    setResult(null);
  }, [studyId]);

  // reload when the study changes (list polling updates `updated_at`)
  useEffect(() => {
    void load();
  }, [load, updatedAt]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const d = detail;
  const active = status ? isActive(status) : false;
  const progress = summary?.progress ?? d?.progress ?? 0;
  const previews = new Map(d?.images.map((i) => [i.id, i.preview_url]) ?? []);
  const hasResult = !!result && result.rows.length > 0;

  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" aria-label="Карточка исследования">
        <div className="drawer__head">
          <div className="drawer__title">
            <h3 title={d?.source_path}>{d?.name ?? summary?.name ?? "Исследование"}</h3>
            {d?.source_path && d.source_path !== d.name && <div className="cell-sub mono">{d.source_path}</div>}
          </div>
          <button className="icon-btn" onClick={onClose} aria-label="Закрыть">
            <IconClose />
          </button>
        </div>

        <div className="drawer__body">
          {error && (
            <div className="alert alert--error">
              <IconAlert size={14} /> {error}
            </div>
          )}
          {!d && !error && <div className="empty">Загрузка…</div>}

          {d && status && (
            <>
              <section className="panel">
                <div className="panel__row">
                  <StatusBadge status={status} progress={progress} />
                  {result?.is_mock && <MockBadge />}
                  <div className="spacer" />
                  <button className="btn btn--primary btn--sm" disabled={active} onClick={() => onProcess(d.id)}>
                    {d.status === "uploaded" ? <IconPlay size={14} /> : <IconRefresh size={14} />}
                    {d.status === "uploaded" ? "Запустить обработку" : "Обработать повторно"}
                  </button>
                </div>
                {(active || status === "completed") && <ProgressBar value={active ? progress : 100} status={status} />}
                <dl className="kv kv--grid">
                  <dt>Загружено</dt>
                  <dd>{formatDate(d.created_at)}</dd>
                  <dt>Старт</dt>
                  <dd>{formatDate(d.started_at)}</dd>
                  <dt>Завершено</dt>
                  <dd>{formatDate(d.finished_at)}</dd>
                  <dt>Время обработки</dt>
                  <dd>{formatSeconds(d.processing_time_sec)}</dd>
                </dl>
                {d.error_message && !active && (
                  <div className={`alert ${status === "failed" ? "alert--error" : "alert--warn"}`}>
                    <IconAlert size={14} /> {d.error_message}
                  </div>
                )}
              </section>

              {result?.is_mock && (
                <div className="alert alert--info">
                  <span>
                    Результат сформирован <b>mock-процессором</b> ({result.processor_version}) — это тестовые данные,
                    не медицинское заключение.
                  </span>
                </div>
              )}

              {hasResult && result && (
                <section className="panel">
                  <div className="panel__row">
                    <h4>Результат</h4>
                    <div className="spacer" />
                    <button className="btn btn--ghost btn--sm" onClick={() => onDownload(d.id, "csv")}>
                      <IconDownload size={14} /> CSV
                    </button>
                    <button className="btn btn--ghost btn--sm" onClick={() => onDownload(d.id, "xlsx")}>
                      <IconDownload size={14} /> XLSX
                    </button>
                  </div>
                  <div className="summary-row">
                    <div>
                      <b>{result.summary.total}</b> изобр.
                    </div>
                    <div className="ok-text">
                      <b>{result.summary.acceptable}</b> качественных
                    </div>
                    <div className="bad-text">
                      <b>{result.summary.unacceptable}</b> с нарушениями
                    </div>
                    {result.summary.errors > 0 && (
                      <div className="warn-text">
                        <b>{result.summary.errors}</b> ошибок
                      </div>
                    )}
                  </div>
                  <div className="results">
                    {result.rows.map((r, i) => (
                      <ResultCard key={`${r.image_id}-${i}`} row={r} preview={previews.get(r.image_id ?? "") ?? null} />
                    ))}
                  </div>
                </section>
              )}

              {!hasResult && (
                <section className="panel">
                  <h4>Изображения ({d.images.length})</h4>
                  <div className="thumbs">
                    {d.images.map((img) => (
                      <figure key={img.id} className="thumb">
                        {img.preview_url ? (
                          <img src={api.assetUrl(img.preview_url)} alt={img.original_filename} loading="lazy" />
                        ) : (
                          <div className="result__noimg">нет пикселей</div>
                        )}
                        <figcaption>
                          <div className="cell-title" title={img.original_filename}>
                            {img.original_filename.split("/").pop()}
                          </div>
                          <div className="cell-sub">
                            {img.body_part_examined ?? "—"} · {img.columns ?? "?"}×{img.rows ?? "?"} ·{" "}
                            {formatBytes(img.size_bytes)}
                          </div>
                        </figcaption>
                      </figure>
                    ))}
                  </div>
                </section>
              )}

              <section className="panel">
                <h4>Метаданные</h4>
                <dl className="kv">
                  <dt>Study UID</dt>
                  <dd className="mono break">{d.study_instance_uid ?? "—"}</dd>
                  <dt>Модальность</dt>
                  <dd>{d.modality ?? "—"}</dd>
                  <dt>Производитель</dt>
                  <dd>{d.manufacturer ?? "—"}</dd>
                  <dt>Процессор</dt>
                  <dd>{d.processor_name ? `${d.processor_name} (${d.processor_version})` : "—"}</dd>
                  <dt>ID</dt>
                  <dd className="mono">{d.id}</dd>
                </dl>
                {d.warnings.map((w) => (
                  <div key={w} className="alert alert--warn">
                    <IconAlert size={14} /> {w}
                  </div>
                ))}
              </section>

              <div className="drawer__footer">
                <button
                  className="btn btn--danger-ghost btn--sm"
                  disabled={status === "processing"}
                  onClick={() => summary && onDelete(summary)}
                >
                  <IconTrash size={14} /> Удалить исследование
                </button>
              </div>
            </>
          )}
        </div>
      </aside>
    </>
  );
}
