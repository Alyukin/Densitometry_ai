import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api, downloadFile } from "./api/client";
import type { ExportFormat, Health, StudySummary, UploadResponse } from "./api/types";
import { ConfirmDialog } from "./components/ConfirmDialog";
import { Header } from "./components/Header";
import { IconInfo } from "./components/Icons";
import { StatsBar } from "./components/StatsBar";
import { StudiesTable, type Filter } from "./components/StudiesTable";
import { StudyDrawer } from "./components/StudyDrawer";
import { useToast } from "./components/Toast";
import { UploadPanel } from "./components/UploadPanel";
import { usePolling } from "./hooks/usePolling";
import { isActive, pluralRu } from "./utils/format";

const SKIP_REASONS: Record<string, string> = {
  not_found: "не найдено",
  already_completed: "уже обработано",
  already_queued: "уже в очереди",
  already_processing: "уже обрабатывается",
};

export default function App() {
  const toast = useToast();
  const [studies, setStudies] = useState<StudySummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [health, setHealth] = useState<Health | null>(null);
  const [healthError, setHealthError] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [activeId, setActiveId] = useState<string | null>(null);
  const [busy, setBusy] = useState<Set<string>>(new Set());
  const [toDelete, setToDelete] = useState<StudySummary | null>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await api.listStudies({ limit: 500 });
      setStudies(list.items);
      const ids = new Set(list.items.map((s) => s.id));
      setSelected((prev) => {
        const next = new Set([...prev].filter((id) => ids.has(id)));
        return next.size === prev.size ? prev : next;
      });
      setActiveId((cur) => (cur && !ids.has(cur) ? null : cur));
    } catch (e) {
      // keep old data on transient errors; health indicator shows API state
      console.warn("Failed to load studies", e);
    } finally {
      setLoading(false);
    }
  }, []);

  const checkHealth = useCallback(async () => {
    try {
      setHealth(await api.health());
      setHealthError(false);
    } catch {
      setHealthError(true);
    }
  }, []);

  useEffect(() => {
    void refresh();
    void checkHealth();
  }, [refresh, checkHealth]);

  const anyActive = useMemo(() => studies.some((s) => isActive(s.status)), [studies]);
  usePolling(refresh, 1500, anyActive);
  usePolling(refresh, 15000, !anyActive);
  usePolling(checkHealth, 15000, true);

  const withBusy = async (ids: string[], fn: () => Promise<void>) => {
    setBusy((b) => new Set([...b, ...ids]));
    try {
      await fn();
    } finally {
      setBusy((b) => new Set([...b].filter((x) => !ids.includes(x))));
    }
  };

  const processOne = (id: string) =>
    withBusy([id], async () => {
      try {
        await api.process(id);
        toast.info("Исследование поставлено в очередь");
      } catch (e) {
        toast.error((e as ApiError).message);
      }
      await refresh();
    });

  const batchProcess = async (ids: string[] | null) => {
    try {
      const res = await api.batchProcess(ids ? { study_ids: ids, force: true } : { all_pending: true });
      const n = res.accepted.length;
      if (n) toast.success(`В очередь: ${n} ${pluralRu(n, "исследование", "исследования", "исследований")}`);
      if (res.skipped.length) {
        const reasons = [...new Set(res.skipped.map((s) => SKIP_REASONS[s.reason] ?? s.reason))].join(", ");
        toast.info(`Пропущено: ${res.skipped.length} (${reasons})`);
      }
      if (ids) setSelected(new Set());
    } catch (e) {
      toast.error((e as ApiError).message);
    }
    await refresh();
  };

  const download = async (id: string, format: ExportFormat) => {
    try {
      await downloadFile(api.downloadUrl(id, format));
    } catch (e) {
      toast.error((e as ApiError).message);
    }
  };

  const batchDownload = async (format: ExportFormat, ids: string[] | null) => {
    try {
      await downloadFile(api.batchDownloadUrl(format, ids ?? undefined));
    } catch (e) {
      toast.error((e as ApiError).message);
    }
  };

  const confirmDelete = async () => {
    const s = toDelete;
    setToDelete(null);
    if (!s) return;
    await withBusy([s.id], async () => {
      try {
        await api.remove(s.id);
        toast.success(`«${s.name}» удалено`);
        if (activeId === s.id) setActiveId(null);
      } catch (e) {
        toast.error((e as ApiError).message);
      }
    });
    await refresh();
  };

  const onUploaded = async (res: UploadResponse, autoProcess: boolean) => {
    await refresh();
    if (autoProcess && res.studies.length) {
      await batchProcess(res.studies.map((s) => s.id));
    }
    if (res.studies.length === 1) setActiveId(res.studies[0].id);
  };

  const closeDrawer = useCallback(() => setActiveId(null), []);

  return (
    <div className="app">
      <Header health={health} healthError={healthError} />

      <main className="container">
        {(health?.is_mock ?? true) && (
          <div className="banner">
            <IconInfo size={16} />
            <span>
              <b>Прототип (этап 1).</b> Обработка выполняется mock-процессором: результаты тестовые и не являются
              медицинским заключением. AI-модель будет подключена на следующем этапе.
            </span>
          </div>
        )}

        <StatsBar studies={studies} />

        <div className="layout">
          <UploadPanel onUploaded={onUploaded} />
          <StudiesTable
            studies={studies}
            loading={loading}
            filter={filter}
            onFilter={setFilter}
            selected={selected}
            onSelectedChange={setSelected}
            activeId={activeId}
            onOpen={setActiveId}
            onProcess={processOne}
            onDelete={setToDelete}
            onDownload={download}
            onBatchProcess={batchProcess}
            onBatchDownload={batchDownload}
            onRefresh={refresh}
            busy={busy}
          />
        </div>
      </main>

      <footer className="footer container">
        Densitometry AI {health ? `v${health.version}` : ""} · локальный сервис · данные не покидают ваш контур
      </footer>

      {activeId && (
        <StudyDrawer
          studyId={activeId}
          summary={studies.find((s) => s.id === activeId)}
          onClose={closeDrawer}
          onProcess={processOne}
          onDownload={download}
          onDelete={setToDelete}
        />
      )}

      {toDelete && (
        <ConfirmDialog
          title="Удалить исследование?"
          text={`«${toDelete.name}» и все его файлы и результаты будут удалены без возможности восстановления.`}
          confirmLabel="Удалить"
          danger
          onConfirm={confirmDelete}
          onCancel={() => setToDelete(null)}
        />
      )}
    </div>
  );
}
