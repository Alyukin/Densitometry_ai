import { useRef, useState, type DragEvent } from "react";
import { ApiError, api, type UploadItem } from "../api/client";
import type { UploadResponse } from "../api/types";
import { formatBytes, pluralRu } from "../utils/format";
import { IconAlert, IconClose, IconFile, IconFolder, IconUpload } from "./Icons";
import { useToast } from "./Toast";

interface Props {
  onUploaded: (res: UploadResponse, autoProcess: boolean) => void | Promise<void>;
}

const IGNORED = /(^|\/)(\.ds_store|thumbs\.db|desktop\.ini|\._.*)$/i;

async function readEntry(entry: FileSystemEntry, prefix = ""): Promise<UploadItem[]> {
  if (entry.isFile) {
    const file = await new Promise<File>((res, rej) => (entry as FileSystemFileEntry).file(res, rej));
    return [{ file, name: prefix + file.name }];
  }
  if (entry.isDirectory) {
    const reader = (entry as FileSystemDirectoryEntry).createReader();
    const entries: FileSystemEntry[] = [];
    // readEntries returns results in chunks
    for (;;) {
      const chunk = await new Promise<FileSystemEntry[]>((res, rej) => reader.readEntries(res, rej));
      if (!chunk.length) break;
      entries.push(...chunk);
    }
    const nested = await Promise.all(entries.map((e) => readEntry(e, `${prefix}${entry.name}/`)));
    return nested.flat();
  }
  return [];
}

export function UploadPanel({ onUploaded }: Props) {
  const toast = useToast();
  const [items, setItems] = useState<UploadItem[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [progress, setProgress] = useState<number | null>(null);
  const [autoProcess, setAutoProcess] = useState(true);
  const [report, setReport] = useState<UploadResponse | null>(null);
  const [rejectedFromError, setRejectedFromError] = useState<UploadResponse["rejected"]>([]);
  const fileInput = useRef<HTMLInputElement>(null);
  const dirInput = useRef<HTMLInputElement>(null);

  const uploading = progress !== null;
  const totalSize = items.reduce((a, i) => a + i.file.size, 0);

  const addItems = (next: UploadItem[]) => {
    const filtered = next.filter((i) => !IGNORED.test(i.name));
    setItems((prev) => {
      const seen = new Set(prev.map((p) => `${p.name}:${p.file.size}`));
      return [...prev, ...filtered.filter((i) => !seen.has(`${i.name}:${i.file.size}`))];
    });
    setReport(null);
    setRejectedFromError([]);
  };

  const onInput = (files: FileList | null) => {
    if (!files) return;
    addItems(
      Array.from(files).map((file) => ({
        file,
        name: (file as File & { webkitRelativePath?: string }).webkitRelativePath || file.name,
      })),
    );
  };

  const onDrop = async (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const entries = Array.from(e.dataTransfer.items)
      .map((it) => it.webkitGetAsEntry?.())
      .filter((x): x is FileSystemEntry => !!x);
    if (entries.length) {
      const all = (await Promise.all(entries.map((en) => readEntry(en)))).flat();
      addItems(all);
    } else {
      onInput(e.dataTransfer.files);
    }
  };

  const submit = async () => {
    if (!items.length) return;
    setProgress(0);
    setReport(null);
    setRejectedFromError([]);
    try {
      const res = await api.upload(items, setProgress);
      setReport(res);
      setItems([]);
      const n = res.studies.length;
      toast.success(`Загружено ${n} ${pluralRu(n, "исследование", "исследования", "исследований")}`);
      await onUploaded(res, autoProcess);
    } catch (err) {
      const e = err as ApiError;
      const payload = e.payload as { detail?: { rejected?: UploadResponse["rejected"] } } | undefined;
      if (payload?.detail?.rejected) setRejectedFromError(payload.detail.rejected);
      toast.error(e.message);
    } finally {
      setProgress(null);
      if (fileInput.current) fileInput.current.value = "";
      if (dirInput.current) dirInput.current.value = "";
    }
  };

  const rejected = report?.rejected.length ? report.rejected : rejectedFromError;

  return (
    <section className="card upload">
      <div className="card__head">
        <div>
          <h2>Загрузка исследований</h2>
          <p className="muted">DICOM-файлы (.dcm или без расширения), папки или ZIP-архивы. Файлы группируются по StudyInstanceUID.</p>
        </div>
      </div>

      <div
        className={`dropzone ${dragOver ? "dropzone--over" : ""} ${uploading ? "dropzone--busy" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => !uploading && fileInput.current?.click()}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && fileInput.current?.click()}
        aria-label="Выбрать DICOM-файлы"
      >
        <div className="dropzone__icon">
          <IconUpload size={26} />
        </div>
        <div className="dropzone__title">Перетащите файлы или папку сюда</div>
        <div className="dropzone__hint">или нажмите, чтобы выбрать файлы</div>
        <input
          ref={fileInput}
          type="file"
          multiple
          accept=".dcm,.dicom,.zip,application/dicom,application/zip,*/*"
          hidden
          onChange={(e) => onInput(e.target.files)}
        />
        <input
          ref={dirInput}
          type="file"
          hidden
          // @ts-expect-error non-standard but widely supported attribute
          webkitdirectory=""
          onChange={(e) => onInput(e.target.files)}
        />
      </div>

      <div className="upload__actions">
        <button className="btn btn--ghost btn--sm" onClick={() => dirInput.current?.click()} disabled={uploading}>
          <IconFolder size={15} /> Выбрать папку
        </button>
        <label className="checkbox">
          <input type="checkbox" checked={autoProcess} onChange={(e) => setAutoProcess(e.target.checked)} />
          Сразу запустить обработку
        </label>
      </div>

      {items.length > 0 && (
        <div className="queue">
          <div className="queue__head">
            <span>
              {items.length} {pluralRu(items.length, "файл", "файла", "файлов")} · {formatBytes(totalSize)}
            </span>
            <button className="link" onClick={() => setItems([])} disabled={uploading}>
              Очистить
            </button>
          </div>
          <ul className="queue__list">
            {items.slice(0, 50).map((i) => (
              <li key={`${i.name}:${i.file.size}`}>
                <IconFile size={14} />
                <span className="queue__name" title={i.name}>
                  {i.name}
                </span>
                <span className="muted">{formatBytes(i.file.size)}</span>
                {!uploading && (
                  <button
                    className="icon-btn icon-btn--sm"
                    aria-label="Убрать"
                    onClick={() => setItems((xs) => xs.filter((x) => x !== i))}
                  >
                    <IconClose size={12} />
                  </button>
                )}
              </li>
            ))}
            {items.length > 50 && <li className="muted">…и ещё {items.length - 50}</li>}
          </ul>
          {uploading && (
            <div className="progress progress--processing">
              <div className="progress__fill" style={{ width: `${Math.round((progress ?? 0) * 100)}%` }} />
            </div>
          )}
          <button className="btn btn--primary btn--block" onClick={submit} disabled={uploading}>
            <IconUpload size={16} />
            {uploading
              ? (progress ?? 0) >= 1
                ? "Проверка DICOM…"
                : `Загрузка… ${Math.round((progress ?? 0) * 100)}%`
              : "Загрузить"}
          </button>
        </div>
      )}

      {(report || rejected.length > 0) && (
        <div className="report">
          {report && (
            <div className="report__ok">
              Создано исследований: <b>{report.studies.length}</b>
              {report.studies.length > 0 && (
                <> ({report.studies.reduce((a, s) => a + s.image_count, 0)} изобр.)</>
              )}
            </div>
          )}
          {report?.warnings.map((w) => (
            <div key={w} className="report__warn">
              <IconAlert size={14} /> {w}
            </div>
          ))}
          {rejected.length > 0 && (
            <details className="report__rejected" open={rejected.length <= 5}>
              <summary>
                Отклонено файлов: <b>{rejected.length}</b>
              </summary>
              <ul>
                {rejected.map((r, idx) => (
                  <li key={`${r.filename}-${idx}`}>
                    <span className="mono">{r.filename}</span> — {r.reason}
                  </li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
    </section>
  );
}
