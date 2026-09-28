import type {
  BatchProcessResponse,
  ExportFormat,
  ExportSource,
  Health,
  ReviewOut,
  ReviewRequest,
  StudyDetail,
  StudyList,
  StudyResult,
  StudyStatus,
  StudyStatusOut,
  UploadResponse,
} from "./types";

/** Empty by default: the UI is served from the same origin as the API (nginx / vite proxy). */
export const API_BASE = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.replace(/\/$/, "") ?? "";

export interface UploadItem {
  file: File;
  /** relative path, e.g. "study_01/IM0001.dcm" */
  name: string;
}

export class ApiError extends Error {
  status: number;
  payload: unknown;
  constructor(status: number, message: string, payload?: unknown) {
    super(message);
    this.status = status;
    this.payload = payload;
  }
}

function extractMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === "object" && "detail" in payload) {
    const detail = (payload as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      return String((detail as { message: unknown }).message);
    }
    if (Array.isArray(detail)) {
      return detail.map((d) => (d && typeof d === "object" && "msg" in d ? String(d.msg) : String(d))).join("; ");
    }
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init?.body ? { "Content-Type": "application/json" } : {}), ...init?.headers },
    });
  } catch {
    throw new ApiError(0, "Сервер недоступен. Проверьте, что backend запущен.");
  }
  if (res.status === 204) return undefined as T;
  const text = await res.text();
  const payload = text ? safeJson(text) : null;
  if (!res.ok) {
    throw new ApiError(res.status, extractMessage(payload, `Ошибка ${res.status}`), payload);
  }
  return payload as T;
}

function safeJson(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

export const api = {
  health: () => request<Health>("/health"),

  listStudies: (params: { status?: StudyStatus; limit?: number; offset?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.status) q.set("status", params.status);
    q.set("limit", String(params.limit ?? 200));
    if (params.offset) q.set("offset", String(params.offset));
    return request<StudyList>(`/api/v1/studies?${q}`);
  },

  getStudy: (id: string) => request<StudyDetail>(`/api/v1/studies/${id}`),
  getStatus: (id: string) => request<StudyStatusOut>(`/api/v1/studies/${id}/status`),
  getResult: (id: string) => request<StudyResult>(`/api/v1/studies/${id}/result`),
  process: (id: string) => request<{ study_id: string; status: StudyStatus }>(`/api/v1/studies/${id}/process`, { method: "POST" }),
  remove: (id: string) => request<void>(`/api/v1/studies/${id}`, { method: "DELETE" }),

  batchProcess: (body: { study_ids?: string[]; all_pending?: boolean; force?: boolean }) =>
    request<BatchProcessResponse>("/api/v1/batch/process", { method: "POST", body: JSON.stringify(body) }),

  downloadUrl: (id: string, format: ExportFormat, source: ExportSource = "auto") =>
    `${API_BASE}/api/v1/studies/${id}/download?format=${format}&source=${source}`,

  /** Заключение в виде DICOM SR и ZIP-пакет (SR + вторичная серия с разметкой + таблица). */
  srUrl: (id: string) => `${API_BASE}/api/v1/studies/${id}/sr`,
  packageUrl: (id: string) => `${API_BASE}/api/v1/studies/${id}/package`,

  /** Подтверждение или исправление вердикта специалистом. */
  review: (studyId: string, imageId: string, body: ReviewRequest) =>
    request<ReviewOut>(`/api/v1/studies/${studyId}/images/${imageId}/review`, {
      method: "POST",
      body: JSON.stringify(body),
    }),

  batchDownloadUrl: (format: ExportFormat, ids?: string[]) => {
    const q = new URLSearchParams({ format });
    ids?.forEach((id) => q.append("study_ids", id));
    return `${API_BASE}/api/v1/batch/download?${q}`;
  },

  assetUrl: (path: string) => `${API_BASE}${path}`,

  /**
   * Загрузка любого числа файлов: частями, по очереди, с общим прогрессом.
   * Сервер принимает не больше MAX_FILES_PER_UPLOAD файлов за запрос (по умолчанию 500),
   * а исследование, пришедшее в нескольких частях, сам собирает обратно по StudyInstanceUID.
   */
  uploadAll: async (files: UploadItem[], onProgress?: (fraction: number) => void): Promise<UploadResponse> => {
    const chunks = chunkUploads(files);
    const total = files.reduce((a, f) => a + f.file.size, 0) || 1;
    const studies = new Map<string, UploadResponse["studies"][number]>();
    const rejected: UploadResponse["rejected"] = [];
    const warnings: string[] = [];
    let done = 0;
    for (const [k, chunk] of chunks.entries()) {
      const size = chunk.reduce((a, f) => a + f.file.size, 0);
      try {
        const res = await api.upload(chunk, (f) => onProgress?.((done + f * size) / total));
        res.studies.forEach((s) => studies.set(s.id, s)); // позже пришедшая часть — свежее
        rejected.push(...res.rejected);
        warnings.push(...res.warnings);
      } catch (e) {
        const err = e as ApiError;
        const detail = (err.payload as { detail?: { rejected?: UploadResponse["rejected"] } } | undefined)?.detail;
        // «в этой части нет новых снимков» — не повод бросать остальные части
        if (err.status === 422 && detail?.rejected) {
          rejected.push(...detail.rejected);
        } else {
          const partial = k ? ` (отправлено частей: ${k} из ${chunks.length})` : "";
          throw new ApiError(err.status, `${err.message}${partial}`, err.payload);
        }
      }
      done += size;
    }
    if (!studies.size && rejected.length) {
      const message = rejected.some((r) => r.reason.startsWith("Уже загружен"))
        ? "Новых снимков нет: эти файлы уже загружены"
        : "Ни один файл не распознан как DICOM";
      throw new ApiError(422, message, { detail: { message, rejected } });
    }
    return { studies: [...studies.values()], rejected, warnings: [...new Set(warnings)] };
  },

  /** Multipart upload with progress (fetch has no upload progress events). */
  upload: (files: UploadItem[], onProgress?: (fraction: number) => void) =>
    new Promise<UploadResponse>((resolve, reject) => {
      const form = new FormData();
      // `name` keeps the relative folder path so studies get a meaningful path_to_study
      for (const { file, name } of files) form.append("files", file, name);
      const xhr = new XMLHttpRequest();
      xhr.open("POST", `${API_BASE}/api/v1/studies/upload`);
      xhr.responseType = "text";
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable && onProgress) onProgress(e.loaded / e.total);
      };
      xhr.onload = () => {
        const payload = xhr.responseText ? safeJson(xhr.responseText) : null;
        if (xhr.status >= 200 && xhr.status < 300) resolve(payload as UploadResponse);
        else reject(new ApiError(xhr.status, extractMessage(payload, `Ошибка загрузки (${xhr.status})`), payload));
      };
      xhr.onerror = () => reject(new ApiError(0, "Сервер недоступен"));
      xhr.send(form);
    }),
};

/** Файлов за запрос — с запасом меньше серверного MAX_FILES_PER_UPLOAD (500 по умолчанию). */
const CHUNK_FILES = 200;
/** Байт за запрос — с запасом меньше MAX_UPLOAD_SIZE_MB (1024 по умолчанию). */
const CHUNK_BYTES = 256 * 1024 * 1024;

/** Делит файлы на части; файлы одной папки идут подряд, чтобы исследование реже рвалось. */
export function chunkUploads(files: UploadItem[]): UploadItem[][] {
  const sorted = [...files].sort((a, b) => a.name.localeCompare(b.name));
  const chunks: UploadItem[][] = [];
  let cur: UploadItem[] = [];
  let size = 0;
  for (const item of sorted) {
    if (cur.length && (cur.length >= CHUNK_FILES || size + item.file.size > CHUNK_BYTES)) {
      chunks.push(cur);
      cur = [];
      size = 0;
    }
    cur.push(item);
    size += item.file.size;
  }
  if (cur.length) chunks.push(cur);
  return chunks;
}

/** Trigger a file download and surface API errors (e.g. 409) instead of navigating to JSON. */
export async function downloadFile(url: string): Promise<void> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiError(0, "Сервер недоступен");
  }
  if (!res.ok) {
    const payload = safeJson(await res.text());
    throw new ApiError(res.status, extractMessage(payload, `Ошибка ${res.status}`), payload);
  }
  const blob = await res.blob();
  const cd = res.headers.get("Content-Disposition") ?? "";
  const match = /filename="?([^";]+)"?/i.exec(cd);
  const name = match?.[1] ?? url.split("/").pop() ?? "result";
  const href = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(href), 1000);
}
