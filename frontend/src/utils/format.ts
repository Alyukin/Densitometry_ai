import type { StudyStatus } from "../api/types";

export const STATUS_LABELS: Record<StudyStatus, string> = {
  uploaded: "Загружено",
  queued: "В очереди",
  processing: "Обработка",
  completed: "Готово",
  failed: "Ошибка",
};

export const ACTIVE: StudyStatus[] = ["queued", "processing"];
export const isActive = (s: StudyStatus) => ACTIVE.includes(s);

// NOTE: коды ниже соответствуют mock-процессору. После анализа разметки датасета
// словарь нужно синхронизировать с реальными классами модели.
export const REGION_LABELS: Record<string, string> = {
  lumbar_spine: "Поясничный отдел позвоночника",
  proximal_femur: "Проксимальный отдел бедра",
};

export const REGION_SHORT: Record<string, string> = {
  lumbar_spine: "Позвоночник",
  proximal_femur: "Бедро",
};

export const QUALITY_LABELS: Record<string, string> = {
  acceptable: "Качественное",
  unacceptable: "Есть нарушения",
};

export const VIOLATION_LABELS: Record<string, string> = {
  iliac_crest_not_visible: "Не видны верхние края подвздошных костей",
  th12_not_visible: "Не видна половина Th12",
  spine_axis_tilt: "Наклон оси позвоночника > 5°",
  artifacts_present: "Артефакты / металлические предметы",
  greater_trochanter_not_visible: "Не виден большой вертел",
  femoral_neck_not_visible: "Не видна шейка бедра",
  ischium_not_visible: "Не видна седалищная кость",
  over_rotation: "Переротация бедра",
  under_rotation: "Недоротация бедра",
  roi_margin_insufficient: "Недостаточные отступы ROI",
};

export const label = (dict: Record<string, string>, code: string | null | undefined) =>
  code ? (dict[code] ?? code) : "—";

export const splitViolations = (v: string | null | undefined) => (v ? v.split(";").filter(Boolean) : []);

const dtf = new Intl.DateTimeFormat("ru-RU", {
  day: "2-digit",
  month: "2-digit",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

export const formatDate = (iso: string | null | undefined) => (iso ? dtf.format(new Date(iso)) : "—");

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} Б`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} КБ`;
  return `${(n / 1024 ** 2).toFixed(1)} МБ`;
}

export const formatSeconds = (s: number | null | undefined) =>
  s == null ? "—" : s < 60 ? `${s.toFixed(1)} с` : `${Math.floor(s / 60)} мин ${Math.round(s % 60)} с`;

export function pluralRu(n: number, one: string, few: string, many: string): string {
  const m10 = n % 10;
  const m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
