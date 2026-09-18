export type StudyStatus = "uploaded" | "queued" | "processing" | "completed" | "failed";
export type ExportFormat = "csv" | "xlsx";

export interface QualitySummary {
  total: number;
  success: number;
  errors: number;
  acceptable: number;
  unacceptable: number;
  regions: string[];
  overall_quality: string | null;
}

export interface StudySummary {
  id: string;
  name: string;
  source_path: string;
  study_instance_uid: string | null;
  modality: string | null;
  manufacturer: string | null;
  status: StudyStatus;
  progress: number;
  image_count: number;
  is_mock: boolean;
  processing_time_sec: number | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
  finished_at: string | null;
  summary: QualitySummary | null;
}

export interface StudyImage {
  id: string;
  original_filename: string;
  size_bytes: number;
  sop_instance_uid: string | null;
  series_instance_uid: string | null;
  modality: string | null;
  body_part_examined: string | null;
  rows: number | null;
  columns: number | null;
  has_pixel_data: boolean;
  preview_url: string | null;
}

export interface StudyDetail extends StudySummary {
  warnings: string[];
  processor_name: string | null;
  processor_version: string | null;
  queued_at: string | null;
  started_at: string | null;
  images: StudyImage[];
}

export interface StudyList {
  items: StudySummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface RejectedFile {
  filename: string;
  reason: string;
}

export interface UploadResponse {
  studies: StudySummary[];
  rejected: RejectedFile[];
  warnings: string[];
}

export interface StudyStatusOut {
  study_id: string;
  status: StudyStatus;
  progress: number;
  processed_images: number;
  total_images: number;
  error_message: string | null;
  queued_at: string | null;
  started_at: string | null;
  finished_at: string | null;
  processing_time_sec: number | null;
}

export interface CheckDetail {
  code: string;
  title: string;
  passed: boolean;
  value?: unknown;
  unit?: string;
  threshold?: unknown;
}

export interface ResultRow {
  path_to_study: string;
  study_uid: string | null;
  image_uid: string | null;
  anatomical_region: string | null;
  quality_class: string | null;
  violation_type: string | null;
  processing_status: string;
  time_of_processing: number;
  image_id: string | null;
  original_filename: string | null;
  confidence: number | null;
  error_message: string | null;
  details: { checks?: CheckDetail[]; note?: string; mock?: boolean; [k: string]: unknown };
}

export interface StudyResult {
  study_id: string;
  status: StudyStatus;
  processor: string | null;
  processor_version: string | null;
  is_mock: boolean;
  processing_time_sec: number | null;
  finished_at: string | null;
  summary: QualitySummary;
  rows: ResultRow[];
}

export interface BatchProcessResponse {
  accepted: string[];
  skipped: { study_id: string; reason: string }[];
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  environment: string;
  processor: string;
  processor_version: string | null;
  is_mock: boolean | null;
  database: "ok" | "error";
  workers: number;
  tasks_in_progress: number;
}
