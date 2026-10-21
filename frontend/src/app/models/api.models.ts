/**
 * Shapes of the backend responses this app uses. They mirror the API exactly - fields the backend may omit are
 * optional/nullable here, and nothing is given an invented default.
 */

/** The three states every screen speaks (backend `display_status`). */
export type MeetingState = 'completed' | 'processing' | 'failed';

export interface UserError {
  code: string;
  message: string;
}

/** One row of GET /meetings?page=… */
export interface MeetingListItem {
  id: number;
  meeting_id: string | null;
  title: string | null;
  filename: string | null;
  file_type: string | null;
  status: MeetingState;
  duration_seconds: number | null;
  speaker_count: number | null;
  language: string | null;
  uploaded_at: string | null;
  error: UserError | null;
}

export interface MeetingListPage {
  items: MeetingListItem[];
  total: number;
  page: number;
  limit: number;
  pages: number;
  status_counts: Record<MeetingState, number>;
}

export type MeetingSort = 'newest' | 'oldest' | 'title' | 'duration';

export interface MeetingListQuery {
  page?: number;
  limit?: number;
  status?: MeetingState | null;
  search?: string;
  sort?: MeetingSort;
}

/** GET /meetings/{id}/status */
export type StepState = 'pending' | 'in_progress' | 'completed' | 'failed';

export interface ProcessingStep {
  key: string;
  label: string;
  state: StepState;
}

export interface ProcessingStatus {
  meeting_id: number;
  status: MeetingState;
  progress_percent: number;
  has_warnings: boolean;
  steps: ProcessingStep[];
  error: string | null;
  error_code: string | null;
}

/** GET /meetings/{id}/final */
export interface TranscriptSegment {
  speaker: string | null;
  start_time: string | number | null;
  end_time: string | number | null;
  text: string;
  original_text?: string | null;
}

export interface AgendaItem {
  agenda_id: string;
  title: string;
  order: number;
}

export interface Decision {
  agenda_id: string;
  decision: string;
  reason?: string | null;
  evidence?: string | null;
  speaker?: string | null;
  timestamp?: string | null;
}

export interface AgendaDecisionMapping {
  agenda_id: string;
  agenda_title: string;
  decision: string;
  has_decision: boolean;
  is_grounded: boolean;
  evidence?: string | null;
  speaker?: string | null;
  timestamp?: string | null;
}

export interface ActionItem {
  item_order?: number | null;
  task: string;
  description?: string | null;
  owner?: string | null;
  deadline?: string | null;
  due_date?: string | null;
  priority?: string | null;
  status?: string | null;
  evidence?: string | null;
  speaker?: string | null;
  timestamp_start?: string | null;
}

export interface SpeakerDuration {
  speaker_id: string;
  duration_seconds: number;
  percentage: number;
  segment_count: number;
}

export interface Sentiment {
  label?: string | null;
}

export interface FinalMeetingMetadata {
  meeting_pk: number;
  title?: string | null;
  meeting_date?: string | null;
  meeting_time?: string | null;
  duration_seconds?: number | null;
  speaker_count?: number | null;
  participant_count?: number | null;
  source_language?: string | null;
  source_language_name?: string | null;
  translation_applied?: boolean | null;
}

export interface FinalMeeting {
  processing_status: string;
  display_status?: MeetingState | null;
  user_error?: UserError | null;
  /** Plain-language notes when parts of the run finished with reduced quality. */
  warnings?: string[];
  meeting: FinalMeetingMetadata;
  transcript: TranscriptSegment[];
  speaker_duration: SpeakerDuration[];
  agenda: AgendaItem[];
  decisions: Decision[];
  agenda_decision_mapping: AgendaDecisionMapping[];
  action_items: ActionItem[];
  summary?: string | null;
  mom?: string | null;
  sentiment?: Sentiment | null;
}

/** The 202 answer of /final while results are not ready. */
export interface FinalPending {
  processing_status: 'PENDING';
  status: MeetingState;
  user_error?: UserError | null;
}

export type FinalResult = { kind: 'ready'; meeting: FinalMeeting } | { kind: 'pending'; pending: FinalPending };

/** GET /search */
export interface SearchResult {
  meeting_id: number;
  meeting_title: string | null;
  uploaded_at: string | null;
  speaker: string | null;
  start_seconds: number | null;
  end_seconds: number | null;
  text: string;
  similarity: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  total: number;
}

/** GET /upload/options */
export interface UploadOptions {
  accepted_extensions: string[];
  max_file_size_mb: number;
  language_detection: boolean;
  default_language: string | null;
  language_required: boolean;
  languages: { code: string; name: string }[];
}

export interface UploadResult {
  id: number;
  meeting_id: string;
  status: string;
  potential_duplicate?: boolean;
  existing_duplicate_meeting_id?: number | null;
}

/** GET /system/status */
export interface ServiceStatus {
  status: 'available' | 'degraded' | 'unavailable';
  can_view_meetings: boolean;
  can_upload: boolean;
  can_search: boolean;
}

/** POST /meetings/{id}/chat */
export interface ChatSource {
  speaker?: string | null;
  /** Retrieval chunks name their speakers as one comma-separated string. */
  speakers?: string | null;
  start_time?: string | number | null;
  end_time?: string | number | null;
  text?: string | null;
}

export interface ChatReply {
  content: string;
  session_id?: string | null;
  sources?: ChatSource[];
}
