import { MeetingState, StepState } from '../models/api.models';

/** The only words users see for a meeting's state. */
export const STATUS_LABELS: Record<MeetingState, string> = {
  completed: 'Completed',
  processing: 'Processing meeting',
  failed: 'Needs attention',
};

export function statusLabel(status: string | null | undefined): string {
  return STATUS_LABELS[normalizeStatus(status)];
}

export function statusBadgeClass(status: string | null | undefined): string {
  const s = normalizeStatus(status);
  return s === 'completed' ? 'badge-completed' : s === 'failed' ? 'badge-failed' : 'badge-processing';
}

/** Any lifecycle value the API may return, reduced to the three states (unknown counts as still processing). */
export function normalizeStatus(status: string | null | undefined): MeetingState {
  const s = (status ?? '').trim().toLowerCase();
  if (s === 'completed' || s === 'success') return 'completed';
  if (s === 'failed' || s === 'error' || s === 'failure') return 'failed';
  return 'processing';
}

export const STEP_LABELS: Record<StepState, string> = {
  pending: 'Waiting',
  in_progress: 'In progress',
  completed: 'Done',
  failed: 'Stopped',
};
