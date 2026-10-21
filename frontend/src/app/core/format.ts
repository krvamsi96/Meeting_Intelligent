/** Display helpers. Missing values render as an explicit "not available" marker, never as an invented number. */

export const NOT_AVAILABLE = '—';

/** Seconds (number or numeric string) or null when the value is absent or not a number. */
export function toSeconds(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined || value === '') return null;
  if (typeof value === 'string' && value.includes(':')) {
    // "mm:ss" or "hh:mm:ss[.fff]"
    const parts = value.trim().split(':').map(Number);
    if (parts.length < 2 || parts.length > 3 || parts.some((p) => !Number.isFinite(p) || p < 0)) return null;
    return parts.reduce((acc, p) => acc * 60 + p, 0);
  }
  const n = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(n) && n >= 0 ? n : null;
}

/** "1h 02m", "12m 05s", "45s"; "—" when unknown. */
export function formatDuration(seconds: number | null | undefined): string {
  const s = toSeconds(seconds ?? null);
  if (s === null || s === 0) return NOT_AVAILABLE;
  const total = Math.round(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
}

/** Clock position in a recording: "04:07" or "1:02:09"; "—" when unknown. */
export function formatTimestamp(value: string | number | null | undefined): string {
  const s = toSeconds(value ?? null);
  if (s === null) return NOT_AVAILABLE;
  const total = Math.floor(s);
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const sec = total % 60;
  const mm = String(m).padStart(2, '0');
  const ss = String(sec).padStart(2, '0');
  return h > 0 ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** "Speaker_1" / "SPEAKER_00" -> "Speaker 1"; real names are kept as they are. */
export function speakerLabel(raw: string | null | undefined): string {
  const value = (raw ?? '').trim();
  if (!value || value.toLowerCase() === 'unknown') return 'Unidentified speaker';
  const m = /^speaker[_\s-]*(\d+)$/i.exec(value);
  if (m) {
    // Zero-based machine labels (SPEAKER_00, SPEAKER_01) become 1-based for people; Speaker_1 stays Speaker 1.
    const zeroBased = m[1].startsWith('0');
    return `Speaker ${Number(m[1]) + (zeroBased ? 1 : 0)}`;
  }
  return value;
}

/** Owners the backend writes when nobody was named. */
const NO_OWNER = new Set(['', 'unassigned', 'unknown', 'none', 'n/a', 'not specified', 'tbd']);

export function ownerLabel(raw: string | null | undefined): string | null {
  const v = (raw ?? '').trim();
  if (NO_OWNER.has(v.toLowerCase())) return null;
  return /^speaker[_\s-]*\d+$/i.test(v) ? speakerLabel(v) : v;
}

export function deadlineLabel(deadline: string | null | undefined, dueDate: string | null | undefined): string | null {
  const v = (dueDate || deadline || '').trim();
  return NO_OWNER.has(v.toLowerCase()) || v.toLowerCase() === 'not_specified' ? null : v;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

/** A display name for a meeting: its title, else its file name, else its number. */
export function meetingName(m: { title?: string | null; filename?: string | null; id?: number } | null): string {
  if (!m) return 'Meeting';
  return (m.title || m.filename || (m.id !== undefined ? `Meeting ${m.id}` : 'Meeting')).trim();
}
