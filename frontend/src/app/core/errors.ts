import { HttpErrorResponse } from '@angular/common/http';

/**
 * Turns a failed request into a sentence an ordinary user can act on.
 *
 * Server text is shown only for the status codes where the API returns messages written for users (validation of
 * an upload, a conflicting action, a temporarily unavailable feature) AND only when it carries nothing technical.
 * Everything else - stack traces, SQL errors, URLs, file paths, component names - is replaced by `fallback`.
 */

const USER_MESSAGE_STATUSES = new Set([400, 404, 409, 410, 413, 415, 422, 429, 503]);

const TECHNICAL_PATTERNS: RegExp[] = [
  /https?:\/\//i,
  /[a-z]:\\|\/(home|usr|var|tmp|opt|etc)\//i,
  /traceback|exception|stack|errno|sqlstate/i,
  /psycopg|sqlalchemy|postgres|chroma|fastapi|uvicorn|python|ffmpeg|ffprobe|nemo|whisper|langchain/i,
  /airawat|groq|gemini|ollama|openai|llama|qwen|gpt/i,
  /\b(select|insert|update|delete)\b.+\b(from|into|set)\b/i,
  /[{}[\]<>]|\w+Error\b/,
];

export function isUserSafeText(text: unknown): text is string {
  return typeof text === 'string' && text.trim().length > 0 && text.length <= 300 &&
    !TECHNICAL_PATTERNS.some((p) => p.test(text));
}

/** The message field of the API's error body ({message}, {detail} or {detail: {message}}). */
function serverMessage(err: HttpErrorResponse): unknown {
  const body = err.error;
  if (!body || typeof body !== 'object') return null;
  if (typeof body.message === 'string') return body.message;
  if (typeof body.detail === 'string') return body.detail;
  if (body.detail && typeof body.detail.message === 'string') return body.detail.message;
  return null;
}

export function friendlyError(err: unknown, fallback: string): string {
  if (!(err instanceof HttpErrorResponse)) return fallback;
  if (err.status === 0) return 'The service could not be reached. Check your connection and try again.';
  if (err.status === 401 || err.status === 403) return 'You do not have permission to do this.';
  if (USER_MESSAGE_STATUSES.has(err.status)) {
    const msg = serverMessage(err);
    if (isUserSafeText(msg)) return msg;
  }
  if (err.status === 404) return 'This item could not be found. It may have been deleted.';
  if (err.status === 429) return 'Too many requests. Please wait a moment and try again.';
  if (err.status === 503) return 'This feature is temporarily unavailable. Please try again later.';
  return fallback;
}
