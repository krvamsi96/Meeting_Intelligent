"""
Transcription Service — Dynamic Chunking for Any File Size
───────────────────────────────────────────────────────────
Works for any video/audio size: 1 min, 25 min, 2 hours, 3 hours, 2 GB+.

Strategy:
  - Use ffprobe to get exact audio duration and bitrate
  - Dynamically compute chunk count so each chunk is safely under 20 MB
  - Process all chunks sequentially with retry logic
  - Merge all timestamps, words, logprobs into a single unified result
"""

import io
import logging
import math
import os
import random
import re
import subprocess
import time
import uuid

import requests

from backend.config.config import settings
from backend.utils.exceptions import (
    AIProviderError,
    AIProviderNotConfiguredError,
    AirawatError,
    AirawatNotConfiguredError,
    LanguageIdentificationError,
)
from backend.utils.media_tools import FFMPEG, FFPROBE

logger = logging.getLogger(__name__)

# ─── Audio chunking constants loaded dynamically from settings ───────────────
# Do NOT redeclare these locally — use settings.CLOUD_MAX_BYTES etc. everywhere.
from backend.config.config import settings as _audio_settings  # noqa: E402

CLOUD_MAX_BYTES: int = _audio_settings.CLOUD_MAX_BYTES
BYTES_PER_SEC_WAV: int = _audio_settings.BYTES_PER_SEC_WAV
CHUNK_OVERLAP_SEC: float = _audio_settings.CHUNK_OVERLAP_SEC


def get_audio_info(filepath: str) -> dict:
    """
    Uses ffprobe to get exact duration and audio bitrate of any audio/video file.
    Returns {'duration': float_seconds, 'size_bytes': int}
    Works for any container: mp4, mkv, avi, mov, webm, mp3, wav, m4a, etc.
    """
    info = {"duration": 0.0, "size_bytes": 0}
    try:
        cmd = [
            FFPROBE,
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "default=noprint_wrappers=1",
            filepath,
        ]
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60
        )
        for line in res.stdout.strip().splitlines():
            if line.startswith("duration="):
                try:
                    info["duration"] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
            elif line.startswith("size="):
                try:
                    info["size_bytes"] = int(line.split("=", 1)[1])
                except ValueError:
                    pass
    except Exception as e:
        logger.warning(f"ffprobe info failed for {filepath}: {e}")

    # Fallback: estimate from actual file size on disk
    if info["size_bytes"] == 0 and os.path.exists(filepath):
        info["size_bytes"] = os.path.getsize(filepath)
    return info


def compute_chunk_duration(total_duration_sec: float) -> int:
    """
    Dynamically compute optimal chunk duration in seconds so that each WAV
    chunk stays under CLOUD_MAX_BYTES (from settings — default 18 MB).
    16KHz mono PCM = 32000 bytes/sec → max safe seconds = 18MB / 32KB/s = 562.5s ≈ 9 min.
    Always returns a sensible chunk size regardless of total duration.
    """
    # Reserve headroom for the 44-byte WAV header + multipart/form-data encoding
    # overhead so the *actual* uploaded body never exceeds CLOUD_MAX_BYTES.
    usable_bytes = max(1_000_000, int(settings.CLOUD_MAX_BYTES) - 64 * 1024)
    max_safe_secs = max(30, usable_bytes // settings.BYTES_PER_SEC_WAV)

    if total_duration_sec <= max_safe_secs:
        # File fits in a single chunk
        return int(total_duration_sec) + 1

    # Calculate number of chunks needed and adjust duration evenly.
    num_chunks = math.ceil(total_duration_sec / max_safe_secs)
    chunk_secs = math.ceil(total_duration_sec / num_chunks)

    # Floor to the safe max — a chunk of `max_safe_secs` is already at the byte
    # budget, so `chunk_secs` must never be allowed to exceed it (the old code
    # ceil()'d then clamped, which could still round a chunk over the limit).
    chunk_secs = min(chunk_secs, max_safe_secs)
    logger.info(
        f"Dynamic chunking: {total_duration_sec/60:.1f} min audio → "
        f"{num_chunks} chunks × ~{chunk_secs//60}m{chunk_secs%60}s each "
        f"(~{(chunk_secs * BYTES_PER_SEC_WAV) / (1024*1024):.1f} MB each)"
    )
    return chunk_secs


def split_audio_into_chunks(wav_filepath: str) -> list:
    """
    Splits any WAV file into dynamically-sized chunks that each fit under the cloud provider's 25 MB limit.
    Returns list of (chunk_filepath, start_offset_seconds, is_first_chunk).
    Works for any length: 1 min, 25 min, 2 hours, 5 hours, 10 hours.

    Each chunk (except the first) starts CHUNK_OVERLAP_SEC seconds earlier than the nominal
    boundary. This prevents timestamp drift at chunk boundaries from ffmpeg PCM alignment.
    The caller must subtract CHUNK_OVERLAP_SEC from word timestamps when chunk_index > 0.
    """
    info = get_audio_info(wav_filepath)
    total_duration = info["duration"]
    file_size = info["size_bytes"]

    if total_duration <= 0:
        logger.warning(
            f"Could not determine audio duration for {wav_filepath}. Sending whole file."
        )
        return [(wav_filepath, 0.0, True)]

    chunk_duration = compute_chunk_duration(total_duration)

    # If file fits in one chunk — no splitting needed
    if file_size <= CLOUD_MAX_BYTES and total_duration <= chunk_duration:
        logger.info(
            f"Audio fits in single request ({file_size/(1024*1024):.1f} MB, {total_duration/60:.1f} min). No chunking."
        )
        return [(wav_filepath, 0.0, True)]

    logger.info(
        f"Splitting audio: {file_size/(1024*1024):.1f} MB, "
        f"{total_duration/60:.1f} min → chunks of {chunk_duration}s (overlap={CHUNK_OVERLAP_SEC}s)"
    )

    chunks = []
    start = 0.0
    idx = 0
    attempted = 0
    failed_ranges: list = []   # (start_sec, end_sec) of every stretch of audio no chunk could be cut for
    base_no_ext = os.path.splitext(wav_filepath)[0]
    hard_cap = int(getattr(settings, "CLOUD_HARD_MAX_BYTES", 24 * 1024 * 1024))

    while start < total_duration:
        attempted += 1
        chunk_file = f"{base_no_ext}_chunk_{idx:04d}.wav"
        # Apply overlap: start each non-first chunk slightly before its nominal position
        actual_start = max(0.0, start - (CHUNK_OVERLAP_SEC if idx > 0 else 0.0))
        actual_duration = chunk_duration + (CHUNK_OVERLAP_SEC if idx > 0 else 0.0)
        cmd = [
            FFMPEG,
            "-y",
            "-nostdin",
            "-ss",
            str(actual_start),
            "-t",
            str(actual_duration),
            "-i",
            wav_filepath,
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            chunk_file,
        ]
        ok = False
        for cut_attempt in (1, 2):      # one retry: a transient ffmpeg failure must not drop 7 minutes of speech
            try:
                res = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600
                )
                ok = res.returncode == 0 and os.path.exists(chunk_file)
            except subprocess.TimeoutExpired:
                logger.error(f"FFmpeg timed out creating chunk {idx} at offset {start}s (attempt {cut_attempt}/2)")
                ok = False
            except Exception as ff_err:
                logger.error(f"FFmpeg error creating chunk {idx} at offset {start}s: {ff_err} (attempt {cut_attempt}/2)")
                ok = False
            if ok:
                break

        if ok:
            chunk_size = os.path.getsize(chunk_file)
            if chunk_size > hard_cap and actual_duration > 60:
                # Produced chunk still too big for the endpoint — split it in half
                # by halving the working chunk_duration and retrying from `start`.
                logger.warning(
                    f"  Chunk {idx} is {chunk_size/1e6:.1f} MB (> {hard_cap/1e6:.0f} MB cap) — "
                    f"halving chunk duration {chunk_duration}s -> {max(30, chunk_duration // 2)}s and re-splitting."
                )
                try:
                    os.remove(chunk_file)
                except Exception:
                    pass
                new_dur = max(30, chunk_duration // 2)
                if new_dur < chunk_duration:
                    chunk_duration = new_dur
                    continue  # retry same `start` with the smaller duration
            logger.info(
                f"  Chunk {idx}: {actual_start/60:.1f}–{(actual_start+actual_duration)/60:.1f} min → {chunk_size/(1024*1024):.1f} MB"
            )
            # Store nominal start (not actual_start) as the offset for timestamp correction
            chunks.append((chunk_file, start, idx == 0))
        else:
            logger.error(f"FFmpeg failed creating chunk {idx} at offset {start}s")
            failed_ranges.append((start, min(total_duration, start + chunk_duration)))
        start += chunk_duration
        idx += 1

    if failed_ranges:
        # A chunk that could not be cut used to be skipped silently: the remaining chunks were transcribed and the
        # meeting finished as SUCCESS with minutes of speech missing. An incomplete recording is never transcribed.
        gaps = ", ".join(f"{a / 60:.1f}-{b / 60:.1f} min" for a, b in failed_ranges)
        raise RuntimeError(
            f"Audio chunking could not cut {len(failed_ranges)} of {attempted} chunk(s) ({gaps}) from a "
            f"{total_duration / 60:.1f}-min file; refusing to transcribe an incomplete recording."
        )

    if not chunks:
        # Never silently POST a multi-hundred-MB file to a 25 MB endpoint — that
        # is a guaranteed empty/failed transcription. Surface a real error; the caller
        # (transcribe_audio_airawat) then sends the file as one chunk and AIRAWAT rejects it clearly.
        raise RuntimeError(
            f"Audio chunking produced 0 usable chunks from {attempted} attempt(s) "
            f"for a {total_duration/60:.1f}-min file — cannot transcribe via cloud."
        )

    return chunks


_TOKEN_STRIP_RE = re.compile(r"[\W_]+", re.UNICODE)


def _merge_chunk_words(combined_words: list, new_words: list, boundary_sec: float, is_first: bool) -> None:
    """Append one chunk's word timestamps, removing the duplicates created by chunk overlap.

    Every non-first chunk starts ``CHUNK_OVERLAP_SEC`` BEFORE its nominal boundary, so the words
    spoken in that window are transcribed twice (end of the previous chunk, start of this one) —
    previously both copies were kept, duplicating a word or two at every boundary of any recording
    longer than one chunk (~8 min). Cut at the MIDDLE of the overlap: the previous chunk keeps
    everything before it, this chunk everything from it on, so neither engine's clipped edge word
    survives while the fully-heard one does.
    """
    if is_first or CHUNK_OVERLAP_SEC <= 0:
        combined_words.extend(new_words)
        return
    cut = boundary_sec - CHUNK_OVERLAP_SEC / 2.0
    while combined_words and float(combined_words[-1].get("start", 0.0) or 0.0) >= cut:
        combined_words.pop()
    combined_words.extend(w for w in new_words if float(w.get("start", 0.0) or 0.0) >= cut)


def overlap_word_budget(overlap_sec: float, words_per_sec: float = 5.0) -> int:
    """The most words that can be spoken inside the ``overlap_sec`` window (fast speech, 5 words/s), at least 1."""
    return max(1, int(math.ceil(max(0.0, float(overlap_sec)) * words_per_sec)))


def _trim_overlap_prefix(prev_text: str, new_text: str, max_words: int = 4) -> str:
    """Drop the leading words of ``new_text`` that merely repeat the tail of ``prev_text``.

    Text counterpart of ``_merge_chunk_words``: the overlap window makes the last word(s) of one
    chunk's transcript reappear at the start of the next. Compared case/punctuation-insensitively.
    """
    prev_tokens = prev_text.split()
    new_tokens = new_text.split()
    if not prev_tokens or not new_tokens:
        return new_text
    prev_norm = [_TOKEN_STRIP_RE.sub("", t).lower() for t in prev_tokens[-max_words:]]
    new_norm = [_TOKEN_STRIP_RE.sub("", t).lower() for t in new_tokens[:max_words]]
    for n in range(min(len(prev_norm), len(new_norm)), 0, -1):
        if all(prev_norm[-n:]) and prev_norm[-n:] == new_norm[:n]:
            return " ".join(new_tokens[n:])
    return new_text


# HTTP statuses that mean "this request can never succeed": authentication, unknown model/language, payload too
# large, malformed. Only 408/429/5xx and transport failures are worth another attempt.
_PERMANENT_HTTP_ERRORS = frozenset({400, 401, 403, 404, 405, 413, 415, 422})


def _require_language(language, cfg) -> str:
    """The language code speech-to-text must be given, or LanguageIdentificationError."""
    eff_lang = (language or (cfg or {}).get("language") or "").strip().lower()
    if not eff_lang or eff_lang in ("auto", "any", "detect", "multi"):
        raise LanguageIdentificationError(
            "Speech-to-text needs the spoken language and none is known: upload the "
            "meeting with the form field `language` (e.g. hi, te, ml, en) or set DEFAULT_WHISPER_LANGUAGE (AIRAWAT_WHISPER_LANGUAGE)."
        )
    return eff_lang


def _describe_http_error(err: Exception, api_key: str = "") -> str:
    """One-line, secret-free description of an HTTP failure, including AIRAWAT's own explanation."""
    resp = getattr(err, "response", None)
    if resp is None:
        return str(err)
    try:
        body = resp.json()
        detail = body.get("detail") or (body.get("error") or {}).get("message") or body.get("message") or ""
    except Exception:
        detail = (resp.text or "")[:200]
    text = f"HTTP {resp.status_code}{' ' + resp.reason if getattr(resp, 'reason', None) else ''}: {detail}".strip()
    return text.replace(api_key, "<key>") if api_key else text


class EmptyCloudTranscription(RuntimeError):
    """Raised when the cloud Whisper endpoint answers 2xx but with no usable text.

    Distinct from a transport/HTTP failure so the caller reports "AIRAWAT returned no text" rather than a
    network error.
    """


def _extract_text_from_transcription(transcription) -> str:
    """Pull the transcript string out of a Whisper-style response defensively.

    Accepts every shape the AIRAWAT / OpenAI-compatible gateways have been seen
    to return: ``{"text": ...}``, a bare string, ``{"transcription": ...}``,
    ``{"data": {"text": ...}}``, ``{"result"/"results": [...] | {...}}`` and
    segment lists. Returns "" for anything blank / whitespace-only.
    """
    if transcription is None:
        return ""
    if isinstance(transcription, str):
        return transcription.strip()
    if not isinstance(transcription, dict):
        return (getattr(transcription, "text", "") or "").strip()

    for key in ("text", "transcription", "transcript"):
        val = transcription.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()

    nested = transcription.get("data") or transcription.get("result") or transcription.get("results")
    if isinstance(nested, dict):
        t = _extract_text_from_transcription(nested)
        if t:
            return t
    if isinstance(nested, list):
        parts = [_extract_text_from_transcription(item) for item in nested]
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            return joined

    segments = transcription.get("segments")
    if isinstance(segments, list):
        parts = []
        for seg in segments:
            if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                parts.append(seg["text"].strip())
        joined = " ".join(p for p in parts if p).strip()
        if joined:
            return joined
    return ""


def _whisper_timeouts(chunk_bytes: int) -> tuple:
    """(connect, read) timeout for one AIRAWAT Whisper attempt; the read timeout grows with the chunk's audio length.

    urllib3 also applies the CONNECT timeout to each send of the request body; the upload is streamed in blocks
    (``_StreamingMultipartBody``), so it is the longest an upload may stall, not a limit on the whole upload.
    """
    connect = max(1.0, float(settings.AIRAWAT_WHISPER_CONNECT_TIMEOUT_SEC))
    audio_min = max(0, chunk_bytes) / float(settings.BYTES_PER_SEC_WAV or 32000) / 60.0
    read = float(settings.AIRAWAT_WHISPER_READ_TIMEOUT_SEC) + float(settings.AIRAWAT_WHISPER_READ_TIMEOUT_PER_MIN_SEC) * audio_min
    read = min(read, max(float(settings.AIRAWAT_WHISPER_READ_TIMEOUT_MAX_SEC), float(settings.AIRAWAT_WHISPER_READ_TIMEOUT_SEC)))
    return connect, round(max(10.0, read), 1)


class _StreamingMultipartBody:
    """multipart/form-data upload that reads the audio file from disk block by block while it is sent.

    requests encodes ``files=`` into ONE bytes object sent with a single ``sendall``, which urllib3 bounds with the
    connect timeout from start to end: a healthy upload slower than that is cut off ("The write operation timed
    out") and a stalled one is only noticed when the whole allowance has passed (252 s measured for a 14.5 MB
    chunk). Sent in blocks, the timeout applies to each block: a stall fails within it, a slow upload that keeps
    moving completes. ``len()`` gives requests the Content-Length, so the body is not sent chunked.
    """

    def __init__(self, fields: dict, file_name: str, file_obj, file_type: str):
        boundary = uuid.uuid4().hex
        self.content_type = f"multipart/form-data; boundary={boundary}"
        pairs = list(fields.items()) if isinstance(fields, dict) else list(fields)
        self.fields = dict(pairs)
        safe_name = file_name.replace('"', "_").replace("\r", "_").replace("\n", "_")
        head = b"".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8")
            for k, v in pairs
        ) + (
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{safe_name}"\r\n'
            f"Content-Type: {file_type}\r\n\r\n"
        ).encode("utf-8")
        tail = f"\r\n--{boundary}--\r\n".encode("utf-8")
        file_size = os.fstat(file_obj.fileno()).st_size - file_obj.tell()
        self._parts = [io.BytesIO(head), file_obj, io.BytesIO(tail)]
        self._len = len(head) + file_size + len(tail)
        self.upload_done_at = None  # time.monotonic() when the last byte was handed to the socket

    def __len__(self) -> int:
        return self._len

    def read(self, size: int = -1) -> bytes:
        out = b""
        while self._parts and (size is None or size < 0 or len(out) < size):
            block = self._parts[0].read(-1 if size is None or size < 0 else size - len(out))
            if block:
                out += block
            else:
                self._parts.pop(0)
        if not self._parts and not out and self.upload_done_at is None:
            self.upload_done_at = time.monotonic()
        return out

    def timing(self, started: float) -> str:
        """'upload Xs, Airawat processing Ys' for an attempt that started at ``started`` (monotonic)."""
        if self.upload_done_at is None:
            return f"upload not finished after {time.monotonic() - started:.1f}s"
        return f"upload {self.upload_done_at - started:.1f}s, Airawat processing {time.monotonic() - self.upload_done_at:.1f}s"


def _retry_wait_sec(attempt: int, retry_after=None) -> float:
    """Exponential backoff before attempt ``attempt + 1`` (``attempt`` is 1-based), with jitter; honours Retry-After."""
    cap = float(settings.AIRAWAT_RETRY_BACKOFF_MAX_SEC)
    try:
        if retry_after is not None:
            return round(min(max(0.0, float(retry_after)), cap), 1)
    except (TypeError, ValueError):
        pass
    wait = min(float(settings.AIRAWAT_RETRY_BACKOFF_BASE_SEC) * (2 ** (attempt - 1)), cap)
    return round(wait + random.uniform(0, wait * 0.2), 1)


def post_whisper_chunk(
    chunk_filepath: str,
    chunk_index: int,
    total_chunks: int,
    *,
    url: str,
    api_key: str,
    fields,
    transport_retries: int,
    provider_label: str,
) -> dict:
    """POST one audio chunk to an OpenAI-compatible ``/audio/transcriptions`` endpoint (AIRAWAT, Groq).

    Retries on transport errors (``transport_retries``) AND, separately, on "HTTP 2xx but empty body" responses (per
    CLOUD_EMPTY_RESPONSE_RETRIES), which gateways return intermittently for large / rapid-fire uploads. Each attempt
    uses a separate connect / read timeout (the read timeout scales with the chunk's audio length) and attempts are
    spaced by exponential backoff that honours ``Retry-After`` on 429. A permanent HTTP error (bad key, unknown
    model/language, oversized upload) fails at once. When every attempt fails ``AIProviderError`` is raised, so the
    meeting is never finished with an incomplete transcript; ``EmptyCloudTranscription`` when the endpoint keeps
    answering without text.

    ``fields``: the form fields (dict, or a list of pairs for repeated names). The API key is never logged.
    """
    empty_retries = max(0, int(getattr(settings, "CLOUD_EMPTY_RESPONSE_RETRIES", 2)))
    max_attempts = max(1, int(transport_retries)) + empty_retries

    hard_cap = int(getattr(settings, "CLOUD_HARD_MAX_BYTES", 24 * 1024 * 1024))
    try:
        chunk_bytes = os.path.getsize(chunk_filepath)
    except OSError:
        chunk_bytes = 0
    if chunk_bytes > hard_cap:
        logger.warning(
            f"Chunk {chunk_index+1}/{total_chunks} is {chunk_bytes/1e6:.1f} MB — "
            f"over the {hard_cap/1e6:.0f} MB endpoint hard limit; expect rejection."
        )

    connect_timeout, read_timeout = _whisper_timeouts(chunk_bytes)
    tag = f"Chunk {chunk_index+1}/{total_chunks}"
    last_err, last_desc = None, ""
    started = time.monotonic()
    for attempt in range(1, max_attempts + 1):
        retry_after = None
        body = None
        attempt_started = time.monotonic()
        try:
            logger.info(
                f"Transcribing chunk {chunk_index+1}/{total_chunks} "
                f"({chunk_bytes/(1024*1024):.1f} MB) via STT API — attempt {attempt}/{max_attempts} "
                f"(connect/upload-stall timeout={connect_timeout:.0f}s, read timeout={read_timeout:.0f}s)"
            )
            with open(chunk_filepath, "rb") as f:
                body = _StreamingMultipartBody(fields, os.path.basename(chunk_filepath), f, "audio/wav")
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": body.content_type}
                resp = requests.post(
                    url,
                    headers=headers,
                    data=body,
                    verify=settings.SSL_VERIFY,
                    timeout=(connect_timeout, read_timeout),
                )
            resp.raise_for_status()

            try:
                payload = resp.json()
            except ValueError:
                payload = (resp.text or "").strip()

            if _extract_text_from_transcription(payload):
                logger.info(
                    f"{tag} transcribed successfully on attempt {attempt}/{max_attempts} "
                    f"(attempt {time.monotonic() - attempt_started:.1f}s: {body.timing(attempt_started)}; "
                    f"total {time.monotonic() - started:.1f}s)"
                )
                return payload if isinstance(payload, dict) else {"text": str(payload)}

            # HTTP success but no transcript text — retry a few times, then signal.
            snippet = (resp.text or "")[:180].replace("\n", " ")
            last_err = EmptyCloudTranscription(
                f"chunk {chunk_index+1}/{total_chunks}: HTTP {resp.status_code} "
                f"with no transcript text (body: {snippet!r})"
            )
            last_desc = f"HTTP {resp.status_code} with no transcript text"
            logger.warning(f"{tag} attempt {attempt}/{max_attempts}: EMPTY RESPONSE ({snippet!r}).")
        except requests.exceptions.ConnectTimeout as err:
            # Checked before ConnectionError/Timeout: ConnectTimeout subclasses both.
            last_err, last_desc = err, f"connect timeout (no connection within {connect_timeout:.0f}s)"
            logger.warning(f"{tag} attempt {attempt}/{max_attempts} failed: CONNECT TIMEOUT after {connect_timeout:.0f}s: {err}.")
        except requests.exceptions.ReadTimeout as err:
            last_err, last_desc = err, f"read timeout (no answer within {read_timeout:.0f}s)"
            logger.warning(
                f"{tag} attempt {attempt}/{max_attempts} failed: READ TIMEOUT — audio uploaded but STT API sent no "
                f"answer within {read_timeout:.0f}s ({body.timing(attempt_started) if body else 'no upload'})."
            )
        except requests.exceptions.HTTPError as err:
            code = getattr(err.response, "status_code", None)
            if code in _PERMANENT_HTTP_ERRORS:
                # Bad key / unsupported language / oversized upload: retrying the identical request cannot help.
                raise AIProviderError(
                    f"Speech-to-text rejected chunk {chunk_index+1}/{total_chunks}: {_describe_http_error(err, api_key)}",
                    details={"http_status": code, "provider": provider_label},
                ) from err
            if code == 429 and err.response is not None:
                retry_after = err.response.headers.get("Retry-After")
            last_err, last_desc = err, _describe_http_error(err, api_key)
            logger.warning(f"{tag} attempt {attempt}/{max_attempts} failed: HTTP ERROR {last_desc}.")
        except requests.exceptions.RequestException as err:
            # ConnectionError (DNS, refused, reset, SSL, proxy) and any other transport failure.
            last_err, last_desc = err, f"network error ({type(err).__name__}: {err})"
            logger.warning(f"{tag} attempt {attempt}/{max_attempts} failed: NETWORK ERROR {type(err).__name__}: {err}.")
        except Exception as err:
            last_err, last_desc = err, f"unexpected error ({type(err).__name__}: {err})"
            logger.warning(f"{tag} attempt {attempt}/{max_attempts} failed: UNEXPECTED ERROR {type(err).__name__}: {err}.")

        if attempt < max_attempts:
            wait = _retry_wait_sec(attempt, retry_after)
            logger.info(
                f"{tag}: retry {attempt}/{max_attempts - 1} in {wait:.1f}s (exponential backoff) after {last_desc}; "
                f"next attempt {attempt + 1}/{max_attempts}."
            )
            time.sleep(wait)

    elapsed = time.monotonic() - started
    logger.error(
        f"{tag} FAILED: Speech-to-text gave no transcript after {max_attempts} attempt(s) in {elapsed:.0f}s; "
        f"last error: {last_desc}."
    )
    if isinstance(last_err, EmptyCloudTranscription):
        raise last_err
    raise AIProviderError(
        f"Speech-to-text did not answer chunk {chunk_index+1}/{total_chunks} after {max_attempts} attempt(s) "
        f"in {elapsed:.0f}s; last error: {last_desc}. The transcription was stopped instead of saving an "
        f"incomplete transcript; re-process the meeting once the service answers.",
        details={"chunk": chunk_index + 1, "total_chunks": total_chunks, "attempts": max_attempts,
                 "connect_timeout_sec": connect_timeout, "read_timeout_sec": read_timeout, "last_error": last_desc},
    ) from last_err




def transcribe_single_chunk(
    chunk_filepath: str, chunk_index: int, total_chunks: int, language: str = None,
    vocabulary_hint: str = None,
) -> dict:
    """
    Transcribes a single audio chunk via the AIRAWAT Whisper API (retry policy: ``post_whisper_chunk``).

    ``vocabulary_hint`` (optional): a short, dynamically-built string of this
    meeting's own proper nouns / project terms (see
    ``transcript_correction_service.build_vocabulary_hint``), sent as the
    OpenAI-compatible Whisper ``prompt`` field to bias the decoder toward the
    correct spelling of domain-specific words instead of guessing phonetically.
    """
    from backend.services.model_router import model_router

    cfg = model_router._get_dynamic_config().get("airawat", {})
    api_key = cfg.get("api_key") or settings.AIRAWAT_API_KEY
    if not api_key or not api_key.strip():
        raise AirawatNotConfiguredError("AIRAWAT speech-to-text is not configured: no API key.")

    # AIRAWAT's Whisper gateway has NO auto-detection: it answers "400 need model or language" to a request
    # without a language code and "404 Model not found for language auto" to "auto", and the language it is
    # given decides the output (English audio + hi -> Devanagari transliteration, + ko -> a Korean
    # translation). A `model` field is rejected outright ("bad model: whisper-1"), so none is ever sent.
    eff_lang = _require_language(language, cfg)
    # The SOURCE language is sent (never forced to "en"); translation to English is a later LLM stage, so
    # timestamps stay aligned.
    data = {"language": eff_lang, "language_code": eff_lang}
    if vocabulary_hint:
        data["prompt"] = vocabulary_hint

    base_url = cfg.get("base_url") or settings.AIRAWAT_BASE_URL
    whisper_endpoint = cfg.get("whisper_endpoint") or ""
    # Always fall back to the .env-configured AIRAWAT_WHISPER_URL — never hardcode
    if whisper_endpoint.startswith("http://") or whisper_endpoint.startswith("https://"):
        whisper_url = whisper_endpoint
    elif whisper_endpoint and base_url:
        whisper_url = base_url.rstrip("/") + "/" + whisper_endpoint.lstrip("/")
    else:
        whisper_url = settings.AIRAWAT_WHISPER_URL
    logger.info(f"Using Whisper URL: {whisper_url}")

    return post_whisper_chunk(
        chunk_filepath, chunk_index, total_chunks,
        url=whisper_url,
        api_key=api_key,
        fields=data,
        transport_retries=max(1, int(cfg.get("retry_count") or settings.AIRAWAT_RETRY_COUNT)),
        provider_label="AIRAWAT",
    )


def _estimate_words_from_text(
    text: str, offset_sec: float, chunk_duration: float
) -> list:
    """
    When the Airawat Whisper response does not include word-level timestamps,
    estimate word timestamps using a realistic speaking rate model.

    Uses a weighted distribution: shorter words are faster, longer words are slower,
    with a baseline of ~3.5 words/sec (natural English speaking rate).
    Falls back to even distribution if duration is unavailable.
    """
    words = text.split()
    if not words:
        return []

    # Realistic speaking rate: 130-160 words per minute = ~2.3-2.7 words/sec
    # Use 2.5 wps as baseline, adjust per-word by length

    # Compute per-word durations based on word length
    word_weights = []
    for w in words:
        wlen = len(w)
        if wlen <= 2:
            weight = 0.3  # short words (I, a, the) are fast
        elif wlen <= 5:
            weight = 0.5  # medium words
        elif wlen <= 8:
            weight = 0.7  # longer words
        else:
            weight = 1.0  # very long words take more time
        word_weights.append(weight)

    total_weight = sum(word_weights) if word_weights else 1.0
    out = []
    current_time = offset_sec

    for i, w in enumerate(words):
        word_dur = (word_weights[i] / total_weight) * chunk_duration
        # Clamp: minimum 0.05s, maximum 1.5s per word
        word_dur = max(0.05, min(1.5, word_dur))
        end_time = current_time + word_dur
        out.append(
            {"word": w, "start": round(current_time, 3), "end": round(end_time, 3)}
        )
        current_time = end_time

    return out


def extract_words_from_transcription(
    transcription, offset_sec: float, is_first_chunk: bool = True
) -> list:
    """
    Extracts word-level timestamps from a cloud transcription response,
    adjusting all times by the chunk's start offset.
    For non-first chunks, subtracts CHUNK_OVERLAP_SEC from word times to correct
    the overlap-shifted timestamps back to absolute video time.
    Falls back to segment-level timestamps if word-level is unavailable.
    """
    words = []
    # Correction: non-first chunks were started CHUNK_OVERLAP_SEC early,
    # so their internal timestamps are CHUNK_OVERLAP_SEC too high.
    overlap_correction = 0.0 if is_first_chunk else CHUNK_OVERLAP_SEC
    words_attr = getattr(transcription, "words", None)
    if words_attr is None and isinstance(transcription, dict):
        words_attr = transcription.get("words")

    if words_attr:
        for w in words_attr:
            if w is None:
                continue
            word_str = w.get("word") if isinstance(w, dict) else getattr(w, "word", "")
            start_t = (
                w.get("start") if isinstance(w, dict) else getattr(w, "start", 0.0)
            ) or 0.0
            end_t = (
                w.get("end") if isinstance(w, dict) else getattr(w, "end", 0.0)
            ) or 0.0
            # Apply offset and subtract overlap correction
            adj_start = round(start_t + offset_sec - overlap_correction, 3)
            adj_end = round(end_t + offset_sec - overlap_correction, 3)
            # Skip words that would have negative timestamps (in overlap region of next chunk)
            if adj_start < 0:
                continue
            words.append({"word": word_str, "start": adj_start, "end": adj_end})
        return words

    # Fallback: segment-level timestamps
    segments = getattr(transcription, "segments", None)
    if segments is None and isinstance(transcription, dict):
        segments = transcription.get("segments", [])
    if not segments:
        return words

    for seg in segments:
        if seg is None:
            continue
        seg_text = (
            seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")
        )
        raw_start = (
            seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", 0.0)
        ) or 0.0
        raw_end = (
            seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", 0.0)
        ) or 0.0
        seg_start = round(raw_start + offset_sec - overlap_correction, 3)
        seg_end = round(raw_end + offset_sec - overlap_correction, 3)
        if seg_start < 0:
            continue
        if seg_text:
            seg_words = seg_text.strip().split()
            if seg_words:
                dur = (seg_end - seg_start) / max(1, len(seg_words))
                for i, word in enumerate(seg_words):
                    words.append(
                        {
                            "word": word,
                            "start": round(seg_start + i * dur, 3),
                            "end": round(seg_start + (i + 1) * dur, 3),
                        }
                    )
    return words


def extract_quality_metrics(transcription) -> tuple:
    """Extracts avg_logprob and avg_no_speech_prob from a cloud transcription response.

    Returns ``None`` for a value the response did not carry. It used to substitute the constants
    ``-0.15`` / ``0.05`` — an invented "86 % confident, 5 % silence" that then looked like a
    measured signal, contradicting the platform's no-fabricated-metrics policy.
    """
    logprobs, no_speeches = [], []
    segments_list = getattr(transcription, "segments", None)
    if segments_list is None and isinstance(transcription, dict):
        segments_list = transcription.get("segments", [])
    if not segments_list:
        return None, None

    for seg in segments_list:
        if seg is None:
            continue
        lp = (
            seg.get("avg_logprob")
            if isinstance(seg, dict)
            else getattr(seg, "avg_logprob", None)
        )
        if lp is not None:
            logprobs.append(lp)
        nsp = (
            seg.get("no_speech_prob")
            if isinstance(seg, dict)
            else getattr(seg, "no_speech_prob", None)
        )
        if nsp is not None:
            no_speeches.append(nsp)

    avg_lp = sum(logprobs) / len(logprobs) if logprobs else None
    avg_nsp = sum(no_speeches) / len(no_speeches) if no_speeches else None
    return avg_lp, avg_nsp


def transcribe_audio_chunked(
    audio_filepath: str,
    language: str = None,
    vocabulary_hint: str = None,
    *,
    chunk_transcriber,
    provider_label: str,
    require_language: bool,
) -> dict:
    """Transcribe a recording of any length with ``chunk_transcriber`` (one call per sub-25 MB chunk).

    Shared by every speech-to-text provider so chunk-overlap trimming, timestamp offsets, hallucination cleaning and
    the empty-transcript guard behave identically. A chunk that cannot be transcribed after its retries fails the
    whole transcription - an incomplete transcript is never saved, and no other provider is tried.
    """
    logger.info(f"Starting {provider_label} speech-to-text for: {os.path.basename(audio_filepath)}")
    if require_language:
        language = _require_language(language, {})
    else:
        language = (language or "").strip().lower() or None

    try:
        chunks = split_audio_into_chunks(audio_filepath)
    except Exception as split_err:
        # Only a file the endpoint accepts in ONE request may be sent whole; a bigger one would be rejected by the
        # provider, so the chunking error (which names the missing time ranges) is the honest outcome.
        try:
            whole_bytes = os.path.getsize(audio_filepath)
        except OSError:
            whole_bytes = 0
        if whole_bytes > int(getattr(settings, "CLOUD_HARD_MAX_BYTES", 24 * 1024 * 1024)):
            raise
        logger.warning(f"Audio chunking failed ({split_err}); sending the whole file as a single chunk.")
        chunks = [(audio_filepath, 0.0, True)]
    total_chunks = len(chunks)
    logger.info(f"Processing {total_chunks} chunk(s) for full transcription.")

    detected_language = None
    combined_words = []
    combined_text = []
    all_no_speeches = []
    all_logprobs = []  # Bug 8 fix: accumulate per-chunk logprobs for aggregate mean
    cloud_ok_chunks = 0
    raw_chunk_chars = 0     # total transcript chars returned by the engine
    clean_chunk_chars = 0   # total after per-chunk hallucination cleaning

    for chunk_idx, (chunk_file, offset_sec, is_first) in enumerate(chunks):
        try:
            chunk_dur = get_audio_info(chunk_file).get("duration") or 30.0
            try:
                transcription = chunk_transcriber(
                    chunk_file, chunk_idx, total_chunks, language=language,
                    vocabulary_hint=vocabulary_hint,
                )
                cloud_ok_chunks += 1
            except EmptyCloudTranscription as chunk_err:
                raise AIProviderError(
                    f"Speech-to-text returned no text for chunk {chunk_idx+1}/{total_chunks} "
                    f"after retries ({chunk_err}). Re-process the meeting "
                    f"once the service answers, or check that the recording contains speech. No other engine is tried."
                ) from chunk_err
        finally:
            if chunk_file != audio_filepath and os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except Exception:
                    pass

        words = extract_words_from_transcription(
            transcription, offset_sec, is_first_chunk=is_first
        )
        raw_text_chunk = _extract_text_from_transcription(transcription)

        # ── Per-chunk hallucination cleaning ────────────────────────────────
        # Whisper (esp. large-v3, and the cloud path has NO VAD) emits long
        # repetition loops on silent / low-speech spans. Cleaning EACH chunk in
        # isolation keeps the blast radius to that chunk: a mostly-silent chunk
        # contributes little, a real-speech chunk keeps all its text. The old
        # code only cleaned the whole joined transcript at the end, so one bad
        # chunk could drag the global unique/total ratio under threshold and
        # zero the ENTIRE transcript ("transcription_returned_no_text").
        from backend.utils.text_signals import clean_whisper_hallucinations
        cleaned_text_chunk = clean_whisper_hallucinations(raw_text_chunk) if raw_text_chunk else ""
        text_chunk = cleaned_text_chunk or raw_text_chunk  # never lose real content to over-cleaning
        raw_chunk_chars += len(raw_text_chunk)
        clean_chunk_chars += len(text_chunk)

        # Remove the words / leading text the chunk overlap duplicated from the previous chunk. Only when chunks
        # really overlap (CHUNK_OVERLAP_SEC > 0), and never more words than fit in the overlap window: without that
        # bound a legitimate repetition at a boundary ("no, no, no" / "very very") was eaten as a "duplicate".
        if text_chunk and combined_text and not is_first and CHUNK_OVERLAP_SEC > 0:
            text_chunk = _trim_overlap_prefix(
                combined_text[-1], text_chunk, max_words=overlap_word_budget(CHUNK_OVERLAP_SEC)
            )

        if words:
            _merge_chunk_words(combined_words, words, offset_sec, is_first)
        elif text_chunk:
            # Text but no word timestamps — estimate them. A non-first chunk file starts CHUNK_OVERLAP_SEC before its
            # nominal offset (and ``chunk_dur`` includes that lead-in); the words that remain after the overlap is
            # trimmed span [offset, offset + chunk_dur - overlap]. Spreading them over the whole file duration pushed
            # the last word 0.5 s past the end of the recording (segment end 1820.98 s for a 1820.48 s file) and every
            # chunk's last word past the next chunk's first.
            est_dur = chunk_dur - (0.0 if is_first else CHUNK_OVERLAP_SEC)
            combined_words.extend(
                _estimate_words_from_text(text_chunk, offset_sec, max(est_dur, 0.5))
            )

        if text_chunk and text_chunk.strip():
            combined_text.append(text_chunk.strip())

        if not detected_language and isinstance(transcription, dict) and transcription.get("language"):
            detected_language = str(transcription["language"]).strip().lower()

        lp, nsp = extract_quality_metrics(transcription)
        if nsp is not None:
            all_no_speeches.append(nsp)
        # Bug 8 fix: collect per-chunk avg_logprob so we can return the aggregate
        if lp is not None:
            all_logprobs.append(lp)

        logger.info(
            "Chunk %d/%d parsed | source=%s | raw_chars=%d clean_chars=%d | "
            "running words=%d",
            chunk_idx + 1, total_chunks,
            provider_label,
            len(raw_text_chunk), len(text_chunk), len(combined_words),
        )

    avg_no_speech = (
        sum(all_no_speeches) / len(all_no_speeches) if all_no_speeches else None
    )
    # Bug 8 fix: return the actual mean logprob across all chunks (not a hardcoded None).
    avg_logprob = (
        sum(all_logprobs) / len(all_logprobs) if all_logprobs else None
    )

    from backend.utils.text_signals import clean_whisper_hallucinations
    # Chunks were already individually cleaned; this final pass only trims any
    # repetition that spans a chunk boundary. It must NOT be able to empty a
    # transcript that has real per-chunk content — guard below enforces that.
    joined_raw = " ".join(combined_text)
    full_text = clean_whisper_hallucinations(joined_raw)
    if not full_text and joined_raw.strip():
        # Cross-chunk cleaner still zeroed it, but individual chunks DID have
        # content — keep the joined per-chunk-cleaned text rather than lose the
        # whole transcription.
        logger.warning(
            "Cross-chunk hallucination cleaner emptied a %d-char transcript that "
            "had real per-chunk content — keeping the per-chunk-cleaned text.",
            len(joined_raw),
        )
        full_text = re.sub(r"\s+", " ", joined_raw).strip()
    word_count = len(full_text.split()) if full_text else 0
    logger.info(
        "Speech-to-text transcription complete | chunks=%d (cloud_ok=%d) | raw_chunk_chars=%d clean_chunk_chars=%d | "
        "joined_chars=%d final_chars=%d | word_timestamps=%d final_words=%d",
        total_chunks, cloud_ok_chunks, raw_chunk_chars, clean_chunk_chars, len(joined_raw), len(full_text),
        len(combined_words), word_count,
    )

    # ── Whole-file empty guard: an HONEST empty_reason (no other engine is tried) ──
    if not full_text:
        logger.error("Speech-to-text returned no text for the whole recording.")
        return {
            "text": "",
            "words": [],
            "avg_logprob": None,
            "avg_no_speech": avg_no_speech,
            "language": language or detected_language,
            "empty_reason": f"{provider_label}_returned_no_speech_text",
        }

    return {
        "text": full_text,
        "words": combined_words,
        "avg_logprob": avg_logprob,   # Now the real aggregate mean across all chunks
        "avg_no_speech": avg_no_speech,
        # The language the audio was transcribed as: the requested one, else what the engine reported (None when it
        # reported nothing - never assumed).
        "language": language or detected_language,
    }


def transcribe_audio_airawat(audio_filepath: str, language: str = None, vocabulary_hint: str = None) -> dict:
    """
    Transcribes audio with the AIRAWAT Whisper API, with dynamic chunking (see ``transcribe_audio_chunked``).

    ``language``: ISO code of the spoken language (the upload's, else the configured default); AIRAWAT transcribes in
    that SOURCE language and cannot detect it. Translation to English is a later pipeline stage, not done here.
    """
    # Fail before any audio work if AIRAWAT cannot be asked at all (no key / no language).
    from backend.services.model_router import model_router as _mr

    _cfg = _mr._get_dynamic_config().get("airawat", {})
    if not (_cfg.get("api_key") or settings.AIRAWAT_API_KEY or "").strip():
        raise AIProviderNotConfiguredError("Speech-to-text provider is not configured: no API key.")
    language = _require_language(language, _cfg)
    return transcribe_audio_chunked(
        audio_filepath, language=language, vocabulary_hint=vocabulary_hint,
        # resolved at call time, so a test can substitute the module-level chunk call
        chunk_transcriber=lambda *a, **k: transcribe_single_chunk(*a, **k),
        provider_label="airawat", require_language=True,
    )


def transcribe_audio(audio_filepath: str, language: str = None, vocabulary_hint: str = None) -> dict:
    """
    Main entry point for transcription: speech-to-text via ModelRouter.

    ``language``: ISO code of the spoken language. Output is always
    the SOURCE language here; English translation is a downstream stage.
    """
    from backend.services.model_router import model_router

    return model_router.transcribe(audio_filepath, language=language, vocabulary_hint=vocabulary_hint)
