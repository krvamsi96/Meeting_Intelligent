"""How a meeting's lifecycle ``status`` is reported through the API status fields.

``meetings.api_status`` / ``meetings.api_status_code`` used to be written once, by the column default, as
``"Success"`` / ``200`` and never touched again - so a meeting whose pipeline FAILED (``status="error"``) or was still
running was served with ``api_status="Success"``. They now follow the actual processing outcome:

    completed            -> ("Success", 200)
    error / failed       -> ("Failed", 500)
    processing           -> ("Processing", 202)
    uploaded / queued .. -> ("Pending", 202)

The HTTP status of the GET itself is unchanged (200 - the request succeeded and returns the meeting); the fields
carry the outcome of the PROCESSING. ``status``, ``overall_status`` and ``error_message`` are exposed next to them,
unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

_FAILED = {"error", "failed", "failure"}
_PROCESSING = {"processing", "running", "in_progress", "retrying"}


def api_status_for(status: Optional[str]) -> Tuple[str, int]:
    """(api_status, api_status_code) that truthfully describes a meeting with this lifecycle ``status``."""
    s = (status or "").strip().lower()
    if s == "completed":
        return "Success", 200
    if s in _FAILED:
        return "Failed", 500
    if s in _PROCESSING:
        return "Processing", 202
    return "Pending", 202


def api_status_fields(status: Optional[str]) -> Dict[str, object]:
    """The two meeting columns as a dict, ready for ``meeting_updates`` / ``setattr`` / a bulk UPDATE."""
    label, code = api_status_for(status)
    return {"api_status": label, "api_status_code": code}


# ─── What end users are shown ───────────────────────────────────────────────────────────────────────────────────
# The web app speaks five states. ``error_message`` keeps the technical cause for logs and administrators; users get
# one of the plain messages below instead (it used to show the raw exception, provider names and URLs included).

STATUS_FILTERS: Dict[str, Tuple[str, ...]] = {
    "completed": ("completed",),
    "processing": ("uploaded", "processing", "running", "retrying", "queued", "pending"),
    "failed": ("error", "failed", "failure"),
}


def display_status(status: Optional[str]) -> str:
    """``completed`` | ``processing`` | ``failed`` for a lifecycle ``status`` (unknown values count as processing)."""
    s = (status or "").strip().lower()
    for group, members in STATUS_FILTERS.items():
        if s in members:
            return group
    return "processing"


_USER_ERRORS = (
    # (substrings of the technical cause, lower-case) -> (code, message for the user)
    (("language is unknown", "language_not", "needs the spoken language"),
     "LANGUAGE_REQUIRED", "The spoken language of this recording could not be identified. Upload it again and choose "
                          "its language."),
    (("not configured", "stt_provider", "does not transcribe", "no api key", "not supported. use one of",
      "http 401", "http 403", "rejected the api key", "invalid api key"),
     "SERVICE_NOT_CONFIGURED", "Meeting processing is not set up yet. Please contact your administrator."),
    (("returned_no_speech_text", "returned no text", "no speech", "no text produced"),
     "NO_SPEECH", "No speech could be recognised in this recording."),
    (("ffmpeg", "audio extraction", "could not read", "invalid data", "no audio stream", "audio duration",
      "corrupt", "unsupported file"),
     "MEDIA_UNREADABLE", "This recording could not be read. Check the file and upload it again."),
    (("rate limit", "429", "timeout", "timed out", "did not answer", "unreachable", "connection",
      "provider failed", "failed to generate", "failed while generating", "speech-to-text", "http 5"),
     "AI_SERVICE_UNAVAILABLE", "The AI service is temporarily unavailable. Please try again later."),
)
_GENERIC_ERROR = ("PROCESSING_FAILED", "We couldn't process this meeting. Please try again.")


def user_error(error_message: Optional[str]) -> Dict[str, str]:
    """``{"code", "message"}`` safe to show any user, derived from the stored technical ``error_message``."""
    text = (error_message or "").lower()
    for needles, code, message in _USER_ERRORS:
        if any(n in text for n in needles):
            return {"code": code, "message": message}
    return {"code": _GENERIC_ERROR[0], "message": _GENERIC_ERROR[1]}


# Plain-language note for a stage that finished DEGRADED (completed, but with reduced quality or partly missing).
_STAGE_WARNINGS = (
    (("diarization", "alignment"), "Speakers may not be identified accurately in this recording."),
    (("stt", "transcript_correction"), "Parts of the transcript may be less accurate than usual."),
    (("summary", "mom", "agenda", "decision", "action_items", "topics", "sentiment"),
     "Some meeting results may be incomplete."),
    (("embeddings_vector", "rag_chatbot"), "This meeting may not appear in search results yet."),
)


def user_warnings(stage_states_json) -> list:
    """Distinct user-facing warnings for the DEGRADED stages of a run (empty when everything was full quality)."""
    import json

    try:
        states = json.loads(stage_states_json) if isinstance(stage_states_json, str) else (stage_states_json or {})
    except (TypeError, ValueError):
        return []
    degraded = {str(k) for k, v in (states or {}).items() if str(v).upper() == "DEGRADED"}
    return [msg for stages, msg in _STAGE_WARNINGS if degraded & set(stages)]
