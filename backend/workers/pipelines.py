"""Background ingestion pipelines - media, document, and standalone document processing.

Each pipeline runs as a FastAPI BackgroundTasks callback. The PIPELINE_REGISTRY
maps file types to processing functions dynamically.
"""

import json
import logging
import os
import time

from backend.config.config import DEFAULT_PROJECT_NAME, settings
from backend.database.connection import SessionLocal
from backend.evaluation import runtime_quality as rq
from backend.models.models import (
    ActionItem,
    AgendaKeyDecision,
    AnalysisResult,
    GroundTruth,
    Meeting,
    TranscriptSegment,
)
from backend.services import similarity_service
from backend.services.audio_service import extract_audio
from backend.services.diarization_service import diarize_and_align
from backend.services.document_service import (
    create_document_segments,
    extract_document_metadata_with_llm,
    extract_document_text,
)
from backend.evaluation.metrics import rouge_scores
from backend.services.llm_workflow import process_meeting_report
from backend.services.summary_service import extractive_summary_for_transcript, finalize_summary
from backend.services.transcription_service import transcribe_audio
from backend.services.meeting_vector_metadata import build_meeting_vector_metadata  # noqa: F401 (also imported from here by the API)
from backend.services.vector_service import vector_service
from backend.utils import text_signals as ts
from backend.utils.exceptions import AIProviderError, AirawatError, LanguageIdentificationError
from backend.utils.helpers import check_outputs_generated, clean_agenda_text, count_transcript_words
from backend.utils.logging_setup import PipelineLogContext
from backend.utils.processing_status import api_status_fields, api_status_for

logger = logging.getLogger(__name__)


# ─── ffprobe video quality probe ──────────────────────────────────────────────


def _probe_video_quality(file_path: str):
    """
    Uses ffprobe to extract fps, width, height, and whether a video stream exists.
    Returns (fps, width, height, has_video_stream).
    Returns (None, None, None, False) for audio-only or on any error.
    """
    import json as _json
    import subprocess
    from backend.utils.media_tools import FFPROBE

    try:
        cmd = [
            FFPROBE,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_streams",
            "-select_streams",
            "v:0",
            file_path,
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        if result.returncode != 0:
            return None, None, None, False
        data = _json.loads(result.stdout)
        streams = data.get("streams", [])
        if not streams:
            return None, None, None, False  # Audio-only file
        stream = streams[0]
        width = stream.get("width")
        height = stream.get("height")
        # fps stored as fraction string e.g. "30000/1001" or "30/1"
        fps = None
        r_frame_rate = stream.get("r_frame_rate", "0/1")
        try:
            num, den = r_frame_rate.split("/")
            fps = float(num) / float(den) if float(den) != 0 else None
        except Exception:
            pass
        return fps, width, height, True
    except Exception as e:
        logger.debug(f"ffprobe video quality probe failed for {file_path}: {e}")
        return None, None, None, False


# ─── Weighted accuracy recalculation ──────────────────────────────────────────


def _compute_runtime_quality_signal(meeting, reports: dict, full_transcript_text: str, db=None):
    """Set accuracy from ground truth, or compute a reference-free QUALITY SIGNAL.

    ACCURACY POLICY
    ---------------
    * ``meeting.accuracy`` (and ``transcription_accuracy`` / ``speaker_accuracy``)
      are TRUE ACCURACY numbers. They are populated ONLY by comparing the
      generated output against a ground-truth reference (WER, DER, F1, ...).
    * When NO ground truth exists, those columns are set to ``None`` (NULL) —
      never a percentage. This function does NOT fabricate accuracy.

    When ground truth IS available:
      → run the Universal Evaluation Engine (run_meeting_evaluation), which
        computes real WER/CER/DER/JER/precision/recall/F1/ROUGE and writes both
        ``evaluation_results`` and the legacy ``meeting.*_accuracy`` columns.

    When ground truth is NOT available:
      → compute a reference-free QUALITY SIGNAL (0-100) from pipeline evidence
        (audio / STT completeness / diarization / content grounding / topics)
        and store it ONLY in ``meeting.metrics_json`` under
        ``quality_signal_score``. It is explicitly flagged ``is_accuracy: false``
        and is NEVER written to any ``*accuracy`` field.
    """
    import json as _json
    import math as _math

    # Use a direct DB COUNT query to check ground truth — the ORM relationship
    # may not be loaded at this point (lazy-load would work but is unreliable
    # across session boundaries). A COUNT is lightweight and always accurate.
    # Ground truth is OPTIONAL. The lookup runs in a SAVEPOINT (backend.utils.db_safe): if it fails, only the
    # savepoint is rolled back - the pipeline's report rows already written to this session are kept - and the
    # answer is "no ground truth", so reference-free evaluation below still runs. (The old code ran the query bare:
    # the first failed statement aborted the transaction and the lazy `meeting.ground_truth` load that followed
    # raised InFailedSqlTransaction, which became the meeting's error_message.)
    has_gt = False
    gt_eval_failed = False
    if db is not None:
        from backend.utils.db_safe import has_ground_truth as _has_ground_truth

        has_gt = _has_ground_truth(db, meeting.id)
    else:
        try:
            has_gt = bool(getattr(meeting, "ground_truth", None))
        except Exception:
            has_gt = False

    if has_gt:
        # Ground truth present → compute REAL accuracy now (WER/DER/F1/ROUGE...).
        # This closes the loop: previously the pipeline just skipped and hoped
        # someone later called the /evaluation API, leaving meeting.accuracy NULL.
        from backend.services.true_evaluation_service import run_meeting_evaluation
        from backend.utils.db_safe import optional_db as _optional_db

        _eval_failed = object()
        # commit=False + savepoint: a failure inside the engine rolls back ONLY the evaluation's own rows. The engine's
        # normal commit would otherwise also commit (or, on failure, poison) the pipeline's pending report rows.
        _evaluated = _optional_db(
            db,
            lambda: run_meeting_evaluation(meeting, db, performed_by="pipeline", commit=False),
            default=_eval_failed,
            label=f"ground-truth evaluation for meeting {meeting.id}",
        )
        if _evaluated is not _eval_failed:
            logger.info(
                f"Meeting {meeting.id} has Ground Truth — ran Universal Evaluation Engine; "
                f"meeting.accuracy={getattr(meeting, 'accuracy', None)} (ground-truth based)."
            )
            return
        # Failed (cause already logged by optional_db): accuracy stays NULL - never a fabricated value - and we FALL
        # THROUGH to the reference-free quality signal so summary/MoM/action-item processing is unaffected.
        gt_eval_failed = True
        meeting.accuracy = None
        meeting.transcription_accuracy = None
        meeting.speaker_accuracy = None

    # ── Signal A: Audio Quality (20%) ─────────────────────────────────────────
    # avg_no_speech is the Whisper "silence probability" (0=all speech, 1=all silence)
    # A low no_speech_prob means good, speech-rich audio.
    avg_no_speech = getattr(meeting, "avg_no_speech_prob", None)
    if avg_no_speech is not None:
        audio_score = max(0.0, min(100.0, (1.0 - float(avg_no_speech)) * 100.0))
    else:
        # No signal: use video/audio quality field if set, otherwise neutral 70
        raw_aq = getattr(meeting, "audio_quality", None)
        audio_score = float(raw_aq) if raw_aq is not None else 70.0

    # ── Signal B: STT Quality (25%) ────────────────────────────────────────────
    # Prefer actual Whisper logprob-derived confidence stored in the meeting row.
    # Fall back to vocabulary-diversity heuristic only when confidence is absent.
    # NOTE: word_tokens is defined here unconditionally so the ROUGE fallback
    # block below (Signal D) can always reference it without NameError.
    word_tokens = full_transcript_text.lower().split() if full_transcript_text else []
    avg_stt_conf = getattr(meeting, "avg_stt_confidence", None)
    if avg_stt_conf is not None:
        try:
            # avg_stt_confidence is stored as a fraction in [-1, 0] (Whisper logprob)
            # or directly as [0, 1] depending on the pipeline version.
            conf_f = float(avg_stt_conf)
            if conf_f < 0:  # logprob: convert via exp(-abs(logprob))
                import math as _m
                stt_score = max(0.0, min(98.0, _m.exp(conf_f) * 100.0))
            else:           # already a probability [0, 1]
                stt_score = max(0.0, min(98.0, conf_f * 100.0))
        except (TypeError, ValueError):
            avg_stt_conf = None  # fall through to heuristic

    if avg_stt_conf is None:
        if len(word_tokens) >= 10:
            unique_ratio = len(set(word_tokens)) / len(word_tokens)
            # Scale: unique_ratio in [0.3, 0.9] → stt_score in [65, 95]
            stt_score = max(50.0, min(98.0, 50.0 + unique_ratio * 80.0))
        elif len(word_tokens) > 0:
            stt_score = 50.0  # Very short transcript — degraded
        else:
            stt_score = 0.0   # No transcript at all

    # ── Signal C: Diarization Quality (15%) ───────────────────────────────────
    # Computed dynamically from actual segment timestamps.
    # rq.diarization_quality_signal returns a [0, 100] quality signal (NOT accuracy).
    # diar_status is used as a reliability multiplier inside the function.
    # No hardcoded constants — values change with actual pipeline output.
    diar_status = getattr(meeting, "diarization_status", None)
    audio_dur_for_diar = float(getattr(meeting, "duration", 0) or 0)

    # Build minimal segment dicts from meeting object for signal calculation.
    # The full TranscriptSegment objects are not available here (no DB access);
    # use the diarization_info_json as proxy if segments are unavailable.
    try:
        diar_info_raw = getattr(meeting, "diarization_info_json", None) or "{}"
        diar_info = json.loads(diar_info_raw) if isinstance(diar_info_raw, str) else diar_info_raw
        diag = diar_info.get("diagnostics", {})
        # Build minimal synthetic segments from speaker duration data.
        # The diagnostic dict uses 'speaker_breakdown' → {speaker_id: {speaking_duration_seconds: N, ...}}
        # Keys 'speaker_duration' and 'speakers' do NOT exist in compute_speaker_diagnostics() output.
        speaker_breakdown = diag.get("speaker_breakdown", {})
        seg_proxies = []
        t = 0.0
        for spk, spk_data in (speaker_breakdown.items() if isinstance(speaker_breakdown, dict) else []):
            # spk_data may be a dict (full breakdown) or a raw float (legacy)
            if isinstance(spk_data, dict):
                dur = float(spk_data.get("speaking_duration_seconds", 0) or 0)
            else:
                dur = float(spk_data or 0)
            if dur > 0:
                seg_proxies.append({"speaker": spk, "start_time": t, "end_time": t + dur})
                t += dur
        if not seg_proxies:
            # No speaker data — fall back to simple status-based signal
            diar_score = (
                85.0 if diar_status == "SUCCESS" else
                55.0 if diar_status == "DEGRADED" else
                15.0 if diar_status == "FAILED" else
                50.0
            )
        else:
            diar_score = rq.diarization_quality_signal(seg_proxies, audio_dur_for_diar, diar_status)
    except Exception:
        # Fallback if anything goes wrong — use status-based estimate
        diar_score = (
            80.0 if diar_status == "SUCCESS" else
            50.0 if diar_status == "DEGRADED" else
            15.0 if diar_status == "FAILED" else
            45.0
        )

    # ── Signal D: Content Quality — ROUGE + MoM completeness (25%) ────────────
    summary_text = ""
    mom_text = ""
    if meeting.analysis:
        summary_text = meeting.analysis.summary or ""
        mom_text = meeting.analysis.mom or ""
    elif reports:
        summary_text = reports.get("summary", "")
        mom_text = reports.get("mom", "")

    # ROUGE-L F1 stored in confidence_score by _run_shared_pipeline
    rouge_val = getattr(meeting, "confidence_score", None)
    if rouge_val is not None and 0.0 <= float(rouge_val) <= 100.0:
        rouge_score = float(rouge_val)
    else:
        # Fallback: compute lightweight token overlap instead of a full ROUGE pass
        sum_tokens = set(summary_text.lower().split())
        tx_tokens = set(word_tokens)
        if sum_tokens and tx_tokens:
            overlap = len(sum_tokens & tx_tokens) / len(sum_tokens)
            rouge_score = max(0.0, min(100.0, overlap * 100.0))
        else:
            rouge_score = 0.0 if not summary_text.strip() else 60.0

    # MoM completeness bonus: structured MoM adds quality signal
    mom_words = len(mom_text.split()) if mom_text else 0
    mom_bonus = min(10.0, _math.log10(max(1, mom_words)) * 4.0)  # 0–10 bonus
    content_score = min(98.0, rouge_score + mom_bonus)

    # ── Signal E: Topic Richness (15%) ────────────────────────────────────────
    kw_raw = getattr(meeting, "keyword_frequency", "") or ""
    try:
        import json as _kw_json
        kw_dict = _kw_json.loads(kw_raw) if kw_raw.strip().startswith("{") else {}
        topic_count = len(kw_dict)
    except Exception:
        topic_count = len([k for k in kw_raw.split(",") if k.strip()]) if kw_raw else 0
    # 0 topics → 0, ≥10 topics → 100; log-scaled
    topic_score = min(100.0, _math.log10(max(1, topic_count + 1)) / _math.log10(11) * 100.0)

    # ── Weighted combination ───────────────────────────────────────────────────
    weighted = (
        audio_score   * 0.20
        + stt_score   * 0.25
        + diar_score  * 0.15
        + content_score * 0.25
        + topic_score * 0.15
    )
    quality_signal_score = round(max(0.0, min(100.0, weighted)), 1)

    # NO ground truth → accuracy is genuinely unknown. Store NULL, never a %.
    meeting.accuracy = None
    meeting.transcription_accuracy = None
    meeting.speaker_accuracy = None

    label = (
        "Excellent" if quality_signal_score >= 90 else
        "Good" if quality_signal_score >= 75 else
        "Average" if quality_signal_score >= 60 else
        "Needs Improvement"
    )
    color = (
        "emerald" if quality_signal_score >= 90 else
        "blue" if quality_signal_score >= 75 else
        "amber" if quality_signal_score >= 60 else
        "red"
    )

    metrics_payload = {
        "quality_signal_score": quality_signal_score,
        "is_accuracy": False,
        "accuracy": None,
        "accuracy_status": "not_measured",
        "label": label,
        "color": color,
        "penalty": 0.0,
        "metric_type": "RUNTIME_QUALITY_SIGNAL",
        "note": (
            "Reference-free quality signal from pipeline evidence (audio/STT/diarization/"
            "content/topics). This is NOT accuracy. Upload a ground-truth reference to "
            "compute true WER/DER/F1 accuracy."
        ),
        "warnings": (
            ["A ground-truth reference exists but its evaluation failed; accuracy is not measured (null). "
             "See the server log entry '[OptionalQuery]' for the database error."]
            if gt_eval_failed else []
        ),
        "weights": {
            "audio_quality": 0.20,
            "stt_completeness": 0.25,
            "diarization": 0.15,
            "content_quality": 0.25,
            "topic_richness": 0.15,
        },
        "contributions": {
            "audio_quality": round(audio_score * 0.20, 2),
            "stt_completeness": round(stt_score * 0.25, 2),
            "diarization": round(diar_score * 0.15, 2),
            "content_quality": round(content_score * 0.25, 2),
            "topic_richness": round(topic_score * 0.15, 2),
        },
        "components": {
            "audio_quality": round(audio_score, 1),
            "transcription": round(stt_score, 1),
            "speaker": round(diar_score, 1),
            "summary": round(rouge_score, 1),
            "mom": round(content_score, 1),
            "topic": round(topic_score, 1),
            "action_items": None,   # Requires ground truth to evaluate correctly
            "sentiment": None,
            "similarity": None,
            "video_quality": None,
        },
    }

    if hasattr(meeting, "metrics_json"):
        meeting.metrics_json = _json.dumps(metrics_payload)

    logger.info(
        f"Meeting {meeting.id} reference-free quality signal: {quality_signal_score} ({label}) "
        f"[audio={audio_score:.1f} stt={stt_score:.1f} diar={diar_score:.1f} "
        f"content={content_score:.1f} topics={topic_score:.1f}] "
        f"— accuracy=NULL (no ground truth)"
    )


def silence_from_speech(speaker_diag, media_duration: float):
    """Seconds of silence = recording length - DETECTED speech, or None when speech was not measured.

    Only a speech total taken from the diarization's VAD regions (``speech_source == "vad_regions"``) counts: segment
    spans tile the timeline, so "media - span" is always ~0 (it reported ``silence_duration: 0.0`` for a recording
    that is 12 % silence). Unknown is None, never 0.
    """
    if not isinstance(speaker_diag, dict) or speaker_diag.get("speech_source") != "vad_regions":
        return None
    if not media_duration or media_duration <= 0:
        return None
    return round(max(0.0, float(media_duration) - float(speaker_diag.get("total_speech_duration_sec") or 0.0)), 3)


def stamp_evidence_times(items, segment_rows, evidence_key: str, start_key: str, end_key: str = None) -> int:
    """Give each item whose evidence QUOTE occurs in a stored transcript segment that segment's start (and end) time
    as ``HH:MM:SS``. Returns how many items were stamped.

    ``segment_rows``: ``(text, start_time, end_time)`` rows in transcript order. Only quotes that are really found
    are stamped and a value the model already gave is never replaced; an item with no quote, or a quote that is not
    in the transcript, keeps ``None`` - a time is never guessed. Segment-level precision: AIRAWAT returns no word
    timestamps, so this is "within this segment", not the second the words were spoken.
    """
    from backend.utils.helpers import format_seconds_to_duration

    texts = [r[0] or "" for r in segment_rows]
    stamped = 0
    for item in items or []:
        if not isinstance(item, dict) or item.get(start_key) or not item.get(evidence_key):
            continue
        idx = ts.locate_quote_segment(str(item[evidence_key]), texts)
        if idx is None:
            continue
        try:
            start = float(segment_rows[idx][1])
        except (TypeError, ValueError):
            continue
        item[start_key] = format_seconds_to_duration(start) if start > 0 else "00:00:00"
        if end_key and not item.get(end_key):
            try:
                item[end_key] = format_seconds_to_duration(float(segment_rows[idx][2]))
            except (TypeError, ValueError):
                pass
        stamped += 1
    return stamped


def _save_similar_matches(meeting, similar_matches, db):
    """Save similar match data to meeting record."""
    if similar_matches:
        top_match = similar_matches[0]
        meeting.matched_meeting_id = top_match["meeting_id"]
        meeting.matched_meeting_code = top_match["meeting_code"]
        meeting.similarity_score = top_match["similarity_score"]
        meeting.matched_mom = top_match["matched_mom"]
        meeting.related_meetings_json = json.dumps(similar_matches)
    else:
        meeting.related_meetings_json = json.dumps([])


def _generate_reports_with_retry(meeting_id: int, transcript: str, on_retry=None, **report_kwargs) -> dict:
    """``process_meeting_report`` with the configured retries (LLM_PIPELINE_MAX_RETRIES, default 2, waiting
    LLM_PIPELINE_RETRY_DELAY_SEC * attempt between them).

    Every attempt logs its number, duration, result and reason; ``on_retry(attempt, error)`` (optional) is called
    before each retry so the caller can record it (the pipeline writes a RETRYING activity-log row).

    When every attempt fails: AI_MODE=local (development) returns ``{}`` and the extractive fallbacks
    downstream take over; every other mode RAISES so the meeting is marked failed with the reason instead of
    silently carrying rule-based output that looks like AIRAWAT's.
    """
    max_retries = max(1, int(os.getenv("LLM_PIPELINE_MAX_RETRIES", "2")))
    base_delay = float(os.getenv("LLM_PIPELINE_RETRY_DELAY_SEC", "5.0"))
    last_err = None
    section_cache: dict = {}  # sections that already succeeded are not regenerated by a retry
    for attempt in range(1, max_retries + 1):
        attempt_t0 = time.perf_counter()
        try:
            # FULL transcript: process_meeting_report fits it to the context window per section and condenses a
            # too-long meeting for the summary in parts instead of losing its middle to truncation.
            reports = process_meeting_report(transcript, section_cache=section_cache, **report_kwargs)
            logger.info(
                f"[Pipeline] LLM report attempt {attempt}/{max_retries} for meeting {meeting_id}: OK "
                f"in {time.perf_counter() - attempt_t0:.1f}s."
            )
            return reports
        except Exception as report_err:
            last_err = report_err
            if attempt < max_retries:
                delay = base_delay * attempt
                logger.warning(
                    f"[Pipeline] LLM report attempt {attempt}/{max_retries} for meeting {meeting_id}: FAILED "
                    f"after {time.perf_counter() - attempt_t0:.1f}s ({type(report_err).__name__}: {report_err}). "
                    f"Retrying in {delay:.0f}s; sections that already succeeded are reused."
                )
                if on_retry is not None:
                    try:
                        on_retry(attempt + 1, report_err)
                    except Exception as cb_err:   # bookkeeping must never abort the retry itself
                        logger.warning(f"[Pipeline] retry notification failed: {cb_err}")
                time.sleep(delay)
    logger.error(f"[Pipeline] LLM report FAILED after {max_retries} attempt(s) for meeting {meeting_id}: {last_err}")
    if isinstance(last_err, AIProviderError):
        raise last_err
    raise AIProviderError(f"AI LLM report generation failed after {max_retries} attempt(s): {last_err}") from last_err


def _run_shared_pipeline(
    meeting_id: int,
    full_transcript_text: str,
    meeting,
    db,
    reports: dict = None,
    is_document: bool = False,
    doc_text: str = "",
    wav_path: str = None,
    act=None,  # ActivityLogger instance (optional — no-op if None)
):
    """Shared pipeline logic for both document and media processing.

    ``reports`` MUST be pre-computed by the caller BEFORE opening a DB session.
    This function only performs fast DB writes — no LLM calls are made here.
    """
    if reports is None:
        reports = {}

    # Convenience wrapper — no-op if act is None so document pipeline still works
    def _act_start(step, msg="", details=None):
        return act.start_step(step, message=msg, details=details) if act else None

    def _act_done(log_id, msg="", details=None):
        if act and log_id:
            act.complete_step(log_id, message=msg, details=details)

    def _act_fail(log_id, code="STEP_ERROR", msg="", details=None):
        if act and log_id:
            act.fail_step(log_id, error_code=code, error_message=msg, details=details)

    logger.info(f"Writing reports for meeting_id = {meeting.bharatvc_meeting_id or meeting.meeting_id}")

    # Fallback summaries derived from source text only — no LLM call needed here.
    m_date = meeting.meeting_date or "Unknown Date"

    # ── LLM report timing metadata ────────────────────────────────────────────
    report_elapsed = reports.pop("_elapsed_sec", 0.0)
    section_timings = reports.pop("_section_timings", {}) or {}
    section_durations = reports.pop("_section_durations", {}) or {}
    if act:
        act.record_timing("llm_report_generation (parallel)", report_elapsed)
        for _name, _secs in section_timings.items():
            act.record_timing(f"  - {_name}", _secs)

    def _strip_speaker_prefix(line: str) -> str:
        # Transcript lines are formatted "<speaker label>: <text>" — a raw-text
        # fallback must never echo that label into narrative Summary/MoM prose.
        return line.split(":", 1)[-1].strip() if ":" in line[:40] else line

    def _fallback_summary(text: str, title: str) -> str:
        # Only reached when report generation produced no summary at all. A
        # label-free, length-planned extractive overview — never a dump of the
        # first transcript lines under a heading.
        return extractive_summary_for_transcript(text) or "Not Available"

    def _fallback_mom(text: str, title: str, date: str) -> str:
        sections = [
            line.strip()
            for line in text.split("\n")
            if line.strip().startswith(("1.", "2.", "3.", "4.", "5.", "#", "Section"))
        ][:10]
        sec_str = (
            "\n".join([f"- {s}" for s in sections])
            if sections
            else "- Main Session Overview"
        )
        overview_lines = [_strip_speaker_prefix(line.strip()) for line in text.split("\n") if line.strip()]
        overview = " ".join(overview_lines)[:1500]
        return f"### Minutes of Meeting — {title}\n\n**Date:** {date}\n\n#### Key Sections:\n{sec_str}\n\n#### Overview:\n{overview}" if text.strip() else "Not Available"

    # Prefer full_transcript_text — for both media and documents it is the
    # English, post-translation text, so text-derived fallbacks stay English.
    source_text = full_transcript_text or (doc_text if is_document else "")
    # finalize_summary is the last gate before analysis_results.summary is
    # written: no markdown, repeated sentences, timecodes or speaker labels can
    # reach the DB (or json_data / the GET API built from it) from any path.
    initial_summary = finalize_summary(
        reports.get("summary") or _fallback_summary(
            source_text,
            meeting.title or meeting.filename,
        )
    )
    initial_mom = reports.get("mom") or _fallback_mom(
        source_text,
        meeting.title or meeting.filename,
        m_date,
    )
    initial_action_items = reports.get("action_items") or ""
    initial_decisions = reports.get("decisions") or ""
    initial_risks = reports.get("risks") or ""

    # Record actual elapsed time for the report generation step
    # Each section's own generation time. The sections run in parallel inside one report call, so these overlap and
    # are NOT additive; mom / action items used to be stored as a hardcoded 0.0 although each is timed. A section
    # that was not timed (reports that did not come from process_meeting_report) is None - unknown, not zero.
    meeting.summary_duration_sec = section_durations.get("summary", report_elapsed)
    meeting.mom_duration_sec = section_durations.get("mom")
    meeting.action_items_duration_sec = section_durations.get("action_items")

    meeting.summary_generated = bool(initial_summary.strip() and "Not Available" not in initial_summary)
    meeting.summary_word_count = len(initial_summary.split()) if initial_summary else 0
    meeting.mom_generated = bool(initial_mom.strip() and "Not Available" not in initial_mom)
    meeting.mom_word_count = len(initial_mom.split()) if initial_mom else 0

    # ``action_items_generated`` / ``action_items_count`` / ``db_action_items_saved`` are set below, once the action
    # items that are actually SAVED are known (they come from the dedicated structured extraction, not from the
    # metadata call's bullet text, which is a different LLM answer).

    agenda_decisions_data = reports.get("agenda_key_decisions", [])

    # Stored segments (media meetings only: a document's segment times are synthetic) give a time to every decision
    # / action item whose evidence quote is really in the transcript.
    segment_rows = []
    if not is_document:
        segment_rows = [
            (r.text, r.start_time, r.end_time)
            for r in db.query(TranscriptSegment.text, TranscriptSegment.start_time, TranscriptSegment.end_time)
            .filter(TranscriptSegment.meeting_id == meeting_id).order_by(TranscriptSegment.id).all()
        ]
    if segment_rows:
        stamp_evidence_times(
            [d for d in agenda_decisions_data if d.get("decision_grounded")], segment_rows, "evidence", "timestamp",
        )

    # ── Feature 7: Agenda ──────────────────────────────────────────────────────
    _agenda_log = _act_start("agenda_extraction", "Extracting and structuring agenda topics")
    # Unconditional delete-before-insert (matches TranscriptSegment's pattern):
    # if this run produced zero agenda items, any stale rows from a PRIOR run
    # on this same meeting_id must not survive — otherwise a reprocess that
    # legitimately found nothing would still serve old agenda/decision data.
    db.query(AgendaKeyDecision).filter(AgendaKeyDecision.meeting_id == meeting_id).delete()
    if agenda_decisions_data:
        for idx, item in enumerate(agenda_decisions_data, start=1):
            ag_ord = item.get("agenda_order", idx)
            akd_record = AgendaKeyDecision(
                meeting_id=meeting_id,
                bharatvc_meeting_id=meeting.bharatvc_meeting_id,
                agenda_order=ag_ord,
                agenda_id_str=item.get("agenda_id_str") or f"A{ag_ord}",
                agenda=item.get("agenda", f"Agenda {idx}"),
                agenda_type=item.get("agenda_type", "inferred"),
                key_decision=item.get("key_decision", "No formal decision recorded."),
                reason=item.get("reason"),
                evidence=item.get("evidence"),
                speaker=item.get("speaker"),
                timestamp=item.get("timestamp"),
                confidence=item.get("confidence"),
                match_score=item.get("match_score"),
                embedding_model=item.get("embedding_model"),
                # Must be persisted explicitly — the column default (True) would
                # otherwise silently discard validate_and_format_agenda_key_decisions'
                # real grounding check (evidence-or-decisive-language) for every row.
                decision_grounded=bool(item.get("decision_grounded", False)),
            )
            db.add(akd_record)
    _act_done(_agenda_log, f"{len(agenda_decisions_data)} agenda topic(s) extracted",
              details={"agenda_count": len(agenda_decisions_data)})

    # ── Feature 8: Decision ────────────────────────────────────────────────────
    _dec_log = _act_start("decision_extraction", "Extracting key decisions and rationale")
    valid_decisions = [
        d for d in agenda_decisions_data
        if d.get("key_decision") and "no formal decision" not in str(d.get("key_decision", "")).lower()
    ]
    dec_count = len(valid_decisions)
    _act_done(_dec_log, f"{dec_count} key decision(s) identified",
              details={"decisions_count": dec_count, "total_topics": len(agenda_decisions_data)})

    # ── Feature 9: Agenda → Decision Mapping ──────────────────────────────────
    _map_log = _act_start("decision_mapping", "Mapping decisions to agenda topics with semantic match scores")
    scores = [d.get("match_score") for d in agenda_decisions_data if d.get("match_score") is not None]
    avg_match = round(sum(scores) / len(scores), 3) if scores else None
    _act_done(_map_log, f"Mapped {len(agenda_decisions_data)} agenda-decision relationships",
              details={"mapped_count": len(agenda_decisions_data), "average_match_score": avg_match})

    # ── Feature 10: Action Items ──────────────────────────────────────────────
    _ai_log = _act_start("action_item_extraction", "Extracting and assigning structured action items")
    structured_ai = reports.get("structured_action_items", [])
    if not structured_ai and (str(initial_action_items or "").strip() or full_transcript_text.strip()):
        try:
            from backend.services.action_item_service import extract_action_items_extractive, resolve_action_item_dates
            from backend.utils.date_resolver import get_reference_datetime
            ref_dt = get_reference_datetime(meeting.meeting_date, settings.DEFAULT_TIMEZONE)
            # Parses the action items AIRAWAT returned. Keyword rules are never run over the raw transcript:
            # "no action items" is the model's answer and rules must not invent items.
            extracted = extract_action_items_extractive(str(initial_action_items or ""))
            if extracted:
                structured_ai = resolve_action_item_dates(extracted, ref_dt)
        except Exception as ai_parse_err:
            logger.warning(f"[Pipeline] Failed to parse action items from text: {ai_parse_err}")
    if segment_rows:
        stamp_evidence_times(structured_ai, segment_rows, "evidence", "timestamp_start", "timestamp_end")

    db_analysis = (
        db.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting_id).first()
    )
    if not db_analysis:
        db_analysis = AnalysisResult(
            meeting_id=meeting_id,
            summary=initial_summary,
            mom=initial_mom,
            action_items=initial_action_items,
            decisions=initial_decisions,
            risks=initial_risks,
        )
        db.add(db_analysis)
    else:
        db_analysis.summary = initial_summary
        db_analysis.mom = initial_mom
        db_analysis.action_items = initial_action_items
        db_analysis.decisions = initial_decisions
        db_analysis.risks = initial_risks

    # Unconditional delete-before-insert — a reprocess that legitimately finds
    # zero action items must not leave stale rows from a prior run in place.
    db.query(ActionItem).filter(ActionItem.meeting_id == meeting_id).delete()
    if structured_ai:
        _ai_text_lines = []
        for _ai in structured_ai:
            _task = _ai.get("task") or _ai.get("title") or "Task"
            _owner = _ai.get("owner") or _ai.get("assigned_to") or "Unassigned"
            _deadline = _ai.get("deadline") or _ai.get("due_date") or "Not specified"
            _priority = _ai.get("priority") or "Not specified"
            _ai_text_lines.append(
                f"- [ ] **{_task}** — Owner: {_owner} | Deadline: {_deadline} | Priority: {_priority}"
            )
        db_analysis.action_items = "\n".join(_ai_text_lines)
        db_analysis.structured_action_items = structured_ai
        for idx, ai in enumerate(structured_ai, start=1):
            t_name = ai.get("task") or ai.get("title") or f"Action {idx}"
            ai_record = ActionItem(
                meeting_id=meeting_id,
                bharatvc_meeting_id=meeting.bharatvc_meeting_id,
                item_order=idx,
                task=t_name,
                description=ai.get("description"),
                main_task=ai.get("main_task"),
                owner=ai.get("owner", "Unassigned"),
                assigned_by=ai.get("assigned_by"),
                deadline=ai.get("deadline", "Not specified"),
                deadline_type=ai.get("deadline_type", "not_specified"),
                due_date=ai.get("due_date"),
                deadline_start=ai.get("deadline_start"),
                deadline_end=ai.get("deadline_end"),
                deadline_precision=ai.get("deadline_precision"),
                priority=ai.get("priority", "Not specified"),
                status=ai.get("status", "open"),
                evidence=ai.get("evidence"),
                speaker=ai.get("speaker"),
                timestamp_start=ai.get("timestamp_start"),
                timestamp_end=ai.get("timestamp_end"),
                confidence=ai.get("confidence"),
                source=ai.get("source", "llm"),
                agenda_ref=ai.get("agenda_ref"),
                decision_ref=ai.get("decision_ref"),
            )
            db.add(ai_record)
        logger.info(f"[Pipeline] Saved {len(structured_ai)} structured action item(s) to DB.")
    # One source for every action-item flag: the rows just written. (A meeting whose model found none is a
    # successful stage with 0 items - "generated" is then False because nothing was generated, not because it failed.)
    meeting.action_items_count = len(structured_ai)
    meeting.action_items_generated = bool(structured_ai)
    _act_done(_ai_log, f"{len(structured_ai)} action items saved",
              details={"action_item_count": len(structured_ai)})

    # ── Feature 11: Summary ───────────────────────────────────────────────────
    _sum_log = _act_start("summary_generation", "Generating executive summary")
    _act_done(_sum_log, f"Executive summary generated ({meeting.summary_word_count} words)",
              details={"summary_words": meeting.summary_word_count,
                       "summary_ok": meeting.summary_generated})

    # ── Feature 12: MOM / Minutes of Meeting ──────────────────────────────────
    _mom_log = _act_start("mom_generation", "Generating structured Minutes of Meeting (MoM)")
    _act_done(_mom_log, f"Minutes of Meeting generated ({meeting.mom_word_count} words)",
              details={"mom_words": meeting.mom_word_count,
                       "mom_ok": meeting.mom_generated})

    # ── Feature 13: Sentiment Analysis ────────────────────────────────────────
    _sent_log = _act_start("sentiment_analysis", "Analyzing sentiment, emotions, and tone")
    meeting.sentiment = reports.get("sentiment", "Neutral")
    meeting.sentiment_label = reports.get("sentiment", "Neutral")
    _raw_emotions = reports.get("emotions", "Neutral: 100%")

    # BUG-5 FIX: Normalize emotions from opaque string to structured JSON dict.
    # Handles formats like "Positive: 60%, Neutral: 40%" → {"Positive": 60.0, "Neutral": 40.0}
    # Falls back to storing raw string if parsing fails (backward compat).
    _emotions_structured = None
    if isinstance(_raw_emotions, str) and _raw_emotions.strip():
        try:
            _emotions_dict = {}
            for _part in _raw_emotions.split(","):
                _part = _part.strip()
                if ":" in _part:
                    _k, _v = _part.split(":", 1)
                    _emotions_dict[_k.strip()] = float(_v.strip().rstrip("% "))
            if _emotions_dict:
                _emotions_structured = json.dumps(_emotions_dict)
        except Exception:
            pass  # keep raw string on any parse error
    elif isinstance(_raw_emotions, dict):
        try:
            _emotions_structured = json.dumps(_raw_emotions)
        except Exception:
            pass

    meeting.emotions = _emotions_structured if _emotions_structured else _raw_emotions
    # ``emotions`` is a native object (or the raw text when it could not be parsed) INSIDE this JSON: embedding
    # ``_emotions_structured`` - already a JSON string - double-encoded it ('"{\"Confidence\": 80.0}"').
    meeting.sentiment_breakdown_json = json.dumps({
        "sentiment": meeting.sentiment,
        "emotions": json.loads(_emotions_structured) if _emotions_structured else _raw_emotions,
    })
    meeting.questions_raised = reports.get("questions_raised", "[]")
    meeting.keyword_frequency = reports.get("keyword_frequency", "{}")
    proj_name = reports.get("project_name", DEFAULT_PROJECT_NAME)
    meeting.project_name = proj_name
    _act_done(_sent_log, f"Sentiment: {meeting.sentiment}",
              details={"sentiment": meeting.sentiment, "emotions": meeting.emotions})

    # ── Feature 14: Metrics ───────────────────────────────────────────────────
    _metrics_log = _act_start("ai_metrics", "Computing AI quality metrics, topic keywords, and confidence score")
    kw_raw = meeting.keyword_frequency or "{}"
    try:
        kw_dict = json.loads(kw_raw) if isinstance(kw_raw, str) else kw_raw
        meeting.topics_count = len(kw_dict) if isinstance(kw_dict, dict) else 0
    except Exception:
        meeting.topics_count = 0
    meeting.topics_generated = bool(meeting.topics_count > 0)

    is_sum_ok, is_mom_ok = check_outputs_generated(
        initial_summary, initial_mom, meeting.filename
    )
    if is_sum_ok and is_mom_ok:
        _rouge = rouge_scores(initial_summary, full_transcript_text)
        rouge_val = _rouge.get("rougeL")
        meeting.confidence_score = (
            round(rouge_val, 1) if rouge_val is not None else None
        )
    else:
        meeting.confidence_score = 0.0
        meeting.is_approved = False

    _compute_runtime_quality_signal(meeting, reports, full_transcript_text, db=db)
    _qsig = None
    try:
        _qsig = json.loads(meeting.metrics_json).get("quality_signal_score") if meeting.metrics_json else None
    except Exception:
        _qsig = None
    _acc_txt = f"{meeting.accuracy}% (ground truth)" if meeting.accuracy is not None else "N/A (no ground truth)"
    _act_done(_metrics_log,
              f"Metrics computed (Accuracy: {_acc_txt}, Quality signal: {_qsig}, Topics: {meeting.topics_count})",
              details={"accuracy": meeting.accuracy, "accuracy_basis": "ground_truth" if meeting.accuracy is not None else "not_measured",
                       "quality_signal_score": _qsig, "confidence_score": meeting.confidence_score,
                       "topics_count": meeting.topics_count})

    meeting_code = meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)
    fingerprint = ts.content_fingerprint(full_transcript_text)
    meeting.content_fingerprint = fingerprint

    # Duplicate-content notice: if another meeting already has this exact
    # transcript fingerprint, its Agenda/Decisions/Summary WILL legitimately
    # match this one (same input → same output). Surface it so an identical
    # re-upload is not mistaken for cross-meeting contamination.
    try:
        _dup = (
            db.query(Meeting.id, Meeting.display_id)
            .filter(Meeting.content_fingerprint == fingerprint,
                    Meeting.id != meeting_id, fingerprint != "")
            .first()
        )
        if _dup:
            logger.warning(
                "DUPLICATE_CONTENT | meeting_id=%s has the SAME transcript fingerprint "
                "(%s) as meeting_id=%s (%s). Identical Agenda/Decisions/Summary between "
                "them is expected — the uploaded media content is the same.",
                meeting_id, fingerprint[:16], _dup[0], _dup[1],
            )
    except Exception as dup_err:
        # Informational only, but a failed statement aborts the transaction: say so instead of hiding it, then let
        # the commit below fail loudly with the real cause rather than continue on a poisoned transaction.
        logger.warning("Duplicate-content check failed for meeting %s: %s", meeting_id, dup_err)
        raise

    # Features 15 (RAG indexing) and 16 (related meetings) run in _finish_meeting_processing, AFTER this function's
    # commit and with no transaction open: they embed text and query ChromaDB.

    # ── Feature 17: Database / Persistence ────────────────────────────────────
    _db_log = _act_start("data_persistence", "Persisting meeting intelligence records to relational database")
    meeting.db_meeting_saved = True
    meeting.db_transcript_saved = bool(full_transcript_text.strip())
    meeting.db_summary_saved = meeting.summary_generated
    meeting.db_action_items_saved = meeting.action_items_generated

    # The seven report-stage states go into the same commit (StageTracker used to commit once per transition and
    # the meeting row was re-read after each); the caller closes its session right after this returns.
    from backend.loop_engine import StageTracker, StageState
    tracker = StageTracker(meeting, db)
    tracker.transition("summary", StageState.SUCCESS, "Summary generated", commit=False)
    tracker.transition("mom", StageState.SUCCESS, "MoM generated", commit=False)
    tracker.transition("agenda", StageState.SUCCESS, f"{len(agenda_decisions_data)} agenda items extracted", commit=False)
    tracker.transition("decision", StageState.SUCCESS, "Decisions extracted", commit=False)
    tracker.transition("action_items", StageState.SUCCESS, f"{len(structured_ai)} action items saved", commit=False)
    tracker.transition("topics", StageState.SUCCESS, f"{meeting.topics_count} topics identified", commit=False)
    tracker.transition("sentiment", StageState.SUCCESS, f"Sentiment: {meeting.sentiment}", commit=False)

    # Plain data for the slow steps that follow: read BEFORE the commit (a commit expires the row; reading it
    # afterwards would open a new transaction just to fetch values already in hand).
    context = _SharedContext(
        meeting_id=meeting_id,
        meeting_code=meeting_code,
        filename=meeting.filename,
        title=meeting.title or meeting.filename,
        project_name=meeting.project_name,
        mom_summary=initial_mom[:500],
        vector_metadata=build_meeting_vector_metadata(meeting, fingerprint),
        transcript=full_transcript_text,
        accuracy=meeting.accuracy,
        duration_sec=meeting.duration,
        output_details={
            "word_count": meeting.stt_word_count,
            "speaker_count": meeting.speaker_count,
            "summary_generated": meeting.summary_generated,
            "action_items_count": meeting.action_items_count,
        },
    )
    db.commit()
    _act_done(_db_log, "All meeting intelligence records committed to database", details={"db_saved": True})
    return context


class _SharedContext:
    """What the post-commit steps (RAG indexing, related meetings, completion) need, as plain values."""

    __slots__ = ("meeting_id", "meeting_code", "filename", "title", "project_name", "mom_summary", "vector_metadata",
                 "transcript", "accuracy", "duration_sec", "output_details")

    def __init__(self, **values):
        for name in self.__slots__:
            setattr(self, name, values[name])


def _finish_meeting_processing(
    context: "_SharedContext",
    *,
    act=None,
    use_stage_outcome: bool = True,
    proc_start_dt=None,
    start_t: float = None,
) -> None:
    """Everything after the report rows are committed: RAG index, related meetings, completion, final JSON.

    Transaction discipline (the audit of this pipeline):

    * The embedding, ChromaDB and the similarity computation run with NO database transaction open. They used to run
      inside the session that had already DELETEd/INSERTed the report rows, pinning that connection and its locks
      for the duration.
    * ``status`` becomes ``completed`` in the SAME transaction that stores the final JSON - the meeting is never
      reported complete before its final record exists, and it can no longer flip completed -> error afterwards
      (it used to be committed ``completed`` first, then evaluated by ``mark_completed``).
    * A failure to store the final JSON is retried once and then FAILS the meeting with the real error; it used to
      be logged and ignored, leaving a ``completed`` meeting with no final record.
    """
    from datetime import datetime, timezone
    from backend.database.connection import SessionLocal, pipeline_db_session
    from backend.loop_engine import StageTracker, StageState
    from backend.services.final_aggregation_service import build_final_record

    meeting_id = context.meeting_id

    def _act_start(step, msg="", details=None):
        return act.start_step(step, message=msg, details=details) if act else None

    def _act_done(log_id, msg="", details=None):
        if act and log_id:
            act.complete_step(log_id, message=msg, details=details)

    def _act_fail(log_id, code="STEP_ERROR", msg="", details=None):
        if act and log_id:
            act.fail_step(log_id, error_code=code, error_message=msg, details=details)

    # ── Feature 15: RAG (embedding + ChromaDB: no transaction) ───────────────
    _vec_log = _act_start("vector_indexing", "Indexing meeting embeddings and context for RAG retrieval")
    vector_indexed = vector_service.add_meeting_vector(
        meeting_id=meeting_id,
        meeting_code=context.meeting_code,
        filename=context.filename,
        title=context.title,
        text=context.transcript,
        mom_summary=context.mom_summary,
        project_name=context.project_name,
        extra_metadata=context.vector_metadata,
    )
    if vector_indexed:
        _act_done(_vec_log, "RAG vector embeddings indexed", details={"meeting_code": context.meeting_code})
    else:
        _act_fail(_vec_log, "VECTOR_DB_UNAVAILABLE",
                  "Meeting could not be indexed in ChromaDB (see vector_database in /system/health); it is "
                  "indexed again on the next startup backfill or the first chat.")

    # ── Feature 16: AI Chatbot (related meetings: short read sessions only) ──
    _sim_log = _act_start("similar_meeting_search", "Configuring AI Chatbot context and semantic similarity")
    similar_matches = []
    try:
        analysis = similarity_service.find_similar_meetings(
            repo=None,
            meeting_id=meeting_id,
            top_k=8,
            include_details=True,
            session_factory=SessionLocal,
        )
        similar_matches = analysis.get("meetings", [])
        _act_done(_sim_log, f"{len(similar_matches)} related meeting(s) identified for chatbot context",
                  details={"similar_count": len(similar_matches)})
    except Exception as sim_err:
        logger.warning(f"Similarity analysis failed for meeting {meeting_id}: {sim_err}")
        _act_fail(_sim_log, "SIMILARITY_ERROR", str(sim_err))

    # Report what actually happened: a meeting that is not in ChromaDB is DEGRADED (completed with warnings),
    # never a silent SUCCESS. Chat re-indexes it lazily, so nothing is lost.
    rag_chunks_indexed = vector_service.has_transcript_chunks(meeting_id)

    # ── Feature 18: Final validation — one short write transaction ────────────
    _val_log = _act_start("final_validation", "Validating final outputs and pipeline completion")
    outcome = {"status": "completed", "overall_status": None, "error_message": None}
    with pipeline_db_session() as db:
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if meeting is None:
            raise RuntimeError(f"Meeting {meeting_id} disappeared before it could be completed.")
        _save_similar_matches(meeting, similar_matches, db)
        tracker = StageTracker(meeting, db)
        tracker.transition(
            "embeddings_vector",
            StageState.SUCCESS if vector_indexed else StageState.DEGRADED,
            "Meeting embeddings indexed" if vector_indexed else "Vector database unavailable - meeting embeddings not indexed",
            commit=False,
        )
        tracker.transition(
            "rag_chatbot",
            StageState.SUCCESS if rag_chunks_indexed else StageState.DEGRADED,
            "RAG vector collection updated" if rag_chunks_indexed else "Transcript chunks not indexed - indexed on first chat",
            commit=False,
        )
        if proc_start_dt is not None and start_t is not None:
            meeting.processing_start_time = proc_start_dt
            meeting.processing_end_time = datetime.now(timezone.utc)
            processing_duration = round(time.perf_counter() - start_t, 3)
            meeting.total_processing_time_sec = processing_duration
            meeting.processing_time = processing_duration
        if use_stage_outcome:
            outcome = tracker.evaluate_completion()
        tracker.save(commit=True)   # similar matches + stage states + timings: one commit; status is still "processing"
    # api_status / api_status_code follow the outcome and are written with it (they were left at "Success"/200).
    outcome.update(api_status_fields(outcome["status"]))
    details = {"status": outcome["status"], "accuracy": context.accuracy, "duration_sec": context.duration_sec}
    if outcome["status"] == "completed":
        _act_done(_val_log, "Pipeline execution validated and completed successfully", details=details)
    else:
        _act_fail(_val_log, "STAGE_FAILED", outcome["error_message"] or "A required stage failed", details=details)

    # ── Final Consolidated Table: AI_Meeting_Inteligent_final ─────────────────
    # The last activity step is completed BEFORE the JSON is built so the stored json_data / activity_logs are
    # final (they used to lack it until the next GET rebuilt them); if the JSON then cannot be stored the step is
    # flipped to FAILED below. The meeting's completion is part of the JSON's transaction. Two attempts; then the
    # meeting FAILS with the cause.
    _output_log = _act_start("final_output", "Meeting processing complete")
    if outcome["status"] == "completed":
        _act_done(_output_log, "Meeting fully processed and available",
                  details={**context.output_details, "status": outcome["status"],
                           "total_processing_sec": round(time.perf_counter() - start_t, 3) if start_t is not None else None})
    else:
        _act_fail(_output_log, "PIPELINE_FAILED", outcome["error_message"] or "A required stage failed")
    _t0_final = time.perf_counter()
    last_error = None
    for attempt in (1, 2):
        try:
            with pipeline_db_session() as db:
                record = build_final_record(meeting_id, db, meeting_updates=outcome)
                # A FAILED record is legitimate only when the meeting itself failed (outcome status "error").
                if outcome["status"] == "completed" and (
                    not isinstance(getattr(record, "json_data", None), dict) or record.processing_status == "FAILED"
                ):
                    raise RuntimeError(
                        f"the final record is {getattr(record, 'processing_status', None)} "
                        f"(error_info={getattr(record, 'error_info', None)})"
                    )
                db.commit()
            last_error = None
            break
        except Exception as final_err:
            last_error = final_err
            logger.error(
                "[Pipeline] Final record for meeting %s failed on attempt %d/2 after %.2fs: %s",
                meeting_id, attempt, time.perf_counter() - _t0_final, final_err, exc_info=True,
            )
    if last_error is not None:
        _act_fail(_output_log, "FINAL_JSON_FAILED", str(last_error))
        raise RuntimeError(f"The final result could not be stored: {last_error}") from last_error
    if act:
        act.record_timing("final_json_postgresql_json_data", time.perf_counter() - _t0_final)
    logger.info(
        "[Pipeline] Meeting %s (%s) %s; AI_Meeting_Inteligent_final.json_data saved.",
        meeting_id, context.meeting_code, outcome["status"],
    )


# ActivityLogger of each running media pipeline, so the entry point can print its stage timing table at the end
# of the run whichever phase it stopped in.
_RUN_ACTIVITY_LOGGERS: dict = {}

# Returned by the pipeline functions when this worker did NOT take the meeting (already being processed elsewhere,
# or it does not exist): the entry point must then leave the running pipeline's timing table and SSE stream alone.
_NOT_STARTED = object()


def _claim_meeting(db, meeting_id: int, start_dt) -> bool:
    """Atomically move the meeting to ``processing`` and COMMIT that at once; False if this worker must not run it.

    ``UPDATE ... WHERE status IS DISTINCT FROM 'processing'`` is decided by PostgreSQL under the row lock, so two
    workers racing for one meeting cannot both win - whatever process or thread they are in.
    """
    from sqlalchemy import or_

    claimed = (
        db.query(Meeting)
        .filter(Meeting.id == meeting_id, or_(Meeting.status.is_(None), Meeting.status != "processing"))
        .update(
            {
                Meeting.status: "processing",
                Meeting.processing_start_time: start_dt,
                # A (re)run starts CLEAN. The meetings row is reused on re-upload / reprocess, and nothing used to
                # reset the run state: meeting 96 was served mid-run with the previous run's
                # InFailedSqlTransaction error_message, "Success" api_status and its old stage states.
                Meeting.error_message: None,
                Meeting.overall_status: "PROCESSING",
                Meeting.processing_end_time: None,
                Meeting.pipeline_stage_states_json: None,   # StageTracker re-initialises every stage to PENDING
                Meeting.retry_history_json: None,
                Meeting.api_status: api_status_for("processing")[0],
                Meeting.api_status_code: api_status_for("processing")[1],
            },
            synchronize_session=False,
        )
    )
    if claimed:
        db.commit()   # publish the claim now; nothing slow ever runs inside this transaction
        return True
    current = db.query(Meeting.status).filter(Meeting.id == meeting_id).first()
    db.rollback()
    if current is None:
        logger.error("PIPELINE ABORT | reason=meeting_not_found | meeting_id=%s", meeting_id)
    else:
        logger.warning(
            "PIPELINE ABORT | reason=already_processing | meeting_id=%s | another worker owns this meeting", meeting_id
        )
    return False


# process_meeting_report names a failed section by these words (``AirawatError.details["failed_sections"]``); each
# maps to the stage(s) whose output that section produces.
_REPORT_SECTION_STAGES = {
    "summary": ("summary",),
    "minutes of meeting": ("mom",),
    "metadata": ("topics", "sentiment"),
    "agenda and decisions": ("agenda", "decision"),
    "action items": ("action_items",),
}


def _stage_state_for_diarization(diar_status: str):
    """The diarization STAGE state for the engine's reported status.

    A hard FAILED must surface as FAILED (not be relabeled DEGRADED), so ``evaluate_completion``'s honest-status
    cascade and the audit trail both reflect what happened; anything short of SUCCESS is DEGRADED.
    """
    from backend.loop_engine import StageState

    if diar_status == "SUCCESS":
        return StageState.SUCCESS
    if diar_status == "FAILED":
        return StageState.FAILED
    return StageState.DEGRADED


def _mark_meeting_failed(
    meeting_id: int,
    error: Exception,
    *,
    failed_stage: "str | None" = None,
    stage_states: "dict | None" = None,
) -> None:
    """Record a failed processing run truthfully: ``error`` / ``FAILED`` plus the reason, the states of the stages
    that finished, and the stage that failed.

    * ``stage_states`` - ``{stage: "SUCCESS" | "DEGRADED"}`` for stages this run PROVED finished before it failed
      (their real outcome, not PENDING).
    * ``failed_stage`` - the stage that was running when the error was raised (FAILED, with the reason). Stages after
      it stay ``PENDING``: no output exists for them, and nothing is ever promoted to SUCCESS without evidence.
    * An AIRAWAT report failure additionally marks the report sections it named.

    Used to leave every stage ``PENDING`` and ``overall_status`` at ``uploaded`` next to ``status=error`` (meeting
    1647), and the phase-2/3/5 handlers wrote only ``status`` + ``error_message`` - so a failure in STT or in the
    final write was indistinguishable, per stage, from "nothing ran". ``api_status`` follows the outcome too (it
    stayed "Success" next to ``status=error``). One short transaction, no AI call inside it.
    """
    from backend.database.connection import pipeline_db_session
    from backend.loop_engine import StageState, StageTracker
    from backend.utils.exceptions import AIProviderError as _AIProviderError

    with pipeline_db_session() as db:
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        if not meeting:
            return
        reason = str(error)[:300]
        tracker = StageTracker(meeting, db)
        for stage, state in (stage_states or {}).items():
            if tracker.states.get(stage) in (None, StageState.PENDING.value, StageState.PROCESSING.value):
                tracker.transition(stage, StageState(state), "Finished before the run failed", commit=False)
        if failed_stage:
            tracker.transition(failed_stage, StageState.FAILED, reason, commit=False)
        if isinstance(error, _AIProviderError):
            failed = (getattr(error, "details", None) or {}).get("failed_sections")
            stages = [s for name in (failed or _REPORT_SECTION_STAGES) for s in _REPORT_SECTION_STAGES.get(name, ())]
            for stage in stages:
                tracker.transition(stage, StageState.FAILED, reason, commit=False)
        meeting.status = "error"
        meeting.overall_status = "FAILED"
        meeting.error_message = f"Processing failed: {error}"
        for column, value in api_status_fields("error").items():
            setattr(meeting, column, value)
        db.commit()


def _start_audio_diarization(wav_path: str, num_speakers):
    """Start the diarization engine's audio-only pass on a worker thread; returns a Future of
    ``((segments, status, error), elapsed_sec)``.

    The pass needs no transcript, so it overlaps the speech-to-text request (network-bound, waiting on AIRAWAT)
    instead of following it: measured 36.7 s of NeMo compute and 37-112 s of Whisper wait, previously added up.
    The worker inherits this pipeline's log context; an exception is stored in the Future and re-raised where the
    result is collected, so error handling is unchanged.
    """
    import concurrent.futures
    import contextvars

    from backend.diarization.manager import diarization_manager

    def _run():
        t0 = time.perf_counter()
        result = diarization_manager.run_primary(wav_path, num_speakers=num_speakers)
        return result, round(time.perf_counter() - t0, 3)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="diar-audio")
    future = executor.submit(contextvars.copy_context().run, _run)
    executor.shutdown(wait=False)   # the thread exits when the task does
    return future


def _log_run_timing(meeting_pk: int, total_sec: float) -> None:
    act = _RUN_ACTIVITY_LOGGERS.pop(meeting_pk, None)
    if act is None:
        return
    try:
        from backend.database.connection import pipeline_db_session

        with pipeline_db_session() as db:
            status = db.query(Meeting.status).filter(Meeting.id == meeting_pk).scalar() or ""
    except Exception:  # never let timing output mask a pipeline result
        status = ""
    act.log_timing_summary(total_sec, status=status)


def _close_sse_stream(meeting_pk: int) -> None:
    """End any live activity-log SSE stream for this meeting.

    Previously only the SUCCESS path closed it (and did so BEFORE the final commit), so a failed
    or aborted run left every subscriber connected forever on keepalives, and a client that
    re-fetched the meeting on the "done" event could still read pre-commit state.
    """
    try:
        from backend.database.connection import pipeline_db_session
        from backend.services.sse_manager import sse_manager

        with pipeline_db_session() as db:
            row = db.query(Meeting.status, Meeting.error_message).filter(Meeting.id == meeting_pk).first()
        final_event = {"event": "done", "meeting_pk": meeting_pk,
                       "status": row.status if row else None, "error_message": row.error_message if row else None}
        sse_manager.close_meeting(meeting_pk=meeting_pk, final_event=final_event)
    except Exception as exc:  # never let stream bookkeeping mask a pipeline result
        logger.debug("SSE close for meeting %s skipped: %s", meeting_pk, exc)


def _process_document_pipeline(meeting_id: int, file_path: str):
    """Pipeline for document processing (PDF, DOCX, PPTX, etc.).

    DB connection is only held during fast read/write phases.  The slow
    LLM extraction (``extract_document_metadata_with_llm``) runs between
    context-manager blocks so the pool connection is released during AI work.
    """
    from datetime import datetime, timezone
    from backend.database.connection import pipeline_db_session
    from backend.services.activity_logger import ActivityLogger, build_activity_logger_for_meeting
    start_time = time.time()
    act: ActivityLogger = None

    # ── Phase 1: Mark as processing (fast DB write) ──────────────────────────
    with pipeline_db_session() as db:
        # Atomic claim (see _claim_meeting): a second worker for the same meeting stops here.
        if not _claim_meeting(db, meeting_id, datetime.now(timezone.utc)):
            return _NOT_STARTED
        meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
        act = build_activity_logger_for_meeting(meeting)
        _upload_log = act.start_step(
            "meeting_upload",
            message="Document received and queued for processing",
            details={"filename": meeting.filename, "file_type": meeting.file_type},
        )
        act.complete_step(
            _upload_log,
            message="Document upload confirmed",
            details={"meeting_pk": meeting.id, "bharatvc_meeting_id": meeting.bharatvc_meeting_id},
        )
        # Capture non-DB fields needed for Phase 2
        meeting_filename = meeting.filename
        meeting_title = meeting.title

    # ── Phase 2: Slow extraction (LLM + doc parse) — NO DB connection held ──
    try:
        _text_log = act.start_step("document_text_extraction", message="Extracting text from document")
        doc_text = extract_document_text(file_path)
        if not doc_text.strip():
            act.fail_step(_text_log, error_code="DOC_EMPTY", error_message="No readable text could be extracted")
            raise Exception(
                "No readable text could be extracted from the uploaded document."
            )
        act.complete_step(_text_log, message="Document text extracted", details={"char_count": len(doc_text)})

        _meta_log = act.start_step("document_metadata_extraction", message="Extracting metadata with LLM")
        metadata = extract_document_metadata_with_llm(doc_text, meeting_filename)
        act.complete_step(_meta_log, message="Document metadata extracted", details={"title": metadata.get("title")})

        _seg_log = act.start_step("document_segments_creation", message="Creating document segments")
        doc_segments = create_document_segments(doc_text)
        act.complete_step(_seg_log, message=f"Created {len(doc_segments)} document segments", details={"segment_count": len(doc_segments)})

        # ── English translation (any-language document → English output) ────────
        # Language is auto-detected from the text itself (non-Latin-script share);
        # English documents pass straight through with no accuracy impact.
        _dtr_log = act.start_step("translation", message="Ensuring document text is English")
        try:
            from backend.services.translation_service import translate_segments_to_english
            doc_segments, _doc_tr_meta = translate_segments_to_english(
                doc_segments, "", meeting_context={"title": metadata.get("title", "")}
            )
            act.complete_step(
                _dtr_log,
                message=(f"Translated {_doc_tr_meta.get('translated_segments', 0)} section(s) to English"
                         if _doc_tr_meta.get("translated") else "Document already English"),
                details=_doc_tr_meta,
            )
        except (AIProviderError, AirawatError) as _dtr_err:
            act.fail_step(_dtr_log, error_code="TRANSLATION_FAILED", error_message=str(_dtr_err),
                          details={"error": str(_dtr_err)})
            raise
        except Exception as _dtr_err:
            act.complete_step(_dtr_log, message=f"Translation skipped ({_dtr_err})", details={"error": str(_dtr_err)})
            logger.warning("Document translation stage failed: %s", _dtr_err)

        full_transcript_text = ""
        for seg in doc_segments:
            full_transcript_text += f"{seg['speaker']}: {seg['text']}\n"

        # Generate LLM meeting report here (no DB session held) so Phase 3 only does writes
        m_date = metadata.get("meeting_date") or "Unknown Date"
        m_time_val = metadata.get("meeting_time") or "Unknown Time"
        _t0_report = time.time()
        reports = _generate_reports_with_retry(
            meeting_id, full_transcript_text, meeting_date=m_date, meeting_time=m_time_val, duration_str=""
        )
        reports["_elapsed_sec"] = round(time.time() - _t0_report, 3)

    except Exception as e:
        logger.error(
            f"Error processing document meeting {meeting_id}: {str(e)}", exc_info=True
        )
        if act:
            act.cancel_remaining(from_step_order=0)
        _mark_meeting_failed(meeting_id, e)
        return

    # ── Phase 3: Write results (fast DB writes) ──────────────────────────────
    try:
        with pipeline_db_session() as db:
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            if not meeting:
                return

            # A title the caller typed at upload time must survive. The upload endpoint stores the
            # original filename (or its cleaned "Title Case" form, for URL ingestion) when no title
            # was given; only THEN is the LLM-derived title used.
            _stem = os.path.splitext(meeting_filename or "")[0]
            _auto_titles = {
                (meeting_filename or "").strip(),
                _stem.replace("_", " ").replace("-", " ").title().strip(),
            }
            _user_title = (meeting_title or "").strip()
            if _user_title and _user_title not in _auto_titles:
                meeting.title = _user_title
            else:
                meeting.title = metadata["title"]
            meeting.agenda = metadata["agenda"]
            meeting.participants = metadata["participants"]
            meeting.meeting_date = metadata["meeting_date"]
            meeting.meeting_time = metadata["meeting_time"]

            db.query(TranscriptSegment).filter(TranscriptSegment.meeting_id == meeting_id).delete()
            for seg in doc_segments:
                db_segment = TranscriptSegment(
                    meeting_id=meeting_id,
                    speaker=seg["speaker"],
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=seg["text"],                        # final, English
                    original_text=seg.get("original_text"),  # source-language text if translated
                )
                db.add(db_segment)

            processing_duration = round(time.time() - start_time, 2)
            participant_names = [
                p.strip() for p in metadata["participants"].split(",") if p.strip()
            ]

            meeting.duration = float(len(doc_text.split())) * 0.4
            meeting.participant_count = len(participant_names) if participant_names else 1
            meeting.processing_time = processing_duration
            meeting.total_processing_time_sec = processing_duration
            meeting.audio_quality = None   # document — no audio to probe
            meeting.video_quality = None
            meeting.transcription_accuracy = None
            meeting.speaker_accuracy = None
            meeting.confidence_score = 0.0

            # Segments, meeting columns, report rows and stage states: ONE commit (inside), then the session closes.
            context = _run_shared_pipeline(
                meeting_id,
                full_transcript_text,
                meeting,
                db,
                reports=reports,
                is_document=True,
                doc_text=doc_text,
                act=act,
            )

        # No transaction open from here: chunk embedding + ChromaDB, related meetings, then the completion (status
        # "completed" and the final JSON in one transaction).
        _embed_log = act.start_step("embedding_generation", message="Indexing document segments into vector store")
        vector_service.add_transcript_chunks(meeting_id, doc_segments)
        act.complete_step(_embed_log, message="Document embeddings indexed", details={"chunk_count": len(doc_segments)})

        _finish_meeting_processing(context, act=act, use_stage_outcome=False)

    except Exception as e:
        logger.error(
            f"Error writing document meeting {meeting_id} results: {str(e)}", exc_info=True
        )
        _mark_meeting_failed(meeting_id, e)



def _process_meeting_pipeline(meeting_id: int, file_path: str):
    """Pipeline for media video/audio processing.

    DB connection is released between phases so pool is not exhausted during
    slow Whisper / NeMo / LLaMA processing (which can take 10-15 minutes).

    Phase 1  — fast   : Set status=processing, commit, close DB session.
    Phase 2  — slow   : Extract audio, transcribe (Whisper), diarize (NeMo),
                         correct transcript — NO database connection held.
    Phase 3  — fast   : Write transcript segments to DB, commit, close session.
    Phase 4  — slow   : LLM report generation (LLaMA) — NO database connection held.
    Phase 5  — fast   : Write analysis results, similarity, vectors, commit.
    """
    from datetime import datetime, timezone
    from backend.database.connection import pipeline_db_session
    from backend.loop_engine import StageTracker, StageState
    from backend.services.transcript_correction_service import correct_and_normalize_transcript
    from backend.services.activity_logger import ActivityLogger, build_activity_logger_for_meeting

    job_id = str(meeting_id)
    proc_start_dt = datetime.now(timezone.utc)
    start_t = time.perf_counter()
    wav_path = file_path + ".wav"
    error_msg = None
    act: ActivityLogger = None  # initialized after Phase 1 DB read

    with PipelineLogContext(job_id=job_id, meeting_id=job_id, stage="INIT"):
        logger.info(
            "PIPELINE START | file=%s",
            os.path.basename(file_path),
        )

    # ── Phase 1: Mark meeting as processing (fast DB write) ──────────────────
    with PipelineLogContext(job_id=job_id, meeting_id=job_id, stage="DB_INIT"):
        with pipeline_db_session() as db:
            # Atomic claim: only ONE worker may move a meeting to "processing". The UPDATE's WHERE clause is
            # re-evaluated after a concurrent claimer commits, so of two workers started for the same meeting
            # (a double submit, or two uvicorn processes) exactly one gets a row back; the other stops here
            # instead of running a second pipeline over the same rows and files.
            if not _claim_meeting(db, meeting_id, proc_start_dt):
                return _NOT_STARTED
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            # Initialize ActivityLogger — one instance per pipeline run
            act = build_activity_logger_for_meeting(meeting)
            _RUN_ACTIVITY_LOGGERS[meeting_id] = act
            _upload_log_id = act.start_step(
                "meeting_upload",
                message="Meeting received and queued for processing",
                details={"filename": meeting.filename, "file_type": meeting.file_type},
            )
            act.complete_step(
                _upload_log_id,
                message="Meeting upload confirmed",
                details={"meeting_pk": meeting.id, "bharatvc_meeting_id": meeting.bharatvc_meeting_id},
            )
            # Capture scalar fields needed for Phase 2 AI calls (avoids detached-instance errors)
            meeting_title   = meeting.title or ""
            meeting_agenda  = clean_agenda_text(meeting.agenda) or ""
            meeting_project = meeting.project_name or ""
            meeting_participants = meeting.participants or ""
            meeting_filename = meeting.filename or ""
            upload_time = meeting.upload_time
            # Spoken language named at upload (or kept from an earlier run of this meeting); None = configured default.
            meeting_language = getattr(meeting, "source_language", None)
            # DI-01 FIX: Capture user-specified speaker count to pass into diarization.
            # Previously this was never wired — any num_speakers set by the user had no effect.
            meeting_num_speakers = getattr(meeting, "num_speakers", None)
            if meeting_num_speakers is not None:
                try:
                    meeting_num_speakers = int(meeting_num_speakers)
                    if meeting_num_speakers < 1:
                        meeting_num_speakers = None
                except (TypeError, ValueError):
                    meeting_num_speakers = None
        logger.info("PHASE_1 COMPLETE | status=processing")

    # ── Phase 2: Slow AI processing — NO DB connection held ─────────────────
    # Note: act uses its own pipeline_db_session per write — safe without an open session.
    # Stage bookkeeping for the failure handler: the stage that is running now, and the outcome of every stage this
    # run has PROVEN finished - so a failure is recorded against the right stage instead of leaving all PENDING.
    _p2_stage = "audio_extraction"
    _p2_states: dict = {}
    try:
        with PipelineLogContext(job_id=job_id, meeting_id=job_id, stage="AUDIO_EXTRACTION"):
            logger.info("PHASE_2 START | stage=AUDIO_EXTRACTION | src=%s", os.path.basename(file_path))
            _audio_log_id = act.start_step("audio_extraction", message="Extracting audio from media file")
            success = extract_audio(file_path, wav_path)
            if not success:
                act.fail_step(_audio_log_id, error_code="AUDIO_EXTRACTION_FAILED",
                              error_message="FFmpeg returned non-zero exit code")
                raise Exception("Audio extraction failed — FFmpeg returned non-zero exit code")
            wav_size_mb = round(os.path.getsize(wav_path) / (1024 * 1024), 2) if os.path.exists(wav_path) else 0
            # Real media length from ffprobe — this is the authoritative meeting
            # duration. Do NOT infer it later from the span of diarized speech
            # (that omits leading/trailing silence and understates the timeline).
            from backend.services.transcription_service import get_audio_info
            media_duration = float(get_audio_info(wav_path).get("duration") or 0.0)
            if media_duration <= 0:
                media_duration = float(get_audio_info(file_path).get("duration") or 0.0)
            act.complete_step(_audio_log_id, message="Audio extracted successfully",
                              details={"wav_size_mb": wav_size_mb, "source": os.path.basename(file_path),
                                       "media_duration_sec": round(media_duration, 2)})
            logger.info("AUDIO_EXTRACTION SUCCESS | wav_size_mb=%s | media_duration_sec=%s",
                        wav_size_mb, round(media_duration, 2))
            _p2_states["audio_extraction"] = "SUCCESS"
            _p2_stage = "stt"   # language resolution + speech-to-text

        # ── Spoken language for AIRAWAT Whisper ──────────────────────────────────
        # AIRAWAT Whisper cannot detect a language and the code it is given decides the output, so it is never
        # guessed: the upload's language, else the configured default (admin setting / AIRAWAT_WHISPER_LANGUAGE),
        # else the meeting fails here with a clear error. Translation to English happens downstream.
        with PipelineLogContext(job_id=job_id, meeting_id=job_id, stage="LANGUAGE_DETECTION"):
            _lang_log_id = act.start_step("language_detection", message="Resolving the spoken language")
            from backend.services.language_service import language_name as _lname_default
            from backend.services.language_service import resolve_spoken_language
            from backend.services.model_router import model_router as _mr_lang

            from backend.providers import provider_factory as _pf_lang

            # The configured default: the AIRAWAT admin setting, else AIRAWAT_WHISPER_LANGUAGE / DEFAULT_WHISPER_LANGUAGE.
            _configured_lang = _mr_lang._get_dynamic_config()["airawat"].get("language")
            source_lang, _lang_source = resolve_spoken_language(meeting_language, _configured_lang)
            # The speech-to-text provider must be configured at all; resolving it here fails the meeting before any
            # audio work, with the configuration error (never a silent switch to another provider).
            _stt_name = _pf_lang.stt_provider_name()
            if not source_lang:
                if _stt_name == "airawat":
                    # AIRAWAT cannot detect a language and the code it is given decides the output: never guessed.
                    _msg = ("The spoken language is unknown: AIRAWAT Whisper cannot detect it. Upload the meeting with "
                            "the form field `language` (e.g. hi, te, ml, en) or set DEFAULT_WHISPER_LANGUAGE.")
                    act.fail_step(_lang_log_id, error_code="LANGUAGE_NOT_SPECIFIED", error_message=_msg)
                    raise LanguageIdentificationError(_msg)
                _lang_source = "detected"   # the STT provider identifies it; recorded after transcription
            source_lang_name = _lname_default(source_lang) if source_lang else None
            # 1.0 when the uploader stated the language; None (not measured) otherwise. It was stored as 0.0 and then
            # scored as a measured "0 % language confidence" (language_consistency component 0.0).
            source_lang_conf = 1.0 if _lang_source == "upload" else None
            act.complete_step(
                _lang_log_id,
                message=(f"Spoken language: {source_lang_name} ({source_lang}) from the "
                         f"{'upload' if _lang_source == 'upload' else 'configured default'}") if source_lang
                else "Spoken language not specified; it will be identified during speech-to-text",
                details={"code": source_lang, "name": source_lang_name, "source": _lang_source},
            )
            logger.info("LANGUAGE | %s (%s) source=%s stt=%s", source_lang_name, source_lang, _lang_source, _stt_name)

        # NeMo's speaker pass reads only the audio, while the Whisper request below is network-bound on AIRAWAT:
        # run them at the same time (they used to run one after the other). The alignment with the transcript
        # words still happens after both, in the diarization stage.
        # (DIARIZATION_OVERLAP_WITH_STT=false starts it after transcription instead, as before.)
        diar_future = (
            _start_audio_diarization(wav_path, meeting_num_speakers)
            if settings.DIARIZATION_OVERLAP_WITH_STT else None
        )

        with PipelineLogContext(job_id=job_id, meeting_id=job_id, stage="TRANSCRIPTION"):
            logger.info("PHASE_2 START | stage=TRANSCRIPTION")
            _stt_log_id = act.start_step("transcription", message="Running speech-to-text (Whisper)")
            stt_start_t = time.perf_counter()
            # Root-cause fix for misheard proper nouns / project terms: bias the ASR
            # decoder with a short hint built ONLY from THIS meeting's own metadata
            # (whatever the user has already supplied at upload time). Dynamic per
            # meeting, never hardcoded; a meeting with no metadata yet gets no hint
            # and behaves exactly as before.
            from backend.services.transcript_correction_service import build_vocabulary_hint
            vocab_hint = build_vocabulary_hint(
                meeting_title=meeting_title,
                agenda=meeting_agenda,
                project_name=meeting_project,
                participants=meeting_participants,
            )
            transcript_data = transcribe_audio(wav_path, language=source_lang, vocabulary_hint=vocab_hint)
            stt_elapsed = round(time.perf_counter() - stt_start_t, 3)
            _td_was_dict = isinstance(transcript_data, dict)
            if not _td_was_dict:
                transcript_data = {"text": "", "words": []}
            if not source_lang:
                # No language was given: use the one the speech-to-text provider identified (None if it reported none).
                from backend.services.language_service import code_for_reported_language

                source_lang = code_for_reported_language(transcript_data.get("language"))
                source_lang_name = _lname_default(source_lang) if source_lang else None
                logger.info("LANGUAGE | identified by speech-to-text: %s (%s)", source_lang_name, source_lang)
            # transcribe_audio() returns {"text", "words", ...} — it does NOT emit a
            # "segments" key, so never derive the STT metric from one (that made every
            # meeting log "0 segments"). The honest STT signals are word count + text.
            stt_words = transcript_data.get("words") or []
            stt_word_count = len(stt_words)
            _raw_text = transcript_data.get("text") or ""
            stt_text = _raw_text.strip()
            stt_char_count = len(stt_text)
            lang = transcript_data.get("language") or transcript_data.get("language_code") or "en"

            if not stt_text:
                # Genuine failure OR silent audio — never continue as "SUCCESS" with an
                # empty transcript. Build a full diagnostic so the CAUSE is in the
                # error itself (no server-log access needed to triage it).
                empty_reason = transcript_data.get("empty_reason") or "transcription_returned_no_text"

                # Probe the extracted WAV independently of the transcriber.
                _wav_exists = os.path.exists(wav_path)
                _wav_mb = round(os.path.getsize(wav_path) / (1024 * 1024), 2) if _wav_exists else 0.0
                try:
                    from backend.services.transcription_service import get_audio_info as _gai
                    _wav_dur = round(float(_gai(wav_path).get("duration") or 0.0), 1)
                except Exception:
                    _wav_dur = -1.0
                _route = "speech_to_text"          # active speech-to-text engine

                diag = {
                    "reason": empty_reason,
                    "route": _route,
                    "stt_duration_sec": stt_elapsed,
                    "wav_exists": _wav_exists,
                    "wav_size_mb": _wav_mb,
                    "wav_probed_duration_sec": _wav_dur,
                    "transcriber_returned_dict": _td_was_dict,
                    "return_keys": sorted(transcript_data.keys()) if _td_was_dict else None,
                    "raw_text_len": len(_raw_text),
                    "word_timestamps": stt_word_count,
                    "has_empty_reason_key": "empty_reason" in transcript_data,
                    "language_sent": source_lang,
                    "raw_text_head": _raw_text[:300],
                }
                logger.error("TRANSCRIPTION_EMPTY diagnostic | %s", json.dumps(diag, default=str))

                # Human-readable hint about the most likely cause.
                if not _wav_exists or _wav_mb == 0.0:
                    hint = "audio extraction produced no WAV — check ffmpeg / the source file's audio track"
                elif _wav_dur == 0.0:
                    hint = "extracted WAV has 0s duration — the MP4's audio stream did not decode"
                elif _raw_text and not stt_text:
                    hint = "transcriber returned only whitespace"
                elif stt_word_count > 0 and len(_raw_text) == 0:
                    hint = ("word timestamps were produced but text was emptied downstream — "
                            "hallucination-cleaning over-stripped (update build) ")
                elif empty_reason.endswith("_returned_no_speech_text"):
                    hint = (f"Speech-to-text returned no text for language '{source_lang}' - the recording may "
                            f"contain no speech, or the spoken language differs (upload with the right `language`)")
                elif empty_reason == "transcription_returned_no_text":
                    hint = ("transcriber returned empty text with NO empty_reason — this build predates "
                            "the long-video transcription fix; deploy the current backend")
                else:
                    hint = "audio may contain no audible speech, or the Whisper endpoint returned nothing"

                act.fail_step(
                    _stt_log_id,
                    error_code="TRANSCRIPTION_EMPTY",
                    error_message=(
                        f"Speech-to-text produced no text. Reason: {empty_reason}. "
                        f"Likely cause: {hint}. Diagnostic: {json.dumps(diag, default=str)}"
                    ),
                    details=diag,
                )
                raise Exception(
                    f"Transcription failed — no text produced (reason: {empty_reason}; "
                    f"route={_route}, wav={_wav_mb}MB/{_wav_dur}s, raw_text_len={len(_raw_text)}, "
                    f"word_ts={stt_word_count}, likely_cause: {hint})"
                )

            act.complete_step(
                _stt_log_id,
                message=f"Transcription complete — {stt_word_count} words ({stt_char_count} chars)",
                details={
                    "word_count": stt_word_count,
                    "char_count": stt_char_count,
                    "language": lang,
                    "stt_duration_sec": stt_elapsed,
                },
            )
            logger.info(
                "TRANSCRIPTION SUCCESS | elapsed_sec=%s | words=%s | chars=%s",
                stt_elapsed, stt_word_count, stt_char_count,
            )
            _p2_states["stt"] = "SUCCESS"
            _p2_stage = "diarization"

        with PipelineLogContext(job_id=job_id, meeting_id=job_id, stage="SPEAKER_DIARIZATION"):
            logger.info("PHASE_2 START | stage=SPEAKER_DIARIZATION")
            _diar_log_id = act.start_step("speaker_diarization", message="Running speaker diarization & identification (NeMo)")
            # The engine's audio pass was started before transcription; collect it here (an error in it surfaces at
            # this same point, as the engine's own error did before).
            _join_t0 = time.perf_counter()
            if diar_future is None:   # overlap disabled: the engine's audio pass starts now
                diar_future = _start_audio_diarization(wav_path, meeting_num_speakers)
            audio_pass, audio_pass_sec = diar_future.result()
            join_wait_sec = time.perf_counter() - _join_t0
            # DI-01 FIX: Pass user-specified speaker count so diarization honors it.
            # Previously diarize_and_align() was always called without num_speakers.
            _align_t0 = time.perf_counter()
            diar_res = diarize_and_align(
                wav_path, transcript_data, num_speakers=meeting_num_speakers, primary_result=audio_pass
            )
            diarized_segments, speaker_diag, diar_status = diar_res[0], diar_res[1], diar_res[2]
            diar_metadata = getattr(diar_res, "metadata", {})
            align_sec = time.perf_counter() - _align_t0
            # The diarization WORK time (audio pass + validation/alignment/labelling), not just the tail the
            # pipeline waited for - it is stored as meeting.speaker_duration_sec and feeds the runtime metrics.
            spk_elapsed = round(audio_pass_sec + align_sec, 3)
            logger.info(
                "SPEAKER_DIARIZATION timing | audio_pass=%.1fs (started before transcription) | waited %.1fs for it "
                "after transcription finished | validation+alignment+labels=%.1fs",
                audio_pass_sec, join_wait_sec, align_sec,
            )
            speaker_count = speaker_diag.get("speaker_count", 0) if isinstance(speaker_diag, dict) else 0

            # Bug 3 Fix: diarization_status must not be SUCCESS when speaker_count=0 and
            # no segments survived text-filtering. Downgrade to DEGRADED so the DB does
            # not store the paradoxical SUCCESS+0-speakers state.
            if diar_status == "SUCCESS" and speaker_count == 0 and not diarized_segments:
                diar_status = "DEGRADED"
                logger.warning(
                    "SPEAKER_DIARIZATION downgraded to DEGRADED: reported SUCCESS but "
                    "no valid segments exist after text filtering (speaker_count=0)."
                )

            if diar_status == "FAILED":
                act.fail_step(_diar_log_id, error_code="DIARIZATION_FAILED",
                              error_message="Diarization pipeline returned FAILED status",
                              details={"engine": diar_metadata.get("selected_engine", "NVIDIA_NEMO"),
                                       "detected_speakers": speaker_count})
            else:
                act.complete_step(_diar_log_id,
                                  message=f"Diarization {diar_status} — {speaker_count} speakers detected",
                                  details={"detected_speakers": speaker_count,
                                           "speaker_segments": len(diarized_segments),
                                           "engine": diar_metadata.get("selected_engine", "NVIDIA_NEMO"),
                                           "diarization_status": diar_status,
                                           # this step's duration is the critical-path wait; the engine pass itself
                                           # ran alongside transcription
                                           "audio_pass_sec": audio_pass_sec,
                                           "waited_for_audio_pass_sec": round(join_wait_sec, 3)})
            logger.info(
                "SPEAKER_DIARIZATION %s | elapsed_sec=%s | speakers=%s | segments=%s",
                diar_status, spk_elapsed, speaker_count, len(diarized_segments),
            )
            # The engine's own outcome is what the stage state reports (same mapping as the Phase-3 write below).
            _p2_states["diarization"] = _stage_state_for_diarization(diar_status).value
            _p2_states["alignment"] = "SUCCESS" if diar_status == "SUCCESS" else "DEGRADED"
            _p2_stage = "transcript_correction"

        # ── Feature 5: Speaker Duration ─────────────────────────────────────────
        _dur_log_id = act.start_step("speaker_duration", message="Computing speaker talk time and duration statistics")
        seg_times = []
        unique_speakers = set()
        for seg in diarized_segments:
            try:
                s = float(seg["start_time"])
                e = float(seg["end_time"])
                seg_times.append((s, e))
            except Exception:
                pass
            if seg.get("speaker") and "UNKNOWN" not in str(seg["speaker"]).upper():
                unique_speakers.add(seg["speaker"])

        seg_times.sort(key=lambda x: x[0])
        speech_span = 0.0          # first-utterance-start → last-utterance-end
        total_speech = 0.0         # summed spoken time (excludes gaps)
        inter_segment_silence = 0.0
        interruptions = 0
        if seg_times:
            speech_span = seg_times[-1][1] - seg_times[0][0]
            last_end = seg_times[0][1]
            total_speech = seg_times[0][1] - seg_times[0][0]
            for s, e in seg_times[1:]:
                total_speech += max(0.0, e - s)
                if s > last_end:
                    inter_segment_silence += s - last_end
                elif s < last_end:
                    interruptions += 1
                last_end = max(last_end, e)

        # Authoritative meeting duration = real media length (ffprobe). Fall back to
        # the speech span only if the probe failed.
        # Silence = media length - DETECTED speech (the diarization's VAD regions). It used to be media length minus
        # the summed segment spans, which is 0 whenever segments tile the timeline (they always do for one speaker),
        # i.e. "no silence" for a recording that measured 12 % silence. Without a VAD measurement the value is
        # unknown (None), never 0.
        meeting_duration = media_duration if media_duration > 0 else speech_span
        silence_duration = silence_from_speech(speaker_diag, media_duration)

        act.complete_step(
            _dur_log_id,
            message=f"Speaker duration computed for {len(unique_speakers)} speaker(s)",
            details={
                "speaker_count": len(unique_speakers),
                "meeting_duration_sec": round(meeting_duration, 2),
                "silence_duration_sec": round(silence_duration, 2) if silence_duration is not None else None,
                "interruptions": interruptions,
            },
        )

        # ── Feature 6: Transcript + Speaker Mapping ─────────────────────────────
        with PipelineLogContext(job_id=job_id, meeting_id=job_id, stage="TRANSCRIPT_CORRECTION"):
            logger.info("PHASE_2 START | stage=TRANSCRIPT_CORRECTION")
            _corr_log_id = act.start_step("transcript_correction", message="Aligning speaker mapping and normalizing transcript")
            corrected_segments = correct_and_normalize_transcript(
                segments=diarized_segments,
                meeting_title=meeting_title,
                agenda=meeting_agenda,
                project_name=meeting_project,
                participants=meeting_participants,
                meeting_id=meeting_id,
            )
            # correct_and_normalize_transcript's own catch-all falls back to
            # correction_source="raw_asr_fallback" on total failure — surface
            # that as a degraded step instead of reporting a plain "complete"
            # that looks identical to a real, successful correction pass.
            _fallback_segs = sum(
                1 for s in corrected_segments if s.get("correction_source") == "raw_asr_fallback"
            )
            if _fallback_segs and _fallback_segs == len(corrected_segments):
                act.fail_step(
                    _corr_log_id, error_code="CORRECTION_FAILED",
                    error_message=(corrected_segments[0].get("correction_reason") or "correction pass failed"),
                    details={"corrected_segments": len(corrected_segments), "fallback_segments": _fallback_segs},
                )
                logger.warning("TRANSCRIPT_CORRECTION FAILED — fell back to raw ASR for all %s segment(s)", len(corrected_segments))
            else:
                act.complete_step(_corr_log_id, message="Transcript and speaker mapping complete",
                                  details={"corrected_segments": len(corrected_segments)})
                logger.info("TRANSCRIPT_CORRECTION SUCCESS | corrected_segments=%s", len(corrected_segments))
            # The stage state must agree with the activity log above: SUCCESS was written unconditionally, even
            # while the step itself was logged FAILED because every segment fell back to the raw ASR text.
            if _fallback_segs:
                _corr_state = "DEGRADED"
                _corr_reason = (
                    f"{_fallback_segs} of {len(corrected_segments)} segments kept the raw ASR text "
                    f"(correction unavailable)"
                )
            else:
                _corr_state = "SUCCESS"
                _corr_reason = f"Corrected {len(corrected_segments)} segments"
            _p2_states["transcript_correction"] = _corr_state

        # ── English translation stage (any language → English) ──────────────────
        # Runs AFTER native ASR + correction so word timestamps stay aligned and
        # the LLM has full native context. Speaker labels + timestamps preserved
        # exactly; native text kept in `raw_text` / `original_text`. English
        # source or TRANSLATION_ENABLED=false → passthrough (no accuracy risk).
        translation_meta = {"translated": False, "translated_segments": 0, "partial": False}
        translate_elapsed = 0.0
        with PipelineLogContext(job_id=job_id, meeting_id=job_id, stage="TRANSLATION"):
            _tr_log_id = act.start_step(
                "translation",
                message=f"Translating transcript to English (source: {source_lang_name or 'auto'})",
            )
            try:
                from backend.services.translation_service import translate_segments_to_english
                _tr_t0 = time.perf_counter()
                corrected_segments, translation_meta = translate_segments_to_english(
                    corrected_segments,
                    source_lang or "",
                    meeting_context={
                        "title": meeting_title, "agenda": meeting_agenda,
                        "participants": meeting_participants, "project_name": meeting_project,
                    },
                )
                translate_elapsed = round(time.perf_counter() - _tr_t0, 3)
                if translation_meta.get("translated"):
                    _msg = (f"Translated {translation_meta.get('translated_segments', 0)}/"
                            f"{len(corrected_segments)} segment(s) from {source_lang_name} to English"
                            + (" (partial — some lines kept native)" if translation_meta.get("partial") else ""))
                else:
                    _msg = translation_meta.get("note") or "Transcript already English — no translation needed"
                act.complete_step(_tr_log_id, message=_msg, details=translation_meta)
                logger.info("TRANSLATION | %s | elapsed=%ss", _msg, translate_elapsed)
            except Exception as _tr_err:
                # Never fail the whole PIPELINE on translation — keep going with
                # what we have — but the activity-log STEP itself must be marked
                # FAILED, not "complete", or an operator/UI reading the log has
                # no way to tell this apart from a genuine successful translation.
                act.fail_step(_tr_log_id, error_code="TRANSLATION_FAILED",
                              error_message=f"Translation stage error ({_tr_err})",
                              details={"error": str(_tr_err)})
                logger.error("TRANSLATION stage failed: %s", _tr_err, exc_info=True)
                if isinstance(_tr_err, (AIProviderError, AirawatError)):
                    raise  # AI LLM failed after its retries: outputs must be English, so the meeting fails clearly

        # Build transcript text (English) — no DB needed
        full_transcript_text = ""
        for seg in corrected_segments:
            corr_text = ts.clean_whisper_hallucinations(seg.get("text", ""))
            if corr_text.strip():
                full_transcript_text += f"{seg['speaker']}: {corr_text}\n"

        # ── Per-meeting transcript identity + empty guard (BUG-SAMEOUTPUT) ──────
        # Log a fingerprint so any cross-meeting contamination is visible in logs,
        # and refuse to continue to report generation on a content-less
        # transcript (that path used to emit identical boilerplate for every
        # such meeting).
        _tx_fp = ts.content_fingerprint(full_transcript_text)
        _tx_meaningful = len(ts.normalize_text(full_transcript_text))
        logger.info(
            "TRANSCRIPT_READY | meeting_id=%s | chars=%d | meaningful_chars=%d | "
            "words=%d | fingerprint=%s | source_lang=%s | translated=%s",
            meeting_id, len(full_transcript_text), _tx_meaningful,
            len(full_transcript_text.split()), _tx_fp[:16], source_lang,
            translation_meta.get("translated"),
        )
        if _tx_meaningful < 40:
            raise Exception(
                "Transcript contains no usable content after STT/diarization/"
                f"translation (meaningful_chars={_tx_meaningful}). Not generating "
                "reports — this would otherwise emit generic placeholder Agenda/"
                "Summary/Decisions identical across meetings."
            )

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in AI processing phase for meeting {meeting_id}: {error_msg}", exc_info=True)
        if act:
            act.cancel_remaining(from_step_order=0)
        # The stage that was running is FAILED, the ones this run finished keep their real outcome, and the rest
        # stay PENDING (they never ran) - plus overall_status / api_status, which this handler never set.
        _mark_meeting_failed(meeting_id, e, failed_stage=_p2_stage, stage_states=_p2_states)
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as _fe:
                logger.warning(f"Could not remove temp WAV '{wav_path}': {_fe}")
        return

    # ── Phase 3: Write transcript segments to DB (fast) ─────────────────────
    try:
        with pipeline_db_session() as db:
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            if not meeting:
                return

            tracker = StageTracker(meeting, db)
            tracker.transition("audio_extraction", StageState.SUCCESS, "Audio extraction completed", commit=False)
            tracker.transition("stt", StageState.SUCCESS, f"Transcribed {stt_word_count} words", commit=False)
            meeting.diarization_status = diar_status
            meeting.diarization_info_json = json.dumps({
                "engine": diar_metadata.get("selected_engine", "NVIDIA_NEMO"),
                "status": diar_status,
                "speaker_count": speaker_diag.get("speaker_count", 0),
                "diagnostics": speaker_diag,
            })
            meeting.diarization_metadata_json = json.dumps(diar_metadata)
            # Report the diarization engine's TRUE status to the tracker — a
            # hard FAILED must surface as FAILED (not be silently relabeled
            # DEGRADED), so mark_completed()'s honest-status cascade and the
            # retry/activity audit trail both reflect what actually happened.
            # No speaker information is invented either way: diarize_and_align
            # already returns one real, unassigned Speaker_1 segment spanning
            # the actual transcript on a hard failure — this only fixes how
            # that outcome is *reported*, not what data is produced.
            _diar_stage_state = _stage_state_for_diarization(diar_status)
            tracker.transition(
                "diarization",
                _diar_stage_state,
                f"Engine: {diar_metadata.get('selected_engine', 'NVIDIA_NEMO')}",
                commit=False,
            )
            # Alignment is only as good as the speaker turns it was aligned to.
            tracker.transition(
                "alignment",
                StageState.SUCCESS if diar_status == "SUCCESS" else StageState.DEGRADED,
                "Speaker alignment completed" if diar_status == "SUCCESS"
                else f"Aligned to a {diar_status} diarization - speaker labels are not verified",
                commit=False,
            )
            tracker.transition("transcript_correction", StageState(_corr_state), _corr_reason, commit=False)

            db.query(TranscriptSegment).filter(TranscriptSegment.meeting_id == meeting_id).delete()
            for seg in corrected_segments:
                raw_text = seg.get("raw_text", "")
                corr_text = ts.clean_whisper_hallucinations(seg.get("text", ""))
                db_segment = TranscriptSegment(
                    meeting_id=meeting_id,
                    speaker=seg["speaker"],
                    speaker_id=seg.get("speaker_id"),
                    speaker_name=seg.get("speaker_name"),
                    speaker_confidence=seg.get("speaker_confidence"),
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=corr_text,                                  # final, English
                    raw_text=raw_text,                              # raw ASR (source language)
                    original_text=seg.get("original_text"),         # corrected source-language text
                    stt_confidence=seg.get("stt_confidence"),
                    correction_applied=seg.get("correction_applied", False),
                    correction_confidence=seg.get("correction_confidence"),
                    correction_source=seg.get("correction_source"),
                    correction_reason=seg.get("correction_reason"),
                    correction_metadata=seg.get("correction_metadata"),
                )
                db.add(db_segment)

            clean_name = (
                os.path.splitext(meeting_filename)[0]
                .replace("_", " ").replace("-", " ").title()
            )
            if not meeting.title:
                meeting.title = clean_name
            # No agenda text is made up here: meeting.agenda stays exactly what the uploader supplied (or empty). The
            # agenda topics and their decisions are extracted from the transcript into agenda_key_decisions.
            if not meeting.meeting_date and upload_time:
                meeting.meeting_date = upload_time.strftime("%Y-%m-%d")
            if not meeting.meeting_time and upload_time:
                meeting.meeting_time = upload_time.strftime("%I:%M %p")

            meeting.duration = meeting_duration
            meeting.silence_duration = silence_duration
            meeting.interruptions = interruptions
            meeting.participant_count = len(unique_speakers)
            # Prefer voice-matched registered names; fall back to any non-generic
            # label (LLM transcript-clue names), else the generic speaker set.
            voice_names = {
                seg.get("speaker_name") for seg in diarized_segments if seg.get("speaker_name")
            }
            resolved_names = voice_names or {
                seg["speaker"] for seg in diarized_segments
                if seg.get("speaker")
                and not str(seg["speaker"]).upper().startswith("SPEAKER_")
                and not str(seg["speaker"]).strip().lower().startswith("speaker ")
            }
            if resolved_names:
                meeting.participants = ", ".join(sorted(resolved_names))
            else:
                meeting.participants = ", ".join(sorted(list(unique_speakers)))
            meeting.audio_quality = None
            meeting.video_quality = None
            meeting.transcription_accuracy = None
            meeting.speaker_accuracy = None
            meeting.stt_generated = bool(diarized_segments and full_transcript_text.strip())
            meeting.stt_word_count = count_transcript_words(full_transcript_text)
            meeting.stt_char_count = len(full_transcript_text)
            meeting.stt_duration_sec = stt_elapsed
            # ── Multilingual pipeline audit fields ────────────────────────────
            # Unknown stays unknown (None): it used to be stored as English whatever was spoken.
            meeting.source_language = source_lang or None
            meeting.source_language_name = source_lang_name or None
            meeting.source_language_confidence = source_lang_conf
            meeting.translation_applied = bool(translation_meta.get("translated"))
            try:
                from backend.config.config import settings as _sett
                meeting.output_language = getattr(_sett, "OUTPUT_LANGUAGE", "en")
            except Exception:
                meeting.output_language = "en"
            meeting.translation_duration_sec = translate_elapsed
            meeting.speaker_count = len(unique_speakers)
            meeting.speaker_timeline_generated = bool(diarized_segments)
            meeting.speaker_duration_sec = spk_elapsed
            # Fields Phase 4 needs, read BEFORE the commit (a commit expires the row and would re-SELECT it).
            meeting_title   = meeting.title or ""
            meeting_agenda  = clean_agenda_text(meeting.agenda) or ""
            meeting_date    = meeting.meeting_date or ""
            meeting_time_str = meeting.meeting_time or ""
            meeting_duration_f = meeting.duration
            # ONE commit for the segments, the stage states and the meeting columns (there were seven).
            db.commit()

        # Index transcript chunks into vector DB (part of Feature 15 RAG pipeline). Outside the session: it embeds
        # text and writes to ChromaDB, and must not hold a database transaction open while it does.
        _t0_chunks = time.perf_counter()
        vector_service.add_transcript_chunks(meeting_id, corrected_segments)
        act.record_timing("embeddings_chromadb_transcript_chunks", time.perf_counter() - _t0_chunks)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error writing transcript for meeting {meeting_id}: {error_msg}", exc_info=True)
        # Phase 2 completed: its stage outcomes were being written in the transaction that just failed (rolled back),
        # so they are re-recorded from this run's own bookkeeping. No stage is marked FAILED - none of them is
        # "saving the transcript"; the failure is the meeting's error_message.
        _mark_meeting_failed(meeting_id, e, stage_states=_p2_states)
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as _fe:
                logger.warning(f"Could not remove temp WAV '{wav_path}': {_fe}")
        return

    # ── Free Phase-2/3 buffers before the LLM wait ──────────────────────────
    # transcript_data (raw Whisper words), diarized_segments and corrected_segments
    # are all persisted to the DB by now and not referenced again — only
    # full_transcript_text is still needed. Releasing them here keeps peak RSS
    # low during the multi-minute LLM phase.
    try:
        import gc as _gc
        transcript_data = None
        diarized_segments = None
        corrected_segments = None
        speaker_diag = None
        diar_res = None
        _gc.collect()
    except Exception:
        pass

    # ── Phase 4: LLM report computation (in-memory, no DB connection held) ─
    reports = {}
    try:
        from backend.utils.helpers import format_seconds_to_duration
        m_dur = format_seconds_to_duration(meeting_duration_f)

        def _record_report_retry(attempt: int, error: Exception) -> None:
            act.retry_step(
                "summary_generation",
                message=f"Retrying the AI report (attempt {attempt}) after {type(error).__name__}",
                details={"attempt": attempt, "reason": str(error)[:300]},
            )

        _t0_report = time.perf_counter()
        reports = _generate_reports_with_retry(
            meeting_id, full_transcript_text, on_retry=_record_report_retry, meeting_date=meeting_date,
            meeting_time=meeting_time_str, duration_str=m_dur
        )
        reports["_elapsed_sec"] = round(time.perf_counter() - _t0_report, 3)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in LLM phase for meeting {meeting_id}: {error_msg}", exc_info=True)
        _mark_meeting_failed(meeting_id, e)
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as _fe:
                logger.warning(f"Could not remove temp WAV '{wav_path}': {_fe}")
        return

    # ── Phase 5: Write all analysis results (fast) ───────────────────────────
    try:
        # 5a. One short write transaction: report rows, meeting columns, report-stage states -> COMMIT -> closed.
        with pipeline_db_session() as db:
            meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
            if not meeting:
                return

            context = _run_shared_pipeline(meeting_id, full_transcript_text, meeting, db,
                                           reports=reports, act=act)

        # 5b. No transaction is open from here: RAG indexing (embedding + ChromaDB), related meetings, and then the
        #     completion - the meeting becomes "completed" in the same transaction as its final JSON.
        _finish_meeting_processing(
            context, act=act, use_stage_outcome=True, proc_start_dt=proc_start_dt, start_t=start_t,
        )

        # 5c. Runtime-metrics log line (read-only; a failure here must not undo a completed meeting).
        try:
            with pipeline_db_session() as db:
                meeting = db.query(Meeting).filter(Meeting.id == meeting_id).first()
                from backend.services.runtime_metrics import build_runtime_metrics_dict, log_runtime_metrics
                log_runtime_metrics(meeting_id, build_runtime_metrics_dict(meeting, db))
        except Exception as metrics_err:
            logger.warning(f"[Pipeline] Runtime metrics log skipped for meeting {meeting_id}: {metrics_err}")

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error writing results for meeting {meeting_id}: {error_msg}", exc_info=True)
        # Stage states of this run are already in the database (Phase 3 and 5a committed them); an AIRAWAT section
        # failure names its sections, anything else is recorded as the meeting's error with every stage as it was.
        _mark_meeting_failed(meeting_id, e)
    finally:
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception as fe:
                logger.warning(f"Failed to clean up temp WAV file {wav_path}: {fe}")
        # Drop the last large in-RAM buffers and hand memory back to the OS so a
        # long-running worker does not carry one meeting's peak into the next.
        try:
            import gc as _gc
            full_transcript_text = None
            reports = None
            _gc.collect()
            try:
                import ctypes
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
        except Exception:
            pass



def process_document_pipeline(meeting_id: int, file_path: str):
    """Background entry point for documents; always closes the live SSE stream when done."""
    outcome = None
    try:
        outcome = _process_document_pipeline(meeting_id, file_path)
    finally:
        if outcome is not _NOT_STARTED:   # a worker that did not take the meeting must not close the owner's stream
            _close_sse_stream(meeting_id)


def process_meeting_pipeline(meeting_id: int, file_path: str):
    """Background entry point for audio/video; always closes the live SSE stream when done."""
    t0 = time.perf_counter()
    outcome = None
    try:
        outcome = _process_meeting_pipeline(meeting_id, file_path)
    finally:
        if outcome is not _NOT_STARTED:
            _log_run_timing(meeting_id, time.perf_counter() - t0)
            _close_sse_stream(meeting_id)


# ─── Dynamic pipeline registry ────────────────────────────────────────────────
PIPELINE_REGISTRY = {
    "media": process_meeting_pipeline,
    "document": process_document_pipeline,
}
