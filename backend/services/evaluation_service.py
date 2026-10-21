"""
Evaluation Service — API facade over the Universal Evaluation Engine.
=========================================================
All quality measurements are computed by the engine from real data and
ground truth. Nothing here is estimated, hardcoded, or fabricated: missing
references yield ``None`` which the API layer renders as "Not Measured".
"""

import asyncio
import logging
import re
import time
from typing import Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.config.config import settings
from backend.evaluation.engine import evaluation_engine, MEASURED
from backend.models.models import Meeting, TranscriptSegment
from backend.services.model_router import ModelRouter

logger = logging.getLogger(__name__)

# Legacy weights and thresholds kept as metadata reference
METRIC_WEIGHTS = {
    "transcription": 0.22,
    "speaker": 0.18,
    "summary": 0.15,
    "mom": 0.12,
    "action_items": 0.08,
    "audio_quality": 0.08,
    "topic": 0.05,
    "sentiment": 0.04,
    "similarity": 0.04,
    "video_quality": 0.04,
}

THRESHOLDS = {
    "excellent": 90,
    "good": 75,
    "average": 60,
    "poor": 45,
}

CRITICAL_THRESHOLD: float = settings.CRITICAL_THRESHOLD


# ─── Helper Functions (used by external metrics recalculators) ─────────────────

def _clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def _tokenize(text: str) -> list:
    if not text:
        return []
    return re.findall(r"\w+", text.lower())


def _get_content_words(text: str, min_len: int = 4) -> list:
    words = re.findall(r"\b[a-zA-Z]{%d,}\b" % min_len, text.lower())
    return [w for w in words]


# ─── Evaluation Service Class ──────────────────────────────────────────────────

class EvaluationService:
    @staticmethod
    async def get_database_status(db: Session) -> str:
        """Pings the database asynchronously and returns health status with query latency."""
        loop = asyncio.get_event_loop()
        def _ping():
            try:
                t0 = time.time()
                db.execute(text("SELECT 1"))
                latency = (time.time() - t0) * 1000.0
                return f"Online ({latency:.1f}ms)"
            except Exception as e:
                logger.error(f"Database ping failed: {e}")
                return "Offline"
        return await loop.run_in_executor(None, _ping)

    @staticmethod
    async def get_api_status() -> str:
        """Pings the AIRAWAT API endpoint reachability asynchronously."""
        loop = asyncio.get_event_loop()
        def _check():
            try:
                router = ModelRouter()
                is_healthy = router.validate_airawat_config()
                return "Online" if is_healthy else "Degraded"
            except Exception as e:
                logger.error(f"API ping failed: {e}")
                return "Offline"
        return await loop.run_in_executor(None, _check)

    @classmethod
    async def calculate_runtime_metrics(cls, meeting: Meeting, db: Session) -> dict:
        """
        Computes Module 1 (Runtime Quality) metrics dynamically from actual
        execution data. Every value is derived from real measurements; any
        component without a measurable reference is reported as None.
        """
        loop = asyncio.get_event_loop()
        def _fetch_segments():
            return db.query(TranscriptSegment).filter(TranscriptSegment.meeting_id == meeting.id).all()

        segments = await loop.run_in_executor(None, _fetch_segments)

        db_status_task = asyncio.create_task(cls.get_database_status(db))
        api_status_task = asyncio.create_task(cls.get_api_status())
        db_status, api_status = await asyncio.gather(db_status_task, api_status_task)

        # 1. Speech Recognition Quality — % of segments with text (real)
        if not segments:
            speech_rec_q = None
        else:
            non_empty = sum(1 for s in segments if s.text and s.text.strip())
            speech_rec_q = round((non_empty / len(segments)) * 100.0, 2)

        # 2. Speaker Consistency Score — % of segments with a known speaker
        if not segments:
            speaker_cons = None
        else:
            known = sum(
                1 for s in segments
                if s.speaker and s.speaker.strip()
                and "unknown" not in s.speaker.lower()
                and "unidentified" not in s.speaker.lower()
            )
            speaker_cons = round((known / len(segments)) * 100.0, 2)

        # 3. Transcript quality — % of segments with non-negative start time
        if not segments:
            transcript_q = None
        else:
            valid = sum(1 for s in segments if not (float(getattr(s, 'start_time', 0) or 0) < 0))
            transcript_q = round((valid / len(segments)) * 100.0, 2)

        # Run the engine for the full scientific report (reference-grounded)
        report = None
        try:
            report = await loop.run_in_executor(
                None,
                lambda: evaluation_engine.evaluate_meeting(meeting, db, performed_by="runtime_metrics", persist=True),
            )
        except Exception as e:
            logger.error(f"Engine evaluation failed in runtime metrics: {e}")

        tasks = (report or {}).get("tasks", {})
        overall = (report or {}).get("overall")

        return {
            # 1. STT
            "stt_wer": tasks.get("stt", {}).get("metrics", {}).get("wer"),
            "stt_cer": tasks.get("stt", {}).get("metrics", {}).get("cer"),
            "speech_recognition_quality": speech_rec_q,

            # 2. Diarization & Speaker Consistency
            "diarization_der": tasks.get("diarization", {}).get("metrics", {}).get("der"),
            "diarization_jer": tasks.get("diarization", {}).get("metrics", {}).get("jer"),
            "transcript_quality": transcript_q,
            "speaker_id_accuracy": tasks.get("speaker_id", {}).get("metrics", {}).get("accuracy"),
            "speaker_id_f1": tasks.get("speaker_id", {}).get("metrics", {}).get("f1"),
            "speaker_id_eer": None,
            "speaker_consistency": speaker_cons,

            # 3. Summary Grounding
            "summary_rougel": tasks.get("summary", {}).get("metrics", {}).get("rougeL"),
            "summary_bertscore": tasks.get("summary", {}).get("metrics", {}).get("bertscore_f1"),
            "summary_grounding": tasks.get("summary", {}).get("quality_score"),

            # 4. MoM Quality
            "mom_quality": tasks.get("mom", {}).get("quality_score"),
            "mom_rougel": tasks.get("mom", {}).get("metrics", {}).get("rougeL"),
            "mom_bertscore": tasks.get("mom", {}).get("metrics", {}).get("bertscore_f1"),

            # 5. Action Item Quality
            "action_item_quality": tasks.get("action_items", {}).get("quality_score"),
            "action_precision": tasks.get("action_items", {}).get("metrics", {}).get("precision"),
            "action_recall": tasks.get("action_items", {}).get("metrics", {}).get("recall"),
            "action_f1": tasks.get("action_items", {}).get("metrics", {}).get("f1"),

            # 6. Topic Coverage
            "topic_coherence": tasks.get("topics", {}).get("metrics", {}).get("coherence"),
            "topic_f1": tasks.get("topics", {}).get("metrics", {}).get("f1"),
            "topic_coverage": tasks.get("topics", {}).get("quality_score"),

            # 7. Sentiment Confidence
            "sentiment_confidence": tasks.get("sentiment", {}).get("quality_score"),
            "sentiment_accuracy": tasks.get("sentiment", {}).get("metrics", {}).get("accuracy"),
            "sentiment_f1": tasks.get("sentiment", {}).get("metrics", {}).get("f1"),
            "sentiment_mcc": None,

            # 8. Similarity Confidence (reference-grounded nDCG only)
            "similarity_confidence": tasks.get("similarity", {}).get("quality_score"),
            "similarity_recall_10": None,
            "similarity_mrr": None,
            "similarity_ndcg": tasks.get("similarity", {}).get("metrics", {}).get("ndcg"),

            # 9. Audio Signal Quality (real media telemetry)
            "audio_signal_quality": tasks.get("audio_signal", {}).get("metrics", {}).get("snr_db"),
            "audio_snr": tasks.get("audio_signal", {}).get("metrics", {}).get("snr_db"),
            "audio_silence_ratio": tasks.get("audio_signal", {}).get("metrics", {}).get("silence_ratio"),
            "audio_quality": tasks.get("audio_signal", {}).get("metrics", {}).get("snr_db"),

            # 10. Video Signal Quality (real media telemetry)
            "video_signal_quality": tasks.get("video_signal", {}).get("metrics", {}).get("fps"),
            "video_fps": tasks.get("video_signal", {}).get("metrics", {}).get("fps"),
            "video_resolution": (
                f"{tasks['video_signal']['metrics']['width']}x{tasks['video_signal']['metrics']['height']}"
                if tasks.get("video_signal", {}).get("metrics", {}).get("width")
                else None
            ),
            "video_bitrate": tasks.get("video_signal", {}).get("metrics", {}).get("bitrate"),
            "video_quality": tasks.get("video_signal", {}).get("metrics", {}).get("fps"),

            # Chatbot AI & Infra
            "chatbot_faithfulness": None,
            "chatbot_answer_relevance": None,
            "hallucination_risk": None,
            "processing_time": round(meeting.total_processing_time_sec or meeting.processing_time or 0.0, 2),
            "database_status": db_status,
            "api_status": api_status,
            "evaluation_status": overall.get("status") if overall else None,
            "overall_accuracy": overall.get("score") if overall else None,
        }

    @classmethod
    async def evaluate_meeting(cls, meeting: Meeting, db: Session):
        """
        Executes the Universal Evaluation Engine and persists the latest report.
        """
        loop = asyncio.get_event_loop()
        def _run():
            return evaluation_engine.evaluate_meeting(
                meeting, db, performed_by="system", persist=True
            )
        return await loop.run_in_executor(None, _run)
