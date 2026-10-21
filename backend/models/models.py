"""SQLAlchemy ORM models for the Meeting Intelligence Platform."""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship

from backend.config.config import settings

Base = declarative_base()


def _utcnow() -> datetime:
    """Timezone-aware UTC datetime factory (replaces deprecated datetime.utcnow)."""
    return datetime.now(timezone.utc)

class Meeting(Base):
    __tablename__ = "meetings"

    id = Column(Integer, primary_key=True, index=True)
    bharatvc_meeting_id = Column(String, unique=True, index=True, nullable=True)
    meeting_id = Column(String, unique=True, index=True, nullable=True)
    display_id = Column(String, index=True, nullable=True)
    filename = Column(String, index=True)
    file_type = Column(String, default="media")
    filepath = Column(String, nullable=True)
    upload_time = Column(DateTime, default=_utcnow)
    status = Column(String, default="uploaded")
    error_message = Column(Text, nullable=True)

    # Meeting Metadata
    title = Column(String, nullable=True)
    agenda = Column(Text, nullable=True)
    meeting_date = Column(String, nullable=True)
    meeting_time = Column(String, nullable=True)
    project_name = Column(
        String, nullable=True, default=lambda: settings.APP_NAME
    )

    # Semantic Similarity
    matched_meeting_id = Column(Integer, nullable=True)
    matched_meeting_code = Column(String, nullable=True)
    similarity_score = Column(Float, nullable=True)
    matched_mom = Column(Text, nullable=True)
    related_meetings_json = Column(Text, nullable=True)
    content_fingerprint = Column(String, nullable=True, index=True)

    # Approval
    is_approved = Column(Boolean, default=False)

    # Analytics & Metrics
    # NOTE: All quality/accuracy columns are nullable=True, default=None.
    # A value of None means "not yet evaluated" — NEVER 0.0 unless genuinely computed.
    accuracy = Column(Float, nullable=True, default=None)
    speaker_accuracy = Column(Float, nullable=True, default=None)
    transcription_accuracy = Column(Float, nullable=True, default=None)
    confidence_score = Column(Float, nullable=True, default=None)
    processing_time = Column(Float, default=0.0)  # runtime timer — 0.0 is valid
    audio_quality = Column(Float, nullable=True, default=None)
    video_quality = Column(Float, nullable=True, default=None)
    duration = Column(Float, default=0.0)
    participant_count = Column(Integer, default=0)
    participants = Column(Text, nullable=True)
    interruptions = Column(Integer, default=0)
    silence_duration = Column(Float, default=0.0)
    sentiment = Column(String, nullable=True)
    emotions = Column(Text, nullable=True)
    questions_raised = Column(Text, nullable=True)
    keyword_frequency = Column(Text, nullable=True)
    # Detailed per-metric quality scores — None until engine computes a real value
    summary_quality = Column(Float, nullable=True, default=None)
    mom_quality = Column(Float, nullable=True, default=None)
    action_item_quality = Column(Float, nullable=True, default=None)
    topic_quality = Column(Float, nullable=True, default=None)
    # Full weighted metrics breakdown (JSON)
    metrics_json = Column(Text, nullable=True)

    # Real Runtime Benchmark & Lifecycle Execution Metrics
    file_size_bytes = Column(Integer, default=0)
    upload_start_time = Column(DateTime, nullable=True)
    upload_end_time = Column(DateTime, nullable=True)
    upload_duration_sec = Column(Float, default=0.0)
    processing_start_time = Column(DateTime, nullable=True)
    processing_end_time = Column(DateTime, nullable=True)
    total_processing_time_sec = Column(Float, default=0.0)

    stt_generated = Column(Boolean, default=False)
    stt_word_count = Column(Integer, default=0)
    stt_char_count = Column(Integer, default=0)
    stt_duration_sec = Column(Float, default=0.0)

    # ── Multilingual pipeline (auto-detected; user never selects) ─────────────
    # Spoken language of the uploaded media. ALL user-facing output is produced
    # in `output_language` (default "en") regardless of this value.
    source_language = Column(String, nullable=True)            # ISO code, e.g. "te", "hi", "en"
    source_language_name = Column(String, nullable=True)       # e.g. "Telugu"
    source_language_confidence = Column(Float, nullable=True)  # 0..1
    translation_applied = Column(Boolean, default=False)       # True when native→English ran
    output_language = Column(String, default="en")
    translation_duration_sec = Column(Float, default=0.0)

    # Optional caller-supplied hint for the ACTUAL number of speakers in the
    # recording. None (the default) means "auto-detect" — diarization always
    # estimates the count dynamically either way; this only lets a caller who
    # already knows the true count short-circuit that estimation.
    num_speakers = Column(Integer, nullable=True)
    speaker_count = Column(Integer, default=0)
    speaker_timeline_generated = Column(Boolean, default=False)
    speaker_duration_sec = Column(Float, default=0.0)
    diarization_status = Column(String, default="SUCCESS")  # SUCCESS, FAILED, DEGRADED
    diarization_info_json = Column(Text, nullable=True)
    diarization_metadata_json = Column(Text, nullable=True)

    # Dynamic Universal Loop Engineering State & Retry Audit Tracking
    pipeline_stage_states_json = Column(Text, nullable=True)
    retry_history_json = Column(Text, nullable=True)

    summary_generated = Column(Boolean, default=False)
    summary_word_count = Column(Integer, default=0)
    summary_duration_sec = Column(Float, default=0.0)

    mom_generated = Column(Boolean, default=False)
    mom_word_count = Column(Integer, default=0)
    mom_duration_sec = Column(Float, default=0.0)

    action_items_generated = Column(Boolean, default=False)
    action_items_count = Column(Integer, default=0)
    action_items_duration_sec = Column(Float, default=0.0)

    topics_generated = Column(Boolean, default=False)
    topics_count = Column(Integer, default=0)

    sentiment_label = Column(String, nullable=True)
    sentiment_breakdown_json = Column(Text, nullable=True)

    db_meeting_saved = Column(Boolean, default=False)
    db_transcript_saved = Column(Boolean, default=False)
    db_summary_saved = Column(Boolean, default=False)
    db_action_items_saved = Column(Boolean, default=False)

    export_pdf_generated = Column(Boolean, default=False)
    export_docx_generated = Column(Boolean, default=False)
    export_json_generated = Column(Boolean, default=False)

    api_status = Column(String, default="Success")
    api_status_code = Column(Integer, default=200)
    overall_status = Column(String, default="uploaded")

    segments = relationship(
        "TranscriptSegment", back_populates="meeting", cascade="all, delete-orphan"
    )
    analysis = relationship(
        "AnalysisResult",
        uselist=False,
        back_populates="meeting",
        cascade="all, delete-orphan",
    )
    agenda_key_decisions = relationship(
        "AgendaKeyDecision",
        back_populates="meeting",
        cascade="all, delete-orphan",
        order_by="AgendaKeyDecision.agenda_order",
    )
    action_items = relationship(
        "ActionItem",
        back_populates="meeting",
        cascade="all, delete-orphan",
        order_by="ActionItem.item_order",
    )
    ground_truth = relationship(
        "GroundTruth",
        uselist=False,
        back_populates="meeting",
        cascade="all, delete-orphan",
    )
    evaluation_result = relationship(
        "EvaluationResult",
        uselist=False,
        back_populates="meeting",
        cascade="all, delete-orphan",
    )
    benchmark_evaluation = relationship(
        "BenchmarkEvaluation",
        uselist=False,
        back_populates="meeting",
        cascade="all, delete-orphan",
    )
    evaluation = relationship(
        "MeetingEvaluation",
        uselist=False,
        back_populates="meeting",
        cascade="all, delete-orphan",
    )
    chat_sessions = relationship(
        "ChatSession",
        back_populates="meeting",
        cascade="all, delete-orphan",
    )
    evaluation_history = relationship(
        "EvaluationHistory",
        back_populates="meeting",
        cascade="all, delete-orphan",
    )
    evaluation_metric_records = relationship(
        "EvaluationMetricRecord",
        back_populates="meeting",
        cascade="all, delete-orphan",
    )
    activity_logs = relationship(
        "MeetingActivityLog",
        back_populates="meeting",
        cascade="all, delete-orphan",
        order_by="MeetingActivityLog.step_order, MeetingActivityLog.created_at",
    )


    @property
    def has_ground_truth(self) -> bool:
        """Whether a ground-truth reference exists. Ground truth is OPTIONAL, so this never raises.

        Serialising a meeting (pydantic ``from_attributes``) reads this property; a plain ``self.ground_truth`` is a
        lazy load that ran inside whatever transaction the session was in - if that transaction was already aborted
        the load raised ``InFailedSqlTransaction`` out of response building. The lookup now runs in a savepoint and a
        failed lookup answers False (``backend.utils.db_safe``).
        """
        from sqlalchemy import inspect as _sa_inspect
        from sqlalchemy.orm import object_session as _object_session

        state = _sa_inspect(self)
        if "ground_truth" not in state.unloaded:  # already in memory - no query at all
            return self.ground_truth is not None
        session = _object_session(self)
        if session is None or self.id is None:  # detached / transient: cannot query, so there is nothing to find
            return False
        from backend.utils.db_safe import has_ground_truth as _has_ground_truth

        return _has_ground_truth(session, self.id)

    def __repr__(self) -> str:
        return (
            f"<Meeting id={self.id} bharatvc_meeting_id={self.bharatvc_meeting_id!r} "
            f"status={self.status!r}>"
        )


# Composite indexes for common query patterns
Index("ix_meetings_status_upload_time", Meeting.status, Meeting.upload_time)
Index("ix_meetings_is_approved_status", Meeting.is_approved, Meeting.status)


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), index=True)
    speaker = Column(String)                                 # Final display label (voice-matched name, else "Speaker_N")
    speaker_id = Column(String, nullable=True)               # Stable per-meeting diarization id, e.g. "SPEAKER_00"
    speaker_name = Column(String, nullable=True)             # Voice-matched registered name, else NULL (never guessed)
    speaker_confidence = Column(Float, nullable=True)        # Voice-match cosine confidence [0..1] when a name was assigned
    start_time = Column(String)
    end_time = Column(String)
    text = Column(Text)                                      # Final, English, corrected human-readable text
    raw_text = Column(Text, nullable=True)                  # Immutable raw ASR text output (source language)
    original_text = Column(Text, nullable=True)             # Corrected SOURCE-language text before English translation
    stt_confidence = Column(Float, nullable=True)           # ASR logprob/confidence signal
    correction_applied = Column(Boolean, default=False)     # True if STT or grammar correction altered text
    correction_confidence = Column(Float, nullable=True)    # Correction confidence score [0.0, 1.0]
    correction_source = Column(String, nullable=True)       # e.g. "phonetic_contextual_llm", "grammar_normalization"
    correction_reason = Column(String, nullable=True)       # Explanation e.g. "contextual terminology correction"
    correction_metadata = Column(Text, nullable=True)     # JSON array string of edits made

    meeting = relationship("Meeting", back_populates="segments")


Index(
    "ix_transcript_segments_meeting_id_id",
    TranscriptSegment.meeting_id,
    TranscriptSegment.id,
)


class VoiceProfile(Base):
    """A registered participant's voice: name + a speaker embedding derived from
    one or more reference samples. Used to put real names on diarized speakers.

    Nothing here is hard-coded — every profile is created via the enrollment
    API/CLI from an uploaded voice sample, and the embedding is produced by the
    same speaker embedder (TitaNet) the diarizer uses, so meeting-time matching
    is apples-to-apples.
    """

    __tablename__ = "voice_profiles"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, index=True)            # Human display name, e.g. "Raj"
    speaker_key = Column(String, unique=True, index=True)        # Stable slug of the name, e.g. "raj"
    embedding_json = Column(Text, nullable=False)                # JSON list[float] — L2-normalised mean embedding
    embedding_dim = Column(Integer, default=0)
    engine = Column(String, default="titanet_large")            # Embedder that produced the vector
    sample_count = Column(Integer, default=0)                    # Number of reference samples averaged in
    sample_paths_json = Column(Text, nullable=True)             # JSON list[str] of stored reference sample paths
    total_speech_sec = Column(Float, default=0.0)               # Cumulative enrolled speech seconds
    enrolled_by = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class AnalysisResult(Base):
    __tablename__ = "analysis_results"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), unique=True, index=True)
    summary = Column(Text)
    mom = Column(Text)
    action_items = Column(Text)
    decisions = Column(Text)
    risks = Column(Text)

    meeting = relationship("Meeting", back_populates="analysis")


class AgendaKeyDecision(Base):
    """Structured Agenda & Decisions for a meeting.

    Maintains the chronological order of discussion agenda topics and their
    associated decisions, plus a real semantic Match Score computed by the
    Agenda Service using sentence-transformer cosine similarity.

    ``match_score`` is always a real model-computed value in [0, 100] — never
    random, never hardcoded.  ``None`` means not yet computed.
    """

    __tablename__ = "agenda_key_decisions"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), index=True, nullable=False)
    bharatvc_meeting_id = Column(String, index=True, nullable=True)
    agenda_order = Column(Integer, nullable=False, default=1)
    agenda = Column(Text, nullable=False)          # agenda_text (display alias)
    agenda_type = Column(String, nullable=True, default="inferred")  # "explicit" or "inferred"
    key_decision = Column(Text, nullable=False)     # decision_text (display alias)
    reason = Column(Text, nullable=True, default=None)  # rationale/reason for decision

    # Transcript grounding — direct quote from the transcript that supports the decision.
    # None = not extracted (e.g. extractive fallback path).
    evidence = Column(Text, nullable=True, default=None)
    # Speaker who made / proposed the decision (if identifiable from the transcript).
    speaker = Column(String, nullable=True, default=None)
    timestamp = Column(String, nullable=True, default=None)  # Timestamp [HH:MM:SS]
    confidence = Column(Float, nullable=True, default=None)   # Decision extraction confidence

    agenda_id_str = Column(String, nullable=True, default=None)  # e.g. "A1", "A2"
    agenda_source = Column(String, nullable=True, default="llm_detected")  # "user" or "llm_detected"
    decision_grounded = Column(Boolean, default=True)

    # Semantic Match Score / Relevance Score — computed by agenda_service.compute_agenda_decision_match_score()
    # None = not yet computed; never 0.0 unless genuinely computed as zero.
    match_score = Column(Float, nullable=True, default=None)
    relevance_score = Column(Float, nullable=True, default=None)
    embedding_model = Column(String, nullable=True, default=None)

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    meeting = relationship("Meeting", back_populates="agenda_key_decisions")


Index(
    "ix_agenda_decisions_meeting_id_order",
    AgendaKeyDecision.meeting_id,
    AgendaKeyDecision.agenda_order,
)


class ActionItem(Base):
    """Structured Action Item extracted from a meeting transcript.

    Each row represents a single, deduplicated, validated action item.
    All fields are populated dynamically from the transcript — never hardcoded.

    ``confidence`` is computed by action_item_service based on task/owner/deadline
    clarity and transcript grounding.  None = not yet computed.
    """

    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), index=True, nullable=False)
    bharatvc_meeting_id = Column(String, index=True, nullable=True)

    # Ordering
    item_order = Column(Integer, nullable=False, default=1)

    # Core task fields
    task = Column(Text, nullable=False)                    # What needs to be done
    description = Column(Text, nullable=True)              # Optional longer context
    main_task = Column(Text, nullable=True, default=None)  # Top-level task group / parent task (NULL = not assigned; never hardcoded)
    owner = Column(String, nullable=True, default="Unassigned")  # Who is responsible
    assigned_by = Column(String, nullable=True)            # Who assigned it


    # Deadline — original natural-language phrase
    deadline = Column(String, nullable=True, default="Not specified")  # Original phrase (original_deadline_text)
    deadline_type = Column(String, nullable=True)          # absolute/relative/not_specified

    # Resolved deadline fields — populated by date_resolver, never by the LLM (§9.16)
    due_date = Column(Date, nullable=True, default=None)           # Machine-readable resolved date
    deadline_start = Column(Date, nullable=True, default=None)     # Start of date window
    deadline_end = Column(Date, nullable=True, default=None)       # End of date window
    deadline_precision = Column(String, nullable=True, default=None)  # exact/window/unresolvable

    # Status & Priority (extracted from transcript, never assumed)
    priority = Column(String, nullable=True, default="Not specified")  # high/medium/low/not_specified
    status = Column(String, nullable=True, default="open")  # open/in_progress/completed/blocked/unknown

    # Transcript grounding
    evidence = Column(Text, nullable=True)                 # Exact transcript quote
    speaker = Column(String, nullable=True)                # Speaker label from transcript
    timestamp_start = Column(String, nullable=True)        # Transcript timestamp
    timestamp_end = Column(String, nullable=True)

    # Quality & provenance
    confidence = Column(Float, nullable=True, default=None)  # Dynamic 0–100, never hardcoded
    source = Column(String, nullable=True, default="llm")    # "llm" or "extractive"

    # Relationships to other entities
    agenda_ref = Column(String, nullable=True)             # Linked agenda topic title
    decision_ref = Column(String, nullable=True)           # Linked decision text

    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    meeting = relationship("Meeting", back_populates="action_items")


Index(
    "ix_action_items_meeting_id_order",
    ActionItem.meeting_id,
    ActionItem.item_order,
)


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True, nullable=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), nullable=False, index=True)
    role = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    sources = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=_utcnow)

    meeting = relationship("Meeting", back_populates="chat_sessions")


Index(
    "ix_chat_sessions_meeting_session_time",
    ChatSession.meeting_id,
    ChatSession.session_id,
    ChatSession.timestamp,
)


class RagEvaluation(Base):
    """Reference-free RAG groundedness of ONE chatbot answer, scored when the answer was generated.

    The score needs the context the LLM actually saw (vector chunks + MoM/action-item/metadata tool
    output), which is only known at chat time and is not persisted with the message — so the result
    is stored here. The meeting-level ``metrics.rag_groundedness`` is the mean over this meeting's
    rows. A separate table (rather than a column on ``chat_sessions``) keeps every existing chat query
    untouched on databases that have not been migrated yet. Rows die with their chat message / meeting.
    """

    __tablename__ = "rag_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id", ondelete="CASCADE"), nullable=False, index=True)
    chat_message_id = Column(
        Integer, ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, unique=True, index=True
    )
    score = Column(Float, nullable=True)
    label = Column(String, nullable=True)
    result_json = Column(Text, nullable=False)   # the full per-answer result: components, issues, signals
    created_at = Column(DateTime, default=_utcnow)


class AIProviderConfig(Base):
    __tablename__ = "ai_provider_configs"

    provider = Column(String, primary_key=True, index=True)  # 'airawat'
    enabled = Column(Boolean, default=False)
    encrypted_config = Column(Text, nullable=True)  # encrypted JSON string
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)


class GroundTruth(Base):
    __tablename__ = "ground_truth"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), unique=True, index=True, nullable=False)
    uploaded_by = Column(String, nullable=True, default="system")
    version = Column(Integer, default=1)

    # Reference Data
    ref_transcript = Column(Text, nullable=True)
    ref_speaker_labels = Column(Text, nullable=True)     # JSON string
    ref_speaker_timeline = Column(Text, nullable=True)   # JSON string
    ref_summary = Column(Text, nullable=True)
    ref_mom = Column(Text, nullable=True)
    ref_action_items = Column(Text, nullable=True)       # JSON string or text list
    ref_topics = Column(Text, nullable=True)             # JSON string or comma list
    ref_sentiment = Column(Text, nullable=True)          # JSON string or label
    ref_similarity_labels = Column(Text, nullable=True)  # JSON string {meeting_id/code: 0/1 or grade}

    source = Column(String, default="manual")
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    meeting = relationship("Meeting", back_populates="ground_truth")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), unique=True, index=True, nullable=False)
    ground_truth_id = Column(Integer, ForeignKey("ground_truth.id"), index=True, nullable=True)
    evaluator = Column(String, default="True Evaluation Engine v2.0")
    evaluation_timestamp = Column(DateTime, default=_utcnow)
    evaluation_version = Column(String, default="2.0")

    # STT Accuracy
    wer = Column(Float, nullable=True)
    cer = Column(Float, nullable=True)
    stt_match_pct = Column(Float, nullable=True)
    stt_accuracy = Column(Float, nullable=True)
    stt_evaluated = Column(Boolean, default=False)

    # Speaker Diarization
    der = Column(Float, nullable=True)
    jer = Column(Float, nullable=True)
    diarization_accuracy = Column(Float, nullable=True)
    diarization_evaluated = Column(Boolean, default=False)

    # Speaker Identification
    speaker_accuracy = Column(Float, nullable=True)
    speaker_precision = Column(Float, nullable=True)
    speaker_recall = Column(Float, nullable=True)
    speaker_f1 = Column(Float, nullable=True)
    speaker_confusion_matrix_json = Column(Text, nullable=True)
    speaker_evaluated = Column(Boolean, default=False)

    # Summary Accuracy
    rouge1 = Column(Float, nullable=True)
    rouge2 = Column(Float, nullable=True)
    rougeL = Column(Float, nullable=True)
    bertscore_f1 = Column(Float, nullable=True)
    summary_accuracy = Column(Float, nullable=True)
    summary_evaluated = Column(Boolean, default=False)

    # Minutes of Meeting Accuracy
    mom_coverage = Column(Float, nullable=True)
    mom_missing_count = Column(Integer, nullable=True)
    mom_hallucination_count = Column(Integer, nullable=True)
    mom_accuracy = Column(Float, nullable=True)
    mom_evaluated = Column(Boolean, default=False)

    # Action Items Accuracy
    action_precision = Column(Float, nullable=True)
    action_recall = Column(Float, nullable=True)
    action_f1 = Column(Float, nullable=True)
    action_missing_count = Column(Integer, nullable=True)
    action_false_count = Column(Integer, nullable=True)
    action_accuracy = Column(Float, nullable=True)
    action_evaluated = Column(Boolean, default=False)

    # Topic Accuracy
    topic_accuracy = Column(Float, nullable=True)
    topic_precision = Column(Float, nullable=True)
    topic_recall = Column(Float, nullable=True)
    topic_f1 = Column(Float, nullable=True)
    topic_evaluated = Column(Boolean, default=False)

    # Sentiment Accuracy
    sentiment_accuracy = Column(Float, nullable=True)
    sentiment_precision = Column(Float, nullable=True)
    sentiment_recall = Column(Float, nullable=True)
    sentiment_f1 = Column(Float, nullable=True)
    sentiment_confusion_matrix_json = Column(Text, nullable=True)
    sentiment_evaluated = Column(Boolean, default=False)

    # Overall Combined True Accuracy (mean of evaluated metrics)
    overall_accuracy = Column(Float, nullable=True)
    raw_results_json = Column(Text, nullable=True)

    # New engine fields
    similarity_ndcg = Column(Float, nullable=True)
    overall_system_score = Column(Float, nullable=True)
    evaluation_type_code = Column(String, nullable=True)
    evaluation_status = Column(String, default="completed")
    topics_evaluated = Column(Boolean, default=False)
    similarity_evaluated = Column(Boolean, default=False)
    audio_signal_evaluated = Column(Boolean, default=False)
    video_signal_evaluated = Column(Boolean, default=False)

    meeting = relationship("Meeting", back_populates="evaluation_result")
    ground_truth = relationship("GroundTruth")


class EvaluationHistory(Base):
    __tablename__ = "evaluation_history"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), index=True, nullable=False)
    evaluation_result_id = Column(Integer, ForeignKey("evaluation_results.id"), nullable=True)
    action = Column(String, nullable=False)  # 'gt_uploaded', 'evaluated', 're_evaluated', 'gt_updated'
    performed_by = Column(String, default="system")
    timestamp = Column(DateTime, default=_utcnow)
    details_json = Column(Text, nullable=True)

    meeting = relationship("Meeting", back_populates="evaluation_history")


class BenchmarkEvaluation(Base):
    __tablename__ = "benchmark_evaluations"

    benchmark_id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), unique=True, index=True, nullable=False)
    reference_version = Column(String, nullable=True, default="1.0")
    evaluation_version = Column(String, nullable=True, default="2.0")
    reference_available = Column(Boolean, default=False)

    # STT Accuracy
    stt_accuracy = Column(Float, nullable=True)
    wer = Column(Float, nullable=True)
    cer = Column(Float, nullable=True)

    # Speaker Diarization / Identification
    speaker_accuracy = Column(Float, nullable=True)
    der = Column(Float, nullable=True)
    jer = Column(Float, nullable=True)
    speaker_id_accuracy = Column(Float, nullable=True)
    speaker_id_f1 = Column(Float, nullable=True)
    speaker_id_eer = Column(Float, nullable=True)

    # Summary Accuracy
    summary_accuracy = Column(Float, nullable=True)
    rouge1 = Column(Float, nullable=True)
    rouge2 = Column(Float, nullable=True)
    rougel = Column(Float, nullable=True)
    bertscore = Column(Float, nullable=True)

    # MOM Accuracy
    mom_accuracy = Column(Float, nullable=True)
    mom_rougel = Column(Float, nullable=True)
    mom_bertscore = Column(Float, nullable=True)

    # Action Items Metrics
    action_precision = Column(Float, nullable=True)
    action_recall = Column(Float, nullable=True)
    action_f1 = Column(Float, nullable=True)

    # Topic Detection Metrics
    topic_coherence = Column(Float, nullable=True)
    topic_f1 = Column(Float, nullable=True)

    # Sentiment Analysis Metrics
    sentiment_accuracy = Column(Float, nullable=True)
    sentiment_f1 = Column(Float, nullable=True)
    sentiment_mcc = Column(Float, nullable=True)

    # Similarity Search Metrics
    similarity_recall_10 = Column(Float, nullable=True)
    similarity_mrr = Column(Float, nullable=True)
    similarity_ndcg = Column(Float, nullable=True)

    # Chatbot AI Metrics
    chatbot_faithfulness = Column(Float, nullable=True)
    chatbot_answer_relevance = Column(Float, nullable=True)

    # Audio Telemetry
    audio_snr = Column(Float, nullable=True)
    audio_silence_ratio = Column(Float, nullable=True)

    # Video Telemetry
    video_fps = Column(Float, nullable=True)
    video_resolution = Column(String, nullable=True)
    video_bitrate = Column(Float, nullable=True)

    evaluation_time = Column(DateTime, default=_utcnow)
    evaluation_status = Column(String, default="completed")

    meeting = relationship("Meeting", back_populates="benchmark_evaluation")


class MeetingEvaluation(Base):
    __tablename__ = "meeting_evaluations"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), unique=True, index=True, nullable=False)

    speech_recognition_quality = Column(Float, nullable=True)
    speaker_consistency = Column(Float, nullable=True)
    summary_grounding = Column(Float, nullable=True)
    mom_quality = Column(Float, nullable=True)
    action_item_quality = Column(Float, nullable=True)
    topic_coverage = Column(Float, nullable=True)
    sentiment_confidence = Column(Float, nullable=True)
    similarity_confidence = Column(Float, nullable=True)
    audio_signal_quality = Column(Float, nullable=True)
    video_signal_quality = Column(Float, nullable=True)

    overall_ai_quality = Column(Float, nullable=True)
    processing_time = Column(Float, nullable=True)
    evaluation_timestamp = Column(DateTime, default=_utcnow)
    evaluation_version = Column(String, default="3.0")

    meeting = relationship("Meeting", back_populates="evaluation")


class EvaluationMetricRecord(Base):
    """Per-metric evaluation record.

    Every metric computed by the evaluation engine is logged here with its
    source, value, execution time, and status.  All values are from real
    execution — None means not computed, never fabricated.
    """

    __tablename__ = "evaluation_metric_records"

    id = Column(Integer, primary_key=True, index=True)
    meeting_id = Column(Integer, ForeignKey("meetings.id"), index=True, nullable=False)
    evaluation_result_id = Column(
        Integer, ForeignKey("evaluation_results.id"), index=True, nullable=True
    )

    # Metric identity
    metric_name = Column(String, nullable=False)      # e.g. "Speech Recognition"
    metric_key = Column(String, nullable=False)        # e.g. "stt"
    metric_value = Column(Float, nullable=True)        # actual computed value
    metric_unit = Column(String, nullable=True)        # "%", "dB", "fps", …

    # Provenance
    metric_source = Column(String, nullable=True)      # "whisper_output", "pyannote_output", …
    model_version = Column(String, nullable=True)      # model tag used
    execution_time_sec = Column(Float, nullable=True)  # wall-clock time for this metric

    # Status
    status = Column(String, nullable=False, default="not_measured")  # "measured" | "not_measured"
    reason = Column(String, nullable=True)             # why not_measured, if applicable
    primary_metric_key = Column(String, nullable=True) # which sub-key is the headline metric

    # Versioning
    evaluation_version = Column(String, nullable=True)
    created_at = Column(DateTime, default=_utcnow)

    meeting = relationship("Meeting", back_populates="evaluation_metric_records")


# ─── Meeting Activity Log ──────────────────────────────────────────────────────

class MeetingActivityLog(Base):
    """Tracks every step of the AI processing pipeline for a meeting.

    Table name: AI_Meeting_Activty_Logs
    (Exact spelling per project requirement — 'Activty' is intentional.)

    Identity fields
    ---------------
    meeting_pk          : FK → meetings.id  (int; authoritative join key)
    bharatvc_meeting_id : Denormalized string copy of meetings.bharatvc_meeting_id
    meeting_id          : Denormalized string copy of meetings.meeting_id
                          Always equals bharatvc_meeting_id for the same row.
    processing_id       : Identifies one pipeline execution run.
                          Format: PROC-{meeting_pk}-{uuid4[:8]}
                          Multiple runs of the same meeting have different processing_ids.

    Status values
    -------------
    PENDING | PROCESSING | COMPLETED | FAILED | RETRYING | SKIPPED | CANCELLED
    """

    __tablename__ = "AI_Meeting_Activty_Logs"

    id = Column(Integer, primary_key=True, index=True)

    # ── Meeting identity (three complementary references) ──────────────────────
    meeting_pk = Column(
        Integer, ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    bharatvc_meeting_id = Column(String, nullable=True, index=True)
    meeting_id = Column(String, nullable=True, index=True)

    # ── Processing execution identity ──────────────────────────────────────────
    processing_id = Column(String, nullable=False, index=True)

    # ── Step metadata ──────────────────────────────────────────────────────────
    step_name = Column(String, nullable=False)
    step_order = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="PENDING")
    message = Column(Text, nullable=True)

    # ── Timing ─────────────────────────────────────────────────────────────────
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    progress_percent = Column(Float, nullable=True, default=0.0)

    # ── Step details (structured JSON — step-specific, never contains secrets) ─
    details_json = Column(Text, nullable=True)

    # ── Error tracking ─────────────────────────────────────────────────────────
    error_code = Column(String, nullable=True)
    error_message = Column(Text, nullable=True)

    # ── Audit timestamps ───────────────────────────────────────────────────────
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    meeting = relationship("Meeting", back_populates="activity_logs")

    @property
    def feature(self) -> str:
        return self.step_name

    @feature.setter
    def feature(self, val: str) -> None:
        self.step_name = val

    @property
    def feature_order(self) -> int:
        return self.step_order

    @feature_order.setter
    def feature_order(self, val: int) -> None:
        self.step_order = val

    def __repr__(self) -> str:
        return (
            f"<MeetingActivityLog id={self.id} processing_id={self.processing_id!r} "
            f"step={self.step_name!r} status={self.status!r}>"
        )


# ── Composite indexes for activity log queries ─────────────────────────────────
Index(
    "ix_activity_logs_meeting_proc",
    MeetingActivityLog.meeting_pk,
    MeetingActivityLog.processing_id,
)
Index(
    "ix_activity_logs_bvc_id_status",
    MeetingActivityLog.bharatvc_meeting_id,
    MeetingActivityLog.status,
)
Index(
    "ix_activity_logs_created",
    MeetingActivityLog.created_at,
)


class TranTask(Base):
    """ORM Model representing transaction task management entities (tran_task)."""
    __tablename__ = "tran_task"

    id = Column(Integer, primary_key=True, index=True)
    task_title = Column(Text, nullable=True)
    target_date = Column(Date, nullable=True, index=True)
    col_3 = Column(Text, nullable=True)
    remarks = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    reference_uuid = Column(String(100), nullable=True, index=True)
    is_active = Column(Boolean, default=True, index=True)
    created_at = Column(DateTime, nullable=True)
    ext_task_id = Column(Integer, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    col_11 = Column(Text, nullable=True)
    ministry_id = Column(Integer, nullable=True, index=True)
    section_id = Column(Integer, nullable=True)
    task_type_id = Column(Integer, nullable=True)
    parent_task_id = Column(Integer, nullable=True)
    hierarchy_id = Column(Integer, nullable=True)
    priority_id = Column(Integer, nullable=True)
    assigned_to = Column(Integer, nullable=True, index=True)
    assigned_by = Column(Integer, nullable=True, index=True)
    is_closed = Column(Boolean, default=False, index=True)
    status_id = Column(Integer, nullable=True)
    attachment_filename = Column(Text, nullable=True)
    description = Column(Text, nullable=True)
    source_ref_id = Column(Integer, nullable=True)
    reference_date = Column(Date, nullable=True)
    initiator_id = Column(Integer, nullable=True)
    priority_level = Column(Integer, nullable=True)
    division_id = Column(Integer, nullable=True)
    is_escalated = Column(Boolean, default=False)
    reminder_days = Column(Integer, nullable=True)
    location_id = Column(Integer, nullable=True)
    col_32 = Column(Text, nullable=True)
    attachment_guid = Column(String(100), nullable=True)
    attachment_doc_name = Column(Text, nullable=True)
    source_type = Column(Integer, nullable=True)
    forward_dept_id = Column(Integer, nullable=True)
    forward_officer_id = Column(Integer, nullable=True)
    col_38 = Column(Text, nullable=True)
    is_visible = Column(Boolean, default=True)
    file_number = Column(String(100), nullable=True)
    subject_category_id = Column(Integer, nullable=True)
    reminder_count = Column(Integer, nullable=True)
    escalation_level = Column(Integer, nullable=True)
    col_44 = Column(Text, nullable=True)
    col_45 = Column(Text, nullable=True)
    revised_date = Column(Date, nullable=True)
    is_valid = Column(Boolean, default=True)
    is_synced = Column(Boolean, default=True)

    def __repr__(self) -> str:
        return f"<TranTask id={self.id} title={self.task_title!r} target_date={self.target_date}>"


# ── AI Meeting Intelligence Final Consolidated Table ──────────────────────────

class AIMeetingIntelFinal(Base):
    """Final consolidated read-model for a processed meeting.

    Table name: AI_Meeting_Inteligent_final
    (Exact spelling per project requirement — 'Inteligent' is intentional.)

    This table aggregates validated outputs from ALL feature tables into a
    single row per meeting. It is populated AFTER the processing pipeline
    completes and is the primary data source for the Angular frontend's
    meeting detail view.

    Design rules
    ------------
    - ONE row per meeting (UNIQUE constraint on meeting_pk).
    - All JSON columns store validated, structured data — never raw LLM output.
    - processing_status drives the frontend loading state:
        PENDING   → aggregation not yet started
        BUILDING  → aggregation in progress
        COMPLETED → all mandatory features present and valid
        FAILED    → aggregation encountered a non-recoverable error
    - Scalar columns (meeting_date, duration_seconds, etc.) use native SQL types
      for efficient filtering/sorting.
    - Nested / list data (transcript, speakers, agenda, etc.) uses Text (JSON
      array / object) because the cardinality is dynamic per meeting.

    Source table mapping
    --------------------
    meetings               → meeting metadata, sentiment, speaker counts
    transcript_segments    → transcript, speakers, speaker_duration
    agenda_key_decisions   → agenda, decisions, agenda_decision_mapping
    action_items           → action_items
    analysis_results       → summary, mom
    AI_Meeting_Activty_Logs → activity_logs (filtered, ordered)
    meeting_evaluations    → metrics (reference-free quality scores)
    evaluation_results     → metrics (reference-based, if ground truth exists)
    benchmark_evaluations  → metrics (technical benchmarks)
    """

    __tablename__ = "AI_Meeting_Inteligent_final"

    # ── Primary Key ────────────────────────────────────────────────────────────
    id = Column(Integer, primary_key=True, index=True)

    # ── Meeting identity (three complementary references) ─────────────────────
    # meeting_pk is the authoritative FK to meetings.id.
    # UNIQUE ensures exactly one final record per meeting.
    meeting_pk = Column(
        Integer,
        ForeignKey("meetings.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    # Denormalized string IDs for fast lookups without a JOIN
    meeting_id = Column(String, nullable=True, index=True)          # meetings.meeting_id
    bharatvc_meeting_id = Column(String, nullable=True, index=True) # meetings.bharatvc_meeting_id
    display_id = Column(String, nullable=True, index=True)          # meetings.display_id

    # ── Meeting metadata (scalar — sortable/filterable) ────────────────────────
    title = Column(String, nullable=True)
    meeting_date = Column(String, nullable=True)         # YYYY-MM-DD or formatted
    meeting_time = Column(String, nullable=True)         # HH:MM formatted
    duration_seconds = Column(Float, nullable=True)      # media/audio duration in seconds
    duration_formatted = Column(String, nullable=True)   # human-readable "1h 23m 45s"
    status = Column(String, nullable=True)               # pipeline status from meetings table
    participant_count = Column(Integer, nullable=True)   # total participants
    participants = Column(Text, nullable=True)           # comma-separated names
    speaker_count = Column(Integer, nullable=True)       # diarized unique speaker count

    # ── Aggregation lifecycle ─────────────────────────────────────────────────
    # Tracks the state of the final aggregation process itself (not the pipeline).
    processing_status = Column(
        String, nullable=False, default="PENDING"
    )  # PENDING | BUILDING | COMPLETED | FAILED
    aggregation_version = Column(
        String, nullable=True, default="1.0"
    )  # version of aggregation logic

    # ── Feature outputs (JSON columns for structured nested data) ─────────────

    # Transcript: JSON array of utterance objects
    # [{speaker, start_time, end_time, text, confidence}]
    transcript = Column(Text, nullable=True)

    # Speakers: JSON array of speaker summary objects
    # [{speaker_id, label, segment_count, total_words}]
    speakers = Column(Text, nullable=True)

    # Speaker duration: JSON array per speaker
    # [{speaker_id, duration_seconds, percentage, segment_count}]
    speaker_duration = Column(Text, nullable=True)

    # Agenda: JSON array of agenda topic objects
    # [{agenda_id, title, type, order, source}]
    agenda = Column(Text, nullable=True)

    # Decisions: JSON array of decision objects
    # [{agenda_id, decision, reason, evidence, speaker, timestamp, confidence, match_score}]
    decisions = Column(Text, nullable=True)

    # Agenda → Decision mapping: JSON array of mapping objects
    # [{agenda_id, agenda_title, decision, match_score, confidence_level, is_grounded}]
    agenda_decision_mapping = Column(Text, nullable=True)

    # Action Items: JSON array of validated action item objects
    # [{task, description, owner, deadline, priority, status, evidence, speaker,
    #   timestamp_start, timestamp_end, confidence, agenda_ref, decision_ref}]
    action_items = Column(Text, nullable=True)

    # Summary: plain text executive summary
    summary = Column(Text, nullable=True)

    # MOM: markdown text of Minutes of Meeting
    mom = Column(Text, nullable=True)

    # Sentiment: JSON object
    # {label, emotions, speaker_breakdown: [{speaker_id, sentiment, confidence}]}
    sentiment = Column(Text, nullable=True)

    # Metrics: JSON object with clearly separated sections
    # {reference_based: {...}, reference_free: {...}, technical: {...}, quality_scores: {...}}
    metrics = Column(Text, nullable=True)

    # Activity logs: JSON array of important feature-level pipeline steps
    # Filtered to step_order <= 18, ordered by step_order then started_at
    # [{step_name, step_order, status, message, started_at, completed_at, duration_seconds}]
    activity_logs = Column(Text, nullable=True)

    # ── Error tracking ─────────────────────────────────────────────────────────
    # JSON {stage, message, traceback_hint} — populated when processing_status=FAILED
    # Never silently empty on failure.
    error_info = Column(Text, nullable=True)

    # ── Consolidated JSON API Response (Complete Final Output) ────────────────
    # Stores the exact, validated GET /api/v1/meetings/{meeting_id} response
    json_data = Column(JSON().with_variant(JSONB, "postgresql"), nullable=True)

    # ── Audit timestamps ───────────────────────────────────────────────────────
    created_at = Column(DateTime, default=_utcnow)
    updated_at = Column(DateTime, default=_utcnow, onupdate=_utcnow)

    # ── Relationship ──────────────────────────────────────────────────────────
    meeting = relationship("Meeting", foreign_keys=[meeting_pk])

    def __repr__(self) -> str:
        return (
            f"<AIMeetingIntelFinal id={self.id} meeting_pk={self.meeting_pk} "
            f"processing_status={self.processing_status!r}>"
        )


# Composite indexes for common final table query patterns
Index(
    "ix_final_meeting_status",
    AIMeetingIntelFinal.meeting_pk,
    AIMeetingIntelFinal.processing_status,
)
Index(
    "ix_final_bvc_id",
    AIMeetingIntelFinal.bharatvc_meeting_id,
)
Index(
    "ix_final_meeting_id_str",
    AIMeetingIntelFinal.meeting_id,
)
Index(
    "ix_final_display_id",
    AIMeetingIntelFinal.display_id,
)

