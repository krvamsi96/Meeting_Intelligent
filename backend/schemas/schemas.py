"""Pydantic schemas for API request/response models."""

import re
from datetime import datetime
from datetime import date
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# "No specific action items identified." / "No action items extracted." / "No action items." - what the report generator
# writes when it found none (llm_workflow); anything else is real content and is never matched.
_NO_ACTION_ITEMS_TEXT = re.compile(
    r"^\s*(?:[-*•]\s*)?no\s+(?:specific\s+|formal\s+|explicit\s+)?action\s+items?"
    r"(?:\s+(?:were\s+|was\s+)?(?:identified|extracted|recorded|found|detected))?\s*\.?\s*$",
    re.IGNORECASE,
)


class TranscriptSegmentBase(BaseModel):
    speaker: str                                 # final display label (voice-matched name, else "Speaker_N")
    speaker_id: Optional[str] = None             # stable per-meeting diarization id, e.g. "SPEAKER_00"
    speaker_name: Optional[str] = None           # registered name when voice-matched, else null (never guessed)
    speaker_confidence: Optional[float] = None   # voice-match cosine confidence [0..1] when a name was assigned
    start_time: str
    end_time: str
    text: str                                   # final, English
    raw_text: Optional[str] = None              # raw ASR (source language)
    original_text: Optional[str] = None         # corrected source-language text pre-translation
    stt_confidence: Optional[float] = None
    correction_applied: Optional[bool] = False
    correction_confidence: Optional[float] = None
    correction_source: Optional[str] = None
    correction_reason: Optional[str] = None
    correction_metadata: Optional[str] = None


class TranscriptSegment(TranscriptSegmentBase):
    id: int
    meeting_id: int

    model_config = ConfigDict(from_attributes=True)


class AnalysisResultBase(BaseModel):
    summary: Optional[str] = None
    mom: Optional[str] = None
    action_items: Optional[Any] = None
    structured_action_items: Optional[List[Any]] = None
    decisions: Optional[str] = None
    risks: Optional[str] = None

    @field_validator("action_items", mode="before")
    @classmethod
    def parse_action_items(cls, v):
        if not v:
            return []
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            v_str = v.strip()
            # The pipeline's own "nothing found" sentence ("No specific action items identified." / "No action items
            # extracted.") means an EMPTY list. Left as text, analysis.action_items - and so the stored final json_data - was
            # a string for exactly the meetings that have no action items, and a list for every other meeting.
            if _NO_ACTION_ITEMS_TEXT.match(v_str):
                return []
            if v_str.startswith("[") and v_str.endswith("]"):
                try:
                    import json
                    return json.loads(v_str)
                except Exception:
                    pass
            # If raw markdown string, dynamically parse into structured action items
            try:
                from backend.services.action_item_service import (
                    extract_action_items_extractive,
                    resolve_action_item_dates,
                )
                from backend.utils.date_resolver import get_reference_datetime
                extracted = extract_action_items_extractive(v_str)
                if extracted:
                    from backend.config.config import settings as _settings
                    ref_dt = get_reference_datetime(None, _settings.DEFAULT_TIMEZONE)
                    return resolve_action_item_dates(extracted, ref_dt)
            except Exception:
                pass
        return v


class AnalysisResult(AnalysisResultBase):
    id: int
    meeting_id: int

    model_config = ConfigDict(from_attributes=True)


class AgendaKeyDecisionItem(BaseModel):
    """A single Agenda item with its corresponding Decision and semantic Match Score."""
    id: Optional[Union[str, int]] = None
    agenda_id: Optional[Any] = None
    agenda_id_str: Optional[str] = None
    agenda_text: Optional[str] = None
    decision_text: Optional[str] = None

    title: Optional[str] = None
    agenda: Optional[str] = None
    source: Optional[str] = "llm_detected"
    key_decision: Optional[str] = None
    decision: Optional[str] = None
    decision_grounded: Optional[bool] = True
    agenda_order: Optional[int] = None

    match_score: Optional[float] = None
    relevance_score: Optional[float] = None
    confidence_level: Optional[str] = None
    model_name: Optional[str] = None
    processing_time: Optional[float] = None

    agenda_type: Optional[str] = "inferred"
    reason: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    evidence: Optional[str] = None
    speaker: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AgendaItemSchema(BaseModel):
    id: str
    title: str
    source: str = "llm_detected"


class AgendaListResponse(BaseModel):
    agenda: List[AgendaItemSchema]


class DecisionItemSchema(BaseModel):
    agenda_id: str
    decision: str
    decision_grounded: bool = True
    relevance_score: Optional[float] = None


class DecisionListResponse(BaseModel):
    decisions: List[DecisionItemSchema]


class MasterActionItemSchema(BaseModel):
    title: str = ""
    description: str = ""
    due_date: str = ""
    assigned_to: str = ""
    priority: str = ""


class MasterActionItemsResponse(BaseModel):
    action_items: List[MasterActionItemSchema]


class AgendaKeyDecisionsResponse(BaseModel):
    """Structured response for the Agenda module.

    Returns all Agenda-Decision pairs with real semantic Match Scores.
    """
    meeting_id: int
    bharatvc_meeting_id: Optional[str] = None
    agenda_key_decisions: List[AgendaKeyDecisionItem]
    agenda_items: List[AgendaKeyDecisionItem] = []
    agenda: List[AgendaItemSchema] = []
    decisions: List[DecisionItemSchema] = []
    total_agenda_items: int = 0
    average_match_score: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class AssigneeDetail(BaseModel):
    """Structured assignee metadata for an action item."""
    name: str
    speaker_id: Optional[str] = None
    confidence: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class ActionItemDetail(BaseModel):
    """A single structured Action Item extracted from a meeting transcript.

    Supports both the canonical Add New Task frontend form schema and existing
    backward-compatible API fields.
    """
    id: Optional[int] = None
    item_order: Optional[int] = None

    # Add New Task Form Canonical Fields
    title: Optional[str] = None
    description: Optional[str] = None
    # Parent / grouping task — NULL by default; populated only when explicitly provided.
    # The only correct default value is None (maps to JSON null and DB NULL).
    main_task: Optional[str] = None
    date_of_initiation: Optional[str] = None
    due_date: Optional[Any] = None
    original_deadline_text: Optional[str] = None
    deadline_start: Optional[Any] = None
    deadline_end: Optional[Any] = None
    deadline_precision: Optional[str] = None
    timezone: Optional[str] = "Asia/Kolkata"

    priority: Optional[str] = None
    priority_confidence: Optional[float] = None

    ministry: Optional[str] = None
    department: Optional[str] = None
    concerned_division: Optional[str] = None

    coordinator: Optional[str] = None
    assignees: List[AssigneeDetail] = Field(default_factory=list)

    apply_hierarchy_rules: bool = True
    category: Optional[str] = None
    dependencies: List[Any] = Field(default_factory=list)

    supporting_evidence: Optional[str] = None
    source_speaker: Optional[str] = None
    source_timestamp_start: Optional[Any] = None
    source_timestamp_end: Optional[Any] = None

    status: Optional[str] = "open"
    confidence_score: Optional[float] = None

    # Backward-compatible legacy aliases (synced with DB columns)
    task: Optional[str] = None
    owner: Optional[str] = "Unassigned"
    assigned_by: Optional[str] = None
    deadline: Optional[str] = "Not specified"
    deadline_type: Optional[str] = None
    evidence: Optional[str] = None
    speaker: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    confidence: Optional[float] = None
    source: Optional[str] = None
    agenda_ref: Optional[str] = None
    decision_ref: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)



class ActionItemsResponse(BaseModel):
    """Structured response for the Action Items module."""
    meeting_id: int
    bharatvc_meeting_id: Optional[str] = None
    action_items: List[ActionItemDetail] = []
    total_items: int = 0
    owners_identified: int = 0          # count of items with non-Unassigned owners
    deadlines_identified: int = 0       # count of items with real deadlines
    average_confidence: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class MeetingBase(BaseModel):
    filename: str


class MeetingCreate(MeetingBase):
    pass


class Meeting(MeetingBase):
    id: int
    bharatvc_meeting_id: Optional[str] = None
    meeting_id: Optional[str] = None
    display_id: Optional[str] = None
    file_type: str = "media"
    filepath: Optional[str] = None
    upload_time: datetime
    status: str
    error_message: Optional[str] = None
    title: Optional[str] = None
    # The meeting's CONFIRMED agenda text: what the uploader / document supplied (agenda_source "stored"), else - only
    # in the API response, never written back to the meeting row - the titles of the topics the speaker announced
    # ("announced"). Topics merely inferred from the discussion are not an agenda: ``agenda`` stays null and
    # ``agenda_source`` is "inferred_topics_only". null when there are no agenda items at all.
    agenda: Optional[str] = None
    agenda_source: Optional[str] = None
    meeting_date: Optional[str] = None
    meeting_time: Optional[str] = None
    matched_meeting_id: Optional[int] = None
    matched_meeting_code: Optional[str] = None
    similarity_score: Optional[float] = None
    matched_mom: Optional[str] = None
    related_meetings_json: Optional[str] = None

    @model_validator(mode="after")
    def _api_status_follows_processing_status(self) -> "Meeting":
        """``api_status`` must describe the PROCESSING outcome, never the column default "Success".

        Derived from ``status`` on every serialisation (ORM row or stored dict), so rows written before this rule
        - and any writer that forgot to update the columns - are still reported truthfully. The stored columns are
        not rewritten here; ``status`` / ``overall_status`` / ``error_message`` are returned unchanged beside them.
        """
        from backend.utils.processing_status import api_status_for

        self.api_status, self.api_status_code = api_status_for(self.status)
        return self

    @field_validator("agenda", mode="after")
    @classmethod
    def _no_generated_agenda(cls, value: Optional[str]) -> Optional[str]:
        """A placeholder the platform used to generate ("Discussion recording: ...") is not an agenda."""
        from backend.utils.helpers import clean_agenda_text

        return clean_agenda_text(value)
    is_approved: Optional[bool] = False
    accuracy: Optional[float] = None
    speaker_accuracy: Optional[float] = None
    transcription_accuracy: Optional[float] = None
    confidence_score: Optional[float] = None
    has_ground_truth: Optional[bool] = False
    processing_time: Optional[float] = 0.0
    audio_quality: Optional[float] = None
    video_quality: Optional[float] = None
    duration: Optional[float] = 0.0
    participant_count: Optional[int] = 0
    participants: Optional[str] = None
    interruptions: Optional[int] = 0
    silence_duration: Optional[float] = 0.0
    sentiment: Optional[str] = None
    emotions: Optional[str] = None
    questions_raised: Optional[str] = None
    keyword_frequency: Optional[str] = None

    # Real Runtime Benchmark Fields
    file_size_bytes: Optional[int] = 0
    upload_start_time: Optional[datetime] = None
    upload_end_time: Optional[datetime] = None
    upload_duration_sec: Optional[float] = 0.0
    processing_start_time: Optional[datetime] = None
    processing_end_time: Optional[datetime] = None
    total_processing_time_sec: Optional[float] = 0.0

    stt_generated: Optional[bool] = False
    stt_word_count: Optional[int] = 0
    stt_char_count: Optional[int] = 0
    stt_duration_sec: Optional[float] = 0.0

    # Multilingual pipeline (auto-detected; user never selects the language).
    # All fields above/below are rendered in English regardless of source_language.
    source_language: Optional[str] = None
    source_language_name: Optional[str] = None
    source_language_confidence: Optional[float] = None
    translation_applied: Optional[bool] = False
    output_language: Optional[str] = "en"
    translation_duration_sec: Optional[float] = 0.0

    speaker_count: Optional[int] = 0
    speaker_timeline_generated: Optional[bool] = False
    # HISTORICAL NAME: wall-clock seconds the diarization step took to run, not how long anyone spoke. The two fields
    # below say what is meant (additive; ``speaker_duration_sec`` is unchanged for existing clients). Per-speaker
    # talk time is in the final record's ``speaker_duration`` and runtime_metrics.speaker.speaker_distribution.
    speaker_duration_sec: Optional[float] = 0.0
    speaker_processing_time_sec: Optional[float] = None    # same value as speaker_duration_sec
    speaker_speech_duration_sec: Optional[float] = None    # total seconds of detected speech (None = not measured)
    diarization_status: Optional[str] = "SUCCESS"
    diarization_info_json: Optional[str] = None
    diarization_metadata_json: Optional[str] = None

    pipeline_stage_states_json: Optional[str] = None
    retry_history_json: Optional[str] = None

    summary_generated: Optional[bool] = False
    summary_word_count: Optional[int] = 0
    summary_duration_sec: Optional[float] = 0.0

    mom_generated: Optional[bool] = False
    mom_word_count: Optional[int] = 0
    mom_duration_sec: Optional[float] = 0.0

    action_items_generated: Optional[bool] = False
    action_items_count: Optional[int] = 0
    action_items_duration_sec: Optional[float] = 0.0

    topics_generated: Optional[bool] = False
    topics_count: Optional[int] = 0

    sentiment_label: Optional[str] = None
    sentiment_breakdown_json: Optional[str] = None

    db_meeting_saved: Optional[bool] = False
    db_transcript_saved: Optional[bool] = False
    db_summary_saved: Optional[bool] = False
    db_action_items_saved: Optional[bool] = False

    export_pdf_generated: Optional[bool] = False
    export_docx_generated: Optional[bool] = False
    export_json_generated: Optional[bool] = False

    api_status: Optional[str] = "Success"
    api_status_code: Optional[int] = 200
    overall_status: Optional[str] = "uploaded"

    segments: List[TranscriptSegment] = []
    analysis: Optional[AnalysisResult] = None
    action_items: List[ActionItemDetail] = []
    structured_action_items: List[ActionItemDetail] = []
    agenda_key_decisions: List[AgendaKeyDecisionItem] = []
    structured_agenda_decisions: List[AgendaKeyDecisionItem] = []

    # Reference-free evaluation (see backend.evaluation.reference_free_engine).
    # Both are assembled dynamically per-meeting in build_meeting_response_dict;
    # never persisted verbatim on the Meeting ORM row itself.
    runtime_metrics: Optional[Dict[str, Any]] = None
    evaluation_metrics: Optional[Dict[str, Any]] = None

    # Real-time reference-free headline scores (transcript quality, summary quality, RAG
    # groundedness) — see backend.evaluation.quality_scores. Same block in GET /meetings/{id},
    # GET /meetings/{id}/final and AI_Meeting_Inteligent_final.json_data.
    metrics: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class ApproveRequest(BaseModel):
    summary: Optional[str] = None
    mom: Optional[str] = None
    action_items: Optional[str] = None
    decisions: Optional[str] = None
    risks: Optional[str] = None
    agenda_key_decisions: Optional[List[AgendaKeyDecisionItem]] = None


class UrlRequest(BaseModel):
    url: str = Field(..., min_length=1, max_length=2000)
    bharatvc_meeting_id: Optional[str] = None
    meeting_id: Optional[str] = None

    @field_validator("url")
    def validate_url(cls, v):
        if not v.strip():
            raise ValueError("URL cannot be empty")
        return v.strip()


class ChatMessageCreate(BaseModel):
    session_id: Optional[str] = None
    message: str = Field(..., min_length=1, max_length=10000)
    cross_meeting: bool = False

    @field_validator("message")
    def message_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class ChatMessageResponse(BaseModel):
    id: Optional[int] = None
    meeting_id: Optional[str] = None
    session_id: Optional[str] = None
    role: str
    content: str
    timestamp: datetime
    sources: Optional[List[dict]] = None
    # Reference-free groundedness of THIS answer in the context retrieved for it (assistant messages only).
    rag_groundedness: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True)


class SimilarMeeting(BaseModel):
    meeting_id: int
    meeting_code: str
    filename: Optional[str] = None
    title: Optional[str] = None
    display_name: Optional[str] = None
    similarity_score: float = 0.0
    similarity_percentage: str = "0.0%"
    rank: int = 0
    file_type: Optional[str] = None
    duration: Optional[float] = None
    meeting_date: Optional[str] = None
    meeting_time: Optional[str] = None
    project_name: Optional[str] = None
    participants: Optional[str] = None
    accuracy: Optional[float] = None
    speaker_accuracy: Optional[float] = None
    transcription_accuracy: Optional[float] = None
    status: Optional[str] = None
    upload_time: Optional[str] = None
    matched_mom: Optional[str] = None
    match_explanations: Optional[dict] = None
    content_preview: Optional[str] = None


class SimilarMeetingsResponse(BaseModel):
    meeting_id: int
    total_similar_meetings: int
    average_similarity_score: float
    meetings: List[SimilarMeeting]


# ─── Intelligent Similar Meetings (redesigned) ──────────────────────────────
class IntelligentSimilarMeeting(BaseModel):
    """A single classified, human-explained similar meeting.

    Uses extra='allow' so the rich, evolving payload from the similarity
    service (signals, difference_analysis, timeline, duplicate_info, matching_*)
    passes through without brittle nested schemas.
    """

    model_config = ConfigDict(extra="allow")

    meeting_id: int
    meeting_code: str
    title: Optional[str] = None
    category: str
    category_label: str
    similarity_score: float = 0.0
    similarity_percentage: str = "0.0%"
    confidence_score: float = 0.0
    confidence_label: str = "Low"
    reason: str = ""
    rank: int = 0
    is_duplicate: bool = False
    matching_topics: List[str] = []
    matching_keywords: List[str] = []
    matching_participants: List[str] = []
    matching_decisions: List[str] = []
    matching_action_items: List[str] = []
    matching_documents: List[str] = []


class IntelligentSimilarMeetingsResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    meeting_id: Any
    meeting_code: Optional[str] = None
    total_similar_meetings: int
    average_similarity_score: float
    duplicate_detected: bool = False
    category_counts: Dict[str, int] = {}
    meetings: List[IntelligentSimilarMeeting]


class DuplicateCheckResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    meeting_id: Any
    is_duplicate: bool
    message: str
    duplicates: List[Dict[str, Any]] = []


class MetricsResponse(BaseModel):
    metrics: List[dict]


# ─── True Evaluation Engine Schemas ──────────────────────────────────────────

class GroundTruthUpload(BaseModel):
    ref_transcript: Optional[str] = None
    ref_speaker_labels: Optional[dict] = None          # {segment_idx: "Speaker Name"}
    ref_speaker_timeline: Optional[List[dict]] = None  # [{"speaker": "...", "start": 0.0, "end": 10.0}]
    ref_summary: Optional[str] = None
    ref_mom: Optional[str] = None
    ref_action_items: Optional[List[Any]] = None
    ref_topics: Optional[List[str]] = None
    ref_sentiment: Optional[str] = None
    ref_similarity_labels: Optional[dict] = None       # {meeting_id_or_code: 0/1 or grade}
    source: Optional[str] = "manual"
    notes: Optional[str] = None


class GroundTruthResponse(BaseModel):
    id: int
    meeting_id: int
    uploaded_by: Optional[str]
    version: int
    ref_transcript: Optional[str]
    ref_speaker_labels: Optional[str]
    ref_speaker_timeline: Optional[str]
    ref_summary: Optional[str]
    ref_mom: Optional[str]
    ref_action_items: Optional[str]
    ref_topics: Optional[str]
    ref_sentiment: Optional[str]
    ref_similarity_labels: Optional[str]
    source: str
    notes: Optional[str]
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EvaluationResultResponse(BaseModel):
    id: int
    meeting_id: int
    ground_truth_id: Optional[int]
    evaluator: str
    evaluation_timestamp: datetime
    evaluation_version: str

    # STT Accuracy
    wer: Optional[float] = None
    cer: Optional[float] = None
    stt_match_pct: Optional[float] = None
    stt_accuracy: Optional[float] = None
    stt_evaluated: bool

    # Speaker Diarization
    der: Optional[float] = None
    jer: Optional[float] = None
    diarization_accuracy: Optional[float] = None
    diarization_evaluated: bool

    # Speaker Identification
    speaker_accuracy: Optional[float] = None
    speaker_precision: Optional[float] = None
    speaker_recall: Optional[float] = None
    speaker_f1: Optional[float] = None
    speaker_confusion_matrix_json: Optional[str] = None
    speaker_evaluated: bool

    # Summary Accuracy
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougeL: Optional[float] = None
    bertscore_f1: Optional[float] = None
    summary_accuracy: Optional[float] = None
    summary_evaluated: bool

    # Minutes of Meeting Accuracy
    mom_coverage: Optional[float] = None
    mom_missing_count: Optional[int] = None
    mom_hallucination_count: Optional[int] = None
    mom_accuracy: Optional[float] = None
    mom_evaluated: bool

    # Action Items Accuracy
    action_precision: Optional[float] = None
    action_recall: Optional[float] = None
    action_f1: Optional[float] = None
    action_missing_count: Optional[int] = None
    action_false_count: Optional[int] = None
    action_accuracy: Optional[float] = None
    action_evaluated: bool

    # Topic Accuracy
    topic_accuracy: Optional[float] = None
    topic_precision: Optional[float] = None
    topic_recall: Optional[float] = None
    topic_f1: Optional[float] = None
    topic_evaluated: bool

    # Sentiment Accuracy
    sentiment_accuracy: Optional[float] = None
    sentiment_precision: Optional[float] = None
    sentiment_recall: Optional[float] = None
    sentiment_f1: Optional[float] = None
    sentiment_confusion_matrix_json: Optional[str] = None
    sentiment_evaluated: bool

    # Overall Combined True Accuracy
    overall_accuracy: Optional[float] = None
    raw_results_json: Optional[str] = None

    # New engine fields
    similarity_ndcg: Optional[float] = None
    overall_system_score: Optional[float] = None
    evaluation_type_code: Optional[str] = None
    evaluation_status: Optional[str] = None
    topics_evaluated: Optional[bool] = False
    similarity_evaluated: Optional[bool] = False
    audio_signal_evaluated: Optional[bool] = False
    video_signal_evaluated: Optional[bool] = False

    model_config = ConfigDict(from_attributes=True)


class EvaluationHistoryEntry(BaseModel):
    id: int
    meeting_id: int
    evaluation_result_id: Optional[int]
    action: str
    performed_by: str
    timestamp: datetime
    details_json: Optional[str]

    model_config = ConfigDict(from_attributes=True)


class MeetingEvaluationResponse(BaseModel):
    meeting_id: int
    speech_recognition_quality: Optional[float] = None
    speaker_consistency: Optional[float] = None
    summary_grounding: Optional[float] = None
    mom_quality: Optional[float] = None
    action_item_quality: Optional[float] = None
    topic_coverage: Optional[float] = None
    sentiment_confidence: Optional[float] = None
    similarity_confidence: Optional[float] = None
    audio_signal_quality: Optional[float] = None
    video_signal_quality: Optional[float] = None

    overall_ai_quality: Optional[float] = None
    processing_time: float = 0.0
    evaluation_timestamp: datetime
    evaluation_version: str = "4.0"

    model_config = ConfigDict(from_attributes=True)


class RuntimeMetricsResponse(BaseModel):
    # 1. STT
    stt_wer: Optional[float] = None
    stt_cer: Optional[float] = None
    speech_recognition_quality: Optional[float] = None

    # 2. Diarization
    diarization_der: Optional[float] = None
    diarization_jer: Optional[float] = None
    transcript_quality: Optional[float] = None

    # 3. Speaker Identification
    speaker_id_accuracy: Optional[float] = None
    speaker_id_f1: Optional[float] = None
    speaker_id_eer: Optional[float] = None
    speaker_consistency: Optional[float] = None

    # 4. Summary
    summary_rougel: Optional[float] = None
    summary_bertscore: Optional[float] = None
    summary_grounding: Optional[float] = None

    # 5. MoM
    mom_rougel: Optional[float] = None
    mom_bertscore: Optional[float] = None

    # 6. Action Items
    action_precision: Optional[float] = None
    action_recall: Optional[float] = None
    action_f1: Optional[float] = None

    # 7. Topic Detection
    topic_coherence: Optional[float] = None
    topic_f1: Optional[float] = None
    topic_coverage: Optional[float] = None

    # 8. Sentiment
    sentiment_accuracy: Optional[float] = None
    sentiment_f1: Optional[float] = None
    sentiment_mcc: Optional[float] = None

    # 9. Similarity Search
    similarity_recall_10: Optional[float] = None
    similarity_mrr: Optional[float] = None
    similarity_ndcg: Optional[float] = None

    # 10. Chatbot
    chatbot_faithfulness: Optional[float] = None
    chatbot_answer_relevance: Optional[float] = None
    hallucination_risk: Optional[float] = None

    # 11. Audio Telemetry
    audio_snr: Optional[float] = None
    audio_silence_ratio: Optional[float] = None
    audio_quality: Optional[float] = None

    # 12. Video Telemetry
    video_fps: Optional[float] = None
    video_resolution: Optional[str] = None
    video_bitrate: Optional[float] = None
    video_quality: Optional[float] = None

    # Infrastructure
    processing_time: Optional[float] = None
    database_status: str
    api_status: str
    # New engine fields
    evaluation_status: Optional[str] = None
    overall_accuracy: Optional[float] = None


class BenchmarkEvaluationResponse(BaseModel):
    status: str  # 'available' or 'unavailable'
    reason: Optional[str] = None  # 'no_reference' or None
    benchmark_id: Optional[int] = None
    meeting_id: int
    reference_version: Optional[str] = None
    evaluation_version: Optional[str] = None
    reference_available: bool = False
    
    stt_accuracy: Optional[float] = None
    wer: Optional[float] = None
    cer: Optional[float] = None

    speaker_accuracy: Optional[float] = None
    der: Optional[float] = None
    jer: Optional[float] = None
    speaker_id_accuracy: Optional[float] = None
    speaker_id_f1: Optional[float] = None
    speaker_id_eer: Optional[float] = None

    summary_accuracy: Optional[float] = None
    rouge1: Optional[float] = None
    rouge2: Optional[float] = None
    rougel: Optional[float] = None
    bertscore: Optional[float] = None

    mom_accuracy: Optional[float] = None
    mom_rougel: Optional[float] = None
    mom_bertscore: Optional[float] = None

    action_precision: Optional[float] = None
    action_recall: Optional[float] = None
    action_f1: Optional[float] = None

    topic_coherence: Optional[float] = None
    topic_f1: Optional[float] = None

    sentiment_accuracy: Optional[float] = None
    sentiment_f1: Optional[float] = None
    sentiment_mcc: Optional[float] = None

    similarity_recall_10: Optional[float] = None
    similarity_mrr: Optional[float] = None
    similarity_ndcg: Optional[float] = None

    chatbot_faithfulness: Optional[float] = None
    chatbot_answer_relevance: Optional[float] = None

    audio_snr: Optional[float] = None
    audio_silence_ratio: Optional[float] = None

    video_fps: Optional[float] = None
    video_resolution: Optional[str] = None
    video_bitrate: Optional[float] = None

    evaluation_time: Optional[datetime] = None
    evaluation_status: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class BenchmarkVersionResponse(BaseModel):
    reference_version: str
    evaluation_version: str
    engine_name: str
    last_updated: datetime


class BenchmarkHistoryResponse(BaseModel):
    id: int
    meeting_id: int
    action: str
    performed_by: str
    timestamp: datetime
    details_json: Optional[str]

    model_config = ConfigDict(from_attributes=True)


# ─── Universal Evaluation Engine Report Schema ───────────────────────────────

class TaskEvaluation(BaseModel):
    status: str  # 'measured' | 'not_measured'
    reason: Optional[str] = None
    primary_metric: Optional[str] = None
    metrics: Dict[str, Any] = {}
    quality_score: Optional[float] = None


class OverallEvaluation(BaseModel):
    status: str
    score: Optional[float] = None
    reason: Optional[str] = None
    components: Optional[Dict[str, Any]] = None


class EvaluationReportResponse(BaseModel):
    engine_name: str
    engine_version: str
    meeting_id: Any
    bharatvc_meeting_id: Optional[str] = None
    has_reference: bool = False
    tasks: Dict[str, TaskEvaluation]
    overall: Optional[OverallEvaluation] = None
    telemetry: Dict[str, Any] = {}
    evaluated_at: str


# ─── Live Metrics Canonical Card Schemas ─────────────────────────────────────

class LiveMetricDisplayItem(BaseModel):
    """A single label/value pair shown inside a metric card."""
    label: str
    value: Optional[Any] = None
    unit: Optional[str] = None
    source: Optional[str] = None


class LiveMetricCard(BaseModel):
    """One of the canonical evaluation metric cards.

    ``status`` is the only field always present.  All numeric values are
    ``None`` when not computable — never 0 or a fabricated estimate.
    """
    metric_name: str              # Display name, e.g. "Speech Recognition"
    metric_key: str               # Engine key, e.g. "stt"
    status: str                   # "measured" | "not_measured"
    primary_metric_label: Optional[str] = None    # e.g. "WER" or "Transcript Coverage"
    primary_metric_value: Optional[float] = None  # actual numeric result
    primary_metric_unit: Optional[str] = None     # e.g. "%", "dB", "s"
    display_items: List[LiveMetricDisplayItem] = []
    source: Optional[str] = None                  # provenance tag
    evaluation_type: Optional[str] = None         # "GROUND_TRUTH" | "REFERENCE_FREE" | "RUNTIME_PERFORMANCE"
    sample_count: Optional[int] = None            # number of items used in calculation
    execution_time_sec: Optional[float] = None    # wall-clock time for this task
    reason: Optional[str] = None                  # why not_measured
    notes: Optional[str] = None                   # human-readable clarification


class ProcessingPerformance(BaseModel):
    """Processing timing breakdown from real runtime execution logs."""
    upload_time_sec: Optional[float] = None
    stt_time_sec: Optional[float] = None
    speaker_time_sec: Optional[float] = None
    summary_time_sec: Optional[float] = None
    mom_time_sec: Optional[float] = None
    action_items_time_sec: Optional[float] = None
    total_processing_time_sec: Optional[float] = None
    api_time_sec: Optional[float] = None
    db_time_sec: Optional[float] = None
    file_size_bytes: Optional[int] = None
    video_duration_sec: Optional[float] = None


class LiveEvaluationResponse(BaseModel):
    """Full live evaluation response: 10 metric cards + processing performance.

    Every value originates from backend execution.  The frontend must
    display each card as-is: show ``primary_metric_value`` when present,
    or show a 'Not Measured' label when ``status == 'not_measured'``.
    Zero fabrication is permitted at any layer.
    """
    meeting_id: int
    evaluated_at: str
    engine_version: str
    has_reference: bool
    metrics: List[LiveMetricCard]
    processing_performance: ProcessingPerformance
    overall_score: Optional[float] = None
    overall_status: Optional[str] = None
    overall_reason: Optional[str] = None


# ─── NVIDIA NeMo Speaker Diarization API Schemas ────────────────────────────

class DiarizationSpeakerInfo(BaseModel):
    speaker_id: str
    speaking_duration_seconds: float
    speaking_percentage: float
    turn_count: Optional[int] = None
    avg_turn_duration_sec: Optional[float] = None
    first_appearance: Optional[float] = None
    last_appearance: Optional[float] = None


class DiarizationResponse(BaseModel):
    status: str = "SUCCESS"  # SUCCESS, FAILED, DEGRADED
    engine: str = "NVIDIA_NEMO"
    speaker_count: int = 0
    speakers: List[DiarizationSpeakerInfo] = []
    error_message: Optional[str] = None


# ─── Runtime Quality Metrics Response Schemas ─────────────────────────────────

class SpeakerDistributionItem(BaseModel):
    """Per-speaker runtime stats from actual diarization timestamps (REFERENCE_FREE)."""
    speaker: str
    duration_seconds: float
    percentage: float
    turn_count: int
    average_turn_sec: float
    max_turn_sec: Optional[float] = None
    min_turn_sec: Optional[float] = None
    first_appearance_sec: Optional[float] = None
    last_appearance_sec: Optional[float] = None


class SttRuntimeQualityMetrics(BaseModel):
    """Reference-free STT metrics from actual pipeline segment data."""
    evaluation_type: str = "REFERENCE_FREE"
    total_segments: Optional[int] = None
    transcript_coverage: Optional[float] = None
    empty_segment_rate: Optional[float] = None
    timestamp_validity_rate: Optional[float] = None
    repetition_rate: Optional[float] = None
    correction_rate: Optional[float] = None
    avg_stt_confidence: Optional[float] = None
    confidence_sample_count: Optional[int] = None
    rtf: Optional[float] = None
    stt_runtime_quality_score: Optional[float] = None
    weights: Optional[Dict[str, float]] = None


class SpeakerRuntimeQualityMetrics(BaseModel):
    """Reference-free speaker diarization metrics from actual NeMo output timestamps."""
    evaluation_type: str = "REFERENCE_FREE"
    detected_speakers: Optional[int] = None
    speaker_coverage: Optional[float] = None
    speaker_assignment_rate: Optional[float] = None
    turn_count: Optional[int] = None
    avg_turn_duration_sec: Optional[float] = None
    timestamp_consistency_rate: Optional[float] = None
    unassigned_duration_sec: Optional[float] = None
    micro_segment_count: Optional[int] = None
    speaker_runtime_quality_score: Optional[float] = None
    speaker_distribution: List[SpeakerDistributionItem] = []
    weights: Optional[Dict[str, float]] = None


class ContentRuntimeQualityMetrics(BaseModel):
    """Reference-free content quality metrics."""
    evaluation_type: str = "REFERENCE_FREE"
    # Summary
    summary_word_overlap_ratio: Optional[float] = None
    unsupported_sentence_rate: Optional[float] = None
    summary_word_count: Optional[int] = None
    summary_runtime_quality_score: Optional[float] = None
    # MoM
    mom_word_count: Optional[int] = None
    mom_section_count: Optional[int] = None
    # Agenda / Decisions
    agenda_item_count: Optional[int] = None
    decisions_with_evidence_pct: Optional[float] = None
    avg_decision_match_score: Optional[float] = None
    unsupported_decision_pct: Optional[float] = None


class ActionItemRuntimeQualityMetrics(BaseModel):
    """Reference-free action item completeness metrics."""
    evaluation_type: str = "REFERENCE_FREE"
    action_item_count: Optional[int] = None
    field_completeness_pct: Optional[float] = None
    evidence_coverage_pct: Optional[float] = None
    assigned_to_coverage_pct: Optional[float] = None
    due_date_coverage_pct: Optional[float] = None
    priority_valid_pct: Optional[float] = None
    duplicate_rate_pct: Optional[float] = None
    action_item_runtime_quality_score: Optional[float] = None


class ProcessingPerformanceRuntimeMetrics(BaseModel):
    """Stage-level pipeline timing from real execution columns."""
    evaluation_type: str = "RUNTIME_PERFORMANCE"
    audio_duration_sec: Optional[float] = None
    stt_duration_sec: Optional[float] = None
    speaker_duration_sec: Optional[float] = None
    summary_duration_sec: Optional[float] = None
    mom_duration_sec: Optional[float] = None
    action_items_duration_sec: Optional[float] = None
    total_processing_time_sec: Optional[float] = None
    rtf: Optional[float] = None
    stt_word_count: Optional[int] = None
    speaker_count: Optional[int] = None


class RuntimeQualityMetricsResponse(BaseModel):
    """Full reference-free runtime quality metrics response.

    All values derived from actual pipeline execution.
    None means not enough data; nothing is fabricated or hardcoded.
    evaluation_type is always REFERENCE_FREE or RUNTIME_PERFORMANCE.
    """
    meeting_id: int
    evaluation_type: str = "REFERENCE_FREE"
    evaluated_at: str
    stt: SttRuntimeQualityMetrics
    speaker: SpeakerRuntimeQualityMetrics
    content: ContentRuntimeQualityMetrics
    action_items: ActionItemRuntimeQualityMetrics
    performance: ProcessingPerformanceRuntimeMetrics


# ─────────────────────────────────────────────────────────────────────────────
# Activity Logs Schemas
# ─────────────────────────────────────────────────────────────────────────────

class ActivityLogEntry(BaseModel):
    """Single pipeline step log record from AI_Meeting_Activty_Logs."""

    id: int
    meeting_pk: int
    bharatvc_meeting_id: Optional[str] = None
    meeting_id: Optional[str] = None
    processing_id: str
    step_name: str
    step_order: int
    feature: Optional[str] = None
    feature_order: Optional[int] = None
    status: str
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    duration: Optional[float] = None
    progress_percent: Optional[float] = None
    details: Optional[Dict[str, Any]] = None   # parsed from details_json
    metadata: Optional[Dict[str, Any]] = None  # alias for details
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    error: Optional[str] = None                # alias for error_message
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class ProcessingRunSummary(BaseModel):
    """Summary of one processing_id run (header shown in dropdown)."""

    processing_id: str
    overall_status: str           # COMPLETED | PROCESSING | FAILED | PARTIAL
    step_count: int
    completed_steps: int
    failed_steps: int
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    current_step: Optional[str] = None
    progress_percent: float = 0.0


class ActivityLogsResponse(BaseModel):
    """Full response for GET /meetings/{id}/activity-logs."""

    meeting_pk: int
    bharatvc_meeting_id: Optional[str] = None
    meeting_id: Optional[str] = None
    processing_id: str
    overall_status: str
    current_step: Optional[str] = None
    progress_percent: float = 0.0
    total_duration_seconds: Optional[float] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    logs: List[ActivityLogEntry] = Field(default_factory=list)
    processing_runs: List[ProcessingRunSummary] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# AI_Meeting_Inteligent_final — Consolidated Response Schemas
# ─────────────────────────────────────────────────────────────────────────────

class FinalTranscriptSegment(BaseModel):
    """One utterance from the meeting transcript. ``text`` is always English; ``original_text`` is the same utterance
    in the language it was spoken (absent for an English recording)."""
    speaker: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    text: str = ""
    confidence: Optional[float] = None
    original_text: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FinalSpeaker(BaseModel):
    """Speaker summary derived from transcript segments."""
    speaker_id: str
    label: str
    segment_count: int = 0
    total_words: int = 0

    model_config = ConfigDict(from_attributes=True)


class FinalSpeakerDuration(BaseModel):
    """Per-speaker duration computed from actual segment boundaries."""
    speaker_id: str
    duration_seconds: float
    percentage: float
    segment_count: int = 0

    model_config = ConfigDict(from_attributes=True)


class FinalAgendaItem(BaseModel):
    """A single agenda topic."""
    agenda_id: str
    title: str = ""
    type: Optional[str] = "inferred"
    order: int = 1
    source: Optional[str] = "llm_detected"

    model_config = ConfigDict(from_attributes=True)


class FinalDecision(BaseModel):
    """A grounded decision associated with an agenda topic."""
    agenda_id: str
    decision: str
    reason: Optional[str] = None
    evidence: Optional[str] = None
    speaker: Optional[str] = None
    timestamp: Optional[str] = None
    confidence: Optional[float] = None
    match_score: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class FinalAgendaDecisionMapping(BaseModel):
    """Semantic mapping of an agenda topic to its decision (or lack thereof)."""
    agenda_id: str
    agenda_title: str = ""
    decision: str = ""
    has_decision: bool = False
    # ``match_score`` is the cosine similarity (0-100) of the agenda title and decision TEXT: a supporting signal,
    # not proof of the mapping and not validated against labelled data. What supports the mapping is ``evidence``,
    # the transcript quote the decision was extracted from (verified to occur in the transcript).
    match_score: Optional[float] = None
    confidence_level: Optional[str] = None
    is_grounded: bool = False
    evidence: Optional[str] = None
    speaker: Optional[str] = None
    timestamp: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FinalActionItem(BaseModel):
    """A validated, deduplicated action item from the meeting."""
    item_order: Optional[int] = None
    task: str
    description: Optional[str] = None
    owner: Optional[str] = None
    deadline: Optional[str] = None
    deadline_type: Optional[str] = None
    due_date: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = "open"
    evidence: Optional[str] = None
    speaker: Optional[str] = None
    timestamp_start: Optional[str] = None
    timestamp_end: Optional[str] = None
    confidence: Optional[float] = None
    agenda_ref: Optional[str] = None
    decision_ref: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FinalSpeakerSentiment(BaseModel):
    """Per-speaker sentiment entry."""
    speaker_id: str
    sentiment: Optional[str] = None
    confidence: Optional[float] = None
    # emotions may arrive as a raw label string ("Positive: 60%, Neutral: 40%")
    # or as a normalized breakdown dict ({"Positive": 60.0, "Neutral": 40.0}).
    emotions: Optional[Union[str, Dict[str, Any]]] = None

    model_config = ConfigDict(from_attributes=True)


class FinalSentiment(BaseModel):
    """Meeting sentiment output."""
    label: Optional[str] = None
    emotions: Optional[Union[str, Dict[str, Any]]] = None
    speaker_breakdown: List[FinalSpeakerSentiment] = Field(default_factory=list)
    methodology: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FinalMetrics(BaseModel):
    """Consolidated metrics with clear source distinction."""
    reference_based: Optional[Dict[str, Any]] = None   # null when no ground truth
    reference_free: Optional[Dict[str, Any]] = None
    quality_scores: Optional[Dict[str, Any]] = None
    technical: Optional[Dict[str, Any]] = None

    # Real-time reference-free headline scores — identical block to GET /meetings/{id}.metrics.
    evaluation_type: Optional[str] = None
    ground_truth_available: Optional[bool] = None
    transcript_quality: Optional[Dict[str, Any]] = None
    summary_quality: Optional[Dict[str, Any]] = None
    rag_groundedness: Optional[Dict[str, Any]] = None
    method_version: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FinalActivityLogEntry(BaseModel):
    """One important pipeline step in the final activity log."""
    step_name: str
    step_order: int
    status: str
    message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_seconds: Optional[float] = None

    model_config = ConfigDict(from_attributes=True)


class FinalMeetingMetadata(BaseModel):
    """Core meeting metadata scalars."""
    meeting_pk: int
    meeting_id: Optional[str] = None
    bharatvc_meeting_id: Optional[str] = None
    display_id: Optional[str] = None
    title: Optional[str] = None
    meeting_date: Optional[str] = None
    meeting_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    duration_formatted: Optional[str] = None
    status: Optional[str] = None
    participant_count: Optional[int] = None
    participants: Optional[str] = None
    speaker_count: Optional[int] = None
    # The spoken language is detected automatically; every generated output is English (``output_language``) and the
    # source-language transcript is kept alongside as ``transcript[].original_text``.
    source_language: Optional[str] = None
    source_language_name: Optional[str] = None
    source_language_confidence: Optional[float] = None
    translation_applied: Optional[bool] = None
    output_language: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class FinalMeetingResponse(BaseModel):
    """Consolidated final meeting response — primary data source for Angular meeting detail.

    One call replaces individual GET calls for transcript, speakers, agenda,
    decisions, action items, summary, mom, sentiment, metrics, and activity logs.

    processing_status values:
        PENDING   — aggregation not yet started (meeting may still be processing)
        BUILDING  — aggregation in progress
        COMPLETED — all mandatory features present; data is ready to display
        FAILED    — aggregation failed; check error_info for details
    """
    processing_status: str
    aggregation_version: Optional[str] = None
    error_info: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    meeting: FinalMeetingMetadata
    transcript: List[FinalTranscriptSegment] = Field(default_factory=list)
    speakers: List[FinalSpeaker] = Field(default_factory=list)
    speaker_duration: List[FinalSpeakerDuration] = Field(default_factory=list)
    agenda: List[FinalAgendaItem] = Field(default_factory=list)
    decisions: List[FinalDecision] = Field(default_factory=list)
    agenda_decision_mapping: List[FinalAgendaDecisionMapping] = Field(default_factory=list)
    action_items: List[FinalActionItem] = Field(default_factory=list)
    summary: Optional[str] = None
    mom: Optional[str] = None
    sentiment: Optional[FinalSentiment] = None
    metrics: Optional[FinalMetrics] = None
    activity_logs: List[FinalActivityLogEntry] = Field(default_factory=list)
    # The meeting's CURRENT state (completed | processing | failed), which can differ from this stored record's
    # ``processing_status`` while a re-run is in progress or after it failed; ``user_error`` is the plain-language
    # reason when it failed.
    display_status: Optional[str] = None
    user_error: Optional[Dict[str, str]] = None
    # Plain-language notes when parts of the run finished with reduced quality (e.g. speakers may be inaccurate).
    warnings: List[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

