"""AI_Meeting_Inteligent_final.json_data keeps ONE structure for every meeting: the reference format.

The reference is the stored ``GET /api/v1/meetings/{id}`` object (flat ``schemas.Meeting`` shape). Newer features may ADD keys
(translation, voice identity, metrics) - generated data is never removed - but every reference key must be present with the
reference's kind of value, ``json_data`` must be a real JSON object (never a JSON string) and the lists must be lists.
These tests use the same rolled-back database session as test_transaction_discipline.
"""

import json

import pytest

from backend.tests.test_transaction_discipline import _make_meeting, db_session, fake_embedder  # noqa: F401  (fixtures)

REFERENCE_TOP = set("""id title agenda status accuracy analysis duration emotions filename filepath segments file_type sentiment api_status
display_id meeting_id is_approved matched_mom upload_time action_items meeting_date meeting_time participants topics_count
audio_quality error_message interruptions mom_generated speaker_count stt_generated video_quality mom_word_count overall_status
stt_char_count stt_word_count api_status_code file_size_bytes processing_time sentiment_label upload_end_time confidence_score
db_meeting_saved db_summary_saved has_ground_truth mom_duration_sec questions_raised silence_duration similarity_score
speaker_accuracy stt_duration_sec topics_generated keyword_frequency participant_count summary_generated upload_start_time
action_items_count diarization_status matched_meeting_id retry_history_json summary_word_count bharatvc_meeting_id
db_transcript_saved processing_end_time upload_duration_sec agenda_key_decisions export_pdf_generated matched_meeting_code
speaker_duration_sec summary_duration_sec db_action_items_saved diarization_info_json export_docx_generated export_json_generated
processing_start_time related_meetings_json action_items_generated transcription_accuracy structured_action_items
sentiment_breakdown_json action_items_duration_sec diarization_metadata_json total_processing_time_sec pipeline_stage_states_json
speaker_timeline_generated structured_agenda_decisions""".split())

REFERENCE_ANALYSIS = set("id mom risks summary decisions meeting_id action_items structured_action_items".split())
REFERENCE_SEGMENT = set("""id text speaker end_time raw_text meeting_id start_time stt_confidence correction_reason correction_source
correction_applied correction_metadata correction_confidence""".split())
REFERENCE_AGENDA_ITEM = set("""id title agenda reason source speaker decision evidence agenda_id timestamp confidence model_name
agenda_text agenda_type match_score agenda_order key_decision agenda_id_str decision_text processing_time relevance_score
confidence_level decision_grounded""".split())
REFERENCE_ACTION_ITEM = set("""id task owner title source status speaker category deadline due_date evidence ministry priority timezone
assignees main_task agenda_ref confidence department item_order assigned_by coordinator description deadline_end decision_ref
dependencies deadline_type timestamp_end deadline_start source_speaker timestamp_start confidence_score concerned_division
date_of_initiation deadline_precision priority_confidence supporting_evidence source_timestamp_end apply_hierarchy_rules
source_timestamp_start original_deadline_text""".split())

# The reference stores these as JSON *text* (raw *_json columns of the meeting row) on purpose.
JSON_TEXT_FIELDS = ["diarization_info_json", "diarization_metadata_json", "pipeline_stage_states_json", "retry_history_json",
                    "related_meetings_json", "sentiment_breakdown_json", "questions_raised", "keyword_frequency", "emotions"]
LIST_FIELDS = ["segments", "action_items", "structured_action_items", "agenda_key_decisions", "structured_agenda_decisions"]


def _with_agenda_and_action_item(db, meeting_pk, no_action_items_text=None):
    from datetime import date

    from backend.models.models import ActionItem, AgendaKeyDecision, AnalysisResult

    db.add(AgendaKeyDecision(meeting_id=meeting_pk, agenda_order=1, agenda="Release date", agenda_id_str="A1",
                             key_decision="Release moves to March", decision_grounded=True))
    db.add(ActionItem(meeting_id=meeting_pk, item_order=1, task="Share the migration plan", owner="Priya",
                      deadline="by Friday", due_date=date(2026, 10, 2)))
    if no_action_items_text is not None:
        db.query(AnalysisResult).filter(AnalysisResult.meeting_id == meeting_pk).update({"action_items": no_action_items_text})
    db.commit()


def _build(db, meeting_pk):
    from backend.services.final_aggregation_service import build_final_record

    return build_final_record(meeting_pk, db).json_data


def test_json_data_is_a_json_object_with_every_reference_key(db_session, fake_embedder):
    meeting_pk = _make_meeting(db_session)
    _with_agenda_and_action_item(db_session, meeting_pk)
    data = _build(db_session, meeting_pk)

    assert isinstance(data, dict), "json_data must be a JSON object, not a JSON string"
    assert REFERENCE_TOP <= set(data), sorted(REFERENCE_TOP - set(data))
    json.dumps(data)                                                       # plain JSON, no objects that need a custom encoder
    assert json.loads(json.dumps(data)) == data                            # and it survives a round trip unchanged


def test_json_data_nested_structures_follow_the_reference(db_session, fake_embedder):
    meeting_pk = _make_meeting(db_session)
    _with_agenda_and_action_item(db_session, meeting_pk)
    data = _build(db_session, meeting_pk)

    assert isinstance(data["analysis"], dict) and REFERENCE_ANALYSIS <= set(data["analysis"])
    for field in LIST_FIELDS:
        assert isinstance(data[field], list), field
    assert data["segments"] and REFERENCE_SEGMENT <= set(data["segments"][0])
    assert data["agenda_key_decisions"] and REFERENCE_AGENDA_ITEM <= set(data["agenda_key_decisions"][0])
    assert data["structured_agenda_decisions"] and REFERENCE_AGENDA_ITEM <= set(data["structured_agenda_decisions"][0])
    assert data["action_items"] and REFERENCE_ACTION_ITEM <= set(data["action_items"][0])
    assert isinstance(data["analysis"]["action_items"], list)
    assert isinstance(data["analysis"]["structured_action_items"], list)


def test_the_raw_json_columns_stay_json_text_exactly_like_the_reference(db_session, fake_embedder):
    meeting_pk = _make_meeting(db_session)
    data = _build(db_session, meeting_pk)
    for field in JSON_TEXT_FIELDS:
        assert data[field] is None or isinstance(data[field], str), field


@pytest.mark.parametrize("nothing_found", ["No specific action items identified.", "No action items extracted.", "No action items.", ""])
def test_a_meeting_without_action_items_still_has_a_list(db_session, fake_embedder, nothing_found):
    """analysis.action_items was the sentence itself for exactly the meetings that have none (a list for all others)."""
    meeting_pk = _make_meeting(db_session)
    _with_agenda_and_action_item(db_session, meeting_pk, no_action_items_text=nothing_found)
    data = _build(db_session, meeting_pk)
    assert isinstance(data["analysis"]["action_items"], list)


def test_the_no_action_items_sentence_is_the_only_text_turned_into_an_empty_list():
    from backend.schemas import schemas

    def parsed(text):
        return schemas.AnalysisResult(id=1, meeting_id=1, action_items=text).action_items

    for nothing in ("No specific action items identified.", "No action items extracted.", "  no action items were identified  ",
                    "- No action items."):
        assert parsed(nothing) == [], nothing
    assert parsed(None) == [] and parsed("") == []
    listed = [{"task": "Share the plan"}]
    assert parsed(listed) == listed                                              # a real list is untouched
    assert parsed('[{"task": "Share the plan"}]') == listed                      # JSON text is still parsed
    real = parsed("- [ ] **Share the plan** — Owner: Priya | Deadline: Friday | Priority: High")
    assert isinstance(real, list) and real and "no action items" not in json.dumps(real).lower()   # real items still extracted
    other = "No action items for the design team, but the QA team must retest by Monday."
    assert parsed(other) != []                                                   # a sentence with content is never discarded as "none"
