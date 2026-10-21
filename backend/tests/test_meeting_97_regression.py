"""Regression tests built on the EXACT API output of meeting 97 (``fixtures/meeting_97_api_output.json``).

Meeting 97 is the first run after the transcript-truncation fix: a 30-minute single-speaker recording, 27 segments,
3 action items, 7 agenda topics, no decisions, and a related-meeting entry for 1648 with similarity 0.98. The output
exposed these defects, each pinned below (the fixture is untouched apart from the local file path):

* ``metrics.action_item_quality`` 88 vs ``evaluation_metrics.action_items`` 0 (two grounding methods);
* ``runtime_metrics.speaker.turn_count`` 0 vs 27 "turns" elsewhere vs 73 raw diarization turns;
* ``0.98`` was a floor applied to any exact-duplicate verdict, justified by "content virtually identical" although
  the earlier record held 260 of 3,448 words (the files are byte-identical; the transcripts are not);
* all 7 agenda rows stored ``explicit`` while the API said ``inferred`` and the model's label is not evidence;
* ``mom_duration_sec`` / ``action_items_duration_sec`` hardcoded 0, last segment end 0.5 s past the recording,
  ``silence_duration`` 0.0 for a recording with 12 % silence;
* ``upload_time`` 2026-08-27 next to ``upload_start_time`` 2026-10-13 (a re-upload reuses the row);
* a configured (not detected) language scored as a measured 0 % language confidence.

No network. DB-backed cases skip when PostgreSQL is unreachable.
"""

import json
import os
from types import SimpleNamespace

import pytest

from backend.evaluation import quality_scores as qs
from backend.evaluation import reference_free_engine as rfe
from backend.evaluation import runtime_quality as rq
from backend.schemas import schemas
from backend.services import action_item_service as ais
from backend.services import agenda_service
from backend.services import diarization_service as ds
from backend.services import final_aggregation_service as fas
from backend.services import llm_workflow as wf
from backend.services import similarity_service as sim
from backend.services import transcription_service as tsvc
from backend.services.model_router import model_router
from backend.utils import text_signals as ts

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "meeting_97_api_output.json")


@pytest.fixture(scope="module")
def m97():
    with open(FIXTURE, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def lexical(monkeypatch):
    """No embedding model: deterministic lexical paths."""
    monkeypatch.setattr(rfe, "_embed_batch", lambda texts: [])
    monkeypatch.setattr(rfe, "_embed_one", lambda text: [])
    monkeypatch.setattr(wf, "compute_batch_match_scores",
                        lambda pairs: [{"match_score": None, "confidence_level": None, "model_name": "stub"} for _ in pairs])
    monkeypatch.setattr(agenda_service, "deduplicate_semantic", lambda texts, threshold=None: list(range(len(texts))))


def _transcript(m97):
    return "\n".join(f"{s['speaker']}: {s['text']}" for s in m97["segments"])


def _segments(m97):
    return [{"speaker": s["speaker"], "start_time": s["start_time"], "end_time": s["end_time"], "text": s["text"],
             "stt_confidence": s.get("stt_confidence")} for s in m97["segments"]]


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# the fixture itself is what it claims to be
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def test_the_fixture_is_the_reported_output(m97):
    assert m97["id"] == 97 and m97["duration"] == 1820.48 and len(m97["segments"]) == 27
    assert m97["matched_meeting_id"] == 1648 and m97["similarity_score"] == 0.98
    assert m97["action_items_count"] == 3 and m97["mom_duration_sec"] == 0.0 and m97["silence_duration"] == 0.0
    assert m97["metrics"]["action_item_quality"]["score"] == 88.0
    assert m97["evaluation_metrics"]["action_items"]["score"] == 0.0
    assert m97["runtime_metrics"]["speaker"]["turn_count"] == 0
    assert "krvam" not in json.dumps(m97)                                  # the local path was redacted, nothing else


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 8. JSON types follow the documented contract
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def test_the_stored_output_validates_against_the_api_schema(m97):
    model = schemas.Meeting.model_validate(m97)
    assert model.id == 97 and len(model.segments) == 27 and len(model.agenda_key_decisions) == 7
    assert len(model.structured_action_items) == 3 and isinstance(model.analysis.action_items, list)


def test_string_typed_json_fields_hold_valid_json_of_the_documented_kind(m97):
    """schemas.Meeting declares these as strings (a documented, deliberate contract): each must hold ONE level of
    JSON of the right kind - never a JSON string inside JSON."""
    expected = {"emotions": dict, "questions_raised": list, "keyword_frequency": dict, "retry_history_json": list,
                "diarization_info_json": dict, "diarization_metadata_json": dict, "pipeline_stage_states_json": dict,
                "sentiment_breakdown_json": dict, "related_meetings_json": list}
    for field, kind in expected.items():
        assert isinstance(m97[field], str), field
        assert isinstance(json.loads(m97[field]), kind), field
    breakdown = json.loads(m97["sentiment_breakdown_json"])
    assert isinstance(breakdown["emotions"], dict)                        # native object, not '{"Confidence": 80.0}'


def test_nested_native_types(m97):
    assert isinstance(m97["segments"], list) and isinstance(m97["analysis"], dict)
    assert isinstance(m97["analysis"]["action_items"], list) and isinstance(m97["structured_action_items"], list)
    assert all(isinstance(a["id"], int) for a in m97["structured_action_items"])
    assert isinstance(m97["metrics"], dict) and isinstance(m97["runtime_metrics"], dict)
    assert json.loads(json.dumps(m97)) == m97


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 1. Action items: one grounding method, validated against the transcript
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def test_every_action_item_quote_is_verbatim_in_the_transcript(m97):
    norm = ts.normalize_text(_transcript(m97))
    for item in m97["structured_action_items"]:
        assert ts.quote_in_text(item["evidence"], norm), item["task"]


def test_both_grounding_blocks_now_agree_that_the_three_items_are_grounded(m97):
    items = m97["structured_action_items"]
    text = _transcript(m97)
    evaluation = rfe.action_item_grounding(items, text)
    assert evaluation["total_items"] == 3 and evaluation["grounded_items"] == 3 and evaluation["unsupported_items"] == 0
    quality = qs.action_item_quality(items, text)
    assert quality["components"]["grounding"] == 100.0


def test_the_evaluation_block_reports_the_metrics_score_not_a_second_number(m97):
    items, text = m97["structured_action_items"], _transcript(m97)
    quality = qs.action_item_quality(items, text)
    out = rfe.build_evaluation_metrics(
        segments=_segments(m97), audio_duration_sec=m97["duration"], language_confidence=None, summary_text="",
        mom_text="", action_items=items, agenda_rows=[], chat_messages=[],
        score_overrides={"action_items": quality["score"], "decisions": None})
    assert out["action_items"]["score"] == quality["score"] != 0.0
    assert out["action_items"]["superseded_by"] == "metrics.action_item_quality"
    assert out["action_items"]["grounded_items"] == 3                       # same facts in both blocks


def test_the_items_are_flagged_as_possibly_illustrative_but_not_removed_or_rescored(m97):
    items, text = m97["structured_action_items"], _transcript(m97)
    quality = qs.action_item_quality(items, text)
    flagged = [i for i in quality["issues"] if "example or hypothetical" in i]
    assert len(flagged) >= 2 and quality["signals"]["action_item_count"] == 3
    assert quality["components"]["grounding"] == 100.0                      # a warning, not a re-score


def test_an_invented_quote_is_removed_and_an_unsupported_item_dropped():
    transcript = "Speaker_1: Priya will prepare the migration plan by Friday and share it with the team."
    raw = [
        {"task": "Prepare the migration plan", "owner": "Priya", "deadline": "Friday",
         "evidence": "Priya will prepare the migration plan by Friday"},
        {"task": "Prepare the migration plan", "owner": "Priya", "evidence": "Priya swore an oath to rewrite everything"},
        {"task": "Negotiate the satellite launch contract", "evidence": "we must negotiate the satellite launch contract"},
    ]
    out = ais.validate_action_items(raw, transcript)
    assert [i["task"] for i in out] == ["Prepare the migration plan", "Prepare the migration plan"]
    assert out[0]["evidence"] == "Priya will prepare the migration plan by Friday"
    assert out[1]["evidence"] is None                                       # never a fabricated quote
    assert all(i["owner"] in ("Priya",) for i in out)


def test_a_missing_owner_or_deadline_is_not_a_zero_in_the_confidence():
    transcript = "Speaker_1: please complete the payment testing by Tuesday evening."
    base = {"task": "Complete the payment testing", "deadline": "Tuesday evening", "deadline_type": "relative",
            "evidence": "complete the payment testing by Tuesday evening"}
    no_owner = ais.compute_confidence(dict(base, owner="Unassigned"), transcript)
    with_owner = ais.compute_confidence(dict(base, owner="Priya"), transcript)
    parts = [(ais._score_task_clarity(base["task"]), 0.40), (ais._score_deadline_clarity("Tuesday evening", "relative"), 0.20),
             (ais._score_evidence_quality(base["evidence"], transcript), 0.15)]
    assert no_owner == round(sum(s * w for s, w in parts) / sum(w for _, w in parts) * 100, 1)
    assert no_owner > 57.8 and with_owner >= no_owner - 5                   # the old formula gave 57.8 to all three
    assert ais.compute_confidence(dict(base, owner="Unassigned", evidence=""), transcript) < no_owner   # no quote = weaker


def test_a_quote_with_an_ellipsis_is_checked_fragment_by_fragment():
    norm = ts.normalize_text("Speaker_1: we should integrate the chatbot into the intranet because users asked for it.")
    assert ts.quote_in_text("we should integrate the chatbot... because users asked for it", norm)
    assert not ts.quote_in_text("we should integrate the chatbot... because the CEO demanded it", norm)
    assert not ts.quote_in_text("", norm) and not ts.quote_in_text("a b", norm)


def test_timestamps_come_only_from_the_segment_that_holds_the_quote(m97):
    from backend.workers.pipelines import stamp_evidence_times

    rows = [(s["text"], s["start_time"], s["end_time"]) for s in m97["segments"]]
    items = [{"evidence": "complete the payment testing by tuesday evening"},
             {"evidence": "this sentence was never spoken in the meeting at all"}, {"evidence": None},
             {"evidence": "deploy on friday", "timestamp_start": "00:01:00"}]          # the model's own value stays
    assert stamp_evidence_times(items, rows, "evidence", "timestamp_start", "timestamp_end") == 1
    assert items[0]["timestamp_start"] == "00:25:33" and items[0]["timestamp_end"]       # segment 23 starts 1533.272 s
    assert items[1].get("timestamp_start") is None and items[2].get("timestamp_start") is None
    assert items[3]["timestamp_start"] == "00:01:00"


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 2. Diarization: one definition of turn / segment / speech / coverage / processing time
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def test_one_speaker_with_27_segments_is_one_turn_and_zero_changes(m97):
    out = rq.speaker_runtime_metrics(_segments(m97), m97["duration"], m97["speaker_duration_sec"])
    assert (out["segment_count"], out["turn_count"], out["speaker_change_count"]) == (27, 1, 0)
    dist = out["speaker_distribution"][0]
    assert (dist["turn_count"], dist["segment_count"]) == (1, 27)             # per-speaker == overall (was 27 vs 0)


def test_segments_are_validated_against_the_recording_length(m97):
    out = rq.speaker_runtime_metrics(_segments(m97), m97["duration"], None)
    assert m97["segments"][-1]["end_time"] == "1820.980"                       # 0.5 s past the 1820.48 s recording
    assert out["total_diarized_duration_sec"] == pytest.approx(1820.48, abs=0.01)
    assert out["audio_duration_sec"] == 1820.48 and out["speaker_coverage"] == 100.0
    late = _segments(m97) + [{"speaker": "Speaker_1", "start_time": "1900.000", "end_time": "1910.000", "text": "x"}]
    bad = rq.speaker_runtime_metrics(late, m97["duration"], None)
    assert bad["timestamp_consistency_rate"] < 100.0                           # starts after the recording ended


def test_audio_speech_silence_coverage_and_processing_time_are_distinct_fields(m97):
    info = json.loads(m97["diarization_info_json"])
    out = rfe.build_runtime_metrics(
        segments=_segments(m97), audio_duration_sec=m97["duration"], stt_duration_sec=m97["stt_duration_sec"],
        speaker_duration_sec=m97["speaker_duration_sec"], summary_text="", mom_text="", transcript_text="",
        agenda_decision_rows=[], action_items=[],
        meeting=SimpleNamespace(duration=m97["duration"], stt_duration_sec=m97["stt_duration_sec"],
                                speaker_duration_sec=m97["speaker_duration_sec"],
                                diarization_metadata_json=m97["diarization_metadata_json"]),
        speech_durations=ds.stored_speech_durations(info))["speaker"]
    assert out["audio_duration_sec"] == 1820.48
    assert out["speech_duration_sec"] == 1605.69
    assert out["silence_duration_sec"] == pytest.approx(214.79, abs=0.01)
    assert out["speech_ratio_pct"] == pytest.approx(88.2, abs=0.1)
    assert out["diarization_processing_time_sec"] == 27.302
    assert out["diarization_turn_count"] == 73                                 # the model's raw turns, labelled as such
    assert out["segment_count"] == 27 and out["turn_count"] == 1
    assert out["speaker_distribution"][0]["duration_seconds"] == 1605.69       # speech seconds, not the segment span


def test_the_stored_speaker_diagnostics_use_the_same_definitions(m97):
    info = json.loads(m97["diarization_info_json"])
    diag = ds.compute_speaker_diagnostics(_segments(m97), m97["duration"], speech_durations={"Speaker_1": 1605.69})
    assert (diag["total_turns"], diag["total_segments"], diag["speaker_switches"]) == (1, 27, 0)
    assert diag["speech_source"] == "vad_regions" and diag["total_speech_duration_sec"] == 1605.69
    spk = diag["speaker_breakdown"]["Speaker_1"]
    assert (spk["turn_count"], spk["segment_count"]) == (1, 27)
    assert info["diagnostics"]["total_turns"] == 27                            # what the OLD code stored (27 "turns")


def test_two_speakers_alternate_turns_and_changes_are_consistent():
    segs = [{"speaker": "A", "start_time": "0", "end_time": "10", "text": "a"},
            {"speaker": "A", "start_time": "10", "end_time": "20", "text": "a"},
            {"speaker": "B", "start_time": "20", "end_time": "30", "text": "b"},
            {"speaker": "A", "start_time": "30", "end_time": "40", "text": "a"}]
    out = rq.speaker_runtime_metrics(segs, 40.0, None)
    assert (out["segment_count"], out["turn_count"], out["speaker_change_count"]) == (4, 3, 2)
    by_spk = {d["speaker"]: d for d in out["speaker_distribution"]}
    assert (by_spk["A"]["turn_count"], by_spk["A"]["segment_count"], by_spk["B"]["turn_count"]) == (2, 3, 1)
    assert ds.compute_speaker_diagnostics(segs, 40.0)["total_turns"] == 3


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 3 + 7. Duplicates and related meetings
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def _related(m97, meeting_id):
    return next(r for r in json.loads(m97["related_meetings_json"]) if r["meeting_id"] == meeting_id)


def _pair_from_entry(entry, **overrides):
    s = entry["signals"]
    sig = {"semantic": s["semantic"], "content": s["content_overlap"], "content_jaccard": s["content_jaccard"],
           "keyword": s["keyword_overlap"], "title": s["title_similarity"], "participant": s["participant_overlap"],
           "duration_ratio": s["duration_ratio"], "fingerprint_match": False, "same_recording": True,
           "temporal": {"has_dates": True, "current_is_after": True}, "bullets": 0.0,
           "action_cmp": {"shared": [], "new": [], "resolved": [], "not_carried_over": []},
           "decision_cmp": {"shared": [], "new": [], "resolved": [], "not_carried_over": []},
           "keyword_common": [], "participant_matches": [], "doc_common": []}
    sig.update(overrides)
    cur = sim.MeetingSnapshot(id=97, code="123456789", title=entry["title"])
    other = sim.MeetingSnapshot(id=entry["meeting_id"], code=entry["meeting_code"], title=entry["title"])
    return cur, other, sig


def test_the_reported_1648_pair_is_the_same_file_with_incomplete_content(m97):
    entry = _related(m97, 1648)
    assert entry["category"] == "exact_duplicate" and entry["similarity_score"] == 0.98     # as reported
    assert entry["signals"]["content_overlap"] == 1.0 and entry["signals"]["content_jaccard"] < 0.2
    cur, other, sig = _pair_from_entry(entry)
    assert sim._classify(cur, other, sig, 0.2) == sim.Category.RELATED       # not a content duplicate, not follow-up


def test_a_genuine_duplicate_of_the_same_recording_is_still_a_duplicate(m97):
    entry = _related(m97, 1647)                                              # same file, full transcript both times
    assert entry["signals"]["content_jaccard"] > 0.9
    cur, other, sig = _pair_from_entry(entry)
    assert sim._classify(cur, other, sig, 0.2) == sim.Category.EXACT_DUPLICATE


def test_similarity_score_is_the_measured_blend_not_a_verdict_floor(m97):
    words = _transcript(m97).split()
    cur = sim.MeetingSnapshot(id=97, code="123456789", title="t", filename="rec.mp4", file_size=238134268, duration=1820.48,
                              transcript=" ".join(words), meeting_date="2026-10-13")
    other = sim.MeetingSnapshot(id=1648, code="123456", title="t", filename="rec.mp4", file_size=238134268,
                                duration=1820.48, transcript=" ".join(words[:260]), meeting_date="2026-10-13",
                                action_items="", status="completed")
    res = sim.analyze_pair(cur, other, semantic=0.992, min_score=0.2, include_details=True)
    assert res["signals"]["same_recording"] is True
    assert res["category"] == sim.Category.RELATED and res["is_duplicate"] is False
    assert res["similarity_score"] < 0.98 and res["similarity_score"] == sim._blended_score(
        sim._compute_signals(cur, other, 0.992))
    assert "same recording" in res["reason"] and "differ" in res["reason"] and "260" in res["reason"]
    assert "virtually identical" not in res["reason"]
    assert res["signals"]["words_current"] > 3000 and res["signals"]["words_other"] == 260


def test_identical_transcripts_keep_the_certain_duplicate_floor():
    text = " ".join(f"word{i}" for i in range(120))
    a = sim.MeetingSnapshot(id=1, code="1", title="x", transcript=text, fingerprint="f")
    b = sim.MeetingSnapshot(id=2, code="2", title="y", transcript=text, fingerprint="f")
    res = sim.analyze_pair(a, b, semantic=0.5, min_score=0.2, include_details=False)
    assert res["category"] == sim.Category.EXACT_DUPLICATE and res["similarity_score"] >= 0.98


def test_recording_identity_needs_name_size_and_duration():
    base = dict(id=1, code="1", title="t", filename="rec.mp4", file_size=100, duration=60.0)
    a = sim.MeetingSnapshot(**base)
    assert sim._same_recording(a, sim.MeetingSnapshot(**{**base, "id": 2}))
    assert not sim._same_recording(a, sim.MeetingSnapshot(**{**base, "id": 2, "file_size": 101}))
    assert not sim._same_recording(a, sim.MeetingSnapshot(**{**base, "id": 2, "filename": "other.mp4"}))
    assert not sim._same_recording(a, sim.MeetingSnapshot(**{**base, "id": 2, "duration": 75.0}))
    assert not sim._same_recording(sim.MeetingSnapshot(**{**base, "filename": "", "file_size": 0}),
                                   sim.MeetingSnapshot(**{**base, "id": 2, "filename": "", "file_size": 0}))


def test_historical_outputs_come_from_the_historical_meeting_only():
    current = sim.MeetingSnapshot(
        id=2, code="B", title="Release sync part two", transcript="we shipped the api", meeting_date="2026-10-12",
        action_items="- [ ] **Write the audit summary** — Owner: Z", mom="CURRENT MOM", summary="CURRENT SUMMARY")
    previous = sim.MeetingSnapshot(
        id=1, code="A", title="Release sync part one", transcript="we planned the api", meeting_date="2026-10-01",
        action_items="- [ ] **Prepare the testing report** — Owner: Anita\n- [x] **Ship the gateway** — Owner: Ravi",
        mom="PREVIOUS MOM", summary="PREVIOUS SUMMARY about planning")
    res = sim.analyze_pair(current, previous, semantic=0.6, min_score=0.2, include_details=True)
    assert res["matched_mom"] == "PREVIOUS MOM" and "PREVIOUS SUMMARY" in res["quick_summary"]
    assert "CURRENT" not in res["matched_mom"] + res["quick_summary"]
    diff = res["difference_analysis"]
    assert any("audit summary" in t for t in diff["new_action_items"])               # the current meeting's task
    assert not any("audit summary" in t for t in diff["resolved_action_items"] + diff["not_carried_over_action_items"])
    assert any("testing report" in t for t in diff["not_carried_over_action_items"])  # the previous meeting's task
    assert any("Ship the gateway" in t for t in diff["resolved_action_items"])


def test_a_meeting_without_analysis_never_borrows_the_mom_of_the_meeting_it_matched():
    m = SimpleNamespace(id=5, display_id="5", bharatvc_meeting_id="5", title="x", filename="x.mp4", file_type="media",
                        status="error", is_approved=False, upload_time=None, meeting_date=None, meeting_time=None,
                        duration=10.0, project_name="p", participants=None, participant_count=0,
                        matched_mom="MINUTES OF A DIFFERENT MEETING", keyword_frequency=None, content_fingerprint="",
                        file_size_bytes=1)
    snap = sim._snapshot_from_parts(m, "some transcript", None)
    assert snap.mom == "" and snap.file_size == 1


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 4. Agenda and decisions
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def _akd_items(m97):
    return [{"agenda": a["agenda"], "key_decision": a["key_decision"], "evidence": a["evidence"],
             "agenda_type": "explicit"} for a in m97["agenda_key_decisions"]]          # the model labelled all explicit


def test_the_transcript_not_the_model_decides_which_topics_were_announced(m97):
    out = wf.validate_and_format_agenda_key_decisions(_akd_items(m97), transcript=_transcript(m97))
    types = {i["agenda"]: i["agenda_type"] for i in out}
    assert len(out) == 7 and set(types.values()) == {"explicit", "inferred"}
    assert types["System Design and Architecture"] == "explicit"        # "this meeting is about system design and architecture"
    assert types["Tech Stack"] == "inferred"                            # never announced with a cue phrase
    again = wf.validate_and_format_agenda_key_decisions(list(reversed(_akd_items(m97))), transcript=_transcript(m97))
    assert {i["agenda"]: i["agenda_type"] for i in again} == types      # same answer whatever the model said or the order


def test_a_topic_nobody_discussed_is_dropped_and_no_decision_is_invented(m97):
    items = _akd_items(m97) + [{"agenda": "Quarterly hiring plan", "key_decision": "Hire three engineers",
                                "evidence": "we decided to hire three engineers"}]
    out = wf.validate_and_format_agenda_key_decisions(items, transcript=_transcript(m97))
    assert "Quarterly hiring plan" not in [i["agenda"] for i in out]
    assert all(i["key_decision"] == "No formal decision recorded." and i["decision_grounded"] is False for i in out)
    assert all(i["evidence"] is None and i["confidence"] is None for i in out)


def test_zero_decisions_stay_zero_through_the_final_record_builders(m97):
    rows = [SimpleNamespace(agenda_id_str=a["id"], agenda_order=a["agenda_order"], agenda=a["agenda"],
                            key_decision=a["key_decision"], decision_grounded=a["decision_grounded"],
                            match_score=a["match_score"], evidence=a["evidence"], speaker=a["speaker"],
                            timestamp=a["timestamp"], reason=a["reason"], confidence=a["confidence"],
                            agenda_type=a["agenda_type"], agenda_source=a["source"])
            for a in m97["agenda_key_decisions"]]
    assert fas._build_decisions(rows) == []
    mapping = fas._build_agenda_decision_mapping(rows)
    assert len(mapping) == 7 and not any(m["has_decision"] for m in mapping)
    assert not any(k in m for m in mapping for k in ("decision", "evidence", "match_score", "timestamp"))
    assert qs.decision_quality([], _transcript(m97))["score"] is None


def test_the_api_item_shows_the_stored_agenda_type_not_the_schema_default():
    from backend.api.meetings import _build_agenda_item

    row = SimpleNamespace(id=1, agenda="System Design", key_decision="", match_score=None, embedding_model=None,
                          speaker="Speaker_1", agenda_id_str="A2", agenda_order=2, agenda_source="llm_detected",
                          decision_grounded=False, evidence=None, agenda_type="explicit", reason=None,
                          timestamp="00:01:00", confidence=0.9)
    item = _build_agenda_item(row)
    assert item.agenda_type == "explicit" and item.timestamp == "00:01:00" and item.confidence == 0.9
    assert _build_agenda_item(SimpleNamespace(**{**row.__dict__, "agenda_type": None})).agenda_type == "inferred"


def test_a_model_given_decision_confidence_is_kept_and_an_absent_one_is_not_invented():
    transcript = "Speaker_1: We decided to adopt PostgreSQL and move the staging environment first."
    base = {"agenda": "PostgreSQL staging environment", "key_decision": "Adopt PostgreSQL",
            "evidence": "We decided to adopt PostgreSQL and move the staging environment first."}
    (without,) = wf.validate_and_format_agenda_key_decisions([base], transcript=transcript)
    (with_,) = wf.validate_and_format_agenda_key_decisions([dict(base, confidence=0.8)], transcript=transcript)
    assert without["confidence"] is None and with_["confidence"] == 0.8      # was 0.90 for "has a quote"


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 5. Quality metrics: missing signals are missing, heuristics are labelled
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def test_a_configured_language_is_not_a_measured_zero_confidence(m97):
    segs = _segments(m97)
    zero = qs.transcript_quality(segs, audio_duration_sec=m97["duration"], language_confidence=0.0)
    none = qs.transcript_quality(segs, audio_duration_sec=m97["duration"], language_confidence=None)
    assert zero["components"]["language_consistency"] == none["components"]["language_consistency"] == 100.0
    assert zero["signals"]["language_detection_confidence"] is None
    legacy = rfe.transcription_quality(segs, m97["duration"], 0.0)
    assert legacy["language_detection_confidence"] is None and legacy["components"]["language_consistency"] is None
    real = rfe.transcription_quality(segs, m97["duration"], 0.9)
    assert real["components"]["language_consistency"] == 90.0               # a real detector value still counts


def test_the_metrics_block_says_its_scores_are_heuristics(m97):
    block = qs.build_quality_metrics(
        segments=_segments(m97), summary_text="The team discussed the platform.", transcript_text=_transcript(m97),
        agenda_topics=[], rag_evaluations=[], assistant_answer_count=0, audio_duration_sec=m97["duration"])
    assert block["scores_are_heuristic"] is True and block["ground_truth_available"] is False
    assert "not measured accuracy" in block["score_meaning"]


def test_unmeasured_components_are_dropped_from_the_overall_score():
    out = rfe.overall_quality({"transcription": 80.0, "summary": 70.0, "action_items": None, "decisions": None,
                               "agenda": None, "rag": None})
    assert set(out["weights_used"]) == {"transcription", "summary"}
    assert out["components"]["action_items"] is None and out["components"]["rag"] is None


# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 6. Timing and coverage
# ═════════════════════════════════════════════════════════════════════════════════════════════════
def test_the_last_word_ends_with_the_recording_not_half_a_second_after_it(monkeypatch, tmp_path):
    """4 chunks of the real recording (456 s nominal, 0.5 s overlap): the last estimated word used to end at 1820.98."""
    durations = [456.0, 456.5, 456.5, 452.98]
    chunk_files = []
    for i, _ in enumerate(durations):
        f = tmp_path / f"c{i}.wav"
        f.write_bytes(b"x")
        chunk_files.append((str(f), i * 456.0, i == 0))
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    by_path = {c[0]: d for c, d in zip(chunk_files, durations)}
    texts = {c[0]: " ".join(f"c{i}w{j}" for j in range(400)) + "." for i, c in enumerate(chunk_files)}
    monkeypatch.setattr(tsvc, "CHUNK_OVERLAP_SEC", 0.5)
    monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": "k"}})
    monkeypatch.setattr(tsvc, "split_audio_into_chunks", lambda p: list(chunk_files))
    monkeypatch.setattr(tsvc, "get_audio_info", lambda p: {"duration": by_path.get(p, 1820.48)})
    monkeypatch.setattr(tsvc, "transcribe_single_chunk", lambda path, i, n, **kw: {"text": texts[path]})
    out = tsvc.transcribe_audio_airawat(str(audio), language="en")
    ends = [w["end"] for w in out["words"]]
    assert max(ends) <= 1820.48 + 0.01 and max(ends) >= 1820.0
    starts = [w["start"] for w in out["words"]]
    assert starts == sorted(starts)                                          # chunks do not run into each other
    assert len(out["words"]) == 1600


def test_silence_is_recording_minus_detected_speech_and_unknown_is_none():
    from backend.workers.pipelines import silence_from_speech

    vad = {"speech_source": "vad_regions", "total_speech_duration_sec": 1605.69}
    assert silence_from_speech(vad, 1820.48) == pytest.approx(214.79, abs=0.001)
    assert silence_from_speech({"speech_source": "segment_spans", "total_speech_duration_sec": 1820.98}, 1820.48) is None
    assert silence_from_speech(None, 1820.48) is None and silence_from_speech(vad, 0) is None
    assert silence_from_speech({"speech_source": "vad_regions", "total_speech_duration_sec": 5000.0}, 1820.48) == 0.0


def test_section_durations_are_each_sections_own_time_and_survive_a_retry(monkeypatch):
    """process_meeting_report returns every section's real generation time (mom / action items were stored as 0);
    a section reused from an earlier attempt keeps ITS measured time in ``_section_durations`` while this attempt's
    ``_section_timings`` correctly shows 0.0 for it."""
    import time as _time

    def slow(seconds, value):
        def _fn(*a, **k):
            _time.sleep(seconds)
            return value
        return _fn

    summary = SimpleNamespace(text="The team reviewed the platform design and agreed on the architecture.",
                              method="llm", llm_calls=1, violations=[])
    metadata = {"action_items": "", "decisions": "", "agenda_key_decisions": [], "risks": "none noted",
                "project_name": "Platform", "sentiment": "Positive", "emotions": "Focus: 100%",
                "questions_raised": "[]", "keyword_frequency": "{}"}
    monkeypatch.setattr(wf, "generate_meeting_summary", slow(0.05, summary))
    monkeypatch.setattr(wf.model_router, "get_llm", lambda **kw: object())
    monkeypatch.setattr(wf, "get_llm", lambda *a, **k: object())
    monkeypatch.setattr(wf, "_generate_section",
                        lambda llm, prompt, transcript, max_chars=None: (
                            _time.sleep(0.12), "### Minutes\n- point one\n- point two")[1])
    monkeypatch.setattr(wf, "_extract_metadata_json", lambda llm, t: dict(metadata))
    monkeypatch.setattr(wf, "_extract_agenda_key_decisions", lambda llm, t, user_agenda="": [])
    calls = {"actions": 0}

    def actions(llm, t, reference_dt=None):
        calls["actions"] += 1
        if calls["actions"] == 1:
            raise TimeoutError("AIRAWAT did not answer")        # first attempt fails, mom + summary succeed
        _time.sleep(0.05)
        return []

    monkeypatch.setattr(wf, "_extract_action_items", actions)
    text = "Speaker_1: " + " ".join(f"topic{i} was reviewed during the platform design session" for i in range(30))
    cache = {}
    with pytest.raises(wf.AirawatError):
        wf.process_meeting_report(text, "2026-10-12", "10:00", "00:30:00", section_cache=cache)
    assert cache["_durations"]["mom"] >= 0.1 and "action_items" not in cache["_durations"]
    out = wf.process_meeting_report(text, "2026-10-12", "10:00", "00:30:00", section_cache=cache)
    d, t = out["_section_durations"], out["_section_timings"]
    assert d["mom"] >= 0.1 and d["action_items"] >= 0.05 and d["summary"] >= 0.05    # all measured, none 0
    assert t["mom"] == 0.0 and t["summary"] == 0.0                          # reused: no time spent THIS attempt
    assert d["mom"] == cache["_durations"]["mom"]



# ═════════════════════════════════════════════════════════════════════════════════════════════════
# 8. Upload dates (DB)
# ═════════════════════════════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def db_session():
    from sqlalchemy import inspect
    from sqlalchemy.orm import Session

    from backend.database.connection import engine

    try:
        if not inspect(engine).has_table("meetings"):
            pytest.skip("meetings table does not exist yet")
    except Exception as exc:
        pytest.skip(f"database not reachable: {exc}")
    conn = engine.connect()
    outer = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint", autoflush=False)
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        conn.close()


def test_a_re_upload_under_the_same_public_id_refreshes_the_upload_date(db_session):
    from datetime import datetime, timedelta, timezone

    from backend.repositories.meeting_repository import MeetingRepository

    pid = "98" + str(abs(hash(os.urandom(4))))[:10]
    repo = MeetingRepository(db_session)
    first_start = datetime(2026, 8, 27, 11, 28, tzinfo=timezone.utc)
    first = repo.create(bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename="a.mp4", file_type="media",
                        filepath="/tmp/a.mp4", title="a", status="uploaded", upload_start_time=first_start)
    first_pk = first.id
    again_start = first_start + timedelta(days=47)
    again = repo.create(bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename="a.mp4", file_type="media",
                        filepath="/tmp/a.mp4", title="a", status="uploaded", upload_start_time=again_start)
    assert again.id == first_pk                                               # the row is reused ...
    assert again.upload_start_time == again.upload_time or (
        abs((again.upload_time - again.upload_start_time).total_seconds()) < 1)   # ... and its dates agree


def test_a_meeting_without_analysis_in_the_database_has_an_empty_mom_in_its_snapshot(db_session):
    from backend.models.models import Meeting
    from backend.repositories.meeting_repository import MeetingRepository
    from backend.services import similarity_service

    pid = "97" + str(abs(hash(os.urandom(4))))[:10]
    m = Meeting(bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename="h.mp4", file_type="media",
                filepath="/tmp/h.mp4", title="Historical", status="error", matched_mom="MOM OF ANOTHER MEETING")
    db_session.add(m)
    db_session.commit()
    snaps = similarity_service.build_snapshots(MeetingRepository(db_session), [m.id])
    assert snaps[m.id].mom == "" and "ANOTHER" not in snaps[m.id].summary


def test_ai_response_time_is_the_longest_parallel_section_not_their_sum(db_session):
    from backend.api import meetings as api
    from backend.models.models import Meeting
    from backend.repositories.meeting_repository import MeetingRepository

    pid = "96" + str(abs(hash(os.urandom(4))))[:10]
    m = Meeting(bharatvc_meeting_id=pid, meeting_id=pid, display_id=pid, filename="t.mp4", file_type="media",
                filepath="/tmp/t.mp4", title="t", status="completed", duration=100.0,
                summary_duration_sec=25.0, mom_duration_sec=18.0, action_items_duration_sec=7.0)
    db_session.add(m)
    db_session.commit()
    report = api.get_meeting_metrics(m.id, db=db_session, repo=MeetingRepository(db_session))
    runtime = report["real_runtime_data"]
    assert runtime["performance"]["ai_response_time_sec"] == 25.0              # the sum (50.0) would double-count
    assert runtime["mom"]["generation_time_sec"] == 18.0                       # mom's own time, not a hardcoded 0


def test_a_true_duplicate_is_scored_at_least_its_measured_transcript_agreement(m97):
    words = _transcript(m97).split()
    cur = sim.MeetingSnapshot(id=97, code="123456789", title="t", filename="rec.mp4", file_size=238134268,
                              duration=1820.48, transcript=" ".join(words), keyword_frequency=m97["keyword_frequency"])
    other = sim.MeetingSnapshot(id=1647, code="123", title="t", filename="rec.mp4", file_size=238134268,
                                duration=1820.48, transcript=" ".join(words[:-20]), keyword_frequency=None)
    res = sim.analyze_pair(cur, other, semantic=0.99, min_score=0.2, include_details=False)
    assert res["category"] == sim.Category.EXACT_DUPLICATE and res["is_duplicate"] is True
    assert res["similarity_score"] >= res["signals"]["content_jaccard"] >= 0.8
