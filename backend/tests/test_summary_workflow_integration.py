"""process_meeting_report wiring for the summary — LLM calls and heavy scorers are mocked."""

import re

import pytest

from backend.services import llm_workflow as wf
from backend.services.summary_service import SummaryResult, plan_summary_length, prepare_summary_source

LABEL_RE = re.compile(r"(?i)\b(?:speaker|spk)[\s_\-]*\d+\b")

TX = "\n".join([
    "Speaker_1: So we are here to review the speaker diarization feature that is not working in production.",
    "Speaker_2: Yeah the diarization output merges two voices and the summary then attributes wrong statements.",
    "Speaker_1: We decided to retrain the clustering step and add a fallback for short recordings.",
    "Speaker_2: Also the API needs a new field for the speaker count so the frontend can display it correctly.",
])

MOM = (
    "### 📅 MEETING INFORMATION\n- **Participants:** Speaker_1, Speaker_2\n\n"
    "### 🎯 EXECUTIVE SUMMARY\nSpeaker_1 opened the review and Speaker_2 agreed with the plan.\n\n"
    "### 🗣️ KEY DISCUSSION POINTS\n- **Diarization**\n  - *Detail:* Speaker_2 raised the merged-voices issue.\n"
)


@pytest.fixture()
def stubbed_workflow(monkeypatch):
    """Route process_meeting_report through fakes; return what the summary worker received."""
    seen = {}

    def fake_summary(transcript, *, llm_call=None, meeting_ref=""):
        seen["transcript"] = transcript
        seen["meeting_ref"] = meeting_ref
        return SummaryResult(
            text="### Executive Summary\n\nSpeaker_1 said the plan slipped. The team agreed to retrain the clustering step.",
            method="llm", llm_calls=1, plan=plan_summary_length(60),
        )

    monkeypatch.setattr(wf, "generate_meeting_summary", fake_summary)
    monkeypatch.setattr(wf.model_router, "validate_airawat_config", lambda: True)
    monkeypatch.setattr(wf.model_router, "get_llm", lambda **kw: object())
    monkeypatch.setattr(wf, "get_llm", lambda *a, **k: object())
    monkeypatch.setattr(wf, "_generate_section", lambda llm, prompt, transcript, max_chars=None: MOM)
    monkeypatch.setattr(wf, "_extract_metadata_json", lambda llm, t: {
        "action_items": "- [ ] **Retrain clustering** - @Not Specified", "decisions": "Retrain the clustering step.",
        "agenda_key_decisions": [{"agenda": "Diarization fix", "key_decision": "Retrain the clustering step."}],
        "risks": "Short recordings may still fail.", "project_name": "X", "sentiment": "Neutral",
        "emotions": "Neutral: 100%", "questions_raised": "[]", "keyword_frequency": "{}",
    })
    monkeypatch.setattr(wf, "_extract_agenda_key_decisions", lambda llm, t, user_agenda="": [
        {"agenda": "Diarization fix", "key_decision": "Retrain the clustering step."}])
    monkeypatch.setattr(wf, "_extract_action_items", lambda llm, t, reference_dt=None: [
        {"task": "Retrain clustering", "owner": "Unassigned"}])
    return seen


def test_summary_worker_gets_full_transcript_and_output_is_finalized(stubbed_workflow):
    long_tx = TX + "\n" + "\n".join(f"Speaker_1: Extra line {i} about topic {i}." for i in range(4000))
    assert len(long_tx) > wf.get_report_max_chars()          # would have been truncated before
    out = wf.process_meeting_report(long_tx, meeting_date="2026-09-01", meeting_time="10:00", duration_str="00:10:00")

    assert stubbed_workflow["transcript"] == long_tx           # full transcript, not the truncated one
    assert stubbed_workflow["meeting_ref"]                      # tagged with the transcript fingerprint
    assert not LABEL_RE.search(out["summary"])                  # write-time gate removed the label
    assert "###" not in out["summary"] and "Executive Summary" not in out["summary"]
    assert "retrain the clustering step" in out["summary"]


def test_mom_executive_summary_is_label_free_but_discussion_keeps_attribution(stubbed_workflow):
    out = wf.process_meeting_report(TX, meeting_date="2026-09-01", meeting_time="10:00", duration_str="00:10:00")
    head, tail = out["mom"].split("KEY DISCUSSION POINTS")
    exec_section = head.split("EXECUTIVE SUMMARY")[1]
    assert not LABEL_RE.search(exec_section)
    assert "Speaker_2 raised the merged-voices issue" in tail
    assert "Speaker_1, Speaker_2" in head                       # MoM participant list is unchanged


def test_extractive_report_summary_is_a_summary_not_a_transcript_dump(monkeypatch):
    monkeypatch.setattr(wf, "compute_batch_match_scores", lambda pairs: [{} for _ in pairs])
    out = wf.generate_extractive_meeting_report(
        TX, meeting_date="2026-09-01", meeting_time="10:15 AM", duration_str="00:12:00")
    s = out["summary"]
    assert s and not LABEL_RE.search(s)
    assert "Executive Summary" not in s and "participants including" not in s
    assert "2026-09-01" not in s and "10:15" not in s and "00:12:00" not in s   # metadata is not the summary
    assert len(s.split()) <= plan_summary_length(prepare_summary_source(TX).word_count).hard_max_words
