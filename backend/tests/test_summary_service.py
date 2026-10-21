"""Tests for backend.services.summary_service — no network, no database.

The LLM is replaced by a scripted fake so the validate -> rewrite -> last-resort
flow is exercised deterministically.
"""

import random
import re

import pytest

from backend.services import summary_service as ss
from backend.services.summary_service import (
    check_summary,
    extractive_summary_for_transcript,
    finalize_summary,
    generate_meeting_summary,
    neutralize_speaker_labels,
    normalize_summary_text,
    plan_summary_length,
    prepare_summary_source,
    scrub_mom_executive_summary,
    summary_preview,
)

LABEL_RE = re.compile(r"(?i)\b(?:speaker|spk)[\s_\-]*\d+\b")


def distinct_text(n_sentences, seed=1):
    """Sentences that share only a few template words, so sentence de-duplication keeps all of them."""
    rng = random.Random(seed)

    def w():
        return "".join(rng.choice("bcdfghklmnprstvz") + rng.choice("aeiou") for _ in range(4))

    return " ".join(f"The group examined {w()} and {w()} for {w()} during {w()}." for _ in range(n_sentences))


def codes(text, plan=None, **kw):
    plan = plan or plan_summary_length(400)
    return {v.code for v in check_summary(text, plan, **kw)}


# ── source preparation ────────────────────────────────────────────────────────
def test_source_strips_id_labels_and_timecodes():
    tx = (
        "Speaker_1: We need to fix the diarization module.\n"
        "[00:12] SPEAKER_00: Agreed, and the API too.\n"
        "Speaker 2: Let's ship it next sprint.\n"
    )
    src = prepare_summary_source(tx)
    assert "Speaker" not in src.text and "SPEAKER" not in src.text and "00:12" not in src.text
    assert "fix the diarization module" in src.text
    assert {"Speaker_1", "SPEAKER_00", "Speaker 2"} <= set(src.id_labels)
    assert src.word_count == len(src.text.split())


def test_source_strips_repeated_real_names_but_keeps_lone_field_label():
    tx = (
        "Ravi: The release slipped by a week.\n"
        "Priya: Then we move the demo.\n"
        "Ravi: Fine, I will update the plan.\n"
        "Priya: I will tell the client tomorrow.\n"
        "Continuing on the vendor side, contracts are still pending.\n"
        "Procurement is waiting on legal review before anything is signed.\n"
        "Note: the vendor call is on Friday.\n"
        "Nobody has confirmed the attendee list yet.\n"
        "The venue booking still needs a signature from finance.\n"
        "Catering quotes arrived but none has been compared.\n"
    )
    src = prepare_summary_source(tx)
    assert "Ravi:" not in src.text and "Priya:" not in src.text
    assert "Note: the vendor call is on Friday." in src.text   # a lone field label in prose survives
    assert set(src.labels) == {"Ravi", "Priya"}
    assert src.id_labels == ()      # names are not ID-shaped


def test_source_treats_every_line_prefix_as_a_speaker_when_all_lines_have_one():
    src = prepare_summary_source("Ravi: One thing.\nPriya: Another thing.\nNote: third thing.")
    assert ":" not in src.text


def test_source_fingerprint_differs_between_meetings():
    a = prepare_summary_source("Speaker_1: alpha beta gamma delta epsilon zeta")
    b = prepare_summary_source("Speaker_1: one two three four five six seven")
    assert a.fingerprint != b.fingerprint


# ── dynamic length ────────────────────────────────────────────────────────────
def test_plan_scales_sublinearly_and_is_bounded():
    sizes = [12, 68, 300, 1000, 4000, 12000, 60000, 400000]
    plans = [plan_summary_length(n) for n in sizes]
    targets = [p.target_words for p in plans]
    assert targets == sorted(targets)                       # never shrinks as meetings grow
    assert plans[0].target_words <= 6 or plans[0].target_words <= 12 * 0.5 + 2
    assert all(p.target_words <= p.source_words * 0.5 + 1 for p in plans)
    assert plans[-1].target_words == ss.settings.SUMMARY_MAX_WORDS   # capped, not proportional
    # 10x more content must not mean 10x longer summary
    assert plans[5].target_words / plans[3].target_words < 4
    assert all(p.min_words <= p.target_words <= p.max_words <= p.hard_max_words for p in plans)


def test_plan_tiny_meeting_gets_tiny_summary():
    p = plan_summary_length(68)
    assert p.target_words <= 34 and p.paragraphs == 1


# ── speaker-label detection ───────────────────────────────────────────────────
@pytest.mark.parametrize("text", [
    "Speaker_0 raised a concern about latency and the team agreed to revisit it next week.",
    "The plan was proposed by Speaker_1 and accepted by Speaker_2 after a short review.",
    "SPEAKER_00 and SPEAKER_01 discussed the migration plan and the rollout order in detail.",
    "As speaker_2 noted the release date depends on the security review and vendor sign off.",
    "Speaker 3 suggested moving the demo, which the group accepted after some discussion.",
    "Speaker A described the outage and Speaker B proposed a rollback procedure for it.",
    "An Unknown Speaker mentioned the budget while the rest of the team reviewed the numbers.",
    "Participant 2 asked about timelines and the team promised a revised schedule by Friday.",
    "The spk3 segment covered the roadmap and the team then agreed on the quarterly priorities.",
])
def test_detects_speaker_labels(text):
    assert "speaker_label" in codes(text)


@pytest.mark.parametrize("text", [
    "The meeting focused on speaker diarization, speaker identification and meeting summarization.",
    "The team reported that the speaker direction feature is not working and agreed to fix it.",
    "Speaker Identification accuracy was reviewed and the team agreed to add a fallback model.",
    "A speaker at the conference covered API integration and action-item extraction quality.",
])
def test_topic_words_are_not_flagged(text):
    assert "speaker_label" not in codes(text)


def test_detects_this_transcripts_own_id_labels():
    text = "The proposal from Person_A was accepted after review and the team moved on to budget items."
    assert "speaker_label" not in codes(text)
    assert "speaker_label" in codes(text, id_labels=("Person_A",))


def test_detects_timestamps_but_not_clock_times():
    assert "timestamp" in codes("At [00:12] the team agreed to ship the release on the planned date.")
    assert "timestamp" in codes("The discussion ran from 00:12:30 and covered the release schedule and risks.")
    assert "timestamp" not in codes("The workshop is scheduled for 10:30 AM and covers the release schedule.")


def test_detects_dialogue_and_mom_structure():
    assert "dialogue_format" in codes(
        "Ravi: the release slipped.\nPriya: then move the demo.", labels=("Ravi", "Priya"),
        raw="Ravi: the release slipped.\nPriya: then move the demo.",
    )
    assert "mom_or_metadata" in codes("The team met. Action Items: fix the build and update the plan today.")
    assert "mom_or_metadata" in codes("Meeting ID: 12345 covered the budget and the release schedule.")
    assert "mom_or_metadata" not in codes("The agenda was reviewed and the key decision was to fix the build.")


def test_detects_length_copy_language_and_truncation():
    plan = plan_summary_length(400)
    long = distinct_text(60)
    assert "too_long" in codes(long, plan)

    source = "we should migrate the reporting service to the new cluster before the end of the quarter " * 3
    copied = "we should migrate the reporting service to the new cluster before the end of the quarter."
    assert "verbatim_copy" in codes(copied + " " + copied, plan, source_text=source)

    assert "non_english" in codes("बैठक में परियोजना की समीक्षा की गई और अगले कदम तय किए गए हैं सभी ने सहमति दी।")
    assert "incomplete" in codes("The team reviewed the migration plan and agreed to proceed with")


def test_clean_summary_has_no_violations():
    text = (
        "The meeting focused on improving the AI Meeting Intelligence platform, including speaker "
        "diarization, summarization and action-item extraction. The team agreed to improve speaker "
        "identification and the quality of meeting outputs."
    )
    assert codes(text, plan_summary_length(600)) == set()


# ── normalisation ─────────────────────────────────────────────────────────────
def test_normalize_flattens_markdown_bullets_and_preamble():
    raw = (
        "Here is the summary:\n\n### Executive Summary\n\n"
        "**The meeting** covered the release plan.\n\n"
        "- Budget was approved\n- Hiring was deferred\n"
    )
    out, structured = normalize_summary_text(raw)
    assert not structured
    assert "###" not in out and "**" not in out and "Here is" not in out and "Executive Summary" not in out
    assert "Budget was approved." in out and "Hiring was deferred." in out
    assert "\n-" not in out


def test_normalize_flags_json_and_unwraps_simple_summary_key():
    _, structured = normalize_summary_text('{"Speaker_0": "we talked", "Speaker_1": "we agreed"}')
    assert structured
    out, structured = normalize_summary_text('{"summary": "The team agreed on the plan."}')
    assert out == "The team agreed on the plan." and not structured


def test_normalize_dedupes_and_caps_paragraphs():
    raw = ("The team approved the budget. The team approved the budget. Hiring waits.\n\n"
           "Second para here.\n\nThird para here.")
    out, _ = normalize_summary_text(raw, max_paragraphs=2)
    assert out.count("approved the budget") == 1
    assert len(out.split("\n\n")) == 2


def test_dedupe_never_drops_sentences_that_differ_by_a_number_or_name():
    raw = ("Sales rose 5% in Q1. Sales rose 7% in Q2. Team A approved the budget. Team B approved the budget. "
           "The rollout moves to March. The rollout moves to April.")
    out, _ = normalize_summary_text(raw)
    for fragment in ("5% in Q1", "7% in Q2", "Team A", "Team B", "March", "April"):
        assert fragment in out


def test_product_names_are_not_mistaken_for_metadata():
    text = "The team reviewed the BharatVC integration and the Zoom connector during the planning session today."
    assert "mom_or_metadata" not in codes(text)
    assert finalize_summary(text) == text
    assert "mom_or_metadata" in codes("The plan covers the bharatvc_meeting_id field mapping for the new export step.")


def test_normalize_strips_think_blocks_and_fences():
    out, _ = normalize_summary_text("<think>hidden reasoning</think>```text\nThe plan was approved.\n```")
    assert out == "The plan was approved."


# ── deterministic label neutralisation ────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("Speaker_1: We agreed to ship.", "We agreed to ship."),
    ("Speaker_1 and Speaker_2 agreed to ship.", "The participants agreed to ship."),
    ("Speaker_3's concern was latency.", "A participant's concern was latency."),
    ("The budget was raised by SPEAKER_00 (SPEAKER_00) yesterday.", "The budget was raised by a participant yesterday."),
    ("Speaker 1, Speaker 2 & Speaker 3 agreed.", "The participants agreed."),
])
def test_neutralize_replaces_not_deletes(text, expected):
    assert neutralize_speaker_labels(text) == expected


def test_finalize_cleans_any_path():
    out = finalize_summary("### Summary\n\nSpeaker_1 said the plan slipped [00:04:10]. Speaker_2 agreed.")
    assert not LABEL_RE.search(out) and "00:04" not in out and "###" not in out
    assert "plan slipped" in out and "agreed" in out


def test_finalize_leaves_placeholder_alone():
    msg = "Not Available — transcript did not contain enough content to analyse."
    assert finalize_summary(msg) == msg


def test_summary_preview_ends_on_sentence_boundary_and_is_label_free():
    text = ("Speaker_1 opened the review of the release. " + "The team went through the open issues in detail. " * 20)
    out = summary_preview(text, 200)
    assert len(out) <= 200 and out.endswith(".") and not LABEL_RE.search(out)


def test_scrub_mom_executive_summary_only_touches_that_section():
    mom = (
        "### 🎯 EXECUTIVE SUMMARY\nSpeaker_1 opened the review and Speaker_2 agreed.\n\n"
        "### 🗣️ KEY DISCUSSION POINTS\n- **Latency**\n  - *Detail:* Speaker_2 raised a concern [00:12].\n"
    )
    out = scrub_mom_executive_summary(mom)
    head, tail = out.split("### 🗣️")
    assert not LABEL_RE.search(head)
    assert "Speaker_2 raised a concern" in tail       # discussion section keeps attribution


@pytest.mark.parametrize("head,nxt", [
    ("**EXECUTIVE SUMMARY**", "**KEY DISCUSSION POINTS**"),
    ("🎯 EXECUTIVE SUMMARY:", "🗣️ KEY DISCUSSION POINTS"),
    ("## Executive Summary", "## Key Discussion Points"),
])
def test_scrub_mom_executive_summary_handles_any_heading_style(head, nxt):
    mom = f"{head}\nSpeaker_1 opened and SPEAKER_02 agreed.\n\n{nxt}\nSpeaker_2 raised a concern.\n"
    out = scrub_mom_executive_summary(mom)
    body, rest = out.split(nxt)
    assert not LABEL_RE.search(body) and "participant" in body.lower()
    assert "Speaker_2 raised a concern" in rest            # the discussion section keeps attribution
    # Layout must survive: the next heading still starts its own line, after a blank line.
    assert out.endswith(mom[mom.index(nxt) - 2:]) and "\n\n" + nxt in out
    assert out.count("\n") == mom.count("\n")


def test_scrub_mom_executive_summary_never_glues_the_next_heading_onto_the_paragraph():
    mom = ("### 🎯 EXECUTIVE SUMMARY\nThe team met to review the plan.\n\n"
           "### 🗣️ KEY DISCUSSION POINTS\n- **Plan**\n  - *Detail:* Speaker_1 objected.\n")
    assert scrub_mom_executive_summary(mom) == mom         # nothing to scrub -> byte-identical
    dirty = mom.replace("The team met", "Speaker_1 said the team met")
    out = scrub_mom_executive_summary(dirty)
    assert "\n\n### 🗣️ KEY DISCUSSION POINTS" in out and "review the plan.###" not in out
    assert "A participant said the team met" in out and "Speaker_1 objected" in out


# ── extractive fallback ───────────────────────────────────────────────────────
TX_A = "\n".join([
    "Speaker_1: So we are here to review the speaker diarization feature that is not working in production.",
    "Speaker_2: Yeah the diarization output merges two voices and the summary then attributes wrong statements.",
    "Speaker_1: We decided to retrain the clustering step and add a fallback for short recordings.",
    "Speaker_2: Also the API needs a new field for the speaker count so the frontend can display it correctly.",
    "Speaker_1: Okay, um, next week we test the fix on the recorded meetings and report the results.",
])


def test_extractive_fallback_is_label_free_and_bounded():
    out = extractive_summary_for_transcript(TX_A)
    plan = plan_summary_length(prepare_summary_source(TX_A).word_count)
    assert out and not LABEL_RE.search(out)
    assert len(out.split()) <= plan.hard_max_words
    assert not check_summary(out, plan)


# ── generate_meeting_summary (scripted LLM) ───────────────────────────────────
class FakeLLM:
    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    def __call__(self, system_prompt, user_text, max_tokens):
        self.calls.append((system_prompt, user_text, max_tokens))
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        return reply


CLEAN = (
    "The meeting reviewed the speaker diarization feature, which merges voices in production, and agreed to "
    "retrain the clustering step with a fallback for short recordings. The team also planned an API field "
    "for speaker count and a test on recorded meetings next week."
)


def test_clean_draft_needs_one_call_and_llm_never_sees_labels():
    llm = FakeLLM(CLEAN)
    res = generate_meeting_summary(TX_A, llm_call=llm, meeting_ref="A")
    assert res.method == "llm" and res.llm_calls == 1 and not res.violations
    system_prompt, user_text, _ = llm.calls[0]
    assert "Speaker_1" not in user_text and "Speaker_2" not in user_text
    assert "retrain the clustering step" in user_text
    assert "about %d words" % res.plan.target_words in system_prompt


def test_labelled_draft_is_rewritten_by_llm_not_blindly_edited():
    bad = ("Speaker_1 said the diarization feature is failing. Speaker_2 added that the API needs a speaker "
           "count field and Speaker_1 decided to retrain the clustering step.")
    llm = FakeLLM(bad, CLEAN)
    res = generate_meeting_summary(TX_A, llm_call=llm)
    assert res.method == "llm_repaired" and res.llm_calls == 2 and not res.violations
    assert res.text == CLEAN
    rewrite_prompt, rewrite_input, _ = llm.calls[1]
    assert "Remove every speaker label" in rewrite_prompt and "Speaker_1" in rewrite_prompt
    assert rewrite_input.startswith("Speaker_1 said")           # rewrites the draft itself


def test_persistent_labels_fall_back_to_meaning_preserving_neutralisation():
    bad = ("Speaker_1 said the diarization feature is failing in production for short recordings. "
           "Speaker_2 added that the API needs a new speaker count field for the frontend.")
    llm = FakeLLM(bad)                                   # every call returns labels
    res = generate_meeting_summary(TX_A, llm_call=llm)
    assert res.method == "llm_neutralized"
    assert not LABEL_RE.search(res.text) and not res.violations
    assert "diarization feature is failing" in res.text and "speaker count field" in res.text
    assert res.llm_calls == 1 + ss.settings.SUMMARY_MAX_REPAIRS


def test_overlong_draft_is_compressed_by_rewrite():
    long = distinct_text(70)
    llm = FakeLLM(long, CLEAN)
    res = generate_meeting_summary(TX_A, llm_call=llm)
    assert res.method == "llm_repaired" and res.text == CLEAN
    assert "too long" in llm.calls[1][0]


def test_overlong_that_never_shrinks_is_bounded_on_sentence_boundaries():
    long = distinct_text(70)
    res = generate_meeting_summary(TX_A, llm_call=FakeLLM(long))
    assert len(res.text.split()) <= res.plan.hard_max_words
    assert res.text.endswith(".") and not res.violations


def test_llm_failure_degrades_to_label_free_extractive():
    res = generate_meeting_summary(TX_A, llm_call=FakeLLM(RuntimeError("boom")))
    assert res.method == "extractive" and res.text and not LABEL_RE.search(res.text)


def test_empty_or_tiny_source_gives_empty_result():
    assert generate_meeting_summary("Speaker_1: ok", llm_call=FakeLLM(CLEAN)).method == "empty"
    assert generate_meeting_summary("", llm_call=FakeLLM(CLEAN)).text == ""


def test_summaries_come_only_from_their_own_transcript():
    tx_b = "\n".join([
        "Speaker_1: The quarterly budget for the marketing campaign needs approval before the launch date.",
        "Speaker_2: We approved the campaign budget and the launch moves to the first week of March.",
        "Speaker_1: Procurement will confirm the vendor contracts by Friday so the creative work can start.",
    ])

    def echo_llm(system_prompt, user_text, max_tokens):
        return "The meeting covered " + ("the marketing campaign budget" if "campaign" in user_text
                                        else "the diarization feature") + " and what was agreed."

    a = generate_meeting_summary(TX_A, llm_call=echo_llm, meeting_ref="A")
    b = generate_meeting_summary(tx_b, llm_call=echo_llm, meeting_ref="B")
    assert "diarization" in a.text and "campaign" not in a.text
    assert "campaign" in b.text and "diarization" not in b.text


def test_very_long_meeting_is_condensed_in_parts_not_truncated(monkeypatch):
    monkeypatch.setattr(ss, "CLOUD_MAX_TRANSCRIPT_CHARS", 3000)
    monkeypatch.setattr(ss, "_PART_CHARS", 1200)
    lines = [f"Speaker_{i % 3 + 1}: Topic {i} covers the unique subsystem alpha{i} and its rollout plan details."
             for i in range(120)]
    seen = {"notes": 0, "final_input": ""}

    def llm(system_prompt, user_text, max_tokens):
        if "ONE PART of a longer meeting" in system_prompt:
            seen["notes"] += 1
            assert len(user_text) <= 1300                        # each part fits comfortably
            first = re.search(r"alpha(\d+)", user_text).group(1)
            return f"Part starting at subsystem alpha{first} was reviewed."
        seen["final_input"] = user_text
        assert "condensed working notes" in system_prompt
        return CLEAN

    res = generate_meeting_summary("\n".join(lines), llm_call=llm)
    assert seen["notes"] >= 3 and res.text == CLEAN
    assert len(seen["final_input"]) <= 3000
    assert "alpha0" in seen["final_input"] and "alpha119" not in seen["final_input"] or "alpha1" in seen["final_input"]
    assert "Speaker_" not in seen["final_input"]
