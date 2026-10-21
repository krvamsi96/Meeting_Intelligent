"""Tests for backend.evaluation.quality_scores — the three reference-free quality scores.

Most tests run with embeddings disabled (lexical fallback) so they are fast and deterministic; the
tests marked ``semantic`` use the real embedding model and skip when it is unavailable.
"""

import copy
import json

import pytest

from backend.evaluation import quality_scores as qs
from backend.evaluation import reference_free_engine as rfe


@pytest.fixture()
def lexical(monkeypatch):
    """Disable embeddings: every check falls back to its deterministic lexical path."""
    monkeypatch.setattr(rfe, "_embed_batch", lambda texts: [])
    monkeypatch.setattr(rfe, "_embed_one", lambda text: [])


# ── fixtures ────────────────────────────────────────────────────────────────
_SENTENCES = [
    "Thanks everyone for joining, let us start with the status of the payment gateway migration.",
    "The staging cluster was provisioned last week and the first smoke tests passed without errors.",
    "However the database replication lag is still higher than we expected during peak hours.",
    "Our vendor promised a patch by Thursday, so we should not change the cutover date yet.",
    "Finance confirmed that the quarterly budget covers the extra storage and the licence renewals.",
    "I would like two more engineers on the security review because the audit window is short.",
    "Recruiting has three strong candidates and interviews are scheduled for early next week.",
    "Please make sure every new hire gets access to the ticketing system on their first day.",
    "On the customer side, the onboarding emails need to be reviewed by legal before they go out.",
    "Support reported that response times improved noticeably after we changed the queue priorities.",
    "We still have an open question about how long the audit logs must be retained.",
    "Compliance will answer that by Friday and then we can finalise the retention settings.",
    "The mobile team asked for a stable API contract before they start building the new screens.",
    "That sounds reasonable, so let us freeze the contract on Monday and document every endpoint.",
    "Performance testing showed the search service handles four thousand requests per second comfortably.",
    "We should still add a cache in front of it to keep the costs down during the sales campaign.",
    "The design review raised concerns about accessibility on the checkout page for keyboard users.",
    "Priya will fix the focus order and the contrast issues before the next release candidate.",
    "There was a short outage on Tuesday caused by an expired certificate on the internal gateway.",
    "To prevent that, we agreed to add automated expiry alerts and a monthly certificate check.",
    "The analytics dashboard now shows weekly active users, churn and the revenue per account.",
    "Marketing wants a version of it that can be shared with partners without exposing raw data.",
    "Let us prepare a redacted export and review it together before we send anything outside.",
    "Training for the support team is planned for the second week of the month at the main office.",
    "Remote colleagues can join through the video link and receive the recording afterwards.",
    "A risk we should track is the dependency on a single supplier for the hardware sensors.",
    "Procurement is negotiating a second source, and we expect an answer within two weeks.",
    "On documentation, the runbooks are outdated and several commands no longer work as written.",
    "Arun volunteered to rewrite them and to add a troubleshooting section for common failures.",
    "We also need clearer ownership for the on call rotation over the holiday period.",
    "Let us circulate a draft schedule and collect swaps by the end of the week.",
    "Finally, the steering group asked for a short written update every Friday afternoon.",
    "I will send the first update this week and keep it to one page with risks and decisions.",
    "If nothing else is urgent, we can close here and meet again the same time next week.",
    "Great, thank you all, please send me any blockers by email before the end of the day.",
    "Have a good afternoon and talk to you next week.",
]


def make_segments(n=12, start_gap=20.0):
    segs, t = [], 0.0
    for i in range(n):
        text = " ".join(_SENTENCES[(3 * i + k) % len(_SENTENCES)] for k in range(3))
        segs.append({"speaker": f"Speaker_{i % 3 + 1}", "text": text, "start_time": t,
                     "end_time": t + start_gap - 1.0, "stt_confidence": None})
        t += start_gap
    return segs


GOOD = make_segments()
DURATION = 12 * 20.0


def tq(segments=None, **kw):
    kw.setdefault("audio_duration_sec", DURATION)
    return qs.transcript_quality(GOOD if segments is None else segments, **kw)


# ═════════════════════════════ transcript quality ═════════════════════════════
def test_good_transcript_scores_excellent_with_no_issues(lexical):
    r = tq()
    assert r["score"] >= 90 and r["label"] == "Excellent" and r["issues"] == []
    assert r["status"] == "ok" and set(r) >= {"score", "label", "issues", "components", "signals"}


@pytest.mark.parametrize("segments", [[], [{"speaker": "S", "text": "  ", "start_time": 0, "end_time": 1}]])
def test_empty_transcript_is_a_clear_failure(segments):
    r = qs.transcript_quality(segments, audio_duration_sec=60)
    assert r["score"] == 0.0 and r["label"] == "Failed" and r["status"] == "failed"
    assert "empty" in r["issues"][0].lower()


def test_repetition_loops_and_duplicates_are_penalised_and_capped(lexical):
    looped = copy.deepcopy(GOOD)
    for s in looped[:8]:
        s["text"] = "thank you thank you thank you thank you thank you thank you"
    r = tq(looped)
    assert r["score"] <= 55 and any("repetition loop" in i or "duplicate" in i for i in r["issues"])
    dup = [copy.deepcopy(GOOD[0]) for _ in range(12)]
    for i, s in enumerate(dup):
        s["start_time"], s["end_time"] = i * 20.0, i * 20.0 + 19.0
    r2 = tq(dup)
    assert r2["score"] <= 55 and any("duplicates" in i for i in r2["issues"])


def test_broken_sentences_lower_the_score(lexical):
    broken = copy.deepcopy(GOOD)
    for s in broken:
        s["text"] = s["text"].rstrip(".") + " and then we"
    r = tq(broken)
    assert r["score"] < tq()["score"] and any("mid-sentence" in i for i in r["issues"])


def test_abnormal_characters_are_flagged(lexical):
    dirty = copy.deepcopy(GOOD)
    for s in dirty[:6]:
        s["text"] = s["text"] + " \x00\x01 ��" + " ~~~###@@@!!!???" * 3
    r = tq(dirty)
    assert r["score"] < tq()["score"] and any("corrupted" in i or "symbols" in i for i in r["issues"])


def test_timestamp_problems_are_reported(lexical):
    bad = copy.deepcopy(GOOD)
    bad[3]["start_time"], bad[3]["end_time"] = 500, 400          # end before start
    bad[7]["start_time"], bad[7]["end_time"] = 5, 15             # goes back in time
    r = tq(bad)
    assert r["components"]["timestamp_consistency"] < 100
    assert any("timestamps" in i.lower() for i in r["issues"])


def test_timeline_that_does_not_fit_the_media_is_reported_not_scored_as_negative_coverage(lexical):
    r = tq(GOOD, audio_duration_sec=30.0)
    assert any("does not match the media" in i for i in r["issues"])
    assert r["components"]["speech_coverage"] is None
    assert all("-" not in i.split("%")[0][-6:] for i in r["issues"] if "%" in i)     # no negative percentages
    assert r["score"] < tq()["score"]


def test_low_coverage_of_the_recording_is_penalised(lexical):
    r = tq(GOOD[:1], audio_duration_sec=3600.0)
    assert r["score"] <= 55 and any("covers only" in i for i in r["issues"])


def test_constant_placeholder_confidence_is_not_used_but_a_real_signal_is(lexical):
    const = copy.deepcopy(GOOD)
    for s in const:
        s["stt_confidence"] = 0.85
    r = tq(const)
    assert r["components"]["transcription_confidence"] is None
    assert r["signals"]["stt_confidence_available"] is False
    varied = copy.deepcopy(GOOD)
    for i, s in enumerate(varied):
        s["stt_confidence"] = 0.55 + 0.03 * i
    r2 = tq(varied)
    assert r2["components"]["transcription_confidence"] is not None
    low = copy.deepcopy(GOOD)
    for i, s in enumerate(low):
        s["stt_confidence"] = 0.30 + 0.02 * i
    assert tq(low)["score"] < r2["score"]


def test_documents_skip_timestamp_coverage_and_confidence_signals(lexical):
    doc = [{"speaker": f"Document Section {i}", "text": s["text"], "start_time": f"{i:02d}:00",
            "end_time": f"{i + 1:02d}:00", "stt_confidence": None} for i, s in enumerate(GOOD)]
    r = qs.transcript_quality(doc, is_document=True)
    assert r["components"]["timestamp_consistency"] is None and r["components"]["speech_coverage"] is None
    assert r["score"] >= 85


def test_untranslated_non_english_text_is_flagged(lexical):
    hindi = copy.deepcopy(GOOD)
    for s in hindi:
        s["text"] = "बैठक में परियोजना की समीक्षा की गई और अगले कदम तय किए गए हैं सभी ने सहमति दी।"
    r = tq(hindi)
    assert r["score"] <= 50 and any("English" in i for i in r["issues"])


def test_score_falls_as_the_transcript_degrades(lexical):
    s0 = tq()["score"]
    half_empty = copy.deepcopy(GOOD)
    for s in half_empty[::2]:
        s["text"] = ""
    s1 = tq(half_empty)["score"]
    all_empty_but_one = copy.deepcopy(GOOD)
    for s in all_empty_but_one[1:]:
        s["text"] = ""
    s2 = tq(all_empty_but_one)["score"]
    assert s0 > s1 > s2 >= 0


def test_very_short_transcripts_are_capped(lexical):
    tiny = [{"speaker": "A", "text": "Yes we agreed to the plan.", "start_time": 0, "end_time": 3}]
    assert qs.transcript_quality(tiny)["score"] <= 30
    short = [{"speaker": "A", "text": " ".join(["word"] * 3 + ["we agreed on the plan."] * 4),
              "start_time": 0, "end_time": 10}]
    assert qs.transcript_quality(short)["score"] <= 70


# ═════════════════════════════ summary quality ═════════════════════════════════
TX = "\n".join([
    "Speaker_1: Today we review the payment gateway migration to the new cluster and the budget of 15000 dollars.",
    "Speaker_2: The migration must finish before the end of March because the old vendor contract expires then.",
    "Speaker_1: We decided to hire two engineers for the security review and to start testing next Monday.",
    "Speaker_2: Acme will provide the staging environment and the deployment checklist by Friday.",
    "Speaker_1: The main risk is the database latency during the cutover so we agreed on a rollback plan.",
    "Speaker_2: Fine. Let us also schedule weekly reporting to the steering group about progress and risks.",
] * 3)

GOOD_SUMMARY = ("The meeting reviewed the payment gateway migration and its 15000 dollar budget, which must finish before "
                "the vendor contract expires in March. The team agreed to hire two engineers for the security review, "
                "prepare a rollback plan for the database cutover, and report weekly to the steering group.")


def sq(summary, transcript=TX, **kw):
    return qs.summary_quality(summary, transcript, **kw)


def test_good_summary_scores_high_without_issues(lexical):
    r = sq(GOOD_SUMMARY)
    assert r["score"] >= 85 and r["label"] in ("Excellent", "Good")
    assert not any("speaker" in i.lower() or "timestamp" in i.lower() for i in r["issues"])


def test_speaker_labels_in_the_summary_reduce_the_score_and_are_reported(lexical):
    labelled = ("Speaker_1 said the payment gateway migration has a 15000 dollar budget. Speaker_2 added that it must "
                "finish before March. Speaker_1 agreed to hire two engineers and Speaker_2 asked for a rollback plan.")
    clean = ("The payment gateway migration has a 15000 dollar budget and must finish before March. The team agreed to hire "
             "two engineers and to prepare a rollback plan.")
    a, b = sq(labelled), sq(clean)
    assert a["score"] < b["score"] - 20
    assert any("speaker labels" in i.lower() for i in a["issues"]) and "speaker_label" in a["signals"]["rule_violations"]


@pytest.mark.parametrize("label", ["Speaker_0", "Speaker_1", "Speaker_2", "SPEAKER_00", "SPEAKER_01", "speaker_2"])
def test_each_speaker_label_format_is_caught(lexical, label):
    text = f"{label} explained that the payment gateway migration must finish before March and needs two engineers."
    r = sq(text)
    assert "speaker_label" in r["signals"]["rule_violations"] and any("speaker" in i.lower() for i in r["issues"])


def test_timestamps_and_transcript_formatting_are_penalised(lexical):
    stamped = "[00:12] The payment gateway migration must finish before March. At 00:45:10 the budget of 15000 dollars was set."
    r = sq(stamped)
    assert "timestamp" in r["signals"]["rule_violations"] and any("timestamp" in i.lower() for i in r["issues"])
    dialogue = "Speaker_1: the migration must finish before March.\nSpeaker_2: we agreed to hire two engineers."
    r2 = sq(dialogue)
    assert r2["score"] < 60 and any("transcript" in i.lower() or "dialogue" in i.lower() or "speaker" in i.lower() for i in r2["issues"])


def test_raw_transcript_as_summary_scores_poorly(lexical):
    raw = "\n".join(TX.splitlines()[:6])
    r = sq(raw)
    assert r["score"] < 55 and r["signals"]["rule_violations"]


def test_speaker_by_speaker_account_is_flagged(lexical):
    text = ("Ravi said the migration must finish before March. Priya mentioned the budget of 15000 dollars. "
            "Arun suggested hiring two engineers. Meena asked about the rollback plan.")
    r = sq(text)
    assert "speaker_by_speaker" in r["signals"]["rule_violations"]


def test_invented_numbers_and_names_are_reported_as_hallucination_indicators(lexical):
    made_up = ("The meeting reviewed the payment gateway migration and its 90000 dollar budget, run by Globex Corporation, "
               "which must finish before March. The team agreed to hire two engineers for the security review.")
    a, b = sq(made_up), sq(GOOD_SUMMARY)
    assert a["score"] < b["score"]
    assert any("do not appear in the transcript" in i for i in a["issues"])
    assert "90000" in " ".join(a["issues"]) and "Globex" in " ".join(a["issues"])


def test_summary_of_a_different_meeting_scores_low(lexical):
    other = ("The marketing team planned the autumn campaign, approved the creative budget and scheduled the product launch "
             "event with the regional partners for October.")
    assert sq(other)["score"] < sq(GOOD_SUMMARY)["score"] - 30


def test_summary_that_is_far_too_long_is_penalised(lexical):
    bloated = " ".join(GOOD_SUMMARY for _ in range(6))
    r = sq(bloated)
    assert any("too long" in i.lower() for i in r["issues"]) and r["score"] < sq(GOOD_SUMMARY)["score"] - 10


def test_redundant_summary_is_flagged(lexical):
    repeated = (GOOD_SUMMARY + " The migration of the payment gateway must finish before March and needs a budget of 15000 dollars. "
                "The team agreed to hire two engineers for the security review and prepare a rollback plan.")
    r = sq(repeated)
    assert r["components"]["redundancy"] < sq(GOOD_SUMMARY)["components"]["redundancy"]


def test_markdown_structure_is_penalised(lexical):
    md = "### Summary\n\n- The migration must finish before March.\n- **Budget** is 15000 dollars.\n- Hire two engineers."
    r = sq(md)
    assert "markdown" in r["signals"]["rule_violations"]


def test_missing_summary_and_missing_transcript_are_reported_distinctly(lexical):
    assert sq("")["label"] == "Failed" and sq("")["score"] == 0.0
    assert sq("Not Available - transcript did not contain enough content to analyse.")["label"] == "Failed"
    na = sq(GOOD_SUMMARY, transcript="")
    assert na["score"] is None and na["label"] == "Not Available" and na["reason"]


def test_speaker_labels_never_count_as_supported_by_the_transcript(lexical):
    """The transcript lines contain 'Speaker_1:' — a leaked label must not be rewarded for matching them."""
    with_label = GOOD_SUMMARY.replace("The meeting", "Speaker_1 and the meeting", 1)
    assert sq(with_label)["score"] < sq(GOOD_SUMMARY)["score"]


def test_summary_scores_are_dynamic_across_meetings(lexical):
    scores = {
        sq(GOOD_SUMMARY)["score"],
        sq(GOOD_SUMMARY[:120] + ".")["score"],
        sq(GOOD_SUMMARY + " Speaker_2 disagreed.")["score"],
        sq("The team met.")["score"],
        sq("The marketing team planned the autumn campaign for October.")["score"],
    }
    assert len(scores) >= 4


def test_semantic_path_ranks_faithful_above_unrelated_summaries():
    if not rfe._embed_batch(["probe sentence"]):
        pytest.skip("embedding model unavailable")
    faithful, unrelated = sq(GOOD_SUMMARY), sq("The company celebrated its anniversary with a picnic and awards ceremony.")
    assert faithful["signals"]["semantic_checks"] is True
    assert faithful["score"] > unrelated["score"] + 25


# ═════════════════════════════ RAG groundedness ═════════════════════════════════
CTX = ["The payment gateway migration must finish before the end of March because the old vendor contract expires then.",
       "The budget for the migration is 15000 dollars and two engineers will be hired for the security review."]
Q = "When must the payment gateway migration finish and what is the budget?"


def rag(answer, ctx=CTX, sources=(), meeting_id=7, question=Q):
    return qs.rag_answer_groundedness(question=question, answer=answer, context_chunks=ctx,
                                      source_chunks=list(sources), meeting_id=meeting_id)


def test_supported_answer_scores_high(lexical):
    r = rag("The migration must finish before the end of March, and the budget is 15000 dollars.")
    assert r["score"] >= 80 and r["answer_type"] == "answer" and not any("not supported" in i for i in r["issues"])


def test_unsupported_claims_and_invented_details_lower_the_score(lexical):
    good = rag("The migration must finish before the end of March, and the budget is 15000 dollars.")
    bad = rag("The migration finishes in June and the budget is 75000 dollars, approved by Globex Corporation.")
    assert bad["score"] < good["score"] - 25
    assert any("not in the retrieved context" in i for i in bad["issues"])


def test_abstention_is_grounded_behaviour(lexical):
    r = rag("The meeting does not mention who the new hires report to.")
    assert r["answer_type"] == "abstention" and r["score"] >= 90


def test_answer_with_no_context_is_ungrounded(lexical):
    r = rag("The migration finishes in June and costs 75000 dollars.", ctx=[])
    assert r["score"] == 0.0 and r["answer_type"] == "ungrounded" and any("No retrieved context" in i for i in r["issues"])


def test_chunks_from_another_meeting_are_excluded_and_cap_the_score(lexical):
    foreign_text = "The marketing budget for the autumn campaign is 90000 dollars and the launch is in October."
    sources = [{"meeting_id": 7, "text": CTX[0], "similarity_score": 0.8},
               {"meeting_id": 99, "text": foreign_text, "similarity_score": 0.7}]
    r = rag("The marketing budget for the autumn campaign is 90000 dollars.", ctx=CTX + [foreign_text], sources=sources)
    assert r["signals"]["foreign_meeting_chunks_excluded"] == 1
    assert r["score"] <= 40 and any("different meeting" in i for i in r["issues"])
    same = rag("The migration must finish before March.", ctx=CTX + [foreign_text], sources=sources)
    assert same["signals"]["foreign_meeting_chunks_excluded"] == 1 and any("different meeting" in i for i in same["issues"])


def test_only_this_meetings_chunks_are_used_as_support(lexical):
    foreign = "Project Falcon launches on 12 May with a 500000 dollar budget owned by the Zurich team."
    r = rag("Project Falcon launches on 12 May with a 500000 dollar budget.", ctx=[CTX[0], foreign],
            sources=[{"meeting_id": 5, "text": foreign}], meeting_id=7,
            question="What is the plan for Project Falcon?")
    assert r["score"] <= 40          # the foreign chunk cannot support this answer


def test_rag_not_applicable_matches_the_documented_shape():
    r = qs.aggregate_rag_groundedness([])
    assert r == {"score": None, "label": "Not Applicable", "reason": "RAG was not used for this meeting"}


def test_legacy_answers_without_stored_evaluations_are_not_scored_or_faked():
    r = qs.aggregate_rag_groundedness([], assistant_answer_count=5)
    assert r["score"] is None and r["label"] == "Not Evaluated" and "5 chatbot answer" in r["reason"]


def test_aggregate_is_the_mean_of_scored_answers(lexical):
    evals = [rag("The migration must finish before the end of March, and the budget is 15000 dollars."),
             rag("The migration finishes in June and costs 75000 dollars."),
             rag("The meeting does not mention who the new hires report to.")]
    agg = qs.aggregate_rag_groundedness(evals)
    assert agg["evaluated_answers"] == 3 and agg["abstained_answers"] == 1
    assert agg["score"] == pytest.approx(sum(e["score"] for e in evals) / 3, abs=0.1)
    assert agg["label"] == qs.score_label(agg["score"]) and agg["context_isolation"]["isolated"] is True


# ═════════════════════════════ the assembled block ═══════════════════════════════
def _walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield str(k).lower()
            yield from _walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_keys(v)


def test_metrics_block_shape_and_no_ground_truth_metrics(lexical):
    block = qs.build_quality_metrics(
        segments=GOOD, summary_text=GOOD_SUMMARY, transcript_text=TX, agenda_topics=["payment gateway migration"],
        rag_evaluations=[], assistant_answer_count=0, audio_duration_sec=DURATION)
    assert block["evaluation_type"] == "reference_free" and block["ground_truth_available"] is False
    for name in ("transcript_quality", "summary_quality"):
        assert isinstance(block[name]["score"], float) and block[name]["label"] and isinstance(block[name]["issues"], list)
    assert block["rag_groundedness"] == {"score": None, "label": "Not Applicable", "reason": "RAG was not used for this meeting"}
    forbidden = {"wer", "cer", "der", "rouge", "rouge1", "rouge2", "rougel", "f1", "bertscore"}
    assert forbidden.isdisjoint(set(_walk_keys(block)))
    json.dumps(block)                                              # must be JSON-serialisable for json_data


def test_one_failing_metric_never_removes_the_others(monkeypatch, lexical):
    monkeypatch.setattr(qs, "summary_quality", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    block = qs.build_quality_metrics(segments=GOOD, summary_text=GOOD_SUMMARY, transcript_text=TX, agenda_topics=[],
                                     rag_evaluations=[], assistant_answer_count=0, audio_duration_sec=DURATION)
    assert block["summary_quality"]["label"] == "Error" and block["summary_quality"]["score"] is None
    assert block["transcript_quality"]["score"] is not None


@pytest.mark.parametrize("score,label", [(100, "Excellent"), (90, "Excellent"), (89.9, "Good"), (75, "Good"),
                                         (74.9, "Average"), (60, "Average"), (59.9, "Needs Improvement"), (0, "Needs Improvement")])
def test_label_bands(score, label):
    assert qs.score_label(score) == label


def test_legacy_evaluation_engine_reports_the_same_value_for_the_three_headline_metrics(lexical):
    """One value per metric: evaluation_metrics.* takes its score from the new computation."""
    common = dict(segments=GOOD, audio_duration_sec=DURATION, language_confidence=None, summary_text=GOOD_SUMMARY,
                  mom_text="", action_items=[], agenda_rows=[], chat_messages=[], meeting_id=1)
    base = rfe.build_evaluation_metrics(**common)
    ev = rfe.build_evaluation_metrics(**common, score_overrides={"transcription": 12.5, "summary": 34.5, "rag": None})
    assert ev["transcription"]["score"] == 12.5 and ev["transcription"]["superseded_by"] == "metrics.transcript_quality"
    assert ev["summary"]["score"] == 34.5 and ev["summary"]["superseded_by"] == "metrics.summary_quality"
    assert ev["rag"]["score"] is None
    assert ev["quality_metrics"]["components"]["transcription"] == 12.5 and ev["quality_metrics"]["components"]["summary"] == 34.5
    assert ev["quality_metrics"]["overall_quality_score"] != base["quality_metrics"]["overall_quality_score"]
    assert "superseded_by" not in base["transcription"]          # untouched when no override is supplied


def test_low_language_detection_confidence_alone_is_not_an_issue_on_a_clean_english_transcript(lexical):
    r = tq(language_confidence=0.33)
    assert r["issues"] == [] and r["score"] >= 90                     # weak hint only; the text itself is fine
    assert r["signals"]["language_detection_confidence"] == 0.33
    mixed = copy.deepcopy(GOOD)
    for s in mixed[:6]:
        s["text"] = s["text"] + " बैठक में परियोजना की समीक्षा की गई और अगले कदम तय किए गए।"
    r2 = tq(mixed, language_confidence=0.33)
    assert any("Language detection confidence is low" in i for i in r2["issues"])
