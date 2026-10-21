"""Quality metrics tell the truth about a bad transcript (fixture = the real API output of meeting 1648).

Measured BEFORE the fix, for a 30-minute recording whose stored transcript was 259 words (8.5 words/minute):

* ``transcript_quality`` was "Excellent 92.8" while its own issues said "Very little speech was transcribed";
* ``agenda_coverage`` was "Excellent 93.9" next to ``evaluation_metrics.agenda`` 33.3 (2 of 6 topics supported);
* ``summary_quality`` was "Excellent 91.7" with 16.7 % agenda-topic coverage;
* ``overall_quality_score`` was 86.2;
* ``avg_stt_confidence`` was 0.85 - a default written on every segment, never measured;
* speaker time was 23.453 (processing seconds) / 1820.98 (segment span) / 1690.01 (speech) under similar names;
* the same transcript was 259 / 260 words depending on the report.

All reference-free heuristics; none of these is WER/DER/ROUGE or an accuracy claim.
"""

from types import SimpleNamespace

import pytest

from backend.evaluation import quality_scores as qs
from backend.evaluation import reference_free_engine as rfe
from backend.evaluation import runtime_quality as rq
from backend.utils.helpers import count_transcript_words


@pytest.fixture(autouse=True)
def lexical(monkeypatch):
    """No embedding model: every check uses its deterministic lexical path."""
    monkeypatch.setattr(rfe, "_embed_batch", lambda texts: [])
    monkeypatch.setattr(rfe, "_embed_one", lambda text: [])


SENTENCES = [
    "The payment gateway migration moved to the staging cluster last week and the smoke tests passed.",
    "Finance confirmed that the quarterly budget covers the extra storage and the licence renewals.",
    "Security asked for an audit of the access logs before the production cutover happens.",
    "Recruiting has three candidates for the platform team and interviews start on Monday.",
    "The mobile team needs a stable interface contract before building the new screens.",
    "Support reported that response times improved after the queue priorities were changed.",
]


_counter = [0]


def _text(n_words):
    """``n_words`` words of varied, repetition-free text (every sentence carries its own unique tokens)."""
    out = []
    while len(" ".join(out).split()) < n_words:
        _counter[0] += 1
        k = _counter[0]
        base = SENTENCES[k % len(SENTENCES)].rstrip(".")
        out.append(f"{base} for workstream{k} owned by team{k} under ticket{k}.")
    return " ".join(" ".join(out).split()[:n_words]).rstrip(".")


def _one_giant_segment(words=259, duration=1820.5):
    return [{"speaker": "Speaker_1", "text": _text(words) + ".", "start_time": 0.0, "end_time": duration - 0.5,
             "stt_confidence": None}]


def _healthy_segments(minutes=30, words_per_min=120, seg_sec=20.0):
    segs, t = [], 0.0
    per_seg = int(words_per_min * seg_sec / 60)
    while t + seg_sec <= minutes * 60:
        segs.append({"speaker": f"Speaker_{int(t // 200) % 2 + 1}", "text": _text(per_seg) + ".", "start_time": t,
                     "end_time": t + seg_sec - 1, "stt_confidence": None})
        t += seg_sec
    return segs


# ── transcript quality ──────────────────────────────────────────────────────────────────────────────

def test_a_thirty_minute_recording_with_259_words_is_not_excellent():
    r = qs.transcript_quality(_one_giant_segment(), audio_duration_sec=1820.5)
    assert r["signals"]["words_per_minute"] < 10
    assert r["score"] <= 55.0 and r["label"] == "Needs Improvement"
    assert any("Very little speech" in i for i in r["issues"])
    assert any("capped at 55" in i and "words per minute" in i for i in r["issues"])


def test_a_healthy_transcript_is_not_penalised_by_the_new_gate():
    r = qs.transcript_quality(_healthy_segments(), audio_duration_sec=1800.0)
    assert r["signals"]["words_per_minute"] >= 100
    assert r["score"] >= 75.0 and not any("capped" in i for i in r["issues"])


def test_a_slow_but_real_meeting_is_capped_below_good_not_failed():
    segs = _healthy_segments(words_per_min=20)
    r = qs.transcript_quality(segs, audio_duration_sec=1800.0)
    assert 15 <= r["signals"]["words_per_minute"] < 30
    assert r["score"] <= 74.0 and r["label"] in ("Average", "Needs Improvement")


# ── summary quality ──────────────────────────────────────────────────────────────────────────────────

def test_a_summary_that_covers_little_of_the_meeting_cannot_be_excellent():
    transcript = " ".join(
        f"Speaker_1: {s} {s} We will track {t} closely." for s, t in zip(
            SENTENCES, ["payments", "budget", "security", "recruiting", "mobile", "support"]))
    summary = "The team reviewed the payment gateway migration and the staging cluster smoke tests."
    r = qs.summary_quality(summary, transcript, agenda_topics=["Budget planning", "Security audit", "Recruiting plan",
                                                                 "Mobile contract", "Support queues"])
    assert r["components"]["topic_coverage"] is not None
    assert r["components"]["topic_coverage"] < 60 or r["score"] > 74      # the property
    if r["components"]["topic_coverage"] < 60:
        assert r["score"] <= 74.0 and r["label"] != "Excellent"


# ── agenda: one score, consistent with the per-topic check ──────────────────────────────────────────

def test_agenda_coverage_includes_the_strict_topic_support_check():
    transcript = " ".join(SENTENCES)
    topics = ["Payment gateway migration", "Quarterly budget", "Zeppelin procurement", "Underwater basket weaving",
              "Quantum cafeteria roadmap", "Interstellar parking"]
    r = qs.agenda_coverage(topics, transcript)
    assert r["signals"]["supported_topics"] + r["signals"]["unsupported_topics"] == 6
    assert r["signals"]["unsupported_topics"] >= 3
    assert r["score"] < 75.0                                              # 4 of 6 topics unsupported is not "Excellent"
    assert any("no matching statement" in i for i in r["issues"])


def test_the_evaluation_block_reports_the_same_agenda_score_as_the_metrics_block():
    segs = _healthy_segments(minutes=5)
    rows = [{"agenda": "Payment gateway migration", "decision": None}, {"agenda": "Underwater basket weaving",
                                                                         "decision": None}]
    out = rfe.build_evaluation_metrics(
        segments=segs, audio_duration_sec=300.0, language_confidence=None, summary_text="", mom_text="",
        action_items=[], agenda_rows=rows, chat_messages=[], score_overrides={"agenda": 41.0})
    assert out["agenda"]["score"] == 41.0 and out["agenda"]["superseded_by"] == "metrics.agenda_coverage"
    assert "strict_topic_support_score" in out["agenda"]
    assert out["quality_metrics"]["components"]["agenda"] == 41.0


# ── overall quality: measured components only, and a bad transcript cannot be averaged away ─────────

def test_overall_is_the_weighted_mean_of_measured_components_only():
    r = rfe.overall_quality({"transcription": 80.0, "summary": 60.0, "action_items": None, "decisions": None})
    w = rfe._OVERALL_WEIGHTS
    expected = (80.0 * w["transcription"] + 60.0 * w["summary"]) / (w["transcription"] + w["summary"])
    assert r["overall_quality_score"] == pytest.approx(expected, abs=0.01)
    assert set(r["weights_used"]) == {"transcription", "summary"} and r["gates_applied"] == []


def test_a_poor_transcript_caps_the_overall_score():
    r = rfe.overall_quality({"transcription": 50.0, "text_integrity": 100.0, "diarization": 100.0, "summary": 92.0,
                             "mom": 90.0})
    assert r["overall_quality_score"] <= 60.0 and r["gates_applied"]
    assert "cannot exceed transcription + 10" in r["formula"]


def test_a_good_transcript_is_not_capped():
    r = rfe.overall_quality({"transcription": 80.0, "summary": 90.0})
    assert r["gates_applied"] == [] and r["overall_quality_score"] > 80.0


def test_overall_is_none_when_nothing_was_measured():
    assert rfe.overall_quality({"transcription": None, "summary": None})["overall_quality_score"] is None


# ── STT confidence is measured or absent, never a default ───────────────────────────────────────────

def _segs_with_confidence(values):
    return [{"text": "hello there", "start_time": i * 5.0, "end_time": i * 5.0 + 4.0, "stt_confidence": v}
            for i, v in enumerate(values)]


def test_a_constant_legacy_085_is_not_a_confidence_signal():
    out = rq.stt_runtime_metrics(_segs_with_confidence([0.85] * 4), 100.0, 5.0)
    assert out["avg_stt_confidence"] is None and out["confidence_sample_count"] == 0
    assert out["stt_quality_components"]["confidence"] is None


def test_missing_confidence_stays_missing_and_real_values_are_used():
    assert rq.stt_runtime_metrics(_segs_with_confidence([None, None]), 100.0, 5.0)["avg_stt_confidence"] is None
    real = rq.stt_runtime_metrics(_segs_with_confidence([0.7, 0.9]), 100.0, 5.0)
    assert real["avg_stt_confidence"] == pytest.approx(0.8)


# ── speaker time: processing time and talk time are different fields ────────────────────────────────

def _runtime(speech_durations=None, segments=None):
    segs = segments or [{"speaker": "Speaker_1", "text": "hello there everyone", "start_time": "0.000",
                         "end_time": "1820.980"}]
    meeting = SimpleNamespace(duration=1820.48, stt_duration_sec=234.21, speaker_duration_sec=23.453,
                              total_processing_time_sec=278.4, upload_duration_sec=0.5, summary_duration_sec=22.3,
                              mom_duration_sec=0.0, action_items_duration_sec=0.0, file_size_bytes=1,
                              stt_word_count=3, stt_char_count=20, speaker_count=1)
    return rfe.build_runtime_metrics(
        segments=segs, audio_duration_sec=1820.48, stt_duration_sec=234.21, speaker_duration_sec=23.453,
        summary_text="", mom_text="", transcript_text="Speaker_1: hello there everyone",
        agenda_decision_rows=[], action_items=[], meeting=meeting, speech_durations=speech_durations)


def test_speaker_distribution_uses_speech_seconds_when_the_diarization_has_them():
    out = _runtime({"Speaker_1": 1690.01})["speaker"]
    assert out["speaker_distribution"][0]["duration_seconds"] == 1690.01
    assert out["speaker_distribution"][0]["percentage"] == 100.0
    assert out["speech_duration_source"] == "diarization_speech_regions" and out["total_speech_duration_sec"] == 1690.01


def test_speaker_distribution_falls_back_to_segment_spans_and_says_so():
    out = _runtime(None)["speaker"]
    # the segment ends at 1820.98 s but the recording is 1820.48 s: a segment cannot end after the recording does
    assert out["speaker_distribution"][0]["duration_seconds"] == 1820.48
    assert out["speech_duration_source"] == "segment_spans"
    assert out["speech_duration_sec"] is None and out["silence_duration_sec"] is None     # not measured: null, not 0


def test_speech_durations_for_other_speakers_are_not_forced_onto_these_segments():
    out = _runtime({"Speaker_9": 5.0})["speaker"]
    assert out["speech_duration_source"] == "segment_spans"


def test_processing_time_has_its_own_honest_name_and_the_old_key_is_kept():
    out = _runtime({"Speaker_1": 1690.01})
    assert out["speaker"]["diarization_processing_time_sec"] == out["speaker"]["diarization_duration_sec"] == 23.453
    assert out["performance"]["speaker_processing_time_sec"] == out["performance"]["speaker_duration_sec"] == 23.453


# ── one word count ──────────────────────────────────────────────────────────────────────────────────

def test_speaker_labels_are_not_counted_as_words():
    text = "Speaker_1: hello there\nSpeaker_2: ok then\nSpeaker_1: bye"
    assert count_transcript_words(text) == 5
    assert count_transcript_words("") == 0 and count_transcript_words(None) == 0
    assert count_transcript_words("no label here at all") == 5


# ── runtime STT score uses the same words-per-minute gate ───────────────────────────────────────────

def test_the_runtime_stt_score_cannot_be_high_for_a_transcript_of_8_words_per_minute():
    seg = [{"text": _text(259) + ".", "start_time": "0.000", "end_time": "1820.980", "stt_confidence": None}]
    out = rq.stt_runtime_metrics(seg, 1820.5, 234.0)
    assert out["transcript_coverage"] == 100.0                      # the timeline IS spanned ...
    assert out["words_per_minute"] < 10 and out["stt_runtime_quality_score"] == 55.0   # ... but little was said
    assert "words per minute" in out["score_gate"]


def test_the_runtime_stt_score_is_untouched_for_normal_speech():
    segs = [{"text": _text(40) + ".", "start_time": f"{i * 20:.3f}", "end_time": f"{i * 20 + 19:.3f}",
             "stt_confidence": None} for i in range(90)]
    out = rq.stt_runtime_metrics(segs, 1800.0, 200.0)
    assert out["words_per_minute"] >= 100 and out["score_gate"] is None and out["stt_runtime_quality_score"] > 74.0


def test_short_clips_are_not_gated_on_words_per_minute():
    out = rq.stt_runtime_metrics([{"text": "hello there", "start_time": "0.0", "end_time": "5.0"}], 5.0, 1.0)
    assert out["words_per_minute"] is None and out["score_gate"] is None
