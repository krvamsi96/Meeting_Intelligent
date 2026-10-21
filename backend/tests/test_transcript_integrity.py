"""The transcript is complete, ordered and never silently cut (upload -> chunks -> AIRAWAT -> correction -> segments).

Defects pinned (all measured on real meetings, not hypothesised):

* meeting 1648: STT returned 21,504 chars / 3,494 words; the LLM correction answered with the first 1,673 chars
  ("Corrected minor punctuation and grammar errors"), scored 0.73 >= 0.70 and REPLACED the transcript - 92 % of the
  meeting gone while every stage said SUCCESS;
* one speaker + 73 diarization turns became ONE 21,000-character segment;
* a chunk ffmpeg could not cut was skipped silently, the rest was transcribed, the meeting finished as SUCCESS;
* a legitimate repeated word at a chunk boundary was trimmed as an "overlap duplicate" even with overlap disabled;
* a fabricated stt_confidence of 0.85 was stored on every segment.

No network, no database.
"""

import json
import os
from types import SimpleNamespace

import pytest
import requests

from backend.services import diarization_service as ds
from backend.services import transcript_correction_service as tcs
from backend.services import transcription_service as ts
from backend.services import translation_service as tr
from backend.services.model_router import model_router
from backend.utils.exceptions import AirawatError


def _sentences(n_words, per_sentence=12):
    words = [f"w{i}" for i in range(n_words)]
    out = []
    for i in range(0, n_words, per_sentence):
        chunk = words[i:i + per_sentence]
        chunk[-1] += "."
        out.extend(chunk)
    return out


LONG_WORDS = _sentences(600)
LONG_WORDS[0] = "Opening"          # starts upper-case: grammar normalisation is then a no-op
LONG_TEXT = " ".join(LONG_WORDS)


# ═════════════════════════════════════════════════════════════════════════════
# 1. LLM correction can never replace the transcript with less of it
# ═════════════════════════════════════════════════════════════════════════════
def test_a_truncated_rewrite_is_not_a_correction():
    ok, why = tcs.correction_preserves_content(LONG_TEXT, " ".join(LONG_WORDS[:45]))
    assert not ok and "word count changed" in why


def test_punctuation_casing_and_a_spelling_fix_are_a_correction():
    ok, _ = tcs.correction_preserves_content("i think the fast api server is ready", "I think the FastAPI server is ready.")
    assert ok
    ok, _ = tcs.correction_preserves_content("okay so yeah", "Okay, so yeah.")
    assert ok


def test_an_empty_or_unrelated_candidate_is_rejected_but_an_empty_original_is_fine():
    assert not tcs.correction_preserves_content("some words here please", "")[0]
    same_length_other_words = "alpha beta gamma delta epsilon zeta eta theta"
    assert not tcs.correction_preserves_content("one two three four five six seven eight", same_length_other_words)[0]
    assert tcs.correction_preserves_content("", "anything")[0]


def _llm_answering(payload):
    return SimpleNamespace(invoke=lambda messages: SimpleNamespace(content=json.dumps(payload)))


def _correct(monkeypatch, segments, llm_payload):
    monkeypatch.setattr(tcs.settings, "TRANSCRIPT_CORRECTION_ENABLED", True)
    monkeypatch.setattr(model_router, "get_llm", lambda **kw: _llm_answering(llm_payload))
    return tcs.correct_and_normalize_transcript(segments, meeting_title="Planning")


def test_a_valid_but_truncated_llm_answer_leaves_the_raw_transcript_intact(monkeypatch):
    """The exact failure of meeting 1648, with a valid-JSON answer that is 7.5 % of the segment."""
    seg = {"speaker": "Speaker_1", "start_time": "0.000", "end_time": "1820.980", "text": LONG_TEXT, "stt_confidence": None}
    payload = {"segments": [{"index": 0, "corrected_text": " ".join(LONG_WORDS[:45]), "confidence": 0.95,
                             "reason": "Corrected minor punctuation and grammar errors", "correction_applied": True}]}
    out = _correct(monkeypatch, [seg], payload)
    assert out[0]["text"] == LONG_TEXT
    assert len(out[0]["text"].split()) == 600
    assert out[0]["raw_text"] == LONG_TEXT
    assert out[0]["correction_applied"] is False
    assert out[0]["correction_reason"].startswith("llm_correction_rejected")


def test_a_real_correction_is_still_applied(monkeypatch):
    seg = {"speaker": "Speaker_1", "start_time": "0.000", "end_time": "5.000",
           "text": "i think the fast api server is ready.", "stt_confidence": None}
    payload = {"segments": [{"index": 0, "corrected_text": "I think the FastAPI server is ready.", "confidence": 0.95,
                             "reason": "product name", "correction_applied": True}]}
    out = _correct(monkeypatch, [seg], payload)
    assert out[0]["text"] == "I think the FastAPI server is ready."
    assert out[0]["raw_text"] == "i think the fast api server is ready."


def test_a_missing_stt_confidence_stays_missing(monkeypatch):
    seg = {"speaker": "Speaker_1", "start_time": "0.000", "end_time": "5.000", "text": "Hello there everyone."}
    out = _correct(monkeypatch, [seg], {"segments": []})
    assert out[0]["stt_confidence"] is None                      # no invented 0.85
    seg_with = dict(seg, stt_confidence=0.62)
    assert _correct(monkeypatch, [seg_with], {"segments": []})[0]["stt_confidence"] == 0.62


def test_a_failing_correction_falls_back_to_the_raw_transcript(monkeypatch):
    monkeypatch.setattr(tcs.settings, "TRANSCRIPT_CORRECTION_ENABLED", True)
    monkeypatch.setattr(model_router, "get_llm", lambda **kw: (_ for _ in ()).throw(RuntimeError("AIRAWAT down")))
    seg = {"speaker": "Speaker_1", "start_time": "0.000", "end_time": "5.000", "text": "Hello there everyone."}
    out = tcs.correct_and_normalize_transcript([seg])
    assert out[0]["text"] == "Hello there everyone."


# ═════════════════════════════════════════════════════════════════════════════
# 2. Translation: a cut-off answer is a failure, not a translation
# ═════════════════════════════════════════════════════════════════════════════
def test_a_truncated_translation_is_reported_as_failed_not_stored(monkeypatch):
    source = " ".join(f"word{i}" for i in range(120))
    monkeypatch.setattr(tr, "_get_llm", lambda temperature=0.0: _llm_answering(
        {"translations": [{"i": 0, "en": "The team met and talked."}]}))
    errors = []
    assert tr._translate_batch([source], "Hindi", errors=errors) is None
    assert errors and "cut off" in errors[0]


def test_a_full_translation_passes_the_completeness_check(monkeypatch):
    source = " ".join(f"word{i}" for i in range(120))
    english = " ".join(f"term{i}" for i in range(110))
    monkeypatch.setattr(tr, "_get_llm", lambda temperature=0.0: _llm_answering({"translations": [{"i": 0, "en": english}]}))
    assert tr._translate_batch([source], "Hindi") == [english]


def test_short_utterances_are_not_subject_to_the_completeness_check(monkeypatch):
    monkeypatch.setattr(tr, "_get_llm", lambda temperature=0.0: _llm_answering({"translations": [{"i": 0, "en": "Yes."}]}))
    assert tr._translate_batch(["haan ji bilkul theek hai"], "Hindi") == ["Yes."]


# ═════════════════════════════════════════════════════════════════════════════
# 3. One speaker, many diarization turns: bounded segments, same text, right order
# ═════════════════════════════════════════════════════════════════════════════
def _timed(words, step=0.3):
    return [{"word": w, "start": round(i * step, 3), "end": round(i * step + step, 3)} for i, w in enumerate(words)]


@pytest.fixture()
def segment_limits(monkeypatch):
    monkeypatch.setattr(ds.settings, "TRANSCRIPT_MAX_SEGMENT_SEC", 45.0)
    monkeypatch.setattr(ds.settings, "TRANSCRIPT_MAX_SEGMENT_WORDS", 120)


def test_a_one_speaker_meeting_is_not_one_giant_segment(segment_limits):
    words = _timed(LONG_WORDS)                                       # 600 words, 180 s, one speaker, 73 turns
    turns = [{"start": 2.5 * i, "end": 2.5 * i + 2.4, "speaker": "Speaker_1"} for i in range(73)]
    segs = ds.align_words_to_nemo(words, turns)
    assert len(segs) >= 5
    assert " ".join(s["text"] for s in segs) == LONG_TEXT           # nothing lost, nothing added, same order
    assert all(s["speaker"] == "Speaker_1" for s in segs)
    starts = [float(s["start_time"]) for s in segs]
    assert starts == sorted(starts)
    assert all(s["text"].rstrip().endswith(".") for s in segs)      # cut only at sentence ends
    assert max(len(s["text"].split()) for s in segs) <= 120 + 12    # bounded by the limit + one sentence
    assert all("_hard_break" not in s for s in segs)


def test_the_no_turns_fallback_is_bounded_too(segment_limits):
    segs = ds.align_words_to_nemo(_timed(LONG_WORDS), [])
    assert len(segs) >= 5 and " ".join(s["text"] for s in segs) == LONG_TEXT


def test_unpunctuated_speech_is_still_cut_and_loses_nothing(segment_limits):
    words = [f"u{i}" for i in range(700)]
    segs = ds.align_words_to_nemo(_timed(words), [{"start": 0.0, "end": 300.0, "speaker": "Speaker_1"}])
    assert len(segs) >= 3 and " ".join(s["text"] for s in segs).split() == words
    assert max(len(s["text"].split()) for s in segs) <= 2 * 120


def test_short_meetings_and_speaker_changes_keep_their_segments(segment_limits):
    words = _timed(["Hello", "there.", "Hi", "back."])
    turns = [{"start": 0.0, "end": 0.6, "speaker": "Speaker_1"}, {"start": 0.6, "end": 1.2, "speaker": "Speaker_2"}]
    segs = ds.align_words_to_nemo(words, turns)
    assert [(s["speaker"], s["text"]) for s in segs] == [("Speaker_1", "Hello there."), ("Speaker_2", "Hi back.")]


# ═════════════════════════════════════════════════════════════════════════════
# 4. Chunking never drops audio silently
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def cut_audio(monkeypatch, tmp_path):
    """split_audio_into_chunks with a fake ffmpeg. ``fail`` = {chunk_index: number_of_failures_before_success}."""
    wav = tmp_path / "meeting.wav"
    wav.write_bytes(b"RIFF")
    monkeypatch.setattr(ts, "get_audio_info", lambda p: {"duration": 1800.0, "size_bytes": 57_600_000})
    state = {"fail": {}, "seen": {}}

    def fake_run(cmd, **kw):
        out = cmd[-1]
        idx = int(os.path.basename(out).rsplit("_chunk_", 1)[1].split(".")[0])
        state["seen"][idx] = state["seen"].get(idx, 0) + 1
        if state["seen"][idx] <= state["fail"].get(idx, 0):
            return SimpleNamespace(returncode=1)
        with open(out, "wb") as fh:
            fh.write(b"\0" * 1000)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(ts.subprocess, "run", fake_run)
    return SimpleNamespace(path=str(wav), state=state)


def test_every_chunk_is_cut_in_order(cut_audio):
    chunks = ts.split_audio_into_chunks(cut_audio.path)
    assert len(chunks) == 4
    assert [c[1] for c in chunks] == sorted(c[1] for c in chunks) and chunks[0][2] is True


def test_a_chunk_that_cannot_be_cut_fails_the_transcription_instead_of_being_skipped(cut_audio):
    cut_audio.state["fail"] = {2: 99}
    with pytest.raises(RuntimeError) as err:
        ts.split_audio_into_chunks(cut_audio.path)
    assert "refusing to transcribe an incomplete recording" in str(err.value) and "1 of 4" in str(err.value)


def test_a_transient_ffmpeg_failure_is_retried_once_and_nothing_is_lost(cut_audio):
    cut_audio.state["fail"] = {3: 1}                                # the LAST chunk fails once
    assert len(ts.split_audio_into_chunks(cut_audio.path)) == 4


# ═════════════════════════════════════════════════════════════════════════════
# 5. Merging chunks: ordered, no duplicates, repeated words kept
# ═════════════════════════════════════════════════════════════════════════════
@pytest.fixture()
def stt(monkeypatch, tmp_path):
    """transcribe_audio_airawat over scripted per-chunk AIRAWAT answers (``stt.texts``)."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"x")
    ctx = SimpleNamespace(texts=[], audio=str(audio), calls=[])
    monkeypatch.setattr(model_router, "_get_dynamic_config", lambda: {"airawat": {"api_key": "k"}})

    def split(path):
        files = []
        for i in range(len(ctx.texts)):
            f = tmp_path / f"c{i}.wav"
            f.write_bytes(b"x")
            files.append((str(f), i * 450.0, i == 0))
        return files

    def one_chunk(path, idx, total, language=None, vocabulary_hint=None):
        ctx.calls.append(idx)
        value = ctx.texts[idx]
        if isinstance(value, Exception):
            raise value
        return {"text": value}

    monkeypatch.setattr(ts, "split_audio_into_chunks", split)
    monkeypatch.setattr(ts, "get_audio_info", lambda p: {"duration": 450.0})
    monkeypatch.setattr(ts, "transcribe_single_chunk", one_chunk)
    return ctx


def test_chunk_texts_are_joined_in_order(stt):
    stt.texts = ["First chunk talks about the budget.", "Second chunk covers hiring.", "Third chunk closes the meeting."]
    out = ts.transcribe_audio_airawat(stt.audio, language="en")
    assert out["text"] == "First chunk talks about the budget. Second chunk covers hiring. Third chunk closes the meeting."
    assert stt.calls == [0, 1, 2]
    starts = [w["start"] for w in out["words"]]
    assert starts == sorted(starts) and starts[0] == 0.0


def test_a_long_recording_keeps_every_chunk_once_and_in_order(stt):
    stt.texts = [f"Chunk{i} alpha{i} beta{i} gamma{i} delta{i}." for i in range(60)]
    out = ts.transcribe_audio_airawat(stt.audio, language="en")
    words = out["text"].split()
    assert len(words) == 60 * 5 and len(out["words"]) == 60 * 5
    assert [w for w in words if w.startswith("Chunk")] == [f"Chunk{i}" for i in range(60)]
    assert len(set(words)) == len(words)                                 # no duplicated word anywhere


def test_the_overlap_duplicate_is_trimmed_when_chunks_overlap(stt, monkeypatch):
    monkeypatch.setattr(ts, "CHUNK_OVERLAP_SEC", 0.5)
    stt.texts = ["We agreed on the plan", "the plan today is final"]
    assert ts.transcribe_audio_airawat(stt.audio, language="en")["text"] == "We agreed on the plan today is final"


def test_a_legitimate_repeat_is_kept_when_there_is_no_overlap(stt, monkeypatch):
    monkeypatch.setattr(ts, "CHUNK_OVERLAP_SEC", 0.0)
    stt.texts = ["We said no no", "no no way at all"]
    assert ts.transcribe_audio_airawat(stt.audio, language="en")["text"] == "We said no no no no way at all"


def test_the_trim_never_exceeds_what_fits_in_the_overlap_window(stt, monkeypatch):
    monkeypatch.setattr(ts, "CHUNK_OVERLAP_SEC", 0.5)                  # budget: 3 words
    assert ts.overlap_word_budget(0.5) == 3 and ts.overlap_word_budget(0) == 1
    stt.texts = ["one two three four", "one two three four five"]      # 4 identical words cannot be overlap
    text = ts.transcribe_audio_airawat(stt.audio, language="en")["text"]
    assert text.split().count("four") == 2


def test_an_empty_chunk_answer_fails_the_whole_transcription(stt):
    stt.texts = ["First part is fine.", ts.EmptyCloudTranscription("no text"), "Third part."]
    with pytest.raises(AirawatError) as err:
        ts.transcribe_audio_airawat(stt.audio, language="en")
    assert "returned no text for chunk 2/3" in str(err.value)


def test_a_chunk_that_times_out_fails_the_transcription_instead_of_saving_a_partial_one(stt):
    stt.texts = ["First part is fine.", AirawatError("AIRAWAT Whisper did not answer chunk 2/3"), "Third part."]
    with pytest.raises(AirawatError):
        ts.transcribe_audio_airawat(stt.audio, language="en")
    assert stt.calls == [0, 1]                                         # chunk 3 is never attempted after the failure


# ═════════════════════════════════════════════════════════════════════════════
# 6. Retries are idempotent (a retried chunk appears once)
# ═════════════════════════════════════════════════════════════════════════════
def test_a_chunk_retried_after_a_timeout_is_merged_exactly_once(monkeypatch, tmp_path):
    chunk = tmp_path / "c.wav"
    chunk.write_bytes(b"\0" * 2048)
    monkeypatch.setattr(model_router, "_get_dynamic_config",
                        lambda: {"airawat": {"api_key": "k", "retry_count": 3, "whisper_endpoint": "https://example.invalid/stt"}})
    monkeypatch.setattr(ts.time, "sleep", lambda s: None)
    attempts = []

    def fake_post(url, headers=None, data=None, verify=None, timeout=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise requests.exceptions.ReadTimeout("no answer")
        return SimpleNamespace(status_code=200, text='{"text": "hello world"}', raise_for_status=lambda: None,
                               json=lambda: {"text": "hello world"})

    monkeypatch.setattr(ts.requests, "post", fake_post)
    monkeypatch.setattr(ts, "split_audio_into_chunks", lambda p: [(str(chunk), 0.0, True)])
    monkeypatch.setattr(ts, "get_audio_info", lambda p: {"duration": 5.0})
    out = ts.transcribe_audio_airawat(str(chunk), language="en")
    assert len(attempts) == 2 and out["text"] == "hello world" and len(out["words"]) == 2


def test_a_rejected_rewrite_stays_in_the_audit_trail_when_the_grammar_pass_also_ran(monkeypatch):
    text = " ".join(["we", "reviewed", "the", "release", "plan", "and", "assigned", "owners", "for", "each",
                     "open", "task", "before", "the", "next", "checkpoint", "on", "friday", "afternoon"])
    seg = {"speaker": "Speaker_1", "start_time": "0.000", "end_time": "30.000", "text": text, "stt_confidence": None}
    payload = {"segments": [{"index": 0, "corrected_text": "We reviewed the release plan.", "confidence": 0.95,
                             "reason": "tidy", "correction_applied": True}]}
    out = _correct(monkeypatch, [seg], payload)[0]
    assert out["raw_text"] == text and len(out["text"].split()) == 19             # nothing was cut
    assert out["text"].startswith("We reviewed")                                  # the grammar pass still capitalised
    assert "llm_correction_rejected" in out["correction_reason"]
    assert "grammar and punctuation restoration" in out["correction_reason"]
