"""Chunk-overlap de-duplication and honest quality metrics in the ASR path."""

from types import SimpleNamespace

import pytest

from backend.services import transcription_service as ts

OVERLAP = ts.CHUNK_OVERLAP_SEC  # 0.5 s by default


def _w(word, start):
    return {"word": word, "start": start, "end": start + 0.2}


# ── word merge ────────────────────────────────────────────────────────────────


def test_first_chunk_words_are_appended_untouched():
    combined = []
    ts._merge_chunk_words(combined, [_w("a", 0.0), _w("b", 1.0)], boundary_sec=0.0, is_first=True)
    assert [w["word"] for w in combined] == ["a", "b"]


def test_overlap_window_words_are_not_duplicated():
    """Chunk 1 covers [.., 100); chunk 2 starts 0.5 s early. 'the plan' is heard by both."""
    boundary = 100.0
    combined = [_w("we", 98.0), _w("agree", 99.0), _w("the", 99.6), _w("plan", 99.8)]   # end of chunk 1
    chunk2 = [_w("the", 99.6), _w("plan", 99.8), _w("today", 100.4), _w("okay", 101.0)]  # start of chunk 2
    ts._merge_chunk_words(combined, chunk2, boundary, is_first=False)

    words = [w["word"] for w in combined]
    assert words == ["we", "agree", "the", "plan", "today", "okay"], words
    assert [w["start"] for w in combined] == sorted(w["start"] for w in combined), "timeline must stay ordered"


def test_merge_keeps_everything_when_the_boundary_falls_in_silence():
    boundary = 100.0
    combined = [_w("done", 97.0)]
    ts._merge_chunk_words(combined, [_w("next", 100.9)], boundary, is_first=False)
    assert [w["word"] for w in combined] == ["done", "next"]


def test_merge_with_empty_chunks_is_safe():
    combined = [_w("a", 1.0)]
    ts._merge_chunk_words(combined, [], 100.0, is_first=False)
    assert [w["word"] for w in combined] == ["a"]
    empty = []
    ts._merge_chunk_words(empty, [_w("z", 200.0)], 200.0, is_first=False)
    assert [w["word"] for w in empty] == ["z"]


# ── text merge ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "prev,new,expected",
    [
        ("we agreed on the plan", "the plan today is final", "today is final"),
        ("we agreed on the plan.", "The plan, today is final", "today is final"),      # case / punctuation ignored
        ("okay let us start", "the agenda first", "the agenda first"),                 # no overlap -> untouched
        ("", "hello there", "hello there"),
        ("hello", "", ""),
        ("we shipped it", "it works", "works"),                                        # single-word overlap
    ],
)
def test_trim_overlap_prefix(prev, new, expected):
    assert ts._trim_overlap_prefix(prev, new) == expected


def test_trim_overlap_never_swallows_a_whole_chunk_by_accident():
    # Only the *shared* prefix goes; the rest of a chunk's text always survives.
    assert ts._trim_overlap_prefix("thanks everyone", "everyone please join the call") == "please join the call"


# ── honest metrics ────────────────────────────────────────────────────────────


def test_missing_quality_metrics_are_none_not_invented_constants():
    assert ts.extract_quality_metrics({"text": "hello"}) == (None, None)
    assert ts.extract_quality_metrics({"segments": []}) == (None, None)
    assert ts.extract_quality_metrics(SimpleNamespace(segments=None)) == (None, None)


def test_quality_metrics_are_averaged_when_present():
    lp, nsp = ts.extract_quality_metrics({"segments": [
        {"avg_logprob": -0.2, "no_speech_prob": 0.1},
        {"avg_logprob": -0.4, "no_speech_prob": 0.3},
    ]})
    assert lp == pytest.approx(-0.3) and nsp == pytest.approx(0.2)


def test_partial_metrics_only_fill_what_exists():
    assert ts.extract_quality_metrics({"segments": [{"avg_logprob": -0.5}]}) == (-0.5, None)


# ── chunk sizing invariants ───────────────────────────────────────────────────


@pytest.mark.parametrize("minutes", [1, 8, 9, 25, 60, 180, 600])
def test_every_chunk_stays_under_the_byte_budget(minutes):
    total = minutes * 60.0
    dur = ts.compute_chunk_duration(total)
    assert dur >= 1
    bytes_per_chunk = (dur + OVERLAP) * ts.settings.BYTES_PER_SEC_WAV
    assert bytes_per_chunk <= ts.settings.CLOUD_HARD_MAX_BYTES, "chunk (incl. overlap) must fit the endpoint hard cap"


# ── AIRAWAT Whisper is the only speech-to-text engine ─────────────────────────


def test_no_local_speech_to_text_engine_exists():
    for name in ("_get_local_whisper_model", "_transcribe_audio_local_single", "transcribe_audio_local",
                 "_transcribe_chunk_locally"):
        assert not hasattr(ts, name), name
