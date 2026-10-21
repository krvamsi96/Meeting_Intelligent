"""The speaker COUNT is discovered from the embeddings, never assumed (1 speaker -> 1, 3 speakers -> 3 ...).

TitaNet itself needs NeMo (Linux) and real audio, so these tests exercise the part that decides the count: the
clustering of speaker-embedding windows. Each synthetic speaker is a distinct voice direction plus per-window noise
(same speaker ~0.2-0.35 cosine distance apart, different speakers ~0.8-1.0), laid out as alternating turns in time.
"""

import numpy as np
import pytest

from backend.services import diarization_service as ds

DIM = 192            # TitaNet embedding size
HOP = 1.5            # seconds per embedding window


def _meeting(true_speakers: int, turns: int, windows_per_turn: int = 8, noise: float = 0.06, seed: int = 0):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((true_speakers, DIM))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    feats, truth, spans = [], [], []
    t = 0.0
    for turn in range(turns):
        speaker = turn % true_speakers
        for _ in range(windows_per_turn):
            v = centers[speaker] + noise * rng.standard_normal(DIM)
            feats.append(v / np.linalg.norm(v))
            truth.append(speaker)
            spans.append((t, t + HOP))
            t += HOP
    return np.array(feats, dtype=np.float32), np.array(truth), spans


@pytest.mark.parametrize("true_speakers", [1, 2, 3, 4, 5])
def test_the_number_of_speakers_is_discovered_not_assumed(true_speakers):
    feats, truth, spans = _meeting(true_speakers, turns=max(6, true_speakers * 3))
    labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == true_speakers, f"{true_speakers} real speaker(s) -> detected {k}"
    assert len(set(labels.tolist())) == k


@pytest.mark.parametrize("true_speakers", [2, 3, 4])
def test_windows_are_assigned_to_the_right_speaker(true_speakers):
    feats, truth, spans = _meeting(true_speakers, turns=true_speakers * 4, seed=3)
    labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == true_speakers
    # cluster ids are arbitrary: every true speaker must map to exactly one cluster, and different speakers to different ones
    mapping = {s: set(labels[truth == s].tolist()) for s in range(true_speakers)}
    assert all(len(ids) == 1 for ids in mapping.values()), mapping
    assert len({next(iter(ids)) for ids in mapping.values()}) == true_speakers


def test_one_noisy_speaker_is_not_split_into_two():
    """A lone talker on noisy windows must stay ONE speaker (the platform must not invent a second participant)."""
    feats, truth, spans = _meeting(1, turns=10, windows_per_turn=10, noise=0.09, seed=5)
    _labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == 1


def test_a_caller_who_knows_the_count_can_state_it_but_it_is_optional():
    feats, truth, spans = _meeting(3, turns=12)
    _labels, k = ds._select_speaker_labels(feats, spans, 3, is_neural=True)
    assert k == 3
    _labels, k_auto = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k_auto == 3


def test_a_recording_with_almost_no_speech_is_one_speaker_not_an_error():
    feats, _truth, spans = _meeting(1, turns=1, windows_per_turn=1)
    labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == 1 and labels.tolist() == [0]


def _unit(v):
    return v / np.linalg.norm(v)


def _two_voices(cos_between: float, windows_per_turn: int = 10, turns: int = 8, noise: float = 0.05, seed: int = 0):
    """Two speakers whose true voice directions have the given cosine similarity, alternating turns."""
    rng = np.random.default_rng(seed)
    a = _unit(rng.standard_normal(DIM))
    b_orth = rng.standard_normal(DIM)
    b_orth = _unit(b_orth - (b_orth @ a) * a)
    b = _unit(cos_between * a + np.sqrt(1.0 - cos_between ** 2) * b_orth)
    feats, spans, truth, t = [], [], [], 0.0
    for turn in range(turns):
        for _ in range(windows_per_turn):
            feats.append(_unit((a if turn % 2 == 0 else b) + noise * rng.standard_normal(DIM)))
            spans.append((t, t + HOP))
            truth.append(turn % 2)
            t += HOP / 2                     # 50 % overlapping windows, like the real pipeline
    return np.array(feats, dtype=np.float32), spans, np.array(truth)


def test_overlapping_windows_are_not_counted_twice_as_talk_time():
    spans = [(0.0, 1.5), (0.75, 2.25), (1.5, 3.0), (10.0, 11.5)]      # last window follows a pause
    eff = ds._effective_durations(spans)
    assert eff.tolist() == pytest.approx([0.75, 0.75, 1.5, 1.5])
    assert eff.sum() == pytest.approx(4.5)          # naive sum of (end - start) would be 6.0


def test_a_short_burst_of_boundary_blend_windows_is_not_a_phantom_speaker():
    """Real audio: windows straddling fast alternating turns are blends of the two voices. They used to form a third
    'speaker' because their overlapping windows were counted as twice the talk time."""
    feats, spans, labels = _two_voices(cos_between=0.15, seed=1)
    a_win, b_win = feats[5], feats[15]                                                # a window of each voice
    blend = [_unit(a_win + b_win), _unit(a_win + 1.1 * b_win)]                        # two mixed windows in a row
    feats = np.vstack([feats, np.array(blend, dtype=np.float32)])
    spans = spans + [(200.0, 201.5), (200.75, 202.25)]                                # 2.25 s of unique audio
    labels = np.append(labels, [2, 2])
    seg_durs = ds._effective_durations(spans)
    out, k = ds._postprocess_clusters(labels, feats, seg_durs, float(seg_durs.sum()), merge_cos=0.30, min_talk=3.0)
    assert k == 2
    assert set(out[-2:].tolist()) <= set(out[:-2].tolist())       # re-homed onto a real speaker


def test_two_different_people_with_similar_voices_are_not_fused():
    """Cluster centroids are averages: two DIFFERENT speakers with similar voices already reach cosine ~0.55, which the
    old neural merge threshold (0.5 distance) treated as 'the same person' (real meetings: 2 -> 1, 4 -> 3)."""
    feats, spans, _truth = _two_voices(cos_between=0.55, seed=2)
    _labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == 2


def test_one_person_with_a_noisy_but_single_voice_is_still_one_speaker():
    feats, spans, _truth = _two_voices(cos_between=1.0 - 1e-9, seed=3, noise=0.12)   # both 'speakers' are the SAME voice
    _labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == 1


def test_merge_and_dissolve_settle_so_no_two_final_clusters_are_duplicates():
    feats, spans, _truth = _two_voices(cos_between=0.1, seed=4)
    rng = np.random.default_rng(9)
    labels = rng.integers(0, 6, size=len(spans))                 # a badly over-split starting point
    seg_durs = ds._effective_durations(spans)
    out, k = ds._postprocess_clusters(labels, feats, seg_durs, float(seg_durs.sum()), merge_cos=0.30, min_talk=3.0)
    cents = ds._l2_centroids(out, feats)
    assert k == len(cents)
    assert k == 1 or ds._min_centroid_cos_dist(cents) >= 0.30


# ── a recording-condition change is not a second speaker ─────────────────────────────────────────────

def test_one_voice_after_a_microphone_or_room_change_is_not_two_speakers():
    """One voice, then the same voice 14 dB quieter with room reverb (speaker moved away from the microphone): TitaNet
    puts the two halves ~0.43 apart. Every 'speaker' is ONE unbroken stretch (nobody ever answers back), so it is a
    change of recording conditions, not a second participant. Measured: this used to come out as 2 speakers."""
    feats, spans, _truth = _two_voices(cos_between=0.57, windows_per_turn=40, turns=2, seed=6)
    _labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == 1


def test_a_handover_between_clearly_different_people_is_still_two_speakers():
    """The same single-stretch layout, but the voices are clearly different people (cosine distance ~0.85): kept."""
    feats, spans, _truth = _two_voices(cos_between=0.15, windows_per_turn=40, turns=2, seed=7)
    _labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == 2


def test_voices_that_answer_each_other_stay_separate_even_when_they_sound_alike():
    """Same moderate separation as the recording-change case, but they take turns: two real people."""
    feats, spans, _truth = _two_voices(cos_between=0.57, windows_per_turn=10, turns=8, seed=8)
    _labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == 2


# ── VAD: digital silence must not switch it off ──────────────────────────────────────────────────────

def _speechlike(seconds: float, sr: int, rng) -> np.ndarray:
    n = int(seconds * sr)
    envelope = (1 + np.sin(np.linspace(0, 40 * np.pi, n))) / 2               # ~6.7 Hz syllable-like modulation
    return (0.1 * rng.standard_normal(n) * envelope).astype(np.float32)


def test_room_tone_is_not_speech_when_the_recording_also_has_digital_silence():
    """Measured on real recordings: the quietest frames were digital silence (-111..-200 dB), so the 'adaptive' threshold
    sat at -102..-191 dB and EVERY sound counted as speech (94-97 %). Here: speech, exact zeros, -68 dB room tone, speech."""
    sr, rng = 16000, np.random.default_rng(0)
    tone = (0.0004 * rng.standard_normal(3 * sr)).astype(np.float32)          # about -68 dBFS
    audio = np.concatenate([_speechlike(3, sr, rng), np.zeros(2 * sr, np.float32), tone, _speechlike(3, sr, rng)])
    regions = ds._adaptive_energy_vad(audio, "", sr, len(audio) / sr)
    speech_sec = sum(e - s for s, e in regions)
    assert 5.0 <= speech_sec <= 7.5, regions                                   # the two speech blocks, not the tone
    assert not any(s < 7.0 and e > 6.0 for s, e in regions), regions           # nothing inside the room-tone stretch


def test_quiet_but_real_speech_is_still_kept_by_the_vad():
    """The relative floor must not delete a quiet talker: speech 30 dB below the loud talker is well inside the range."""
    sr, rng = 16000, np.random.default_rng(1)
    loud, quiet = _speechlike(3, sr, rng), _speechlike(3, sr, rng) * 0.03      # -30 dB
    audio = np.concatenate([loud, np.zeros(sr, np.float32), quiet])
    regions = ds._adaptive_energy_vad(audio, "", sr, len(audio) / sr)
    assert any(s >= 3.5 and e - s > 2.0 for s, e in regions), regions


# ── a speaker model that arrives after start-up is used ──────────────────────────────────────────────

def test_a_model_file_put_in_place_after_start_up_is_picked_up(monkeypatch, tmp_path):
    """The weak numpy fallback used to be cached for the life of the process, so uploading titanet_large.nemo to a running
    server changed nothing until the next restart."""
    monkeypatch.setattr(ds, "_nemo_speaker_model", "NEURAL_ACOUSTIC_EMBEDDING")
    monkeypatch.setattr(ds, "_nemo_failed_local_file", None)
    monkeypatch.setattr(ds.settings, "DIARIZATION_MODEL_PATH", str(tmp_path))
    assert not ds._speaker_model_file_provided_after_failure()                # nothing there yet: stay on the fallback

    model_file = tmp_path / "titanet_large.nemo"
    model_file.write_bytes(b"x")
    assert ds._speaker_model_file_provided_after_failure()

    loaded = object()
    monkeypatch.setattr(ds, "_load_nemo_speaker_model_locked", lambda: loaded)
    assert ds._get_nemo_speaker_model() is loaded

    # a load that failed on this very file is not retried for every meeting
    monkeypatch.setattr(ds, "_nemo_speaker_model", "NEURAL_ACOUSTIC_EMBEDDING")
    monkeypatch.setattr(ds, "_nemo_failed_local_file", ds._local_file_signature(str(model_file)))
    assert not ds._speaker_model_file_provided_after_failure()


def test_a_loaded_model_is_reused_not_reloaded(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(ds, "_nemo_speaker_model", sentinel)
    monkeypatch.setattr(ds, "_load_nemo_speaker_model_locked", lambda: pytest.fail("the model was loaded a second time"))
    assert ds._get_nemo_speaker_model() is sentinel and ds._get_nemo_speaker_model() is sentinel


# ── a sentence is not cut in two by a drifted word time ─────────────────────────────────────────────

def _words(text: str, start: float = 0.0, per_word: float = 0.4):
    return [{"word": w, "start": start + i * per_word, "end": start + (i + 1) * per_word} for i, w in enumerate(text.split())]


EXAMPLE = "Great. Today we discuss the migration timeline and decided to move production deployment to October 15th."


def test_the_tail_of_a_sentence_is_not_credited_to_the_next_speaker():
    """Measured end to end: '... decided to move production | deployment to October 15th' was split between two speakers
    because the speech-to-text put the last words a second late, inside the next speaker's turn."""
    words = _words(EXAMPLE)
    n = len(words)
    speakers = ["A"] * (n - 4) + ["B"] * 4                       # 'deployment to October 15th.' credited to B
    assert ds._smooth_speakers_within_sentences(words, speakers) == ["A"] * n


def test_the_head_of_a_sentence_is_not_credited_to_the_previous_speaker():
    words = _words("Understood that gives the testing team two more weeks of time to prepare.")
    speakers = ["A"] * 2 + ["B"] * (len(words) - 2)
    assert ds._smooth_speakers_within_sentences(words, speakers) == ["B"] * len(words)


def test_a_reply_that_is_a_sentence_of_its_own_is_never_touched():
    words = _words("Do you agree with the plan that we discussed yesterday? Yes.")
    speakers = ["A"] * (len(words) - 1) + ["B"]
    assert ds._smooth_speakers_within_sentences(words, speakers) == speakers


def test_a_second_speaker_who_really_holds_the_floor_inside_a_sentence_is_kept():
    words = _words("we should move the deadline no wait let me finish the point about the budget first.")
    assert len(words) == 16
    speakers = ["A"] * 8 + ["B"] * 8                              # half of the sentence each: not a stray fragment
    assert ds._smooth_speakers_within_sentences(words, speakers) == speakers


def test_a_long_fragment_is_kept_even_when_it_is_a_small_share_of_a_long_sentence():
    words = _words("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
                   "seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four.", per_word=0.5)
    speakers = ["A"] * 19 + ["B"] * 5                             # 5 words but 2.5 s > 2 s
    assert ds._smooth_speakers_within_sentences(words, speakers) == speakers


def test_short_sentences_are_left_alone():
    words = _words("Yes that is fine.")
    speakers = ["A", "A", "B", "B"]
    assert ds._smooth_speakers_within_sentences(words, speakers) == speakers


def test_alignment_applies_the_smoothing_and_it_can_be_switched_off(monkeypatch):
    words = _words(EXAMPLE)
    n = len(words)
    boundary = words[n - 4]["start"]                                      # turn change lands 4 words before the sentence end
    turns = [{"start": 0.0, "end": boundary - 0.01, "speaker": "Speaker_1"},
             {"start": boundary, "end": words[-1]["end"] + 1.0, "speaker": "Speaker_2"}]
    assert {s["speaker"] for s in ds.align_words_to_nemo(words, turns)} == {"Speaker_1"}
    monkeypatch.setattr(ds.settings, "DIARIZATION_SENTENCE_SMOOTHING", False)
    assert [s["speaker"] for s in ds.align_words_to_nemo(words, turns)] == ["Speaker_1", "Speaker_2"]


# ── the speaker COUNT comes from NVIDIA NeMo's own estimator ─────────────────────────────────────────
# Measured on real recordings: the hand-written estimator split ONE real voice into 3-8 "speakers" on 30-60 s clips and
# with background noise (few, noisy 1.5 s windows). NeMo's reference estimator with its short-recording counting did not.

def test_neural_embeddings_are_counted_by_nemos_estimator(monkeypatch):
    pytest.importorskip("nemo.collections.asr.parts.utils.offline_clustering")
    calls = []
    real = ds._estimate_num_speakers_nemo

    def spy(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(ds, "_estimate_num_speakers_nemo", spy)
    feats, _truth, spans = _meeting(2, turns=8)
    _labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert calls and k == 2


def test_the_built_in_estimator_is_the_fallback_when_nemo_is_unavailable(monkeypatch):
    def unavailable(*_a, **_k):
        raise ImportError("nemo not installed")

    monkeypatch.setattr(ds, "_estimate_num_speakers_nemo", unavailable)
    feats, _truth, spans = _meeting(3, turns=9)
    _labels, k = ds._select_speaker_labels(feats, spans, None, is_neural=True)
    assert k == 3


def test_the_numpy_fallback_embedder_never_uses_the_nemo_estimator(monkeypatch):
    monkeypatch.setattr(ds, "_estimate_num_speakers_nemo", lambda *_a, **_k: pytest.fail("TitaNet-only estimator used"))
    feats, _truth, spans = _meeting(1, turns=6)
    ds._select_speaker_labels(feats, spans, None, is_neural=False)


def test_nemo_counting_leaves_torchs_random_state_untouched():
    pytest.importorskip("nemo.collections.asr.parts.utils.offline_clustering")
    import torch

    feats, _truth, _spans = _meeting(2, turns=8)
    torch.manual_seed(1234)
    expected = torch.rand(3)
    torch.manual_seed(1234)
    ds._estimate_num_speakers_nemo(feats, 1, 8)                 # NeMo's anchor counting seeds torch 0..4 internally
    assert torch.equal(torch.rand(3), expected)


# ── speaker DURATION = detected speech inside the speaker's segments, never the pauses ───────────────
# Measured against an independent neural VAD: 22-39 % of the reported talk time was silence, because a transcript
# segment's end - start spans every pause inside it.

def test_speech_seconds_count_only_the_speech_inside_each_speakers_segments():
    segments = [{"speaker": "Speaker_1", "start_time": "0.0", "end_time": "10.0"},
                {"speaker": "Speaker_2", "start_time": "10.0", "end_time": "14.0"},
                {"speaker": "Speaker_1", "start_time": "14.0", "end_time": "20.0"}]
    speech = [(1.0, 4.0), (6.0, 9.0), (10.5, 13.0), (15.0, 19.0), (19.5, 19.5)]      # pauses between phrases
    got = ds.speech_seconds_per_speaker(segments, speech)
    assert got == pytest.approx({"Speaker_1": 3.0 + 3.0 + 4.0, "Speaker_2": 2.5})


def test_overlapping_speech_regions_are_not_counted_twice():
    segments = [{"speaker": "A", "start_time": 0.0, "end_time": 10.0}]
    assert ds.speech_seconds_per_speaker(segments, [(1.0, 5.0), (3.0, 6.0), (5.5, 7.0)]) == pytest.approx({"A": 6.0})


def test_speaker_diagnostics_use_speech_time_and_keep_their_format():
    segments = [{"speaker": "Speaker_1", "start_time": "0.0", "end_time": "10.0"},
                {"speaker": "Speaker_2", "start_time": "10.0", "end_time": "14.0"}]
    span_based = ds.compute_speaker_diagnostics(segments)
    speech_based = ds.compute_speaker_diagnostics(segments, speech_durations={"Speaker_1": 6.0, "Speaker_2": 2.0})
    assert set(speech_based) == set(span_based)                                  # same keys = same JSON format
    assert set(speech_based["speaker_breakdown"]["Speaker_1"]) == set(span_based["speaker_breakdown"]["Speaker_1"])
    b = speech_based["speaker_breakdown"]
    assert b["Speaker_1"]["speaking_duration_seconds"] == 6.0 and b["Speaker_2"]["speaking_duration_seconds"] == 2.0
    assert b["Speaker_1"]["speaking_percentage"] == 75.0 and speech_based["total_speech_duration_sec"] == 8.0
    assert span_based["total_speech_duration_sec"] == 14.0                    # the old, pause-inclusive number


def test_speech_durations_for_different_speakers_are_ignored_not_mixed():
    segments = [{"speaker": "Priya", "start_time": "0", "end_time": "5"}]
    diag = ds.compute_speaker_diagnostics(segments, speech_durations={"Speaker_1": 3.0})   # e.g. renamed afterwards
    assert diag["speaker_breakdown"]["Priya"]["speaking_duration_seconds"] == 5.0


def test_the_stored_durations_are_read_back_from_diarization_info():
    import json

    info = json.dumps({"diagnostics": {"speaker_breakdown": {"Speaker_1": {"speaking_duration_seconds": 30.5},
                                                             "Speaker_2": {"speaking_duration_seconds": 7.25}}}})
    assert ds.stored_speech_durations(info) == {"Speaker_1": 30.5, "Speaker_2": 7.25}
    assert ds.stored_speech_durations(None) is None and ds.stored_speech_durations("not json") is None


def test_the_final_record_speaker_duration_uses_the_stored_speech_time():
    from types import SimpleNamespace

    from backend.services.final_aggregation_service import _compute_speaker_duration

    segs = [SimpleNamespace(speaker="Speaker_1", start_time="0", end_time="10"),
            SimpleNamespace(speaker="Speaker_2", start_time="10", end_time="14")]
    spans_only = {d["speaker_id"]: d["duration_seconds"] for d in _compute_speaker_duration(segs, 14.0)}
    speech = {d["speaker_id"]: d for d in _compute_speaker_duration(segs, 14.0, {"Speaker_1": 6.0, "Speaker_2": 2.0})}
    assert spans_only == {"Speaker_1": 10.0, "Speaker_2": 4.0}
    assert speech["Speaker_1"]["duration_seconds"] == 6.0 and speech["Speaker_1"]["percentage"] == 75.0
    assert speech["Speaker_1"]["segment_count"] == 1                           # other fields unchanged


def test_the_vad_used_for_durations_is_the_diarization_vad(tmp_path):
    import soundfile as sf

    sr, rng = 16000, np.random.default_rng(3)
    audio = np.concatenate([_speechlike(3, sr, rng), np.zeros(2 * sr, np.float32), _speechlike(3, sr, rng)])
    path = tmp_path / "a.wav"
    sf.write(path, audio, sr)
    regions = ds.speech_regions_for_durations(str(path))
    assert regions == ds._adaptive_energy_vad(audio, "", sr, len(audio) / sr, **ds._vad_settings())
    assert 5.0 <= sum(e - s for s, e in regions) <= 6.6                          # the 2 s of silence is not speech


def test_without_readable_audio_the_diarization_turns_are_the_speech():
    turns = [{"start": 0.0, "end": 2.0, "speaker": "A"}, {"start": 1.5, "end": 3.0, "speaker": "B"}]
    assert ds.speech_regions_for_durations("/no/such/file.wav", turns) == [(0.0, 3.0)]
