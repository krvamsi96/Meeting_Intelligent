"""Pure metric calculators — industry-standard, mathematically exact.

Every function in this module is a pure computation over its inputs. No DB
access, no AI estimation. A ``None`` or empty reference always yields a
metric that must be reported as ``not_measured`` by the caller.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ─── Tokenization ─────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\b[\w'’-]+\b")


def tokenize_words(text: Optional[str]) -> List[str]:
    """Lowercased word tokens; strips speaker-prefix lines like ``S1:``."""
    if not text:
        return []
    text = re.sub(r"^[A-Z_\d\s]+:", "", text, flags=re.MULTILINE)
    return _WORD_RE.findall(text.lower())


# ─── Edit-distance (Levenshtein) ──────────────────────────────────────────────

def levenshtein_ops(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[int, int, int, int]:
    """Return (substitutions, deletions, insertions, edit_distance).

    Wagner–Fischer dynamic programming over token sequences, tracking the
    operation counts alongside the distance.
    """
    n, m = len(ref), len(hyp)
    dp = [[(0, 0, 0, 0) for _ in range(m + 1)] for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = (i, 0, i, 0)  # deletions
    for j in range(1, m + 1):
        dp[0][j] = (j, 0, 0, j)  # insertions

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub, dele, ins = dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]
                best = min(sub[0], dele[0], ins[0])
                if best == sub[0]:
                    dp[i][j] = (sub[0] + 1, sub[1] + 1, sub[2], sub[3])
                elif best == dele[0]:
                    dp[i][j] = (dele[0] + 1, dele[1], dele[2] + 1, dele[3])
                else:
                    dp[i][j] = (ins[0] + 1, ins[1], ins[2], ins[3] + 1)

    return dp[n][m][1], dp[n][m][2], dp[n][m][3], dp[n][m][0]


# ─── 1. Speech-to-Text: WER / CER ─────────────────────────────────────────────

def stt_metrics(ref_transcript: Optional[str], hyp_transcript: Optional[str]) -> Dict[str, Any]:
    """Word Error Rate (WER) and Character Error Rate (CER).

    WER = (S + D + I) / N_ref ; accuracy = (1 − WER) × 100.
    """
    ref_words = tokenize_words(ref_transcript)
    hyp_words = tokenize_words(hyp_transcript)
    n_ref = len(ref_words)

    if n_ref == 0:
        return {
            "wer": None, "cer": None, "accuracy": None,
            "substitutions": None, "deletions": None, "insertions": None,
        }

    subs, dels, ins, _ = levenshtein_ops(ref_words, hyp_words)
    wer = (subs + dels + ins) / n_ref
    accuracy = max(0.0, (1.0 - wer) * 100.0)

    ref_chars = list((ref_transcript or "").strip())
    hyp_chars = list((hyp_transcript or "").strip())
    cer = None
    if ref_chars:
        _, _, _, dist = levenshtein_ops(ref_chars, hyp_chars)
        cer = dist / len(ref_chars)

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4) if cer is not None else None,
        "accuracy": round(accuracy, 2),
        "substitutions": subs,
        "deletions": dels,
        "insertions": ins,
    }


# ─── 2. Diarization: DER / JER ────────────────────────────────────────────────

_Timeline = List[Dict[str, Any]]  # [{"speaker": str, "start": float, "end": float}]


def diarization_metrics(ref_timeline: _Timeline, hyp_timeline: _Timeline) -> Dict[str, Any]:
    """Diarization Error Rate (DER) and Jaccard Error Rate (JER).

    DER = (false_alarm + missed + speaker_error) / total_ref_duration.
    JER = 1 − mean(Jaccard index per reference speaker).
    """
    if not ref_timeline:
        return {"der": None, "jer": None, "accuracy": None}

    ref = _normalize_timeline(ref_timeline)
    hyp = _normalize_timeline(hyp_timeline)

    if not ref:
        # ref_timeline was non-empty going in, but every segment was malformed
        # (end <= start) and normalization dropped all of them — there is no
        # valid reference to compare against. Falling through to the
        # `len(timestamps) < 2` branch below would otherwise report a
        # fabricated "der=0.0, accuracy=100.0" for a 0-vs-0 comparison.
        return {"der": None, "jer": None, "accuracy": None}

    timestamps = sorted(
        {round(v, 4) for seg in ref + hyp for v in (seg["start"], seg["end"])}
    )
    if len(timestamps) < 2:
        return {"der": 0.0, "jer": 0.0, "accuracy": 100.0}

    mapping = _greedy_speaker_mapping(ref, hyp)

    total_ref = 0.0
    false_alarm = 0.0
    missed = 0.0
    speaker_error = 0.0

    for i in range(len(timestamps) - 1):
        t1, t2 = timestamps[i], timestamps[i + 1]
        dur = t2 - t1
        if dur <= 0:
            continue
        ref_active = {s["speaker"] for s in ref if s["start"] <= t1 and s["end"] >= t2}
        hyp_active = {s["speaker"] for s in hyp if s["start"] <= t1 and s["end"] >= t2}
        n_ref, n_hyp = len(ref_active), len(hyp_active)
        total_ref += n_ref * dur

        if n_ref == 0 and n_hyp > 0:
            false_alarm += n_hyp * dur
        elif n_ref > 0 and n_hyp == 0:
            missed += n_ref * dur
        elif n_ref > 0 and n_hyp > 0:
            matched = sum(1 for p in hyp_active if mapping.get(p) in ref_active)
            speaker_error += (min(n_ref, n_hyp) - matched) * dur
            if n_hyp > n_ref:
                false_alarm += (n_hyp - n_ref) * dur
            elif n_ref > n_hyp:
                missed += (n_ref - n_hyp) * dur

    der = 1.0 if total_ref == 0 and false_alarm > 0 else (false_alarm + missed + speaker_error) / total_ref if total_ref else 0.0
    accuracy = max(0.0, (1.0 - der) * 100.0)

    # JER: 1 − mean Jaccard index over reference speakers
    jaccards = _jaccard_indices(ref, hyp, mapping)
    jer = 1.0 - (sum(jaccards) / len(jaccards)) if jaccards else None

    return {
        "der": round(der, 4),
        "jer": round(jer, 4) if jer is not None else None,
        "accuracy": round(accuracy, 2),
    }


def _normalize_timeline(segments: _Timeline) -> _Timeline:
    out = []
    for seg in segments or []:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        if end <= start:
            continue
        out.append({"speaker": str(seg.get("speaker", "UNKNOWN")), "start": start, "end": end})
    return out


def _greedy_speaker_mapping(ref: _Timeline, hyp: _Timeline) -> Dict[str, str]:
    overlap: Dict[str, Dict[str, float]] = {}
    for r in ref:
        for p in hyp:
            o_start = max(r["start"], p["start"])
            o_end = min(r["end"], p["end"])
            if o_end > o_start:
                overlap.setdefault(r["speaker"], {}).setdefault(p["speaker"], 0.0)
                overlap[r["speaker"]][p["speaker"]] += o_end - o_start

    ranking = sorted(
        ((dur, p, r) for r, pd in overlap.items() for p, dur in pd.items()),
        reverse=True,
    )
    mapping: Dict[str, str] = {}
    mapped_ref = set()
    for _, p_spk, r_spk in ranking:
        if p_spk not in mapping and r_spk not in mapped_ref:
            mapping[p_spk] = r_spk
            mapped_ref.add(r_spk)
    return mapping


def _jaccard_indices(ref: _Timeline, hyp: _Timeline, mapping: Dict[str, str]) -> List[float]:
    ref_speakers = {s["speaker"] for s in ref}
    indices = []
    for r_spk in ref_speakers:
        p_spk = next((p for p, r in mapping.items() if r == r_spk), None)
        ref_dur = sum(max(0.0, s["end"] - s["start"]) for s in ref if s["speaker"] == r_spk)
        if not p_spk:
            indices.append(0.0)
            continue
        hyp_dur = sum(max(0.0, s["end"] - s["start"]) for s in hyp if s["speaker"] == p_spk)
        inter = 0.0
        for r_seg in ref:
            if r_seg["speaker"] != r_spk:
                continue
            for p_seg in hyp:
                if p_seg["speaker"] != p_spk:
                    continue
                inter += max(0.0, min(r_seg["end"], p_seg["end"]) - max(r_seg["start"], p_seg["start"]))
        union = ref_dur + hyp_dur - inter
        indices.append(inter / union if union > 0 else 0.0)
    return indices


# ─── 3. Classification: precision / recall / F1 (macro) ──────────────────────

def classification_metrics(
    ref_labels: Dict[str, Any], hyp_labels: Dict[str, Any]
) -> Dict[str, Any]:
    """Macro-averaged precision, recall, F1 and accuracy over labelled items."""
    ref_keys = set(ref_labels.keys())
    hyp_keys = set(hyp_labels.keys())
    common = ref_keys & hyp_keys
    if not ref_keys or not common:
        return {
            "accuracy": None, "precision": None, "recall": None,
            "f1": None, "support": 0, "confusion_matrix": {},
        }

    classes = sorted(set(ref_labels.values()) | set(hyp_labels.values()))
    conf = {c: {c2: 0 for c2 in classes} for c in classes}
    correct = 0
    for key in common:
        r, h = ref_labels[key], hyp_labels[key]
        conf[r][h] += 1
        if r == h:
            correct += 1

    precisions, recalls = [], []
    for c in classes:
        tp = conf[c][c]
        fp = sum(conf[o][c] for o in classes if o != c)
        fn = sum(conf[c][o] for o in classes if o != c)
        precisions.append(tp / (tp + fp) if tp + fp > 0 else 0.0)
        recalls.append(tp / (tp + fn) if tp + fn > 0 else 0.0)

    precision = sum(precisions) / len(precisions)
    recall = sum(recalls) / len(recalls)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "accuracy": round(correct / len(common) * 100.0, 2),
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "support": len(common),
        "confusion_matrix": conf,
    }


# ─── 4. Summary / MoM: ROUGE-1/2/L + semantic F1 (BERTScore-style) ───────────

def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_scores(ref_text: Optional[str], hyp_text: Optional[str]) -> Dict[str, Any]:
    """ROUGE-1, ROUGE-2, ROUGE-L F1 (macro ROUGE-style union of sets)."""
    ref_tokens = tokenize_words(ref_text)
    hyp_tokens = tokenize_words(hyp_text)
    if not ref_tokens or not hyp_tokens:
        return {"rouge1": None, "rouge2": None, "rougeL": None}

    def _f1(ref_cnt: Counter, hyp_cnt: Counter) -> float:
        overlap = sum((ref_cnt & hyp_cnt).values())
        if overlap == 0:
            return 0.0
        precision = overlap / sum(hyp_cnt.values())
        recall = overlap / sum(ref_cnt.values())
        return 2 * precision * recall / (precision + recall)

    rouge1 = _f1(_ngrams(ref_tokens, 1), _ngrams(hyp_tokens, 1))
    rouge2 = _f1(_ngrams(ref_tokens, 2), _ngrams(hyp_tokens, 2))
    rougeL = _lcs_f1(ref_tokens, hyp_tokens)

    return {
        "rouge1": round(rouge1 * 100.0, 2),
        "rouge2": round(rouge2 * 100.0, 2),
        "rougeL": round(rougeL * 100.0, 2),
    }


def _lcs_f1(ref: Sequence[str], hyp: Sequence[str]) -> float:
    m, n = len(ref), len(hyp)
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    lcs = dp[n]
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def semantic_f1(
    ref_text: Optional[str],
    hyp_text: Optional[str],
    embed_fn,
) -> Dict[str, Any]:
    """BERTScore-style F1: max-cosine alignment of sentence embeddings.

    ``embed_fn(text: str) -> List[float]`` is injected by the caller so this
    function stays pure and testable with any embedder.
    """
    if not ref_text or not hyp_text or embed_fn is None:
        return {"bertscore_f1": None, "bertscore_precision": None, "bertscore_recall": None}

    ref_sents = [s.strip() for s in re.split(r"[.!?\n]", ref_text) if len(s.strip()) > 5]
    hyp_sents = [s.strip() for s in re.split(r"[.!?\n]", hyp_text) if len(s.strip()) > 5]
    if not ref_sents or not hyp_sents:
        ref_sents, hyp_sents = [ref_text], [hyp_text]

    ref_embs = [embed_fn(s) for s in ref_sents]
    hyp_embs = [embed_fn(s) for s in hyp_sents]
    if not ref_embs or not hyp_embs:
        return {"bertscore_f1": None, "bertscore_precision": None, "bertscore_recall": None}

    def _max_sim(vec, others) -> float:
        best = 0.0
        for o in others:
            best = max(best, _cosine(vec, o))
        return best

    recall = sum(_max_sim(r, hyp_embs) for r in ref_embs) / len(ref_embs)
    precision = sum(_max_sim(h, ref_embs) for h in hyp_embs) / len(hyp_embs)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "bertscore_f1": round(f1 * 100.0, 2),
        "bertscore_precision": round(precision * 100.0, 2),
        "bertscore_recall": round(recall * 100.0, 2),
    }


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_matching_f1(
    ref_items: Sequence[str],
    hyp_items: Sequence[str],
    embed_fn,
    threshold: float = 0.75,
) -> Dict[str, Any]:
    """Precision/Recall/F1 for item lists matched by semantic similarity.

    A reference item is a true positive if at least one generated item is
    semantically similar (cosine ≥ threshold). Used for action items and
    reference topics where exact string equality is too brittle.
    """
    if not ref_items or not hyp_items or embed_fn is None:
        return {
            "precision": None, "recall": None, "f1": None,
            "support": min(len(ref_items), len(hyp_items)),
            "reference_count": len(ref_items),
            "generated_count": len(hyp_items),
        }

    ref_embs = [embed_fn(str(x)) for x in ref_items]
    hyp_embs = [embed_fn(str(x)) for x in hyp_items]

    tp = 0
    for r in ref_embs:
        if r and any(_cosine(r, g) >= threshold for g in hyp_embs if g):
            tp += 1

    precision = tp / len(hyp_items) if hyp_items else 0.0
    recall = tp / len(ref_items) if ref_items else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return {
        "precision": round(precision * 100.0, 2),
        "recall": round(recall * 100.0, 2),
        "f1": round(f1 * 100.0, 2),
        "accuracy": round(f1 * 100.0, 2),
        "support": min(len(ref_items), len(hyp_items)),
        "reference_count": len(ref_items),
        "generated_count": len(hyp_items),
    }


# ─── 5. Retrieval: nDCG ───────────────────────────────────────────────────────

def ndcg(
    ranked_ids: Sequence[Any],
    relevance: Dict[Any, float],
    k: Optional[int] = None,
) -> Dict[str, Any]:
    """Normalized Discounted Cumulative Gain at k.

    ``ranked_ids``: candidate ids in retrieved order.
    ``relevance``: mapping id → graded relevance (0/1 binary or 0–3 graded).
    """
    k = k or len(ranked_ids)
    if k <= 0 or not ranked_ids:
        return {"ndcg": None, "dcg": None, "idcg": None, "k": k}

    def _dcg(ids: Sequence[Any]) -> float:
        return sum(
            relevance.get(rid, 0.0) / math.log2(i + 2)
            for i, rid in enumerate(ids[:k])
        )

    dcg = _dcg(ranked_ids)
    ideal = sorted(relevance.get(rid, 0.0) for rid in ranked_ids if rid in relevance)
    ideal = sorted((relevance.get(rid, 0.0) for rid in ranked_ids), reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal[:k]))
    ndcg_val = dcg / idcg if idcg > 0 else 0.0

    return {
        "ndcg": round(ndcg_val, 4),
        "dcg": round(dcg, 4),
        "idcg": round(idcg, 4),
        "k": k,
    }


# ─── 6. Topics: NPMI Topic Coherence (C_v-style) ─────────────────────────────

def topic_coherence_cv(
    topics: Sequence[Sequence[str]],
    corpus: Sequence[str],
    window_size: int = 10,
) -> Dict[str, Any]:
    """C_v-style topic coherence from NPMI over sliding windows.

    For each pair of words (w_i, w_j) in a topic, NPMI is computed from
    co-occurrence in ``window_size``-sized windows over the corpus;
    coherence = mean NPMI. Only measurable when the corpus is non-empty.
    """
    cleaned_topics = [[t.lower() for t in topic if t.strip()] for topic in topics]
    cleaned_topics = [t for t in cleaned_topics if len(t) >= 2]
    if not cleaned_topics or not corpus:
        return {"coherence": None, "topic_count": 0, "window_size": window_size}

    windows: List[List[str]] = []
    for doc in corpus:
        tokens = tokenize_words(doc)
        if len(tokens) < window_size:
            windows.append(tokens)
        else:
            for i in range(len(tokens) - window_size + 1):
                windows.append(tokens[i:i + window_size])
    windows = [w for w in windows if w]
    if not windows:
        return {"coherence": None, "topic_count": len(cleaned_topics), "window_size": window_size}

    total = len(windows)
    term_docs: Counter = Counter()
    pair_docs: Counter = Counter()
    for w in windows:
        uniq = set(w)
        term_docs.update(uniq)
        for a in uniq:
            for b in uniq:
                if a < b:
                    pair_docs[(a, b)] += 1

    def _npmi(a: str, b: str) -> Optional[float]:
        pair = (a, b) if a < b else (b, a)
        p_ab = pair_docs.get(pair, 0) / total
        p_a = term_docs.get(a, 0) / total
        p_b = term_docs.get(b, 0) / total
        if p_ab == 0 or p_a == 0 or p_b == 0:
            return None
        pmi = math.log(p_ab / (p_a * p_b))
        return pmi / -math.log(p_ab) if p_ab < 1.0 else 0.0

    topic_scores = []
    for topic in cleaned_topics:
        vals = [
            npmi
            for i, a in enumerate(topic)
            for b in topic[i + 1:]
            if (npmi := _npmi(a, b)) is not None
        ]
        topic_scores.append(sum(vals) / len(vals) if vals else 0.0)

    coherence = sum(topic_scores) / len(topic_scores) if topic_scores else 0.0
    return {
        "coherence": round(coherence, 4),
        "topic_count": len(cleaned_topics),
        "window_size": window_size,
    }
