"""Voice-based speaker identification.

After diarization has separated a meeting into speakers (Speaker_1, Speaker_2,
…), this module compares each speaker's mean voice embedding against every
registered :class:`VoiceProfile` and assigns a real name **only** when the match
is confident and unambiguous. Speakers with no reliable match keep a generic
label. Nothing is hard-coded and no name is ever guessed.

Pipeline position:

    Diarization → Speaker segments → (this module) → Speaker name / label
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.config.config import settings

logger = logging.getLogger(__name__)


def slugify_name(name: str) -> str:
    """Stable lowercase slug used as a profile's ``speaker_key`` (unique per distinct name).

    ASCII names keep the slug they always had ("Raj Kumar" -> "raj_kumar"), so existing profiles
    still match. A name with characters outside ASCII (Devanagari, Telugu, Tamil, accented Latin …)
    used to lose them: "राज" and "सीता" both became "speaker", so enrolling the second person
    silently AVERAGED their voice into the first person's profile. Such names now get a short
    stable hash of the full (NFKC, case-folded) name appended, keeping distinct people distinct.
    """
    import hashlib
    import unicodedata

    original = (name or "").strip()
    folded = unicodedata.normalize("NFKC", original).casefold()
    slug = re.sub(r"[^a-z0-9]+", "_", original.lower()).strip("_")
    if original and not original.isascii():
        digest = hashlib.sha1(folded.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]  # a label, not a secret
        return f"{slug or 'speaker'}_{digest}"
    return slug or "speaker"


# ─────────────────────────────────────────────────────────────────────────────
# Profile loading
# ─────────────────────────────────────────────────────────────────────────────
def load_active_voice_profiles(db=None) -> List[Dict[str, Any]]:
    """Return active voice profiles as
    ``[{"id", "name", "speaker_key", "embedding": list[float], "engine"}]``.

    Opens its own short-lived session when ``db`` is not supplied. Never raises —
    returns ``[]`` if enrollment is disabled or the table is missing.
    """
    if not getattr(settings, "VOICE_ENROLLMENT_ENABLED", True):
        return []

    close_after = False
    if db is None:
        try:
            from backend.database.connection import SessionLocal
            db = SessionLocal()
            close_after = True
        except Exception as e:  # pragma: no cover - DB not configured
            logger.debug("[SpeakerID] no DB session available: %s", e)
            return []

    try:
        from backend.models.models import VoiceProfile

        rows = (
            db.query(VoiceProfile)
            .filter(VoiceProfile.is_active.is_(True))
            .all()
        )
        profiles: List[Dict[str, Any]] = []
        for r in rows:
            try:
                emb = json.loads(r.embedding_json) if r.embedding_json else None
            except Exception:
                emb = None
            if not emb:
                continue
            profiles.append(
                {
                    "id": r.id,
                    "name": r.name,
                    "speaker_key": r.speaker_key,
                    "embedding": emb,
                    "engine": r.engine,
                }
            )
        return profiles
    except Exception as e:
        logger.debug("[SpeakerID] could not load voice profiles: %s", e)
        return []
    finally:
        if close_after:
            try:
                db.close()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────
def identify_speakers(
    audio_filepath: str,
    speaker_turns: List[Dict[str, Any]],
    db=None,
    profiles: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Map each diarized speaker label to a registered identity when confident.

    Args:
        audio_filepath: the 16 kHz mono meeting WAV.
        speaker_turns:  ``[{"speaker": "Speaker_1", "start": 0.0, "end": 12.3}, …]``
        db:             optional SQLAlchemy session (else one is opened).
        profiles:       optional pre-loaded profile list (skips the DB read).

    Returns ``{speaker_label: {"speaker_name", "confidence", "profile_id",
    "similarity", "runner_up", "reason"}}`` for EVERY input speaker label.
    ``speaker_name`` is ``None`` when no reliable match exists.
    """
    labels = list(dict.fromkeys(t["speaker"] for t in speaker_turns if t.get("speaker")))
    out: Dict[str, Dict[str, Any]] = {
        lb: {"speaker_name": None, "confidence": None, "profile_id": None,
             "similarity": 0.0, "runner_up": 0.0, "reason": "no voice profiles"}
        for lb in labels
    }
    if not labels:
        return out

    if profiles is None:
        profiles = load_active_voice_profiles(db)
    if not profiles:
        return out

    threshold = float(getattr(settings, "VOICE_MATCH_THRESHOLD", 0.62))
    margin = float(getattr(settings, "VOICE_MATCH_MARGIN", 0.06))
    min_speech = float(getattr(settings, "VOICE_MIN_MATCH_SPEECH_SEC", 2.0))

    from backend.services.diarization_service import (
        compute_file_speaker_embedding,
        cosine_similarity,
    )

    # Per-speaker mean embedding, then a full similarity matrix vs every profile.
    sim_matrix: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
    for lb in labels:
        ranges = [
            (float(t["start"]), float(t["end"]))
            for t in speaker_turns
            if t.get("speaker") == lb and float(t.get("end", 0)) > float(t.get("start", 0))
        ]
        emb_res = compute_file_speaker_embedding(
            audio_filepath, time_ranges=ranges, min_speech_sec=min_speech
        )
        if not emb_res.get("ok"):
            out[lb]["reason"] = f"embedding unavailable ({emb_res.get('error')})"
            sim_matrix[lb] = []
            continue
        sims = sorted(
            ((cosine_similarity(emb_res["embedding"], p["embedding"]), p) for p in profiles),
            key=lambda x: x[0],
            reverse=True,
        )
        sim_matrix[lb] = sims

    # Greedy, conflict-free assignment: strongest (speaker, profile) pair first,
    # so one profile is never assigned to two different diarized speakers.
    candidates: List[Tuple[float, str, Dict[str, Any], float]] = []
    for lb, sims in sim_matrix.items():
        if not sims:
            continue
        best_sim, best_p = sims[0]
        second = sims[1][0] if len(sims) > 1 else 0.0
        candidates.append((best_sim, lb, best_p, second))
    candidates.sort(key=lambda x: x[0], reverse=True)

    used_profiles: set = set()
    assigned_labels: set = set()
    for best_sim, lb, best_p, second in candidates:
        if lb in assigned_labels:
            continue
        if best_p["id"] in used_profiles:
            # profile already taken by a stronger speaker — try this speaker's
            # next-best profile that is still free and still clears the gate.
            alt = next(
                (
                    (s, p) for s, p in sim_matrix[lb]
                    if p["id"] not in used_profiles
                ),
                None,
            )
            if alt is None:
                out[lb]["reason"] = "best profile taken by a closer speaker"
                continue
            best_sim, best_p = alt
            free_sims = [s for s, p in sim_matrix[lb] if p["id"] not in used_profiles]
            second = free_sims[1] if len(free_sims) > 1 else 0.0

        if best_sim >= threshold and (best_sim - second) >= margin:
            out[lb].update(
                speaker_name=best_p["name"],
                confidence=round(float(best_sim), 4),
                profile_id=best_p["id"],
                similarity=round(float(best_sim), 4),
                runner_up=round(float(second), 4),
                reason="matched",
            )
            used_profiles.add(best_p["id"])
            assigned_labels.add(lb)
        else:
            why = (
                f"similarity {best_sim:.3f} < threshold {threshold:.2f}"
                if best_sim < threshold
                else f"ambiguous (margin {best_sim - second:.3f} < {margin:.2f})"
            )
            out[lb].update(similarity=round(float(best_sim), 4),
                           runner_up=round(float(second), 4), reason=why)

    named = {k: v["speaker_name"] for k, v in out.items() if v["speaker_name"]}
    logger.info(
        "[SpeakerID] %d/%d diarized speaker(s) matched to a registered voice: %s",
        len(named), len(labels), named or "—",
    )
    return out
