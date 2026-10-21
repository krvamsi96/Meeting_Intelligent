"""Voice enrollment — register a participant's voice as a reusable profile.

    Participant name  +  reference voice sample
              │
              ▼   audio → 16 kHz mono → VAD → TitaNet windows → mean → L2-norm
        Speaker embedding
              │
              ▼
        Stored VoiceProfile  (DB row + reference sample on disk)

Enrolling the same name again folds the new sample into the existing profile
(running mean over samples). Nothing is hard-coded; every embedding is derived
from the uploaded audio by the same embedder the diarizer uses.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from backend.config.config import settings
from backend.services.speaker_identification_service import slugify_name

logger = logging.getLogger(__name__)


class VoiceEnrollmentError(RuntimeError):
    """Raised when a reference sample cannot be turned into a usable profile."""


def _profile_dir() -> str:
    d = getattr(settings, "VOICE_PROFILE_DIR", None) or os.path.join(
        getattr(settings, "UPLOAD_DIR", "uploads"), "voice_profiles"
    )
    os.makedirs(d, exist_ok=True)
    return d


def _remove_quietly(path: str) -> None:
    try:
        if path and os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def _prepare_wav(src_path: str, dest_stub: str) -> str:
    """Return a 16 kHz mono PCM wav for ``src_path`` (transcoding if needed)."""
    ext = os.path.splitext(src_path)[1].lower()
    dest_wav = f"{dest_stub}.wav"
    if ext == ".wav":
        try:
            import soundfile as sf
            info = sf.info(src_path)
            if int(info.samplerate) == 16000 and int(info.channels) == 1:
                shutil.copyfile(src_path, dest_wav)
                return dest_wav
        except Exception:
            pass
    from backend.services.audio_service import extract_audio

    if not extract_audio(src_path, dest_wav):
        raise VoiceEnrollmentError("could not decode / transcode the reference audio")
    return dest_wav


def _embed(wav_path: str) -> Dict[str, Any]:
    from backend.services.diarization_service import compute_file_speaker_embedding

    res = compute_file_speaker_embedding(
        wav_path,
        time_ranges=None,
        min_speech_sec=float(getattr(settings, "VOICE_MIN_ENROLL_SPEECH_SEC", 3.0)),
    )
    if not res.get("ok"):
        raise VoiceEnrollmentError(res.get("error") or "embedding failed")
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
def enroll_participant(
    db,
    name: str,
    sample_path: str,
    enrolled_by: Optional[str] = None,
    notes: Optional[str] = None,
    replace: bool = False,
) -> "Any":
    """Create or update the :class:`VoiceProfile` for ``name`` from ``sample_path``.

    ``replace=True`` discards any previous samples for that name; otherwise the
    new sample is averaged into the existing profile.
    """
    from backend.models.models import VoiceProfile

    name = (name or "").strip()
    if not name:
        raise VoiceEnrollmentError("participant name is required")
    if not sample_path or not os.path.exists(sample_path):
        raise VoiceEnrollmentError(f"sample file not found: {sample_path}")

    key = slugify_name(name)
    stub = os.path.join(_profile_dir(), f"{key}_{uuid.uuid4().hex[:8]}")
    wav_path = _prepare_wav(sample_path, stub)

    try:
        emb_res = _embed(wav_path)
    except BaseException:
        # A sample that yields no usable embedding must not leave its transcoded WAV behind.
        _remove_quietly(wav_path)
        raise
    new_vec = np.asarray(emb_res["embedding"], dtype=np.float64)

    existing = (
        db.query(VoiceProfile).filter(VoiceProfile.speaker_key == key).one_or_none()
    )
    now = datetime.now(timezone.utc)

    if existing and not replace:
        prev_vec = np.asarray(json.loads(existing.embedding_json), dtype=np.float64)
        n_prev = max(1, int(existing.sample_count or 1))
        merged = (prev_vec * n_prev + new_vec) / (n_prev + 1)
        nrm = float(np.linalg.norm(merged)) or 1.0
        merged = merged / nrm
        paths = json.loads(existing.sample_paths_json or "[]")
        paths.append(wav_path)
        existing.embedding_json = json.dumps([float(x) for x in merged.tolist()])
        existing.embedding_dim = int(merged.shape[0])
        existing.engine = emb_res["engine"]
        existing.sample_count = n_prev + 1
        existing.sample_paths_json = json.dumps(paths)
        existing.total_speech_sec = float(existing.total_speech_sec or 0.0) + float(emb_res["speech_sec"])
        existing.updated_at = now
        if notes:
            existing.notes = notes
        db.add(existing)
        db.commit()
        db.refresh(existing)
        logger.info("[VoiceEnroll] '%s' updated (%d samples, engine=%s).",
                    name, existing.sample_count, existing.engine)
        return existing

    if existing and replace:
        # "Discard existing samples" must actually discard them, not just the DB row.
        for old_path in json.loads(existing.sample_paths_json or "[]"):
            _remove_quietly(old_path)
        db.delete(existing)
        db.flush()

    profile = VoiceProfile(
        name=name,
        speaker_key=key,
        embedding_json=json.dumps([float(x) for x in new_vec.tolist()]),
        embedding_dim=int(new_vec.shape[0]),
        engine=emb_res["engine"],
        sample_count=1,
        sample_paths_json=json.dumps([wav_path]),
        total_speech_sec=float(emb_res["speech_sec"]),
        enrolled_by=enrolled_by,
        notes=notes,
        is_active=True,
        created_at=now,
        updated_at=now,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)
    logger.info("[VoiceEnroll] '%s' enrolled (dim=%d, engine=%s, %.1fs speech).",
                name, profile.embedding_dim, profile.engine, emb_res["speech_sec"])
    return profile


def list_profiles(db) -> List[Dict[str, Any]]:
    from backend.models.models import VoiceProfile

    rows = db.query(VoiceProfile).order_by(VoiceProfile.name.asc()).all()
    return [profile_to_dict(r) for r in rows]


def get_profile(db, profile_id: int):
    from backend.models.models import VoiceProfile

    return db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).one_or_none()


def delete_profile(db, profile_id: int) -> bool:
    from backend.models.models import VoiceProfile

    row = db.query(VoiceProfile).filter(VoiceProfile.id == profile_id).one_or_none()
    if not row:
        return False
    # best-effort cleanup of stored samples
    for p in json.loads(row.sample_paths_json or "[]"):
        try:
            if os.path.isfile(p):
                os.remove(p)
        except Exception:
            pass
    db.delete(row)
    db.commit()
    return True


def profile_to_dict(row) -> Dict[str, Any]:
    return {
        "id": row.id,
        "name": row.name,
        "speaker_key": row.speaker_key,
        "embedding_dim": row.embedding_dim,
        "engine": row.engine,
        "sample_count": row.sample_count,
        "total_speech_sec": round(float(row.total_speech_sec or 0.0), 2),
        "enrolled_by": row.enrolled_by,
        "notes": row.notes,
        "is_active": row.is_active,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
