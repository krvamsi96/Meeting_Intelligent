"""Voice enrollment API — register participant voices for speaker identification.

    POST   /api/v1/voice-profiles                 enroll (multipart: name, file)
    GET    /api/v1/voice-profiles                 list all profiles
    GET    /api/v1/voice-profiles/{id}            one profile
    POST   /api/v1/voice-profiles/{id}/samples    add another reference sample
    DELETE /api/v1/voice-profiles/{id}            remove a profile
"""

import logging
import os
import tempfile
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.orm import Session

from backend.api.deps import get_db
from backend.config.config import settings
from backend.utils.rate_limit import upload_rate_limit
from backend.services.voice_enrollment_service import (
    VoiceEnrollmentError,
    delete_profile,
    enroll_participant,
    get_profile,
    list_profiles,
    profile_to_dict,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/voice-profiles", tags=["voice-profiles"])

_ALLOWED_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac", ".opus", ".webm",
                ".mp4", ".mov", ".mkv"}
_MAX_SAMPLE_MB = 50


async def _spool_upload(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "")[1].lower() or ".wav"
    if ext not in _ALLOWED_EXT:
        raise HTTPException(status_code=400, detail=f"Unsupported audio type '{ext}'.")
    tmp_dir = tempfile.gettempdir()
    tmp_path = os.path.join(tmp_dir, f"voice_enroll_{uuid.uuid4().hex}{ext}")
    size = 0
    try:
        with open(tmp_path, "wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > _MAX_SAMPLE_MB * 1024 * 1024:
                    raise HTTPException(status_code=400, detail=f"Sample exceeds {_MAX_SAMPLE_MB} MB.")
                await run_in_threadpool(fh.write, chunk)
    except BaseException:
        # Oversize, client disconnect, disk error: never leave the partial spool file behind.
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return tmp_path


@router.post("", dependencies=[Depends(upload_rate_limit)])
@router.post("/", dependencies=[Depends(upload_rate_limit)])
async def create_voice_profile(
    name: str = Form(..., description="Participant display name, e.g. 'Raj'"),
    file: UploadFile = File(..., description="Reference voice sample (audio or video)"),
    notes: str = Form(None),
    enrolled_by: str = Form(None),
    replace: bool = Form(False, description="Discard existing samples for this name"),
    db: Session = Depends(get_db),
):
    """Enroll (or update) a participant's voice profile from one reference sample."""
    if not getattr(settings, "VOICE_ENROLLMENT_ENABLED", True):
        raise HTTPException(status_code=403, detail="Voice enrollment is disabled.")
    tmp_path = await _spool_upload(file)
    try:
        # Embedding extraction (ffmpeg + TitaNet) is heavy and synchronous: run it in the
        # threadpool. Called directly from this `async def` it froze the whole server.
        profile = await run_in_threadpool(
            enroll_participant,
            db, name=name, sample_path=tmp_path,
            enrolled_by=enrolled_by, notes=notes, replace=replace,
        )
        return {"status": "success", "profile": profile_to_dict(profile)}
    except VoiceEnrollmentError as e:
        raise HTTPException(status_code=422, detail=f"Enrollment failed: {e}") from e
    except Exception as e:
        logger.error("[voice-profiles] enroll error: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Internal error during enrollment.") from e
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@router.post("/{profile_id}/samples", dependencies=[Depends(upload_rate_limit)])
async def add_voice_sample(
    profile_id: int,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Fold an additional reference sample into an existing profile (running mean)."""
    prof = get_profile(db, profile_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    tmp_path = await _spool_upload(file)
    try:
        updated = await run_in_threadpool(
            enroll_participant, db, name=prof.name, sample_path=tmp_path, replace=False
        )
        return {"status": "success", "profile": profile_to_dict(updated)}
    except VoiceEnrollmentError as e:
        raise HTTPException(status_code=422, detail=f"Sample rejected: {e}") from e
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@router.get("")
@router.get("/")
def get_voice_profiles(db: Session = Depends(get_db)):
    profiles = list_profiles(db)
    return {"count": len(profiles), "profiles": profiles}


@router.get("/{profile_id}")
def get_voice_profile(profile_id: int, db: Session = Depends(get_db)):
    prof = get_profile(db, profile_id)
    if not prof:
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    return profile_to_dict(prof)


@router.delete("/{profile_id}")
def remove_voice_profile(profile_id: int, db: Session = Depends(get_db)):
    if not delete_profile(db, profile_id):
        raise HTTPException(status_code=404, detail="Voice profile not found.")
    return {"status": "success", "deleted": profile_id}
