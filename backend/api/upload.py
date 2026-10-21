"""Upload and universal URL ingestion endpoints."""

import hashlib
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from backend.api.deps import end_read_transaction, get_meeting_repository, release_session_before_background_task
from backend.config.config import settings
from backend.models.models import Meeting
from backend.repositories import MeetingRepository
from backend.schemas import schemas
from backend.services.url_service import (
    detect_url_content_type,
    download_video_from_url,
)
from backend.utils.date_resolver import _resolve_timezone
from backend.utils.helpers import generate_numeric_meeting_id
from backend.utils.ingestion import classify_file_type, is_supported_file
from backend.utils.rate_limit import upload_rate_limit
from backend.utils.security import validate_file_size, validate_file_type, validate_media_file, validate_url
from backend.workers import PIPELINE_REGISTRY, process_meeting_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Upload"])


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


# A meeting in these states already has a pipeline running or queued. A second submission of the same id would
# overwrite the file that pipeline is reading and start another pipeline on the same rows.
_BUSY_STATUSES = ("uploaded", "processing")


@router.get("/upload/options")
def upload_options():
    """What the upload form may offer: accepted recording types, the size limit, and the language choice.

    ``language_detection`` is false when the configured speech-to-text cannot identify the language itself and no
    default is configured - the user must then pick it. Nothing here names a provider.
    """
    from backend.config.config import MEDIA_EXTENSIONS
    from backend.providers import provider_factory
    from backend.services.language_service import LANGUAGE_NAMES
    from backend.services.model_router import model_router

    try:
        stt = provider_factory.stt_provider_name()
    except Exception:  # noqa: BLE001 - not configured: processing will fail with that reason, upload still works
        stt = None
    default_language = (model_router._get_dynamic_config()["airawat"].get("language") or "").strip().lower() or None
    detects = stt is not None and stt != "airawat"
    return {
        "accepted_extensions": sorted(MEDIA_EXTENSIONS),
        "max_file_size_mb": settings.MAX_FILE_SIZE_MB,
        "language_detection": detects,
        "default_language": default_language,
        "language_required": not detects and not default_language,
        "languages": [{"code": c, "name": n} for c, n in sorted(LANGUAGE_NAMES.items(), key=lambda kv: kv[1])],
    }


@router.post("/upload", dependencies=[Depends(upload_rate_limit)])
async def upload_meeting(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Audio/Video meeting recording or document file"),
    meeting_id: Optional[str] = Form(None, description="Meeting ID provided by caller"),
    bharatvc_meeting_id: Optional[str] = Form(None, description="Unique BharatVC meeting identifier"),
    meeting_title: Optional[str] = Form(None, description="Optional meeting title"),
    title: Optional[str] = Form(None, description="Alias of meeting_title (used by the web app)"),
    metadata: Optional[str] = Form(None, description="Optional JSON metadata"),
    num_speakers: Optional[int] = Form(
        None,
        description=(
            "Optional: the ACTUAL number of speakers in the recording, if known. "
            "Diarization always auto-detects the speaker count dynamically; this "
            "only lets a caller who already knows the true count skip that "
            "estimation. Omit to let the pipeline determine it automatically."
        ),
    ),
    language: Optional[str] = Form(
        None,
        description=(
            "Spoken language of the recording as an ISO 639-1 code (e.g. hi, te, ml, ta, kn, en). AIRAWAT "
            "Whisper cannot detect the language, so it transcribes in this one. Omit to use the configured "
            "default (AIRAWAT_WHISPER_LANGUAGE)."
        ),
    ),
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """
    Upload an audio/video file OR document with Meeting ID.

    - Accepts multipart/form-data
    - Preserves `meeting_id` or `bharatvc_meeting_id` exactly as received
    - Stores `bharatvc_meeting_id` in PostgreSQL without generating any new random IDs or UUIDs
    """
    clean_num_speakers: Optional[int] = None
    if num_speakers is not None and num_speakers >= 1:
        clean_num_speakers = num_speakers

    from backend.services.language_service import language_name, normalize_language_code

    try:
        spoken_language = normalize_language_code(language)
    except ValueError as lang_err:
        raise HTTPException(status_code=422, detail=str(lang_err))

    raw_id = (bharatvc_meeting_id or meeting_id or "").strip()
    b_id = generate_numeric_meeting_id(raw_id if raw_id else None)
    logger.info(f"Received/Generated numeric meeting_id = {b_id}")

    original_filename = file.filename or "uploaded_file"
    file_ext = os.path.splitext(original_filename)[1].lower() or ".mp4"
    safe_base = "".join(c for c in os.path.splitext(original_filename)[0] if c.isalnum() or c in ("-", "_")).rstrip()
    stored_filename = f"{b_id}_{safe_base}{file_ext}" if safe_base else f"{b_id}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, stored_filename)

    if not is_supported_file(original_filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext or 'unknown'}.",
        )

    file_type = classify_file_type(original_filename)

    # Re-uploading an id whose pipeline is still running would overwrite the file the
    # background worker is reading and start a SECOND pipeline on the same meeting.
    in_flight = repo.get_by_public_id(b_id)
    busy = in_flight is not None and in_flight.status in _BUSY_STATUSES
    # The check above opened a transaction; the upload below can take minutes, so do not sit "idle in transaction".
    end_read_transaction(repo.db)
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"Meeting {b_id} is still being processed. Wait for it to finish before re-uploading.",
        )

    upload_start_time = datetime.now(timezone.utc)
    start_t = time.perf_counter()

    CHUNK_SIZE = settings.UPLOAD_CHUNK_SIZE_BYTES
    MAX_FILE_SIZE = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    bytes_written = 0
    file_hasher = hashlib.sha256()

    # Stream into a temp file in the SAME directory (keeps the extension so type validation
    # still sees it) and move it into place only once it is complete and validated. Writing
    # straight to ``file_path`` meant a failed/aborted upload of an existing meeting id
    # truncated that meeting's original file, and any early exit left a partial file behind.
    tmp_path = os.path.join(settings.UPLOAD_DIR, f".upload-{uuid.uuid4().hex[:8]}-{stored_filename}")
    committed = False
    try:
        with open(tmp_path, "wb") as buffer:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400, detail=f"File too large. Maximum size is {settings.MAX_FILE_SIZE_MB}MB."
                    )
                file_hasher.update(chunk)
                # Disk I/O off the event loop so a big upload does not stall other requests.
                await run_in_threadpool(buffer.write, chunk)

        if bytes_written == 0:
            raise HTTPException(status_code=400, detail="The file is empty.")

        upload_end_time = datetime.now(timezone.utc)
        upload_duration_sec = round(time.perf_counter() - start_t, 3)
        file_sha256 = file_hasher.hexdigest()

        file_size_mb = bytes_written / (1024 * 1024)
        logger.info(
            f"Meeting Uploaded: meeting_id={b_id} filename={original_filename} ({file_size_mb:.1f} MB, sha256={file_sha256[:12]}...) in {upload_duration_sec}s [{file_type}]"
        )

        # Validate file content type
        validate_file_type(tmp_path, file_type)
        validate_file_size(tmp_path, max_size_mb=settings.MAX_FILE_SIZE_MB)
        if file_type == "media":
            # ffprobe off the event loop: a corrupt or audio-less recording is rejected here with a clear 400
            # instead of becoming a meeting that fails later inside the background pipeline.
            await run_in_threadpool(validate_media_file, tmp_path)

        os.replace(tmp_path, file_path)
        committed = True
    finally:
        # Also runs on client disconnect (CancelledError) - never leave a partial file.
        if not committed:
            _remove_quietly(tmp_path)

    requested_title = (meeting_title or title or "").strip()[:300]
    clean_title = requested_title or original_filename

    # The displayed meeting date/time is in the deployment timezone (DEFAULT_TIMEZONE, default Asia/Kolkata) - the same
    # zone action-item deadlines are resolved in. It used to be UTC with no label, so a meeting recorded at 11:19 IST
    # said "05:49 AM" in its MoM and near midnight even got the wrong date.
    local_start = upload_start_time.astimezone(_resolve_timezone(settings.DEFAULT_TIMEZONE))

    # Duplicate pre-check: inspect for previously completed identical files
    is_potential_dup = False
    existing_dup_id = None
    try:
        dup_match = (
            repo.db.query(Meeting)
            .filter(
                Meeting.file_size_bytes == bytes_written,
                Meeting.filename == original_filename,
                Meeting.status == "completed",
            )
            .first()
        )
        if dup_match:
            is_potential_dup = True
            existing_dup_id = dup_match.id
            logger.info(
                f"[Upload] Potential duplicate detected: existing meeting_id={dup_match.id} "
                f"matches filename '{original_filename}' and size {bytes_written} bytes."
            )
    except Exception as dup_err:
        logger.warning(f"[Upload] Duplicate pre-check failed ({type(dup_err).__name__}: {dup_err}); continuing.")
        repo.db.rollback()   # a failed statement leaves the transaction aborted; repo.create() needs a clean one

    logger.info(f"Saving meeting_id = {b_id}")
    meeting = repo.create(
        bharatvc_meeting_id=b_id,
        meeting_id=b_id,
        display_id=b_id,
        filename=original_filename,
        file_type=file_type,
        filepath=file_path,
        title=clean_title,
        meeting_date=local_start.strftime("%Y-%m-%d"),
        meeting_time=local_start.strftime("%I:%M %p"),
        status="uploaded",
        file_size_bytes=bytes_written,
        upload_time=upload_start_time,
        upload_start_time=upload_start_time,
        upload_end_time=upload_end_time,
        upload_duration_sec=upload_duration_sec,
        db_meeting_saved=True,
        num_speakers=clean_num_speakers,
        # The spoken language AIRAWAT Whisper will transcribe in (None: the configured default is used).
        source_language=spoken_language,
        source_language_name=language_name(spoken_language) if spoken_language else None,
    )

    pipeline = PIPELINE_REGISTRY.get(file_type, process_meeting_pipeline)
    meeting_pk = meeting.id   # one refresh SELECT; everything the response needs is read before the session closes
    release_session_before_background_task(repo.db)
    background_tasks.add_task(pipeline, meeting_pk, file_path)

    return {
        "message": "File uploaded successfully. Processing started in background.",
        "id": meeting_pk,
        "meeting_id": b_id,
        "bharatvc_meeting_id": b_id,
        "filename": original_filename,
        "status": "processing",
        "file_type": file_type,
        "file_size_mb": round(file_size_mb, 2),
        "potential_duplicate": is_potential_dup,
        "existing_duplicate_meeting_id": existing_dup_id,
    }


@router.post("/process-url", dependencies=[Depends(upload_rate_limit)])
def process_url(
    request: schemas.UrlRequest,
    background_tasks: BackgroundTasks,
    repo: MeetingRepository = Depends(get_meeting_repository),
):
    """Universal URL Ingestion - auto-detects content type."""
    url = request.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    # SSRF protection — reject private/localhost/non-HTTP(S) URLs
    url = validate_url(url)

    content_type = detect_url_content_type(url)
    logger.info(f"URL Processing: '{url}' auto-detected as [{content_type}]")

    raw_url_id = (request.bharatvc_meeting_id or request.meeting_id or "").strip()
    b_id = generate_numeric_meeting_id(raw_url_id if raw_url_id else None)

    in_flight = repo.get_by_public_id(b_id)
    busy = in_flight is not None and in_flight.status in _BUSY_STATUSES
    end_read_transaction(repo.db)   # the download below can take minutes; do not sit "idle in transaction" through it
    if busy:
        raise HTTPException(
            status_code=409,
            detail=f"Meeting {b_id} is still being processed. Wait for it to finish before re-submitting.",
        )

    if content_type == "video":
        try:
            file_path, orig_filename = download_video_from_url(url, settings.UPLOAD_DIR)
            clean_name = (
                os.path.splitext(orig_filename)[0]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )

            meeting = repo.create(
                bharatvc_meeting_id=b_id,
                meeting_id=b_id,
                display_id=b_id,
                filename=orig_filename,
                file_type="media",
                filepath=file_path,
                title=clean_name,
                status="uploaded",
            )

            # Read everything the response needs, THEN close the session: FastAPI keeps it open until the
            # background pipeline finishes (see release_session_before_background_task).
            meeting_pk = meeting.id
            response = {
                "type": "video",
                "bharatvc_meeting_id": meeting.bharatvc_meeting_id or meeting.meeting_id,
                "meeting_id": meeting.meeting_id,
                "display_id": meeting.display_id,
                "title": meeting.title,
                "message": f"Video URL ingested as Meeting {meeting.display_id}. Processing started.",
            }
            release_session_before_background_task(repo.db)
            background_tasks.add_task(process_meeting_pipeline, meeting_pk, file_path)
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Video URL processing error: {e}", exc_info=True)
            raise HTTPException(
                status_code=400, detail="Failed to process video URL. Please try again."
            ) from e
    else:
        try:
            from backend.services.url_service import download_document_from_url

            local_path, orig_filename, extracted_text = download_document_from_url(
                url, settings.UPLOAD_DIR
            )

            clean_name = (
                os.path.splitext(orig_filename)[0]
                .replace("_", " ")
                .replace("-", " ")
                .title()
            )

            meeting = repo.create(
                bharatvc_meeting_id=b_id,
                meeting_id=b_id,
                display_id=b_id,
                filename=orig_filename,
                file_type="document",
                filepath=local_path,
                title=clean_name,
                status="uploaded",
            )

            pipeline = PIPELINE_REGISTRY.get("document", process_meeting_pipeline)
            meeting_pk = meeting.id
            response = {
                "type": "document",
                "bharatvc_meeting_id": meeting.bharatvc_meeting_id or meeting.meeting_id,
                "meeting_id": meeting.meeting_id,
                "display_id": meeting.display_id,
                "title": meeting.title,
                "message": f"Document URL ingested as {meeting.display_id}. Processing started.",
            }
            release_session_before_background_task(repo.db)
            background_tasks.add_task(pipeline, meeting_pk, local_path)
            return response
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Document URL processing error: {e}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail="Failed to process document/webpage URL. Please check the link and try again.",
            ) from e
