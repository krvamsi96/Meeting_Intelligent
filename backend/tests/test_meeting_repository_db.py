"""Repository behaviour against the real PostgreSQL schema.

Every test runs inside an outer transaction that is ALWAYS rolled back (the session uses
SAVEPOINTs, so the repository's own commit() calls do not escape it): nothing is left behind in
the database the app is configured for. Skipped when the schema has not been created yet.
"""

import uuid

import pytest


@pytest.fixture()
def db_session():
    from sqlalchemy import inspect
    from sqlalchemy.orm import Session

    from backend.database.connection import engine

    try:
        if not inspect(engine).has_table("meetings"):
            pytest.skip("meetings table does not exist yet (start the app once to create the schema)")
    except Exception as exc:  # database unreachable
        pytest.skip(f"database not reachable: {exc}")

    conn = engine.connect()
    outer = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        session.close()
        outer.rollback()
        conn.close()


def _public_id() -> str:
    return "9" + str(uuid.uuid4().int)[:14]  # numeric, like real ids, and unique per test


def _make(repo, **over):
    kwargs = dict(
        bharatvc_meeting_id=_public_id(), filename="x.txt", file_type="document",
        filepath="/tmp/x.txt", title="t", status="uploaded",
    )
    kwargs.update(over)
    return repo.create(**kwargs)


def test_creating_a_meeting_whose_public_id_equals_another_meetings_pk_does_not_overwrite_it(db_session):
    """Regression: create() looked the id up with get_by_id(), whose numeric branch matches the
    internal primary key first — so a caller-chosen id such as "5" silently rewrote meeting #5."""
    from backend.repositories import MeetingRepository

    repo = MeetingRepository(db_session)
    victim = _make(repo, title="victim", filename="victim.txt")
    victim_pk, victim_public = victim.id, victim.meeting_id

    intruder = repo.create(
        bharatvc_meeting_id=str(victim_pk), filename="intruder.txt", file_type="document",
        filepath="/tmp/intruder.txt", title="intruder", status="uploaded",
    )
    db_session.refresh(victim)

    assert intruder.id != victim_pk, "a NEW meeting must be created"
    assert victim.title == "victim" and victim.filename == "victim.txt" and victim.meeting_id == victim_public
    assert intruder.meeting_id == str(victim_pk)


def test_reuploading_the_same_public_id_updates_the_same_meeting(db_session):
    from backend.repositories import MeetingRepository

    repo = MeetingRepository(db_session)
    first = _make(repo, title="v1")
    again = repo.create(bharatvc_meeting_id=first.meeting_id, filename="x2.txt", file_type="document",
                        filepath="/tmp/x2.txt", title="v2", status="uploaded")
    assert again.id == first.id and again.title == "v2"


def test_get_by_public_id_never_matches_the_primary_key(db_session):
    from backend.repositories import MeetingRepository

    repo = MeetingRepository(db_session)
    m = _make(repo)
    assert repo.get_by_public_id(m.meeting_id).id == m.id
    # Looking up by the PRIMARY KEY must not find it (another meeting could legitimately own that
    # string as its public id in a shared dev database, so only "not this row" is asserted).
    by_pk_string = repo.get_by_public_id(str(m.id))
    assert by_pk_string is None or by_pk_string.id != m.id
    assert repo.get_by_public_id(None) is None and repo.get_by_public_id("  ") is None


def test_startup_orphan_recovery_covers_processing_and_uploaded_but_not_finished_meetings(db_session):
    from backend.repositories import MeetingRepository

    repo = MeetingRepository(db_session)
    processing = _make(repo, status="processing")
    uploaded = _make(repo, status="uploaded")
    completed = _make(repo, status="completed")
    errored = _make(repo, status="error")

    orphan_ids = {m.id for m in repo.get_orphaned_processing()}
    assert {processing.id, uploaded.id} <= orphan_ids
    assert completed.id not in orphan_ids and errored.id not in orphan_ids


def test_delete_removes_the_meeting_and_its_children(db_session):
    from backend.models.models import ChatSession, Meeting, TranscriptSegment
    from backend.repositories import MeetingRepository

    repo = MeetingRepository(db_session)
    m = _make(repo, status="completed")
    repo.add_segments(m.id, [{"speaker": "A", "start_time": "0", "end_time": "1", "text": "hello"}])
    repo.add_chat_message(m.id, "user", "hi", session_id="s1")
    pk = m.id

    assert repo.delete(pk) is True
    assert db_session.query(Meeting).filter(Meeting.id == pk).first() is None
    assert db_session.query(TranscriptSegment).filter(TranscriptSegment.meeting_id == pk).count() == 0
    assert db_session.query(ChatSession).filter(ChatSession.meeting_id == pk).count() == 0
    assert repo.delete(pk) is False, "deleting twice reports not-found instead of raising"
