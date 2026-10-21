"""ChromaDB is the only vector database: initialization, persistence, isolation, deduplication, similarity, health.

These tests run the REAL persistent ChromaDB against temporary directories (nothing about the store is mocked).
Only two things are substituted so results are deterministic and no transformer model has to load: the embedding
function (a hashed bag-of-words - texts sharing words are close, exactly what the assertions need) and the lookup of a
meeting's identity in PostgreSQL.
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from backend.config.config import settings
from backend.services import similarity_service, vector_backfill as vb, vector_service as vs
from backend.services.vector_service import (
    CHUNK_COLLECTION,
    MEETING_COLLECTION,
    VectorService,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DIM = 384
BOW_SIGNATURE = "test-bow:384"

REQUIRED_METADATA = {"meeting_id", "bharatvc_meeting_id", "display_id", "chunk_id", "source", "document_type"}

# Three meetings about unrelated things, plus a fourth that is a follow-up of the first.
TOPICS = {
    1: ["database migration to postgres must finish before march",
        "the migration runbook covers schema cutover and rollback",
        "cloud hosting budget approved for the migration project"],
    2: ["autumn marketing campaign launches in october",
        "influencer partnerships and social media spend for the campaign",
        "brand creative approved by the marketing team"],
    3: ["hiring plan for two backend engineers",
        "recruiting pipeline and interview panel schedule",
        "onboarding checklist for the new engineers"],
    4: ["follow up on the database migration to postgres before march",
        "rollback plan and schema cutover for the migration runbook",
        "hosting budget for the migration project"],
}
IDENTITY = {
    1: {"bharatvc_meeting_id": "BV-001", "display_id": "MTG-A", "source": "transcript"},
    2: {"bharatvc_meeting_id": "BV-002", "display_id": "MTG-B", "source": "transcript"},
    3: {"bharatvc_meeting_id": "BV-003", "display_id": "MTG-C", "source": "document"},
    4: {"bharatvc_meeting_id": "BV-004", "display_id": "MTG-D", "source": "transcript"},
}


def bow_embed(text: str):
    vec = np.zeros(DIM)
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm else [1.0] + [0.0] * (DIM - 1)


def transcript(meeting_no: int, n: int = 30, tag: str = ""):
    """n segments (>= 20 -> multi-chunk, structure-aware chunking) drawn only from this meeting's topics."""
    topics = TOPICS[meeting_no]
    return [
        {"speaker": f"Speaker_{i % 3 + 1}", "text": f"{topics[i % len(topics)]} {tag} point {i}",
         "start_time": float(i * 10), "end_time": float(i * 10 + 9)}
        for i in range(n)
    ]


@pytest.fixture()
def make_service(tmp_path, monkeypatch):
    """Factory for a VectorService on its own directory; every service is closed afterwards."""
    created = []
    identities = dict(IDENTITY)

    def _make(subdir: str = "chroma", dim: int = DIM) -> VectorService:
        monkeypatch.setattr(settings, "CHROMA_PERSIST_DIRECTORY", str(tmp_path / subdir))
        monkeypatch.setattr(settings, "VECTOR_DB", "chromadb")
        monkeypatch.setattr("backend.services.model_loader.get_embedding_dimension", lambda: dim)
        svc = object.__new__(VectorService)
        svc._reset_state()
        svc.identities = identities                                   # tests may edit it (simulate a reset database)
        svc._lookup_meeting = lambda mid: identities.get(int(mid))
        svc.generate_embedding = bow_embed
        svc.generate_embeddings_batch = lambda texts: [bow_embed(t) for t in texts]
        svc._embedding_signature = lambda: BOW_SIGNATURE              # the "embedding space" of bow_embed
        svc.ensure_initialized()
        created.append(svc)
        return svc

    yield _make
    for svc in created:
        svc.close()


def index_all(svc, meetings=(1, 2, 3, 4)):
    for m in meetings:
        ident = IDENTITY[m]
        svc.add_transcript_chunks(m, transcript(m))
        svc.add_meeting_vector(
            meeting_id=m, meeting_code=ident["display_id"], filename=f"meeting_{m}.mp4", title=f"Meeting {m}",
            text=" ".join(TOPICS[m] * 3), mom_summary=TOPICS[m][0],
            extra_metadata={"bharatvc_meeting_id": ident["bharatvc_meeting_id"], "display_id": ident["display_id"],
                            "file_type": "document" if ident["source"] == "document" else "media"},
        )


def subprocess_env(directory: Path) -> dict:
    return {**os.environ, "PYTHONPATH": str(REPO_ROOT), "PYTHONUTF8": "1", "VECTOR_DB": "chromadb",
            "CHROMA_PERSIST_DIRECTORY": str(directory)}


def run_python(code: str, directory: Path) -> str:
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, env=subprocess_env(directory),
                          capture_output=True, text=True, timeout=180)
    assert proc.returncode == 0, proc.stderr[-1500:]
    return proc.stdout.strip().splitlines()[-1]


# ── initialization ──────────────────────────────────────────────────────────────────────────────────
def test_startup_creates_the_configured_persistent_directory_and_reports_available(make_service, tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger=vs.logger.name):
        svc = make_service("nested/server/chroma")
    target = tmp_path / "nested" / "server" / "chroma"
    assert target.is_dir() and (target / "chroma.sqlite3").exists()          # persistent, on disk, created on demand
    assert svc.health() == {"backend": "chromadb", "status": "available"}
    assert svc._meetings.name == MEETING_COLLECTION and svc._chunks.name == CHUNK_COLLECTION
    lines = [r.getMessage() for r in caplog.records]
    assert "Vector database: ChromaDB" in lines and "ChromaDB status: available" in lines
    assert f"ChromaDB path: {target}" in lines
    assert [r for r in caplog.records if r.name == vs.logger.name and r.levelno >= logging.WARNING] == []


def test_collections_use_cosine_distance(make_service):
    svc = make_service()
    for col in (svc._meetings, svc._chunks):
        assert vs._collection_space(col) == "cosine"


def test_the_directory_comes_from_the_environment_setting_not_from_the_code(make_service, tmp_path):
    local, server = make_service("local_chroma"), make_service("server_chroma")
    local.add_transcript_chunks(1, transcript(1))
    assert local.path == str(tmp_path / "local_chroma") and server.path == str(tmp_path / "server_chroma")
    assert local._chunks.count() > 0 and server._chunks.count() == 0        # separate stores, chosen purely by config


def test_only_chromadb_is_a_supported_vector_database(make_service, tmp_path, monkeypatch):
    svc = make_service()
    svc.close()
    monkeypatch.setattr(settings, "VECTOR_DB", "pgvector")
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIRECTORY", str(tmp_path / "never_created"))
    svc.ensure_initialized()
    health = svc.health()
    assert health["backend"] == "chromadb" and health["status"] == "unavailable"
    assert "only vector database" in health["error"] and "VECTOR_DB=chromadb" in health["error"]
    assert not (tmp_path / "never_created").exists()


def test_the_placeholder_from_env_example_is_rejected_instead_of_creating_a_bogus_directory(make_service, tmp_path,
                                                                                            monkeypatch):
    svc = make_service()
    svc.close()
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIRECTORY", str(tmp_path / "<configure-for-environment>"))
    svc.ensure_initialized()
    health = svc.health()
    assert health["status"] == "unavailable" and "placeholder" in health["error"]
    assert list(tmp_path.glob("*configure*")) == []


def test_an_unusable_directory_is_reported_with_a_clear_reason_and_never_raises(make_service, tmp_path, monkeypatch):
    svc = make_service()
    svc.close()
    blocker = tmp_path / "a_file"
    blocker.write_text("not a directory")
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIRECTORY", str(blocker / "chroma"))
    svc.ensure_initialized()
    health = svc.health()
    assert health["status"] == "unavailable" and "cannot be created" in health["error"]
    # every operation degrades to an empty result instead of raising
    assert svc.search_transcript_chunks(1, "anything at all") == [] and svc.has_transcript_chunks(1) is False
    assert svc.find_similar_meetings(1, "some meeting text") == [] and svc.get_semantic_candidates(1, "text") == []
    assert svc.add_transcript_chunks(1, transcript(1)) == 0
    svc.add_meeting_vector(meeting_id=1, meeting_code="1", filename="f", title="t", text="hello world")
    svc.delete_meeting_vector(1)


def test_an_unavailable_store_recovers_by_itself_once_the_problem_is_fixed(make_service, tmp_path, monkeypatch):
    svc = make_service()
    svc.close()
    monkeypatch.setattr(settings, "VECTOR_DB", "nonsense")
    svc.ensure_initialized()
    assert svc.health()["status"] == "unavailable"
    monkeypatch.setattr(settings, "VECTOR_DB", "chromadb")
    monkeypatch.setattr(vs, "_RETRY_INTERVAL_SEC", 0.0)
    assert svc.health() == {"backend": "chromadb", "status": "available"}


def test_a_different_embedding_dimension_is_reported_and_reset_rebuilds_the_collections(make_service, tmp_path):
    old = make_service("dims", dim=8)
    old._chunks.upsert(ids=["1:x"], embeddings=[[0.1] * 8], documents=["d"], metadatas=[{"meeting_id": 1}])
    old.close()
    svc = make_service("dims", dim=DIM)
    health = svc.health()
    assert health["status"] == "unavailable" and svc.dimension_mismatch is True
    assert "8-dimensional" in health["error"] and "reindex_vectors --reset" in health["error"]
    svc.reset_collections()                                       # the documented recovery
    assert svc.health()["status"] == "available" and svc._chunks.count() == 0 and svc.dimension_mismatch is False


def test_creating_the_singleton_has_no_side_effects(tmp_path, monkeypatch):
    target = tmp_path / "untouched"
    monkeypatch.setattr(settings, "CHROMA_PERSIST_DIRECTORY", str(target))
    monkeypatch.setattr(VectorService, "_instance", None)
    svc = VectorService()
    assert svc._initialized is False and not target.exists()      # importing/creating never touches the disk


# ── persistence ─────────────────────────────────────────────────────────────────────────────────────
def test_vectors_survive_a_restart_of_the_service(make_service):
    svc = make_service("persist")
    index_all(svc, (1, 2))
    before = (svc._chunks.count(), svc._meetings.count(), svc._chunks.get(include=[])["ids"])
    svc.close()                                                   # process stops

    again = make_service("persist")                               # process starts again on the same directory
    assert (again._chunks.count(), again._meetings.count(), again._chunks.get(include=[])["ids"]) == before
    hit = again.search_transcript_chunks(1, "database migration", top_k=3)
    assert hit and all(h["meeting_id"] == 1 for h in hit)


def test_a_new_python_process_sees_the_persisted_vectors(make_service, tmp_path):
    svc = make_service("persist_proc")
    index_all(svc, (1, 2, 3))
    expected = {"chunks": svc._chunks.count(), "meetings": svc._meetings.count()}
    svc.close()
    out = run_python(
        "import json\n"
        "from backend.services.vector_service import VectorService\n"
        "s = object.__new__(VectorService); s._reset_state()\n"
        "print(json.dumps({'health': s.health(), 'chunks': s._chunks.count(), 'meetings': s._meetings.count()}))",
        tmp_path / "persist_proc")
    result = json.loads(out)
    assert result["health"] == {"backend": "chromadb", "status": "available"}
    assert {"chunks": result["chunks"], "meetings": result["meetings"]} == expected


def test_a_second_process_cannot_use_a_directory_that_is_already_owned(make_service, tmp_path):
    svc = make_service("owned")                                   # this process owns the directory
    code = ("import json\n"
            "from backend.services.vector_service import VectorService\n"
            "s = object.__new__(VectorService); s._reset_state()\n"
            "print(json.dumps(s.health()))")
    blocked = json.loads(run_python(code, tmp_path / "owned"))
    assert blocked["status"] == "unavailable" and "already in use by another process" in blocked["error"]
    assert "UVICORN_WORKERS=1" in blocked["error"]
    svc.close()                                                   # ownership is released on shutdown
    assert json.loads(run_python(code, tmp_path / "owned"))["status"] == "available"


# ── deterministic ids / no duplicates ─────────────────────────────────────────────────────────────────
def test_processing_the_same_meeting_twice_creates_no_duplicates(make_service):
    svc = make_service()
    first = svc.add_transcript_chunks(1, transcript(1))
    ids_before = sorted(svc._chunks.get(include=[])["ids"])
    assert first == len(ids_before) > 3
    assert all(i.startswith("1:") for i in ids_before)            # deterministic "<meeting_id>:<chunk_id>"

    for _ in range(3):
        svc.add_transcript_chunks(1, transcript(1))
        svc.add_meeting_vector(meeting_id=1, meeting_code="MTG-A", filename="a.mp4", title="A", text="alpha beta")
    assert sorted(svc._chunks.get(include=[])["ids"]) == ids_before
    assert svc._meetings.get(include=[])["ids"] == ["1:full_transcript"]


def test_reprocessing_replaces_the_meetings_vectors_and_leaves_other_meetings_alone(make_service):
    svc = make_service()
    svc.add_transcript_chunks(1, transcript(1, n=30, tag="oldtranscript"))
    svc.add_transcript_chunks(2, transcript(2))
    other_before = sorted(svc._chunks.get(where={"meeting_id": 2}, include=[])["ids"])
    old_count = len(svc._chunks.get(where={"meeting_id": 1}, include=[])["ids"])

    svc.add_transcript_chunks(1, transcript(1, n=22, tag="newtranscript"))          # shorter, different transcript
    docs = svc._chunks.get(where={"meeting_id": 1}, include=["documents"])
    assert 0 < len(docs["ids"]) < old_count                                         # stale chunks are gone
    assert all("newtranscript" in d and "oldtranscript" not in d for d in docs["documents"])
    assert sorted(svc._chunks.get(where={"meeting_id": 2}, include=[])["ids"]) == other_before


# ── metadata contract ─────────────────────────────────────────────────────────────────────────────────
def test_every_vector_carries_the_required_metadata(make_service):
    svc = make_service()
    index_all(svc, (1, 3))
    for col in (svc._chunks, svc._meetings):
        for meta in col.get(include=["metadatas"])["metadatas"]:
            assert REQUIRED_METADATA <= set(meta), meta
            ident = IDENTITY[meta["meeting_id"]]
            assert meta["bharatvc_meeting_id"] == ident["bharatvc_meeting_id"] and meta["display_id"] == ident["display_id"]
            assert meta["source"] == ident["source"]
    chunk_types = {m["document_type"] for m in svc._chunks.get(where={"meeting_id": 1}, include=["metadatas"])["metadatas"]}
    doc_types = {m["document_type"] for m in svc._chunks.get(where={"meeting_id": 3}, include=["metadatas"])["metadatas"]}
    assert chunk_types == {"transcript_chunk"} and doc_types == {"document_chunk"}      # document uploads are labelled
    assert {m["document_type"] for m in svc._meetings.get(include=["metadatas"])["metadatas"]} == {"meeting"}
    stored = svc._chunks.get(include=["metadatas"])
    assert all(i == f"{m['meeting_id']}:{m['chunk_id']}" for i, m in zip(stored["ids"], stored["metadatas"]))


# ── meeting isolation ─────────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cross_meeting", [False, True])
def test_meetings_a_b_and_c_only_ever_retrieve_their_own_context(make_service, cross_meeting):
    svc = make_service()
    index_all(svc, (1, 2, 3))
    markers = {1: "migration", 2: "campaign", 3: "engineers"}
    for asking in (1, 2, 3):
        for about in (1, 2, 3):
            hits = svc.search_transcript_chunks(asking, TOPICS[about][0], top_k=8, cross_meeting=cross_meeting)
            assert hits, f"meeting {asking} found nothing"
            assert all(h["meeting_id"] == asking for h in hits)
            others = [markers[m] for m in markers if m != asking]
            assert not any(o in h["text"].lower() for h in hits for o in others), (asking, about)
        own = svc.search_transcript_chunks(asking, TOPICS[asking][1], top_k=3)
        assert markers[asking] in own[0]["text"].lower()                                  # best hit is on-topic
    assert svc.search_transcript_chunks(99, "database migration") == []                  # an unindexed meeting gets nothing


def test_vectors_left_by_a_different_meeting_with_the_same_numeric_id_are_never_returned(make_service):
    svc = make_service()
    svc.add_transcript_chunks(1, transcript(1, tag="oldmeeting"))       # identity BV-001 ...
    assert svc.has_transcript_chunks(1) is True

    svc.identities[1] = {"bharatvc_meeting_id": "BV-NEW", "display_id": "NEW", "source": "transcript"}   # ... then the
    assert svc.has_transcript_chunks(1) is False                          # id is reused by another meeting (reset DB)
    assert svc.search_transcript_chunks(1, "database migration") == []

    svc.add_transcript_chunks(1, transcript(1, tag="freshmeeting"))       # lazy re-index replaces the foreign vectors
    hits = svc.search_transcript_chunks(1, "database migration", top_k=20)
    assert hits and all("freshmeeting" in h["text"] and "oldmeeting" not in h["text"] for h in hits)
    docs = svc._chunks.get(where={"meeting_id": 1}, include=["documents"])["documents"]
    assert not any("oldmeeting" in d for d in docs)


def test_retrieval_falls_back_to_the_meeting_id_filter_when_postgres_cannot_be_read(make_service):
    svc = make_service()
    index_all(svc, (1, 2))
    svc._lookup_meeting = lambda mid: None                                # database unreachable
    hits = svc.search_transcript_chunks(1, "database migration", top_k=5)
    assert hits and all(h["meeting_id"] == 1 for h in hits)


def test_search_results_have_the_shape_the_rag_layer_expects(make_service):
    svc = make_service()
    svc.add_transcript_chunks(1, transcript(1))
    hits = svc.search_transcript_chunks(1, "database migration rollback", top_k=5)
    assert 0 < len(hits) <= 5
    assert {"meeting_id", "start_time", "end_time", "speakers", "similarity_score", "text", "chunk_id",
            "chunk_index"} <= set(hits[0])
    scores = [h["similarity_score"] for h in hits]
    assert scores == sorted(scores, reverse=True) and all(0.0 <= s <= 1.0 for s in scores)
    keys = [(h["speakers"], h["start_time"], h["end_time"]) for h in hits]
    assert len(keys) == len(set(keys))                                    # one hit per speaker/time window


# ── similar meetings ─────────────────────────────────────────────────────────────────────────────────
def test_similar_meetings_exclude_the_current_meeting_and_are_deduplicated_and_ranked(make_service):
    svc = make_service()
    index_all(svc)
    text = " ".join(TOPICS[1] * 3)
    matches = svc.find_similar_meetings(1, text, top_k=5, min_score=0.0)
    ids = [m["meeting_id"] for m in matches]
    assert 1 not in ids and len(ids) == len(set(ids)) == 3                # itself excluded, each meeting once
    assert ids[0] == 4                                                     # the follow-up ranks first
    scores = [m["similarity_score"] for m in matches]
    assert scores == sorted(scores, reverse=True) and scores[0] > scores[-1]
    top = matches[0]                                                       # metadata belongs to meeting 4
    assert (top["meeting_code"], top["filename"], top["title"]) == ("MTG-D", "meeting_4.mp4", "Meeting 4")
    assert top["similarity_percentage"] == f"{top['similarity_score'] * 100:.1f}%"

    cands = svc.get_semantic_candidates(1, text, top_k=10)
    assert [c["meeting_id"] for c in cands][0] == 4 and 1 not in [c["meeting_id"] for c in cands]
    assert cands[0]["metadata"]["bharatvc_meeting_id"] == "BV-004" and cands[0]["semantic_score"] > 0.5
    assert svc.find_similar_meetings(1, text, top_k=5, min_score=0.99) == []          # threshold is applied


def test_similarity_is_calculated_from_the_current_vectors_not_cached(make_service):
    svc = make_service()
    index_all(svc, (1, 2, 3))
    text = " ".join(TOPICS[1] * 3)
    before = {m["meeting_id"] for m in svc.find_similar_meetings(1, text, min_score=0.5)}
    assert 4 not in before
    index_all(svc, (4,))                                                   # a new, very similar meeting arrives
    after = svc.find_similar_meetings(1, text, min_score=0.5)
    assert after and after[0]["meeting_id"] == 4
    svc.delete_meeting_vector(4)                                           # ... and is removed again
    assert 4 not in {m["meeting_id"] for m in svc.find_similar_meetings(1, text, min_score=0.0)}


def test_deleting_a_meeting_removes_its_vectors_from_both_collections(make_service):
    svc = make_service()
    index_all(svc, (1, 2))
    svc.delete_meeting_vector(1)
    assert svc._meetings.get(where={"meeting_id": 1}, include=[])["ids"] == []
    assert svc._chunks.get(where={"meeting_id": 1}, include=[])["ids"] == []
    assert svc.has_transcript_chunks(2) is True and svc._meetings.count() == 1


# ── meeting identifiers: the API accepts numeric / display / bharatvc ids, the vector store only primary keys ─────
def fake_meeting(pk: int, **over):
    ident = IDENTITY[pk]
    base = dict(
        id=pk, bharatvc_meeting_id=ident["bharatvc_meeting_id"], meeting_id=None, display_id=ident["display_id"],
        filename=f"meeting_{pk}.mp4", title=f"Meeting {pk}", file_type="media", status="completed", is_approved=False,
        upload_time=None, meeting_date=None, meeting_time=None, duration=60.0, project_name="Proj", participants=None,
        participant_count=0, keyword_frequency=None, content_fingerprint="", matched_mom="", agenda="", analysis=None)
    base.update(over)
    return SimpleNamespace(**base)


def fake_segments(pk: int):
    return [SimpleNamespace(speaker=s["speaker"], text=s["text"], start_time=s["start_time"], end_time=s["end_time"])
            for s in transcript(pk)]


class FakeRepo:
    """Just enough of MeetingRepository: resolves any identifier like the real one does."""

    def __init__(self, meetings, segments=None):
        self._meetings = list(meetings)
        self._segments = segments if segments is not None else {m.id: fake_segments(m.id) for m in self._meetings}

    def get_by_id(self, ident):
        for m in self._meetings:
            if str(ident) in (str(m.id), m.bharatvc_meeting_id, m.display_id):
                return m
        return None

    def list_all(self):
        return list(self._meetings)

    def get_segments(self, meeting_id):
        return list(self._segments.get(int(meeting_id), []))

    def get_analysis(self, meeting_id):
        return None


@pytest.mark.parametrize("identifier", ["BV-001", "MTG-A", "1"])
def test_similar_meetings_never_list_the_current_meeting_whatever_identifier_the_api_received(make_service, monkeypatch,
                                                                                            identifier):
    svc = make_service()
    index_all(svc)
    monkeypatch.setattr(similarity_service, "vector_service", svc)
    repo = FakeRepo([fake_meeting(m) for m in (1, 2, 3, 4)])
    result = similarity_service.find_similar_meetings(repo, identifier, top_k=8, min_score=0.0)
    ids = [m["meeting_id"] for m in result["meetings"]]
    assert ids and 1 not in ids and len(ids) == len(set(ids))            # never itself, each meeting once
    assert 4 in ids                                                       # the genuine follow-up is found


def test_the_vector_store_is_always_queried_with_the_resolved_primary_key(make_service, monkeypatch):
    svc = make_service()
    index_all(svc)
    monkeypatch.setattr(similarity_service, "vector_service", svc)
    seen = []
    original = svc.get_semantic_candidates
    svc.get_semantic_candidates = lambda meeting_id, *a, **k: (seen.append(meeting_id), original(meeting_id, *a, **k))[1]
    repo = FakeRepo([fake_meeting(m) for m in (1, 2, 3, 4)])
    similarity_service.find_similar_meetings(repo, "BV-004", min_score=0.0)
    similarity_service.detect_duplicate(repo, "MTG-D")
    assert seen == [4, 4]                                                 # not "BV-004" / "MTG-D" / int("007777")-style


def test_a_numeric_looking_meeting_id_that_is_not_the_primary_key_never_lists_the_meeting_itself(make_service, monkeypatch):
    """The live bug: GET /meetings/007777/similar (007777 = bharatvc id of primary key 274) returned meeting 274 itself."""
    svc = make_service()
    index_all(svc)
    monkeypatch.setattr(similarity_service, "vector_service", svc)
    meetings = [fake_meeting(m) for m in (1, 2, 3)] + [fake_meeting(4, bharatvc_meeting_id="007777", display_id="007777")]
    repo = FakeRepo(meetings)
    svc._meetings.upsert(ids=["4:full_transcript"], embeddings=[bow_embed(" ".join(TOPICS[4] * 3))], documents=["x"],
                         metadatas=[{"meeting_id": 4, "bharatvc_meeting_id": "007777", "display_id": "007777",
                                     "chunk_id": "full_transcript", "source": "transcript", "document_type": "meeting",
                                     "meeting_code": "007777", "title": "Meeting 4", "filename": "meeting_4.mp4"}])
    seen = []
    original = svc.get_semantic_candidates
    svc.get_semantic_candidates = lambda meeting_id, *a, **k: (seen.append(meeting_id), original(meeting_id, *a, **k))[1]
    result = similarity_service.find_similar_meetings(repo, "007777", top_k=8, min_score=0.0)
    ids = [m["meeting_id"] for m in result["meetings"]]
    assert seen == [4]                                                    # int("007777") would have been 7777
    assert 4 not in ids and 1 in ids and len(ids) == len(set(ids))


def test_add_meeting_vector_reports_whether_the_vector_was_stored(make_service, monkeypatch):
    svc = make_service()
    assert svc.add_meeting_vector(meeting_id=1, meeting_code="A", filename="a.mp4", title="A", text="alpha beta") is True
    svc.close()
    monkeypatch.setattr(settings, "VECTOR_DB", "nonsense")
    svc.ensure_initialized()
    assert svc.add_meeting_vector(meeting_id=1, meeting_code="A", filename="a.mp4", title="A", text="alpha beta") is False


# ── backfill: meetings that pre-date ChromaDB are added from PostgreSQL ────────────────────────────────────────────
def test_backfill_adds_only_the_meetings_chromadb_does_not_have(make_service, monkeypatch):
    svc = make_service()
    monkeypatch.setattr(vb, "vector_service", svc)
    repo = FakeRepo([fake_meeting(1), fake_meeting(2), fake_meeting(3, status="processing"), fake_meeting(4)],
                    {1: fake_segments(1), 2: fake_segments(2), 3: fake_segments(3), 4: []})
    svc.add_meeting_vector(meeting_id=1, meeting_code="MTG-A", filename="a.mp4", title="A", text="already there",
                           extra_metadata={"bharatvc_meeting_id": "BV-001"})

    stats = vb.backfill(repo=repo)
    assert stats == {"meetings": 1, "chunks": 0, "already_indexed": 1, "skipped_no_transcript": 1,
                     "skipped_in_progress": 1, "failed": 0}
    assert svc.indexed_meeting_vectors() == {1: "BV-001", 2: "BV-002"}
    assert svc._chunks.count() == 0                                        # chunk vectors stay lazy (created on first chat)
    again = vb.backfill(repo=repo)                                         # idempotent: nothing new to add
    assert again["meetings"] == 0 and again["already_indexed"] == 2 and svc._meetings.count() == 2


def test_a_backfilled_meeting_is_immediately_found_by_similar_meeting_search(make_service, monkeypatch):
    svc = make_service()
    monkeypatch.setattr(vb, "vector_service", svc)
    monkeypatch.setattr(similarity_service, "vector_service", svc)
    repo = FakeRepo([fake_meeting(m) for m in (1, 2, 3, 4)])
    svc.add_meeting_vector(meeting_id=4, meeting_code="MTG-D", filename="d.mp4", title="D", text=" ".join(TOPICS[4]),
                           extra_metadata={"bharatvc_meeting_id": "BV-004"})   # only the newest meeting is in ChromaDB
    before = similarity_service.find_similar_meetings(repo, "BV-004", min_score=0.0)["meetings"]
    assert [m["meeting_id"] for m in before] == []                         # older meetings are invisible to it
    vb.backfill(repo=repo)
    after = {m["meeting_id"] for m in similarity_service.find_similar_meetings(repo, "BV-004", min_score=0.0)["meetings"]}
    assert 1 in after and 4 not in after                                   # the earlier meeting is now found, self never


def test_backfill_replaces_a_vector_that_belongs_to_a_different_meeting_with_the_same_id(make_service, monkeypatch):
    svc = make_service()
    monkeypatch.setattr(vb, "vector_service", svc)
    svc.add_meeting_vector(meeting_id=1, meeting_code="OLD", filename="old.mp4", title="Old", text="unrelated old text",
                           extra_metadata={"bharatvc_meeting_id": "BV-OLD"})
    stats = vb.backfill(repo=FakeRepo([fake_meeting(1)]))
    assert stats["meetings"] == 1 and svc.indexed_meeting_vectors() == {1: "BV-001"} and svc._meetings.count() == 1


def test_full_reindex_rebuilds_chunks_without_duplicates(make_service, monkeypatch):
    svc = make_service()
    monkeypatch.setattr(vb, "vector_service", svc)
    repo = FakeRepo([fake_meeting(1), fake_meeting(2)])
    first = vb.backfill(repo=repo, chunks=True, only_missing=False)
    ids = sorted(svc._chunks.get(include=[])["ids"])
    assert first["meetings"] == 2 and first["chunks"] == len(ids) > 2
    second = vb.backfill(repo=repo, chunks=True, only_missing=False)
    assert second["chunks"] == first["chunks"] and sorted(svc._chunks.get(include=[])["ids"]) == ids
    assert svc._meetings.count() == 2


# ── embedding-space signature: vectors from a different engine are re-embedded, never silently compared ───────────────
def test_every_vector_is_tagged_with_the_embedding_signature(make_service):
    svc = make_service()
    index_all(svc)
    for collection in (svc._meetings, svc._chunks):
        metas = collection.get(include=["metadatas"])["metadatas"]
        assert metas and {m["embedding_signature"] for m in metas} == {BOW_SIGNATURE}


def test_vectors_from_another_embedding_engine_are_re_embedded_in_place_without_duplicates(make_service, monkeypatch):
    """The lexical fallback ran while the model file was broken; after the file is fixed the signature changes and
    everything stored under the old one must be rebuilt (same ids), not compared against the new model's queries."""
    svc = make_service()
    monkeypatch.setattr(vb, "vector_service", svc)
    index_all(svc)
    meetings_before, chunk_ids = svc._meetings.count(), sorted(svc._chunks.get(include=[])["ids"])
    assert set(svc.indexed_meeting_vectors()) == {1, 2, 3, 4} and svc.has_transcript_chunks(1) is True

    svc._embedding_signature = lambda: "real-model:384"                  # the model is repaired and now loads
    assert svc.indexed_meeting_vectors() == {}                            # old vectors count as not indexed ...
    assert svc.has_transcript_chunks(1) is False                          # ... chunks too (chat re-indexes lazily)

    repo = FakeRepo([fake_meeting(m) for m in (1, 2, 3, 4)])
    stats = vb.backfill(repo=repo, chunks=True)                           # startup backfill / reindex_vectors
    assert stats["meetings"] == 4 and stats["already_indexed"] == 0
    assert svc._meetings.count() == meetings_before                       # updated in place: no duplicates
    assert {m["embedding_signature"] for m in svc._meetings.get(include=["metadatas"])["metadatas"]} == {"real-model:384"}
    assert {m["embedding_signature"] for m in svc._chunks.get(include=["metadatas"])["metadatas"]} == {"real-model:384"}
    assert svc.has_transcript_chunks(1) is True and set(svc.indexed_meeting_vectors()) == {1, 2, 3, 4}
    assert vb.backfill(repo=repo)["already_indexed"] == 4                 # and it is stable afterwards


def test_untagged_vectors_from_before_signatures_existed_are_re_embedded_too(make_service, monkeypatch):
    svc = make_service()
    index_all(svc, meetings=(1,))
    for collection in (svc._meetings, svc._chunks):                       # rewrite them exactly as an older release stored them
        got = collection.get(include=["metadatas", "embeddings", "documents"])
        collection.delete(ids=got["ids"])
        collection.add(ids=got["ids"], embeddings=got["embeddings"], documents=got["documents"],
                       metadatas=[{k: v for k, v in m.items() if k != "embedding_signature"} for m in got["metadatas"]])
        assert all("embedding_signature" not in m for m in collection.get(include=["metadatas"])["metadatas"])
    assert svc.indexed_meeting_vectors() == {} and svc.has_transcript_chunks(1) is False
    svc.add_transcript_chunks(1, transcript(1))
    assert svc.has_transcript_chunks(1) is True


def test_a_meeting_with_even_one_foreign_chunk_is_not_considered_indexed(make_service):
    svc = make_service()
    svc.add_transcript_chunks(1, transcript(1))
    victim = svc._chunks.get(limit=1, include=["metadatas"])
    svc._chunks.update(ids=victim["ids"], metadatas=[{**victim["metadatas"][0], "embedding_signature": "stale:384"}])
    assert svc.has_transcript_chunks(1) is False                          # partially rebuilt -> rebuild fully


def test_a_failing_model_never_produces_a_vector_from_another_space(monkeypatch):
    """encode_texts falls back to lexical vectors for evaluation callers, but the vector store asks for strict
    encoding: a mid-run model failure must raise, not persist a vector that the stored signature would mislabel."""
    svc = object.__new__(VectorService)
    svc._reset_state()

    def broken(texts, strict=False):
        if strict:
            raise RuntimeError("out of memory")
        return np.ones((len(texts), DIM), dtype=np.float32)

    monkeypatch.setattr("backend.services.model_loader.encode_texts", broken)
    with pytest.raises(RuntimeError):
        svc.generate_embedding("some meeting text")
    with pytest.raises(RuntimeError):
        svc.generate_embeddings_batch(["one", "two"])


def test_backfill_dry_run_writes_nothing_and_an_unavailable_store_is_a_clear_error(make_service, monkeypatch):
    svc = make_service()
    monkeypatch.setattr(vb, "vector_service", svc)
    stats = vb.backfill(repo=FakeRepo([fake_meeting(1), fake_meeting(2)]), dry_run=True)
    assert stats["meetings"] == 2 and svc._meetings.count() == 0 and svc._chunks.count() == 0
    svc.close()
    monkeypatch.setattr(settings, "VECTOR_DB", "nonsense")
    svc.ensure_initialized()
    with pytest.raises(RuntimeError, match="ChromaDB is unavailable"):
        vb.backfill(repo=FakeRepo([fake_meeting(1)]))


def test_the_startup_backfill_runs_in_the_background_and_can_be_switched_off(make_service, monkeypatch):
    svc = make_service()
    monkeypatch.setattr(vb, "vector_service", svc)
    calls = []
    monkeypatch.setattr(vb, "backfill", lambda **kw: calls.append(kw) or
                        {"meetings": 3, "already_indexed": 0, "skipped_no_transcript": 0, "failed": 0})
    monkeypatch.setattr(settings, "VECTOR_BACKFILL_ON_STARTUP", False)
    assert vb.start_background_backfill() is None and calls == []
    monkeypatch.setattr(settings, "VECTOR_BACKFILL_ON_STARTUP", True)
    thread = vb.start_background_backfill()
    thread.join(timeout=30)
    assert thread.daemon and calls == [{"chunks": False, "only_missing": True}]


# ── health payload ─────────────────────────────────────────────────────────────────────────────────────
def test_the_health_endpoint_reports_the_vector_database(make_service, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from unittest.mock import MagicMock

    from backend.api import system as system_api
    from backend.api.deps import get_db

    svc = make_service()
    monkeypatch.setattr(system_api, "vector_service", svc)
    monkeypatch.setattr(system_api, "_get_airawat_health", lambda: True)
    monkeypatch.setattr("backend.services.model_loader.get_all_models_health",
                        lambda: {"status": "healthy", "offline_mode": True, "models": {}})
    session = MagicMock()
    session.execute.return_value.scalar.return_value = 1
    app = FastAPI()
    app.include_router(system_api.router, prefix="/api/v1")
    app.dependency_overrides[get_db] = lambda: session
    client = TestClient(app)

    body = client.get("/api/v1/system/health").json()
    assert body["vector_database"] == {"backend": "chromadb", "status": "available"}
    assert body["services"]["vector_database"] is True and "pgvector" not in body["services"]
    assert body["status"] == "healthy"

    svc.close()
    monkeypatch.setattr(settings, "VECTOR_DB", "nonsense")
    svc.ensure_initialized()
    body = client.get("/api/v1/system/health").json()
    assert body["vector_database"]["backend"] == "chromadb" and body["vector_database"]["status"] == "unavailable"
    assert body["vector_database"]["error"] and body["services"]["vector_database"] is False
    assert body["status"] == "degraded"


# ── nothing of the previous vector store is left ───────────────────────────────────────────────────────────
def test_no_pgvector_code_is_left_in_the_backend():
    banned = re.compile(r"pgvector|CREATE\s+EXTENSION|PGVector", re.IGNORECASE)
    offenders = []
    for path in (REPO_ROOT / "backend").rglob("*.py"):
        if path.name == Path(__file__).name or "__pycache__" in path.parts:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            if banned.search(line):
                offenders.append(f"{path.relative_to(REPO_ROOT)}:{number}")
    assert offenders == []


def test_backfill_stops_once_with_the_reason_when_the_embedding_model_is_unusable(make_service, monkeypatch):
    """A Git LFS pointer instead of the weights must not produce one 'could not be indexed' per meeting (or fake
    vectors): the backfill stops once, naming the model problem, and nothing is written."""
    from backend.services import model_loader

    def broken():
        raise model_loader.EmbeddingModelUnavailable("no usable copy: holds Git LFS pointer file(s). FIX: git lfs pull")

    svc = make_service()
    monkeypatch.setattr(vb, "vector_service", svc)
    monkeypatch.setattr(model_loader, "get_embedding_encoder", broken)
    with pytest.raises(RuntimeError, match=r"4 meeting\(s\) not indexed.*Git LFS pointer"):
        vb.backfill(repo=FakeRepo([fake_meeting(m) for m in (1, 2, 3, 4)]))
    assert svc._meetings.count() == 0
    assert vb.indexing_status()["state"] in ("not_started", "done", "skipped", "disabled", "running", "done_with_failures")
