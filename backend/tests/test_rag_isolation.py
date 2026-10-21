"""Meeting A must never use Meeting B's RAG context — checked at the vector, router and request layers."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from backend.api import chat as chat_api
from backend.rag import agent_router as ar
from backend.schemas import schemas
from backend.services.vector_service import VectorService


class _FakeChunks:
    """A ChromaDB collection that records every query and its ``where`` filter."""

    def __init__(self):
        self.queries = []

    def query(self, **kwargs):
        self.queries.append(kwargs)
        return {"metadatas": [[]], "distances": [[]], "documents": [[]]}


def _service_with_fake_collection(identity=None):
    """A ready VectorService whose chunk collection is a recorder; ``identity`` is the meeting's PostgreSQL row."""
    svc = object.__new__(VectorService)
    svc._reset_state()
    svc._initialized, svc.is_ready, svc._client = True, True, object()
    svc._chunks = _FakeChunks()
    svc._lookup_meeting = lambda meeting_id: identity
    svc.generate_embedding = lambda text: [0.1, 0.2, 0.3]
    return svc


@pytest.mark.parametrize("cross_meeting", [False, True])
def test_chunk_search_is_always_filtered_by_the_meeting_id(cross_meeting):
    svc = _service_with_fake_collection()
    svc.search_transcript_chunks(meeting_id=42, query_text="what was decided", top_k=5, cross_meeting=cross_meeting)
    (query,) = svc._chunks.queries
    assert query["where"] == {"meeting_id": 42}, "the meeting filter must be applied even when cross_meeting=True"


def test_chunk_search_is_also_filtered_by_the_bharatvc_meeting_id_when_it_is_known():
    svc = _service_with_fake_collection({"bharatvc_meeting_id": "BV-42", "display_id": "D", "source": "transcript"})
    svc.search_transcript_chunks(meeting_id=42, query_text="what was decided", cross_meeting=True)
    (query,) = svc._chunks.queries
    assert query["where"] == {"$and": [{"meeting_id": 42}, {"bharatvc_meeting_id": "BV-42"}]}


def test_chunks_that_name_a_different_meeting_are_dropped_even_if_chromadb_returned_them():
    svc = _service_with_fake_collection()
    svc._chunks.query = lambda **kw: {
        "metadatas": [[{"meeting_id": 99, "chunk_id": "c1", "speakers": "A", "start_time": "0", "end_time": "1"},
                       {"meeting_id": 42, "chunk_id": "c2", "speakers": "B", "start_time": "2", "end_time": "3"}]],
        "distances": [[0.1, 0.2]], "documents": [["other meeting text", "own meeting text"]],
    }
    hits = svc.search_transcript_chunks(meeting_id=42, query_text="anything at all")
    assert [h["text"] for h in hits] == ["own meeting text"]


def test_cross_meeting_flag_in_a_chat_request_can_no_longer_widen_retrieval():
    req = schemas.ChatMessageCreate(message="compare with the other meetings", cross_meeting=True)
    chat_api.enforce_meeting_isolation(req, meeting_pk=7)
    assert req.cross_meeting is False
    ok = schemas.ChatMessageCreate(message="hello there")
    chat_api.enforce_meeting_isolation(ok, meeting_pk=7)
    assert ok.cross_meeting is False


class _Reranker:
    def rerank(self, query, chunks, top_k):
        return chunks[:top_k]


def test_router_drops_chunks_tagged_with_another_meeting_even_if_the_vector_layer_returns_them(monkeypatch):
    monkeypatch.setattr(ar, "get_reranker", lambda: _Reranker())
    own = {"meeting_id": 7, "text": "The migration must finish before March.", "similarity_score": 0.9}
    foreign = {"meeting_id": 99, "text": "The autumn marketing campaign launches in October.", "similarity_score": 0.95}
    untagged = {"meeting_id": None, "text": "Hiring two engineers was agreed.", "similarity_score": 0.8}

    vector = MagicMock()
    vector.search_transcript_chunks.return_value = [foreign, own, untagged]
    repo = MagicMock()
    repo.get_by_id.return_value = SimpleNamespace(id=7)
    repo.get_segments.return_value = []

    router = ar.MeetingAgentRouter(repo=repo, vector_service=vector)
    out = router.route(meeting_id=7, query="what was decided about the migration", top_k=5,
                       similarity_threshold=0.1, cross_meeting=True)

    kwargs = vector.search_transcript_chunks.call_args.kwargs
    assert kwargs["meeting_id"] == 7 and kwargs["cross_meeting"] is False
    texts = {c["text"] for c in out["source_chunks"]}
    assert own["text"] in texts and untagged["text"] in texts
    assert foreign["text"] not in texts
    assert not any("marketing campaign" in c for c in out["context_chunks"])


def test_router_uses_the_resolved_primary_key_not_the_raw_path_id(monkeypatch):
    monkeypatch.setattr(ar, "get_reranker", lambda: _Reranker())
    vector = MagicMock()
    vector.search_transcript_chunks.return_value = []
    repo = MagicMock()
    repo.get_by_id.return_value = SimpleNamespace(id=7)         # e.g. resolved from a bharatvc id
    repo.get_segments.return_value = []
    ar.MeetingAgentRouter(repo=repo, vector_service=vector).route(meeting_id="9000123", query="anything at all here")
    assert vector.search_transcript_chunks.call_args.kwargs["meeting_id"] == 7
