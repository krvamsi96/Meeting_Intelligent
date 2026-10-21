"""Verify the ChromaDB vector database exactly as the application sees it (local or server).

Reads VECTOR_DB / CHROMA_PERSIST_DIRECTORY from the same environment the API uses, opens the persistent
store, then does a real write -> similarity query -> delete round trip in a throwaway collection (your
meeting vectors are never touched).

Exit code: 0 = ChromaDB available and working; 1 = unavailable (the reason is printed).

Usage:
    python -m backend.scripts.check_vector_db

Stop the API first: an embedded ChromaDB directory can be owned by one process only.
"""

import sys
import uuid


def main() -> int:
    from backend.config.config import settings
    from backend.services.vector_service import vector_service

    print(f"[1] Configuration     : VECTOR_DB={settings.VECTOR_DB}")
    print(f"    Persist directory : {settings.CHROMA_PERSIST_DIRECTORY}")

    st = vector_service.status()
    print(f"[2] ChromaDB          : {st['status']} (state: {st['state']})")
    if st["status"] != "available":
        print(f"    {st['error']}")
        return 1
    print(f"    Embedding dimension: {st['embedding_dimension']}")
    print(f"    Collections        : {st['collections']}")

    name = f"healthcheck_{uuid.uuid4().hex[:8]}"
    client = vector_service._client
    try:
        probe = client.create_collection(name=name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None)
        vec = vector_service.generate_embedding("vector database round trip")
        probe.upsert(ids=["probe"], embeddings=[vec], documents=["vector database round trip"], metadatas=[{"meeting_id": 0}])
        hit = probe.query(query_embeddings=[vec], n_results=1, include=["distances"])
        distance = float(hit["distances"][0][0])
        print(f"[3] Round trip        : wrote 1 vector, queried it back (cosine distance {distance:.6f})")
        ok = hit["ids"][0] == ["probe"] and distance < 1e-3
    except Exception as exc:
        print(f"[3] Round trip        : FAILED - {type(exc).__name__}: {exc}")
        return 1
    finally:
        try:
            client.delete_collection(name)
        except Exception:
            pass
        vector_service.close()

    print("\nRESULT: ChromaDB is READY." if ok else "\nRESULT: the round trip returned an unexpected result.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
