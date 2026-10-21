"""Rebuild the ChromaDB vector database from PostgreSQL.

PostgreSQL holds every transcript, so vectors are derived data. Use this script to:
  * populate ChromaDB for meetings processed before it became the vector store (data migration),
  * move to a new CHROMA_PERSIST_DIRECTORY (e.g. a new server volume),
  * rebuild after changing the embedding model (--reset).

It READS PostgreSQL and WRITES ChromaDB only. Re-running is always safe: ids are deterministic, so vectors are
updated in place and never duplicated; a meeting's vectors always end up exactly matching its stored transcript.
(The API also adds missing whole-meeting vectors by itself at startup - VECTOR_BACKFILL_ON_STARTUP - this script is
for a full rebuild, chunk vectors included.)

Usage:
    python -m backend.scripts.reindex_vectors                    # every meeting that has a transcript
    python -m backend.scripts.reindex_vectors --only-missing     # only meetings ChromaDB does not have yet
    python -m backend.scripts.reindex_vectors --meeting-id 12 --meeting-id 15
    python -m backend.scripts.reindex_vectors --reset            # drop both collections first (model change)
    python -m backend.scripts.reindex_vectors --dry-run          # show what would be indexed

Stop the API first: an embedded ChromaDB directory can be owned by one process only.
"""

import argparse
import logging
import sys
from typing import Dict, List, Optional


def reindex(meeting_ids: Optional[List[int]] = None, reset: bool = False, dry_run: bool = False,
            only_missing: bool = False) -> Dict[str, int]:
    """Index the given meetings (default: all with a transcript), chunk vectors included; returns counters."""
    from backend.services.vector_backfill import backfill
    from backend.services.vector_service import vector_service

    if reset and not dry_run:
        vector_service.reset_collections()
    elif not vector_service.check_ready():
        raise RuntimeError(f"ChromaDB is unavailable: {vector_service.init_error}")
    return backfill(meeting_ids, chunks=True, only_missing=only_missing, dry_run=dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild ChromaDB vectors from PostgreSQL.")
    parser.add_argument("--meeting-id", type=int, action="append", help="Only this meeting (numeric id); repeatable.")
    parser.add_argument("--only-missing", action="store_true", help="Skip meetings ChromaDB already has.")
    parser.add_argument("--reset", action="store_true", help="Drop and recreate both collections first.")
    parser.add_argument("--dry-run", action="store_true", help="List what would be indexed; write nothing.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    try:
        stats = reindex(args.meeting_id, reset=args.reset, dry_run=args.dry_run, only_missing=args.only_missing)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    finally:
        from backend.services.vector_service import vector_service
        vector_service.close()

    print(f"\nDone{' (dry run)' if args.dry_run else ''}: {stats['meetings']} meeting(s), {stats['chunks']} chunk(s) indexed, "
          f"{stats['already_indexed']} already indexed, {stats['skipped_no_transcript']} without transcript, "
          f"{stats['skipped_in_progress']} still processing, {stats['failed']} failed.")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
