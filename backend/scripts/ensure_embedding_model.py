"""Deployment gate for the embedding model (run by run_backend.sh, start.sh, setup_backend.sh, deploy_docker.sh, Jenkins).

    python3 -m backend.scripts.ensure_embedding_model            # verify; git lfs pull automatically if needed
    python3 -m backend.scripts.ensure_embedding_model --load     # ...and load it with sentence-transformers + embed
    python3 -m backend.scripts.ensure_embedding_model --no-fetch # verify only

Exit code 0 = the real model is present (and loads, with --load); 1 = it is not - the message says why and how to fix.
Without --load only the standard library is used, so it runs before the virtual environment exists.
"""

from __future__ import annotations

import argparse
import os
import sys


def _load_and_embed(model_dir: str) -> str:
    """Load the directory with sentence-transformers (offline) and prove it produces real semantic vectors."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import numpy as np
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_dir, local_files_only=True, device="cpu")
    vecs = model.encode([
        "The team agreed to launch the new website in November.",
        "The website release was scheduled for the following month by everyone.",
        "Bananas are yellow and grow in tropical countries.",
    ], normalize_embeddings=True)
    if not np.isfinite(vecs).all():
        raise RuntimeError("the model produced non-finite values")
    related, unrelated = float(vecs[0] @ vecs[1]), float(vecs[0] @ vecs[2])
    if related <= unrelated + 0.1:
        raise RuntimeError(f"vectors are not semantic (related={related:.2f}, unrelated={unrelated:.2f})")
    return f"loads with sentence-transformers, {vecs.shape[1]}-dim vectors, related={related:.2f} > unrelated={unrelated:.2f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-fetch", action="store_true", help="do not run git lfs pull")
    parser.add_argument("--load", action="store_true", help="also load the model and generate test embeddings")
    args = parser.parse_args(argv)

    from backend.services.embedding_model_guard import ensure_embedding_model

    check = ensure_embedding_model(auto_fetch=not args.no_fetch, log=lambda m: print(f"[INFO]  {m}", flush=True))
    if not check.ok:
        print(f"[ERROR] {check.summary()}", flush=True)
        return 1
    print(f"[OK]    {check.summary()}", flush=True)
    if args.load:
        try:
            print(f"[OK]    embedding model {_load_and_embed(check.model_dir)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - reported, exit 1
            print(f"[ERROR] embedding model files are complete but do not load: {type(exc).__name__}: {exc}. "
                  f"FIX: pip install -r requirements.txt in the venv (sentence-transformers, torch).", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
