"""Cross-encoder reranker for RAG context chunk quality improvement.

Uses sentence-transformers cross-encoder models (local, offline-ready) to
rerank retrieved chunks by relevance to the actual user query, improving
retrieval precision over cosine similarity alone.

Priority:
  1. cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, ~90MB, offline)
  2. cross-encoder/ms-marco-MiniLM-L-2-v2 (tiny fallback)
  3. Cosine similarity passthrough (if no cross-encoder available)

Usage:
    reranker = get_reranker()
    ranked = reranker.rerank(query="What decisions were made?", chunks=[...], top_k=5)
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _PassthroughReranker:
    """Fallback: returns chunks in original cosine-similarity order."""

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        # Sort by existing similarity_score if available
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get("similarity_score", 0),
            reverse=True,
        )
        return sorted_chunks[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers.
    
    Re-scores each (query, chunk) pair with a bi-directional cross-encoder
    model (not just cosine similarity), which significantly improves the
    precision of top-ranked results.
    """

    def __init__(self, model_name: str, model_path: Optional[str] = None):
        self._model = None
        self._model_name = model_name
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        try:
            from sentence_transformers.cross_encoder import CrossEncoder

            load_path = self._model_path or self._model_name
            # Check if local path exists
            if self._model_path and os.path.isdir(self._model_path):
                logger.info(f"[Reranker] Loading cross-encoder from local path: {self._model_path}")
                self._model = CrossEncoder(self._model_path)
            else:
                offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
                if offline:
                    logger.warning(
                        f"[Reranker] HF_HUB_OFFLINE=1 and no local model at {self._model_path}. "
                        "Falling back to passthrough reranker."
                    )
                    self._model = None
                    return
                logger.info(f"[Reranker] Loading cross-encoder from HF Hub: {self._model_name}")
                self._model = CrossEncoder(self._model_name)

            logger.info(f"[Reranker] Cross-encoder loaded: {self._model_name}")
        except ImportError:
            logger.warning("[Reranker] sentence-transformers not installed. Using passthrough reranker.")
        except Exception as e:
            logger.warning(f"[Reranker] Failed to load cross-encoder ({e}). Using passthrough reranker.")

    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Rerank chunks by cross-encoder relevance score.
        
        Args:
            query: The user question
            chunks: List of chunk dicts with 'text' key
            top_k: Number of top chunks to return
            
        Returns:
            Top-k chunks sorted by cross-encoder score (highest first),
            each annotated with 'rerank_score' field.
        """
        if not chunks:
            return []

        if self._model is None:
            return _PassthroughReranker().rerank(query, chunks, top_k)

        try:
            # Build (query, chunk_text) pairs for cross-encoder scoring
            pairs = [(query, c.get("text", "")) for c in chunks]

            # Score all pairs
            scores = self._model.predict(pairs)

            # Annotate each chunk with its cross-encoder score
            scored = []
            for chunk, score in zip(chunks, scores):
                annotated = dict(chunk)
                annotated["rerank_score"] = float(score)
                scored.append(annotated)

            # Sort by rerank score (highest first)
            scored.sort(key=lambda c: c["rerank_score"], reverse=True)

            logger.debug(
                f"[Reranker] Reranked {len(chunks)} chunks → top-{top_k}. "
                f"Top score: {scored[0]['rerank_score']:.4f} if scored else N/A"
            )

            return scored[:top_k]

        except Exception as e:
            logger.warning(f"[Reranker] Reranking failed ({e}). Returning original order.")
            return _PassthroughReranker().rerank(query, chunks, top_k)


# ---------------------------------------------------------------------------
# Singleton lazy loader
# ---------------------------------------------------------------------------

_reranker_instance: Optional[CrossEncoderReranker | _PassthroughReranker] = None
_RERANKER_LOADED = False


def get_reranker() -> CrossEncoderReranker | _PassthroughReranker:
    """Returns the singleton reranker instance (lazy loaded)."""
    global _reranker_instance, _RERANKER_LOADED

    if _RERANKER_LOADED:
        return _reranker_instance  # type: ignore

    _RERANKER_LOADED = True

    try:
        from backend.config.config import settings

        # Determine model path — prefer local offline model
        model_name = os.getenv(
            "RERANKER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )
        # Try local path under models/reranker/
        local_path = os.path.join(
            settings.MODEL_BASE_PATH,
            "reranker",
            model_name.split("/")[-1],
        )

        _reranker_instance = CrossEncoderReranker(
            model_name=model_name,
            model_path=local_path if os.path.isdir(local_path) else None,
        )
    except Exception as e:
        logger.warning(f"[Reranker] Init failed ({e}). Using passthrough.")
        _reranker_instance = _PassthroughReranker()

    return _reranker_instance  # type: ignore
