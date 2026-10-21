"""Dynamic transcript chunking strategy for RAG with context preservation.

Chunking Strategies:
  1. chunk_transcript_segments      — sliding-window (baseline, 4 segs, 50% overlap)
  2. chunk_transcript_by_topic      — time-window (groups by duration)
  3. chunk_transcript_structure_aware — structure-aware (detects decision/agenda/action
                                         boundaries, creates anchor chunks for key moments)
  4. chunk_transcript_smart          — auto-selects best strategy based on transcript size
"""

import logging
import re
from typing import Any, Dict, List

from backend.config.config import RAG_CHUNK_OVERLAP, RAG_CHUNK_SIZE

logger = logging.getLogger(__name__)

# Keywords that signal a boundary (decision/action/agenda moment)
_DECISION_PATTERNS = re.compile(
    r"\b(decided?|decision|agreed?|approved?|resolved?|finalized?|confirmed?|"
    r"conclusion|let'?s proceed|will use|chosen|selected|go with|going with)\b",
    re.IGNORECASE,
)
_ACTION_PATTERNS = re.compile(
    r"\b(action item|todo|follow.?up|need to|will (do|handle|complete|prepare|"
    r"schedule|send|review|check|fix)|assign|take care|responsible for|deadline|"
    r"by (monday|tuesday|wednesday|thursday|friday|next week|tomorrow|end of))\b",
    re.IGNORECASE,
)
_AGENDA_PATTERNS = re.compile(
    r"\b(agenda|next topic|moving on|let'?s (discuss|talk about|cover|start with|look at)|"
    r"the (next|following) item|now (let'?s|we'?ll)|firstly|secondly|thirdly|"
    r"to begin with|wrapping up|in conclusion|any questions)\b",
    re.IGNORECASE,
)


def segment_to_text(seg: Any) -> str:
    """Extract text from a transcript segment, handling both dict and object."""
    if isinstance(seg, dict):
        return seg.get("text", "")
    return getattr(seg, "text", "")


def segment_to_speaker(seg: Any) -> str:
    if isinstance(seg, dict):
        return seg.get("speaker", "Speaker")
    return getattr(seg, "speaker", "Speaker")


def _format_time_str(val: Any) -> str:
    if val is None:
        return "00:00:00"
    if isinstance(val, (int, float)):
        secs = max(0.0, float(val))
        h = int(secs // 3600)
        m = int((secs % 3600) // 60)
        s = int(secs % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    return str(val)


def _parse_time_to_seconds(ts: Any) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            parts = ts.strip().split(":")
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
            elif len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            return float(ts)
        except Exception:
            return 0.0
    return 0.0


def segment_to_start_time(seg: Any) -> str:
    if isinstance(seg, dict):
        return _format_time_str(seg.get("start_time", "00:00:00"))
    return _format_time_str(getattr(seg, "start_time", "00:00:00"))


def segment_to_end_time(seg: Any) -> str:
    if isinstance(seg, dict):
        return _format_time_str(seg.get("end_time", "00:00:00"))
    return _format_time_str(getattr(seg, "end_time", "00:00:00"))


def chunk_transcript_segments(
    segments: List[Any],
    chunk_size: int = RAG_CHUNK_SIZE,
    overlap: int = RAG_CHUNK_OVERLAP,
    meeting_id: Any = None,
) -> List[Dict[str, Any]]:
    """
    Groups transcript segments into overlapping chunks for RAG retrieval.
    Preserves speaker context and temporal continuity for better semantic search.

    Args:
        segments: List of segment objects/dicts with speaker, start_time, end_time, text
        chunk_size: Number of segments per chunk (default: 4)
        overlap: Step size between chunks (default: 2, 50% overlap)
        meeting_id: Optional meeting identifier for scoped chunking

    Returns:
        List of chunk dicts with keys: text, start_time, end_time, speakers, chunk_id, meeting_id
    """
    if not segments:
        return []

    chunks = []
    chunk_index = 0

    for i in range(0, len(segments), overlap):
        chunk_segs = segments[i : i + chunk_size]
        if not chunk_segs:
            break

        formatted_lines = []
        speakers = set()
        start_time = segment_to_start_time(chunk_segs[0])
        end_time = segment_to_end_time(chunk_segs[-1])

        for seg in chunk_segs:
            speaker_name = segment_to_speaker(seg)
            seg_text = segment_to_text(seg).strip()

            # Only add non-empty segments
            if seg_text:
                formatted_lines.append(
                    f"[{segment_to_start_time(seg)} - {segment_to_end_time(seg)}] "
                    f"{speaker_name}: {seg_text}"
                )
                speakers.add(speaker_name)

        if not formatted_lines:
            continue

        chunk_text = "\n".join(formatted_lines)
        chunk_dict = {
            "text": chunk_text,
            "start_time": start_time,
            "end_time": end_time,
            "speakers": (
                ", ".join(sorted(speakers)) if speakers else "Unknown Speaker"
            ),
            "chunk_id": f"chunk_{chunk_index}",
            "chunk_index": chunk_index,
        }
        if meeting_id is not None:
            chunk_dict["meeting_id"] = meeting_id

        chunks.append(chunk_dict)
        chunk_index += 1

        if i + chunk_size >= len(segments):
            break

    logger.debug(
        f"Created {len(chunks)} transcript chunks from {len(segments)} segments"
    )
    return chunks


def chunk_transcript_by_topic(
    segments: List[Any],
    max_chunk_duration_seconds: int = 300,
) -> List[Dict[str, Any]]:
    """
    Alternative chunking strategy: groups segments by time windows.
    Useful for meetings with distinct topic transitions.
    """
    if not segments:
        return []

    chunks = []
    current_chunk_segs = []
    current_start_time = segment_to_start_time(segments[0])
    current_speakers = set()
    chunk_index = 0

    def finalize_chunk():
        nonlocal chunk_index, current_start_time
        if not current_chunk_segs:
            return

        formatted_lines = []
        for seg in current_chunk_segs:
            speaker_name = segment_to_speaker(seg)
            seg_text = segment_to_text(seg).strip()
            if seg_text:
                formatted_lines.append(
                    f"[{segment_to_start_time(seg)} - {segment_to_end_time(seg)}] "
                    f"{speaker_name}: {seg_text}"
                )
                current_speakers.add(speaker_name)

        if formatted_lines:
            chunks.append(
                {
                    "text": "\n".join(formatted_lines),
                    "start_time": current_start_time,
                    "end_time": segment_to_end_time(current_chunk_segs[-1]),
                    "speakers": (
                        ", ".join(sorted(current_speakers))
                        if current_speakers
                        else "Unknown Speaker"
                    ),
                    "chunk_id": f"topic_chunk_{chunk_index}",
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1

    for seg in segments:
        seg_start = segment_to_start_time(seg)
        try:
            seg_start_secs = _parse_time_to_seconds(seg_start)
            current_start_secs = _parse_time_to_seconds(current_start_time)

            if (
                seg_start_secs - current_start_secs > max_chunk_duration_seconds
                and current_chunk_segs
            ):
                finalize_chunk()
                current_chunk_segs = []
                current_start_time = seg_start
                current_speakers = set()
        except Exception:
            # If time parsing fails, just add to current chunk
            pass

        current_chunk_segs.append(seg)

    finalize_chunk()
    logger.debug(
        f"Created {len(chunks)} topic-based chunks from {len(segments)} segments"
    )
    return chunks


def _is_boundary_segment(seg: Any) -> str:
    """Returns the boundary type if segment marks a key moment, else empty string."""
    text = segment_to_text(seg)
    if _DECISION_PATTERNS.search(text):
        return "decision"
    if _ACTION_PATTERNS.search(text):
        return "action"
    if _AGENDA_PATTERNS.search(text):
        return "agenda"
    return ""


def chunk_transcript_structure_aware(
    segments: List[Any],
    chunk_size: int = 6,
    overlap: int = 2,
    create_anchor_chunks: bool = True,
    meeting_id: Any = None,
) -> List[Dict[str, Any]]:
    """Structure-aware transcript chunking that respects topic and decision boundaries.

    Algorithm:
      1. Run the standard sliding-window chunker (size=6, overlap=2).
      2. Additionally, for each segment that matches a decision/action/agenda
         boundary pattern, extract a focused "anchor chunk" of ±2 segments
         centered on that boundary segment (so critical content is never split).

    Anchor chunks are tagged with 'chunk_type': 'anchor_decision' / 'anchor_action'
    / 'anchor_agenda', while regular chunks have 'chunk_type': 'sliding'.

    Args:
        segments: List of segment dicts/objects
        chunk_size: Sliding window size (default 6, up from old default 4)
        overlap: Step between windows (default 2)
        create_anchor_chunks: If True, add focused anchor chunks around key moments
        meeting_id: Optional meeting identifier for scoped chunking

    Returns:
        Combined list of regular + anchor chunks (deduplication is handled at
        retrieval time by the agent router).
    """
    if not segments:
        return []

    # Phase 1: Standard sliding-window chunks with larger size
    regular_chunks = chunk_transcript_segments(
        segments, chunk_size=chunk_size, overlap=overlap, meeting_id=meeting_id
    )
    for c in regular_chunks:
        c["chunk_type"] = "sliding"
        c["section_type"] = "general"

    if not create_anchor_chunks:
        return regular_chunks

    # Phase 2: Anchor chunks for boundary segments
    anchor_chunks = []
    anchor_window = 2  # segments before + after the boundary segment

    for idx, seg in enumerate(segments):
        boundary_type = _is_boundary_segment(seg)
        if not boundary_type:
            continue

        # Extract ±anchor_window segments around this boundary
        start_idx = max(0, idx - anchor_window)
        end_idx = min(len(segments), idx + anchor_window + 1)
        anchor_segs = segments[start_idx:end_idx]

        if not anchor_segs:
            continue

        formatted_lines = []
        speakers = set()
        for s in anchor_segs:
            text = segment_to_text(s).strip()
            speaker = segment_to_speaker(s)
            ts = segment_to_start_time(s)
            ts_end = segment_to_end_time(s)
            if text:
                # Mark the boundary segment with a tag
                tag = f" ← [{boundary_type.upper()}]" if s is seg else ""
                formatted_lines.append(f"[{ts} - {ts_end}] {speaker}: {text}{tag}")
                speakers.add(speaker)

        if not formatted_lines:
            continue

        anchor_dict = {
            "text": "\n".join(formatted_lines),
            "start_time": segment_to_start_time(anchor_segs[0]),
            "end_time": segment_to_end_time(anchor_segs[-1]),
            "speakers": ", ".join(sorted(speakers)) if speakers else "Unknown Speaker",
            "chunk_id": f"anchor_{boundary_type}_{idx}",
            "chunk_index": 10000 + idx,  # High index to distinguish from sliding chunks
            "chunk_type": f"anchor_{boundary_type}",
            "section_type": boundary_type,
        }
        if meeting_id is not None:
            anchor_dict["meeting_id"] = meeting_id

        anchor_chunks.append(anchor_dict)

    total = len(regular_chunks) + len(anchor_chunks)
    logger.debug(
        f"Structure-aware chunking: {len(regular_chunks)} sliding + "
        f"{len(anchor_chunks)} anchor chunks = {total} total "
        f"(from {len(segments)} segments)"
    )
    return regular_chunks + anchor_chunks


def chunk_transcript_smart(
    segments: List[Any],
    chunk_size: int = 6,
    overlap: int = 2,
    meeting_id: Any = None,
) -> List[Dict[str, Any]]:
    """Auto-selects the best chunking strategy based on transcript size.

    Decision:
      - < 20 segments   → use time-window (preserves small meeting context)
      - 20–200 segments → use structure-aware (balanced quality)
      - > 200 segments  → use sliding-window only (speed, memory constraint)

    This is the RECOMMENDED function to call for new indexing.
    """
    n = len(segments)
    if n < 20:
        logger.debug(f"Smart chunker: {n} segments → time-window strategy")
        chunks = chunk_transcript_by_topic(segments, max_chunk_duration_seconds=120)
        if meeting_id is not None:
            for c in chunks:
                c["meeting_id"] = meeting_id
        return chunks
    elif n <= 200:
        logger.debug(f"Smart chunker: {n} segments → structure-aware strategy")
        return chunk_transcript_structure_aware(
            segments,
            chunk_size=chunk_size,
            overlap=overlap,
            create_anchor_chunks=True,
            meeting_id=meeting_id,
        )
    else:
        logger.debug(f"Smart chunker: {n} segments → sliding-window strategy (large transcript)")
        return chunk_transcript_segments(
            segments, chunk_size=chunk_size, overlap=overlap, meeting_id=meeting_id
        )
