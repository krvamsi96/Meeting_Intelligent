"""Central Quality Engine for Universal Loop Engineering.

Evaluates quality decisions across Speech (STT), Speaker (Diarization/Alignment),
and Content (Summary, MoM, Agenda, Decision, Action Items, RAG/Chatbot).

Ground Truth Policy:
- If Ground Truth reference exists: compute GROUND_TRUTH_METRIC (WER, CER, DER, JER, ROUGE).
- If Ground Truth reference does NOT exist: compute RUNTIME_QUALITY_SIGNAL (never fabricate WER/CER).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.config.config import settings
from backend.services.agenda_service import compute_agenda_decision_match_score
from backend.utils.date_resolver import get_reference_datetime, resolve_deadline
from backend.utils import text_signals as ts

logger = logging.getLogger(__name__)


@dataclass
class QualityGateResult:
    """Result of a quality validation gate evaluation."""

    passed: bool
    quality_score: float  # 0.0 to 100.0
    status: str           # "SUCCESS", "DEGRADED", "FAILED"
    failure_reason: Optional[str] = None
    failed_validation: Optional[str] = None
    metric_type: str = "RUNTIME_QUALITY_SIGNAL"  # "GROUND_TRUTH_METRIC" or "RUNTIME_QUALITY_SIGNAL"
    diagnostics: Dict[str, Any] = field(default_factory=dict)


class QualityEngine:
    """Central Quality & Validation Engine for Loop Engineering."""

    # ─── 1. STT Quality Loop ──────────────────────────────────────────────────
    def validate_stt(
        self,
        transcript_data: Dict[str, Any],
        segments: List[Dict[str, Any]],
        ref_transcript: Optional[str] = None,
    ) -> QualityGateResult:
        full_text = transcript_data.get("text", "") or "".join(
            s.get("text", "") for s in segments
        )
        words = transcript_data.get("words", [])

        if not full_text.strip():
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="Empty transcript text returned from ASR",
                failed_validation="transcript_completeness",
            )

        # Signal 1: Ground Truth evaluation if reference exists
        if ref_transcript and ref_transcript.strip():
            from backend.evaluation import metrics as eval_metrics
            stt_res = eval_metrics.stt_metrics(ref_transcript, full_text)
            wer = stt_res.get("wer", 100.0)
            cer = stt_res.get("cer", 100.0)
            acc = stt_res.get("accuracy", 0.0)

            passed = wer <= 40.0
            status = "SUCCESS" if passed else "FAILED"
            return QualityGateResult(
                passed=passed,
                quality_score=round(acc, 2),
                status=status,
                failure_reason=None if passed else f"WER ({wer:.1f}%) exceeded maximum allowable threshold",
                failed_validation=None if passed else "ground_truth_wer",
                metric_type="GROUND_TRUTH_METRIC",
                diagnostics={"wer": wer, "cer": cer, "accuracy": acc},
            )

        # Signal 2: Runtime Quality Signals (No fabrication of WER/CER)
        word_tokens = [w for w in re.findall(r"\b\w+\b", full_text.lower())]
        if not word_tokens:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="No valid word tokens extracted from transcript",
                failed_validation="tokenization",
            )

        # Check for abnormal repetition (e.g. ASR looping bug)
        repetitive_phrase = False
        if len(word_tokens) > 20:
            trigrams = [
                " ".join(word_tokens[i : i + 3])
                for i in range(len(word_tokens) - 2)
            ]
            most_common_count = max(trigrams.count(t) for t in set(trigrams))
            if most_common_count > max(5, len(trigrams) * 0.25):
                repetitive_phrase = True

        if repetitive_phrase:
            return QualityGateResult(
                passed=False,
                quality_score=30.0,
                status="FAILED",
                failure_reason="Abnormal phrase repetition detected in ASR output",
                failed_validation="abnormal_repetition",
                metric_type="RUNTIME_QUALITY_SIGNAL",
            )

        # Check timestamp integrity
        invalid_timestamps = 0
        for s in segments:
            try:
                st = float(s.get("start_time", 0.0))
                et = float(s.get("end_time", 0.0))
                if et < st or st < 0.0:
                    invalid_timestamps += 1
            except Exception:
                invalid_timestamps += 1

        if invalid_timestamps > 0 and len(segments) > 0:
            ts_ratio = invalid_timestamps / len(segments)
            if ts_ratio > 0.10:
                return QualityGateResult(
                    passed=False,
                    quality_score=40.0,
                    status="FAILED",
                    failure_reason="Invalid segment timestamp order detected",
                    failed_validation="timestamp_integrity",
                    metric_type="RUNTIME_QUALITY_SIGNAL",
                )

        # Compute runtime quality score from vocabulary diversity & segment completeness
        unique_ratio = len(set(word_tokens)) / len(word_tokens)
        runtime_score = round(min(98.0, max(70.0, 80.0 + (unique_ratio * 20.0))), 2)

        return QualityGateResult(
            passed=True,
            quality_score=runtime_score,
            status="SUCCESS",
            failure_reason=None,
            metric_type="RUNTIME_QUALITY_SIGNAL",
            diagnostics={
                "word_count": len(word_tokens),
                "unique_words": len(set(word_tokens)),
                "vocabulary_diversity": round(unique_ratio, 3),
            },
        )

    # ─── 2. Transcript Correction Loop ────────────────────────────────────────
    def validate_transcript_correction(
        self,
        raw_segments: List[Dict[str, Any]],
        corrected_segments: List[Dict[str, Any]],
    ) -> QualityGateResult:
        if not corrected_segments:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="Empty corrected segments returned",
                failed_validation="segment_completeness",
            )

        # Validate length ratio to ensure no large text loss or hallucinated additions
        raw_len = sum(len(s.get("text", "") or s.get("raw_text", "")) for s in raw_segments)
        corr_len = sum(len(s.get("text", "")) for s in corrected_segments)

        if raw_len > 50 and corr_len < (raw_len * 0.50):
            return QualityGateResult(
                passed=False,
                quality_score=35.0,
                status="FAILED",
                failure_reason="Correction resulted in >50% text loss",
                failed_validation="meaning_preservation",
            )

        if raw_len > 50 and corr_len > (raw_len * 2.5):
            return QualityGateResult(
                passed=False,
                quality_score=35.0,
                status="FAILED",
                failure_reason="Correction resulted in >250% text expansion (hallucination suspected)",
                failed_validation="hallucination_check",
            )

        # Bug 5 fix: Dynamic score from preservation ratio instead of hardcoded 95.0.
        # preservation_ratio = corrected_len / raw_len, clamped to [0, 1.0].
        # A ratio of 1.0 → 98.0 score; ratio of 0.5 → 60.0 score (linear interpolation).
        if raw_len > 0:
            preservation_ratio = min(1.0, corr_len / raw_len)
            # Map [0.5, 1.0] → [60.0, 98.0]; below 0.5 is already caught as FAILED above.
            dynamic_score = round(60.0 + (preservation_ratio - 0.5) * 76.0, 2)
            dynamic_score = max(60.0, min(98.0, dynamic_score))
        else:
            dynamic_score = 98.0  # No raw content → correction is trivially valid

        return QualityGateResult(
            passed=True,
            quality_score=dynamic_score,
            status="SUCCESS",
            failure_reason=None,
            diagnostics={
                "raw_char_len": raw_len,
                "corrected_char_len": corr_len,
                "preservation_ratio": round(corr_len / raw_len, 3) if raw_len > 0 else 1.0,
            },
        )

    # ─── 3. Summary Loop ──────────────────────────────────────────────────────
    def validate_summary(
        self, summary_text: str, full_transcript: str
    ) -> QualityGateResult:
        if not summary_text or not summary_text.strip() or "Not Available" in summary_text:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="Summary generation failed or returned blank text",
                failed_validation="summary_completeness",
            )

        words = summary_text.split()
        if len(words) < 5:
            return QualityGateResult(
                passed=False,
                quality_score=30.0,
                status="FAILED",
                failure_reason=f"Summary too brief ({len(words)} words)",
                failed_validation="summary_length",
            )

        # Grounding check using token overlap with transcript
        sum_tokens = set(re.findall(r"\b\w{4,}\b", summary_text.lower()))
        tx_tokens = set(re.findall(r"\b\w{4,}\b", full_transcript.lower()))

        if sum_tokens and tx_tokens:
            overlap = len(sum_tokens & tx_tokens) / len(sum_tokens)
            score = round(min(98.0, max(60.0, overlap * 100.0)), 2)
            passed = overlap >= 0.30
            return QualityGateResult(
                passed=passed,
                quality_score=score,
                status="SUCCESS" if passed else "FAILED",
                failure_reason=None if passed else f"Low grounding overlap ({overlap*100:.1f}%) with source transcript",
                failed_validation=None if passed else "summary_grounding",
                diagnostics={"grounding_overlap_pct": round(overlap * 100.0, 1)},
            )

        return QualityGateResult(
            passed=True,
            quality_score=85.0,
            status="SUCCESS",
        )

    # ─── 4. MoM Loop ──────────────────────────────────────────────────────────
    def validate_mom(self, mom_text: str, full_transcript: str) -> QualityGateResult:
        if not mom_text or not mom_text.strip() or "Not Available" in mom_text:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="MoM generation failed or returned empty text",
                failed_validation="mom_completeness",
            )

        sections = [
            line.strip()
            for line in mom_text.splitlines()
            if line.strip().startswith(("#", "1.", "2.", "3.", "4.", "5.", "Section"))
        ]
        if not sections and len(mom_text.split()) < 30:
            return QualityGateResult(
                passed=False,
                quality_score=35.0,
                status="FAILED",
                failure_reason="MoM lacks structured sections or sufficient detail",
                failed_validation="mom_structure",
            )

        # Bug 2 fix: Dynamic MoM quality score instead of hardcoded 90.0.
        # Signal 1: Structural richness — count of recognised section headers.
        #   0 sections → 0.0, ≥5 sections → 1.0 (capped)
        structural_richness = min(1.0, len(sections) / 5.0)

        # Signal 2: Key-term recall — proportion of meaningful MoM tokens that
        #   also appear in the source transcript (grounding check).
        mom_tokens = set(re.findall(r"\b\w{4,}\b", mom_text.lower()))
        tx_tokens = set(re.findall(r"\b\w{4,}\b", full_transcript.lower())) if full_transcript else set()
        if mom_tokens and tx_tokens:
            term_recall = len(mom_tokens & tx_tokens) / len(mom_tokens)
        elif mom_tokens:
            term_recall = 0.7  # No transcript available — give benefit of the doubt
        else:
            term_recall = 0.0

        # Signal 3: Length score — log-scaled word count capped at 1.0 for ≥200 words.
        import math as _math
        word_count = len(mom_text.split())
        length_score = min(1.0, _math.log10(max(1, word_count)) / _math.log10(200))

        # Weighted combination: structural 35%, term recall 40%, length 25%
        dynamic_score = round(
            (structural_richness * 35.0 + term_recall * 40.0 + length_score * 25.0),
            2,
        )
        # Clamp to [35.0, 98.0] so a passing MoM is never below 35%
        dynamic_score = max(35.0, min(98.0, dynamic_score))

        return QualityGateResult(
            passed=True,
            quality_score=dynamic_score,
            status="SUCCESS",
            diagnostics={
                "section_count": len(sections),
                "structural_richness": round(structural_richness, 3),
                "term_recall": round(term_recall, 3),
                "length_score": round(length_score, 3),
                "word_count": word_count,
            },
        )

    # ─── 5. Agenda Loop ───────────────────────────────────────────────────────
    def validate_agenda(
        self, agenda_items: List[Dict[str, Any]], full_transcript: str
    ) -> QualityGateResult:
        if not agenda_items:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="No agenda items extracted from transcript",
                failed_validation="agenda_extraction",
            )

        valid_items = [
            item
            for item in agenda_items
            if isinstance(item, dict) and item.get("agenda") and str(item["agenda"]).strip()
        ]
        if not valid_items:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="Agenda items list contained no non-empty agenda strings",
                failed_validation="agenda_formatting",
            )

        # Bug 3 fix: Dynamic agenda quality from real match scores instead of hardcoded 92.0.
        # Use precomputed match_score when stored; otherwise compute on-the-fly.
        scored_matches: List[float] = []
        for item in valid_items:
            stored_score = item.get("match_score")
            if stored_score is not None:
                try:
                    scored_matches.append(float(stored_score))
                    continue
                except (TypeError, ValueError):
                    pass
            # On-the-fly: compute cosine similarity for items without a cached score
            agenda_txt = str(item.get("agenda", "")).strip()
            dec_txt = str(item.get("key_decision", "") or item.get("decision", "")).strip()
            _NO_DEC = {
                "no formal decision recorded.", "no formal decision recorded",
                "no decision recorded.", "no decision recorded",
                "no decision.", "no decision", "none", "n/a", "",
            }
            if dec_txt and dec_txt.lower() not in _NO_DEC:
                res = compute_agenda_decision_match_score(agenda_txt, dec_txt)
                s = res.get("match_score")
                if s is not None:
                    scored_matches.append(float(s))

        if scored_matches:
            avg_match = round(sum(scored_matches) / len(scored_matches), 2)
            # Map match scores (0–100) onto a quality score clamped to [50, 98]
            dynamic_score = max(50.0, min(98.0, avg_match))
        else:
            # No decisions were concrete — agenda quality is present but not verified
            dynamic_score = 65.0

        return QualityGateResult(
            passed=True,
            quality_score=dynamic_score,
            status="SUCCESS",
            diagnostics={
                "valid_agenda_count": len(valid_items),
                "scored_items": len(scored_matches),
                "avg_match_score": round(sum(scored_matches) / len(scored_matches), 2) if scored_matches else None,
            },
        )

    # ─── 6. Decision Loop ─────────────────────────────────────────────────────
    def validate_decisions(
        self, agenda_decisions: List[Dict[str, Any]], full_transcript: str
    ) -> QualityGateResult:
        if not agenda_decisions:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="Agenda & Key Decisions list is empty",
                failed_validation="decision_extraction",
            )

        # Calculate dynamic Agenda <-> Decision relevance score via sentence-transformers
        evaluated_scores = []
        for item in agenda_decisions:
            agenda_txt = str(item.get("agenda", "")).strip()
            dec_txt = str(item.get("key_decision", "")).strip()

            if not dec_txt or dec_txt.lower() in ("no formal decision recorded.", "no decision identified"):
                item["key_decision"] = "No decision identified"
                item["match_score"] = None
                continue

            # Dynamic cosine similarity scoring
            res = compute_agenda_decision_match_score(agenda_txt, dec_txt)
            score = res.get("match_score")
            item["match_score"] = score
            if score is not None:
                evaluated_scores.append(score)

        # Bug 6 fix: when every agenda item has a "no decision" placeholder,
        # evaluated_scores will be empty.  Returning a hardcoded 85.0 in that
        # case fabricates a quality signal.  Instead, report DEGRADED with an
        # explicit reason so callers know no real decisions were extracted.
        if not evaluated_scores:
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="DEGRADED",
                failure_reason="No concrete decisions found in extracted agenda items — all entries are 'No formal decision recorded.'",
                failed_validation="decision_content_quality",
                metric_type="RUNTIME_QUALITY_SIGNAL",
                diagnostics={
                    "total_items": len(agenda_decisions),
                    "decisions_with_score": 0,
                    "avg_match_score": None,
                },
            )

        avg_score = sum(evaluated_scores) / len(evaluated_scores)

        return QualityGateResult(
            passed=True,
            quality_score=round(avg_score, 2),
            status="SUCCESS",
            diagnostics={
                "total_items": len(agenda_decisions),
                "decisions_with_score": len(evaluated_scores),
                "avg_match_score": round(avg_score, 2),
            },
        )

    # ─── 7. Action Item Loop ──────────────────────────────────────────────────
    def validate_action_items(
        self,
        action_items: List[Dict[str, Any]],
        full_transcript: str,
        meeting_date: str = "",
    ) -> QualityGateResult:
        if not action_items:
            # Having 0 action items can be valid for informational meetings
            return QualityGateResult(
                passed=True,
                quality_score=100.0,
                status="SUCCESS",
                diagnostics={"action_item_count": 0},
            )

        ref_dt = get_reference_datetime(meeting_date or "today")

        invalid_count = 0
        validated_items = []
        _NO_OWNER = {"unassigned", "unknown", "n/a", "not specified", "none", ""}

        for ai in action_items:
            task = str(ai.get("task", "")).strip()
            raw_deadline = str(ai.get("deadline", "Not specified")).strip()

            if len(task.split()) < 2:
                invalid_count += 1
                continue

            # Deterministic Date Resolver — LLM does NOT calculate dates (§9.16)
            if raw_deadline and raw_deadline.lower() not in ("not specified", "n/a", "none", "tbd", ""):
                res = resolve_deadline(raw_deadline, ref_dt)
                ai["due_date"] = res.due_date.isoformat() if res.due_date else None
                ai["deadline_start"] = res.deadline_start.isoformat() if res.deadline_start else None
                ai["deadline_end"] = res.deadline_end.isoformat() if res.deadline_end else None
                ai["deadline_precision"] = res.deadline_precision

            validated_items.append(ai)

        if invalid_count > len(action_items) * 0.50 and len(action_items) > 1:
            return QualityGateResult(
                passed=False,
                quality_score=40.0,
                status="FAILED",
                failure_reason=f"Majority of extracted action items failed WHO/WHAT validation ({invalid_count}/{len(action_items)} invalid)",
                failed_validation="action_item_schema_validation",
            )

        # Bug 1 fix: Dynamic quality score from real content signals instead of hardcoded 92.0.
        #
        # Signal A: Average confidence score of all validated items (primary quality signal).
        #   Already computed by action_item_service.compute_confidence() — 0.0 to 100.0.
        conf_scores = [
            float(ai.get("confidence") or ai.get("confidence_score", 0) or 0)
            for ai in validated_items
            if ai.get("confidence") is not None or ai.get("confidence_score") is not None
        ]
        # Normalise: confidence_score values in [0,1] range must be scaled to [0,100]
        conf_scores_normalised = [c * 100.0 if c <= 1.0 else c for c in conf_scores]
        avg_confidence = (
            sum(conf_scores_normalised) / len(conf_scores_normalised)
            if conf_scores_normalised else 65.0  # default mid-range when missing
        )

        # Signal B: Deadline resolution ratio — proportion with a resolved due_date.
        resolved_deadlines = sum(
            1 for ai in validated_items if ai.get("due_date") is not None
        )
        deadline_ratio = resolved_deadlines / len(validated_items) if validated_items else 0.0

        # Signal C: Owner identification ratio — proportion with a named owner.
        named_owners = sum(
            1 for ai in validated_items
            if str(ai.get("owner") or "").strip().lower() not in _NO_OWNER
        )
        owner_ratio = named_owners / len(validated_items) if validated_items else 0.0

        # Weighted combination: confidence 55%, deadline_ratio 25%, owner_ratio 20%
        dynamic_score = round(
            avg_confidence * 0.55
            + deadline_ratio * 100.0 * 0.25
            + owner_ratio * 100.0 * 0.20,
            2,
        )
        # Clamp: a passing gate is always in [40, 98] to avoid misleading floor/ceiling
        dynamic_score = max(40.0, min(98.0, dynamic_score))

        return QualityGateResult(
            passed=True,
            quality_score=dynamic_score,
            status="SUCCESS",
            diagnostics={
                "total_extracted": len(action_items),
                "valid_count": len(validated_items),
                "avg_confidence": round(avg_confidence, 2),
                "deadline_resolution_ratio": round(deadline_ratio, 3),
                "owner_identification_ratio": round(owner_ratio, 3),
            },
        )

    # ─── 8. RAG / Chatbot Loop ────────────────────────────────────────────────
    def validate_rag_chatbot(
        self, question: str, retrieved_chunks: List[Dict[str, Any]], answer: str
    ) -> QualityGateResult:
        if not question or not question.strip():
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="Empty user question",
                failed_validation="question_validation",
            )

        if not retrieved_chunks:
            return QualityGateResult(
                passed=False,
                quality_score=20.0,
                status="DEGRADED",
                failure_reason="No context chunks retrieved for question",
                failed_validation="context_retrieval",
            )

        if not answer or not answer.strip():
            return QualityGateResult(
                passed=False,
                quality_score=0.0,
                status="FAILED",
                failure_reason="Empty answer generated by LLM",
                failed_validation="answer_generation",
            )

        # Bug 4 fix: Dynamic RAG quality score instead of hardcoded 90.0.
        #
        # Signal A: Context coverage — chunk count score (0–1, capped at 5 chunks).
        chunk_score = min(1.0, len(retrieved_chunks) / 5.0)

        # Signal B: Answer grounding — token overlap between answer and retrieved context.
        context_text = " ".join(
            (c.get("text") or c.get("content") or str(c))
            for c in retrieved_chunks
        )
        answer_tokens = set(re.findall(r"\b\w{4,}\b", answer.lower()))
        context_tokens = set(re.findall(r"\b\w{4,}\b", context_text.lower()))
        if answer_tokens and context_tokens:
            grounding_ratio = len(answer_tokens & context_tokens) / len(answer_tokens)
        else:
            grounding_ratio = 0.5  # Fallback when text is not extractable

        # Signal C: Answer length adequacy — log-scaled word count (≥20 words → 1.0).
        import math as _math
        answer_words = len(answer.split())
        length_score = min(1.0, _math.log10(max(1, answer_words)) / _math.log10(20))

        # Weighted combination: grounding 50%, chunk coverage 30%, length 20%
        dynamic_score = round(
            grounding_ratio * 100.0 * 0.50
            + chunk_score * 100.0 * 0.30
            + length_score * 100.0 * 0.20,
            2,
        )
        # Clamp to [30, 98]
        dynamic_score = max(30.0, min(98.0, dynamic_score))

        return QualityGateResult(
            passed=True,
            quality_score=dynamic_score,
            status="SUCCESS",
            diagnostics={
                "retrieved_chunks": len(retrieved_chunks),
                "answer_word_count": answer_words,
                "grounding_ratio": round(grounding_ratio, 3),
                "chunk_coverage_score": round(chunk_score, 3),
            },
        )


quality_engine = QualityEngine()
