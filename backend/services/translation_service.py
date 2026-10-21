"""
Universal → English Translation Service
──────────────────────────────────────
The product rule: **every user-facing output must be English**, regardless of the
spoken language of the uploaded video/audio.

This service translates the *already-transcribed, already-corrected* native
transcript into fluent English while preserving structure exactly:

  * speaker labels           — untouched
  * segment start/end times  — untouched
  * segment count & order    — 1:1 (no merge / split / drop)
  * numbers, dates, proper nouns, acronyms, product/project names — preserved
  * already-English content   — passed through, not "re-translated"

Design notes
------------
* Language-agnostic. Nothing is hardcoded per language — Telugu, Hindi, Tamil,
  Kannada, Malayalam, Bengali, Marathi, Urdu, French, Japanese … all handled the
  same way. Mixed-language / code-switched segments are normalised to English.
* We translate AFTER native ASR + correction (not by asking Whisper to
  "translate"), so timestamps stay aligned to the native word decode and
  translation quality comes from the LLM understanding full conversational
  context.
* Immutable audit trail: ``raw_text`` (raw native ASR) is never modified;
  ``original_text`` keeps the corrected native text; ``text`` becomes English.
* Fail-safe: if the LLM is unavailable or a batch fails, the pipeline still
  completes — the affected segments keep their native text and ``meta`` reports
  a partial translation instead of raising.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.config.config import settings
from backend.services.language_service import is_english_code, language_name

logger = logging.getLogger(__name__)

# Unicode blocks for the major non-Latin scripts we expect (Indic + CJK + Arabic
# + Cyrillic + Greek + Hebrew + Thai). Used only as a cheap "did any native text
# survive translation?" safety-net signal — never to decide the source language.
_NON_LATIN_RE = re.compile(
    "["
    "ऀ-ॿ"          # Devanagari (Hindi, Marathi, Sanskrit, Nepali)
    "ঀ-৿"          # Bengali / Assamese
    "਀-੿"          # Gurmukhi (Punjabi)
    "઀-૿"          # Gujarati
    "଀-୿"          # Odia
    "஀-௿"          # Tamil
    "ఀ-౿"          # Telugu
    "ಀ-೿"          # Kannada
    "ഀ-ൿ"          # Malayalam
    "඀-෿"          # Sinhala
    "؀-ۿݐ-ݿ"  # Arabic / Urdu / Persian
    "Ѐ-ӿ"          # Cyrillic
    "Ͱ-Ͽ"          # Greek
    "֐-׿"          # Hebrew
    "฀-๿"          # Thai
    "぀-ヿ一-鿿가-힯"  # Hiragana/Katakana + CJK + Hangul
    "]"
)


def looks_non_english(text: str, threshold: float = 0.08) -> bool:
    """Heuristic: does ``text`` contain a meaningful share of non-Latin script?

    Not a language detector — just a guard so a stray untranslated line can be
    caught and re-translated before it reaches the user.
    """
    if not text:
        return False
    non_latin = len(_NON_LATIN_RE.findall(text))
    if non_latin == 0:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return non_latin > 0
    return (non_latin / letters) >= threshold


def _translation_disabled() -> bool:
    return not bool(getattr(settings, "TRANSLATION_ENABLED", True))


def _get_llm(temperature: float = 0.0):
    from backend.services.model_router import model_router
    return model_router.get_llm(temperature=temperature)


# ─────────────────────────────────────────────────────────────────────────────
# Core: batch translate a list of {text,...} segments to English
# ─────────────────────────────────────────────────────────────────────────────
def _system_prompt() -> str:
    return (
        "You are a professional conference interpreter and translator. You are given "
        "individual utterances from a meeting transcript, each with an index. Some "
        "utterances are in a non-English language (any language), some may be "
        "code-switched (mixed with English), some may already be fully English.\n\n"
        "TASK: Return the ENGLISH version of every utterance.\n\n"
        "STRICT RULES:\n"
        "1. Translate meaning faithfully and completely into natural, fluent, "
        "professional English. Do not summarise, shorten, expand, censor, or add "
        "commentary.\n"
        "2. If an utterance is already fully English, return it unchanged (only fix "
        "obvious typos).\n"
        "3. Preserve ALL numbers, dates, times, money amounts, percentages, and "
        "measurements exactly.\n"
        "4. Preserve proper nouns, person names, place names, company/product/"
        "project names, file names, URLs, email addresses, and technical acronyms "
        "(API, KPI, SLA, PostgreSQL, FastAPI, NeMo, …) — transliterate names only "
        "when they are written in a non-Latin script.\n"
        "5. Keep it one-to-one: exactly one English string per input index, same "
        "order. Never merge or split utterances. Never drop an index.\n"
        "6. If an utterance is empty or pure filler ('hmm', 'uh'), return it as-is.\n"
        "7. Output MUST contain no non-English words except untranslatable proper "
        "nouns.\n\n"
        'OUTPUT: ONLY a JSON object: {"translations": [{"i": <index>, "en": "<english text>"}, ...]}\n'
        "No markdown, no prose, no explanation."
    )


# An English rendering below this share of a source utterance's word count (for utterances of at least
# _MIN_WORDS_FOR_COMPLETENESS_CHECK words) is treated as truncated. Real translations are within roughly 0.6-2x.
_MIN_WORDS_FOR_COMPLETENESS_CHECK = 40
_MIN_TRANSLATION_WORD_RATIO = 0.35


def _parse_llm_json(raw: str) -> Optional[Dict[str, Any]]:
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE | re.DOTALL).strip()
    # Grab the outermost JSON object if the model wrapped it in prose.
    if not s.startswith("{"):
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
        if m:
            s = m.group(0)
    try:
        return json.loads(s)
    except Exception:
        return None


def _translate_batch(
    texts: List[str], source_lang_name: str, glossary: str = "", errors: Optional[List[str]] = None
) -> Optional[List[str]]:
    """Translate a list of strings; returns a same-length list or None on failure.

    ``errors`` (optional) collects the failure reasons so a caller that must not swallow them can report them.
    """
    if not texts:
        return []
    payload = [{"i": i, "text": t} for i, t in enumerate(texts)]
    glossary_line = (
        f"Known names / terms for this meeting (keep spelled exactly like this in "
        f"the English output): {glossary}\n" if glossary else ""
    )
    user_msg = (
        f"Source language (auto-detected, for context only): {source_lang_name}\n"
        f"{glossary_line}"
        f"Translate every utterance to English.\n\n"
        f"UTTERANCES (JSON):\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        llm = _get_llm(temperature=0.0)
        res = llm.invoke(
            [
                {"role": "system", "content": _system_prompt()},
                {"role": "user", "content": user_msg},
            ]
        )
        content = getattr(res, "content", None) or str(res)
        parsed = _parse_llm_json(content)
        if not parsed or "translations" not in parsed:
            logger.warning("[Translate] batch response missing 'translations' key.")
            if errors is not None:
                errors.append("the LLM answer had no 'translations' JSON")
            return None
        by_idx: Dict[int, str] = {}
        for item in parsed["translations"]:
            if not isinstance(item, dict):
                continue
            try:
                idx = int(item.get("i"))
            except (TypeError, ValueError):
                continue
            en = item.get("en")
            if isinstance(en, str):
                by_idx[idx] = en.strip()
        out: List[str] = []
        for i, original in enumerate(texts):
            en = by_idx.get(i)
            # A long utterance answered with a fraction of its length is a cut-off answer (the reply outgrew the
            # model's output limit), not a translation: report the batch as failed so it is retried / split,
            # instead of storing the first few sentences as the whole utterance.
            src_words = len(original.split())
            if en and src_words >= _MIN_WORDS_FOR_COMPLETENESS_CHECK and len(en.split()) < src_words * _MIN_TRANSLATION_WORD_RATIO:
                logger.warning(
                    "[Translate] utterance %d looks truncated (%d source words -> %d English words).",
                    i, src_words, len(en.split()),
                )
                if errors is not None:
                    errors.append(f"the translation of utterance {i} was cut off ({src_words} -> {len(en.split())} words)")
                return None
            out.append(en if en is not None and en != "" else original)
        return out
    except Exception as e:
        logger.warning(f"[Translate] batch failed ({type(e).__name__}: {e}).")
        if errors is not None:
            errors.append(f"{type(e).__name__}: {str(e)[:200]}")
        return None


def translate_text_to_english(text: str, source_lang: str = "") -> str:
    """Translate a single free-text block to English (best-effort, never raises)."""
    if not text or not text.strip():
        return text
    if _translation_disabled():
        return text
    if is_english_code(source_lang) and not looks_non_english(text):
        return text
    res = _translate_batch([text], language_name(source_lang) or "unknown")
    if res and res[0]:
        return res[0]
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point used by the media pipeline
# ─────────────────────────────────────────────────────────────────────────────
def translate_segments_to_english(
    segments: List[Dict[str, Any]],
    source_lang: str,
    meeting_context: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Translate diarized+corrected transcript segments to English.

    Input segment dict keys used: ``speaker``, ``start_time``, ``end_time``,
    ``text`` (corrected native), ``raw_text`` (immutable raw ASR), plus any
    correction_* fields which are passed through untouched.

    Returns ``(new_segments, meta)`` where each new segment has:
        text          -> English
        original_text -> the corrected native text (pre-translation)
        raw_text      -> unchanged (raw native ASR)
    ``meta`` = {translated, source_language, source_language_name,
                total_segments, translated_segments, partial, engine}
    """
    src_name = language_name(source_lang)
    meta: Dict[str, Any] = {
        "translated": False,
        "source_language": (source_lang or "").lower(),
        "source_language_name": src_name,
        "total_segments": len(segments or []),
        "translated_segments": 0,
        "partial": False,
        "engine": "enterprise_llm",
    }

    if not segments:
        return segments or [], meta

    any_native = any(looks_non_english(s.get("text", "")) for s in segments)

    # Passthrough (still stamp original_text so the schema stays uniform) when:
    #   * translation is switched off, OR
    #   * the detected source is English AND no non-Latin text slipped through, OR
    #   * the language is unknown/empty (detection failed) AND nothing in the
    #     transcript looks non-English — i.e. almost certainly an English meeting.
    unknown_lang = not (source_lang or "").strip()
    passthrough = (
        _translation_disabled()
        or (is_english_code(source_lang) and not any_native)
        or (unknown_lang and not any_native)
    )
    if passthrough:
        out = []
        for s in segments:
            ns = dict(s)
            ns.setdefault("original_text", s.get("text", ""))
            out.append(ns)
        meta["note"] = (
            "translation disabled" if _translation_disabled()
            else "source already English" if is_english_code(source_lang)
            else "language undetermined but transcript is Latin-script English"
        )
        return out, meta

    if is_english_code(source_lang) and any_native:
        logger.info("[Translate] Source tagged English but non-Latin text present — translating anyway.")
    elif unknown_lang:
        logger.info("[Translate] Source language undetermined but non-English text present — translating.")

    batch_size = max(1, int(getattr(settings, "TRANSLATION_BATCH_SIZE", 20)))
    max_retries = max(0, int(getattr(settings, "TRANSLATION_MAX_RETRIES", 2)))

    glossary = ""
    if meeting_context:
        bits = [str(meeting_context.get(k, "")).strip()
                for k in ("title", "agenda", "participants", "project_name")]
        glossary = " | ".join(b for b in bits if b)[:500]

    texts = [str(s.get("text", "") or "") for s in segments]
    translated: List[Optional[str]] = [None] * len(texts)
    failure_reasons: List[str] = []

    for start in range(0, len(texts), batch_size):
        chunk_idx = list(range(start, min(start + batch_size, len(texts))))
        chunk = [texts[i] for i in chunk_idx]
        result: Optional[List[str]] = None
        for attempt in range(max_retries + 1):
            result = _translate_batch(chunk, src_name, glossary, errors=failure_reasons)
            if result is not None:
                break
            if attempt < max_retries:
                logger.warning(
                    "[Translate] batch %d-%d failed — retry %d/%d",
                    chunk_idx[0], chunk_idx[-1], attempt + 1, max_retries,
                )
        if result is None:
            # Whole-batch failure — try segment-by-segment once before giving up.
            for j, gi in enumerate(chunk_idx):
                single = _translate_batch([chunk[j]], src_name, glossary, errors=failure_reasons)
                translated[gi] = single[0] if single else None
        else:
            for j, gi in enumerate(chunk_idx):
                translated[gi] = result[j]

    # A segment whose translation FAILED (no answer from the LLM at all - as opposed to a model that echoed a
    # proper noun back) would leave native text inside outputs that must be English: that is an AIRAWAT
    # failure to report, not a "partial translation" to carry on with.
    n_failed = sum(1 for t, native in zip(translated, texts) if t is None and native.strip())
    if n_failed:
        from backend.utils.exceptions import AIProviderError

        raise AIProviderError(
            f"AI LLM could not translate {n_failed}/{len(texts)} segment(s) from {src_name} to English after "
            f"{max_retries + 1} attempt(s) ({'; '.join(sorted(set(failure_reasons)))[:300] or 'no answer'}). "
            f"Please check AI provider status and retry."
        )

    out_segments: List[Dict[str, Any]] = []
    n_translated = 0
    n_still_native = 0
    for seg, native_text, en_text in zip(segments, texts, translated):
        ns = dict(seg)
        ns["original_text"] = native_text
        if en_text and en_text.strip():
            final = en_text.strip()
            # Last-resort: a "translation" that is still non-Latin = model echoed
            # the input. Retry that one line individually.
            if looks_non_english(final):
                retry = _translate_batch([native_text], src_name)
                if retry and retry[0] and not looks_non_english(retry[0]):
                    final = retry[0].strip()
            ns["text"] = final
            if final != native_text:
                n_translated += 1
            if looks_non_english(final):
                n_still_native += 1
        else:
            ns["text"] = native_text
            n_still_native += 1
        out_segments.append(ns)

    meta["translated"] = n_translated > 0
    meta["translated_segments"] = n_translated
    meta["partial"] = n_still_native > 0
    if n_still_native:
        logger.warning(
            "[Translate] %d/%d segment(s) could not be translated and kept native text.",
            n_still_native, len(segments),
        )
    logger.info(
        "[Translate] %s → English | %d/%d segments translated%s",
        src_name, n_translated, len(segments),
        " (partial)" if meta["partial"] else "",
    )
    return out_segments, meta
