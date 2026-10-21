"""
Spoken-language handling for AIRAWAT Whisper
────────────────────────────────────────────
AIRAWAT's Whisper gateway is the only speech-to-text engine and it CANNOT identify a language: a request
without a code answers "400 need model or language", `auto` answers "404 Model not found for language
auto", and the code it is given decides the output (English audio + `hi` -> Devanagari transliteration).
So the spoken language is never guessed here - it comes from, in order:

  1. the upload (form field `language`, stored on the meeting as `source_language`),
  2. the configured default (AIRAWAT admin setting, else `AIRAWAT_WHISPER_LANGUAGE`),
  3. otherwise the meeting fails with a clear LanguageIdentificationError.

No local language-ID model is used. Translation to English happens later (translation_service).
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# ISO 639-1 (plus a few 639-2/3 Whisper emits) → English display name.
# Not an allow-list — only used to render a friendly name; unknown codes are
# passed through verbatim.
LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "hi": "Hindi", "te": "Telugu", "ta": "Tamil",
    "kn": "Kannada", "ml": "Malayalam", "bn": "Bengali", "mr": "Marathi",
    "gu": "Gujarati", "pa": "Punjabi", "or": "Odia", "as": "Assamese",
    "ur": "Urdu", "sa": "Sanskrit", "ne": "Nepali", "si": "Sinhala",
    "fr": "French", "es": "Spanish", "de": "German", "pt": "Portuguese",
    "ru": "Russian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
    "ar": "Arabic", "fa": "Persian", "tr": "Turkish", "it": "Italian",
    "nl": "Dutch", "pl": "Polish", "id": "Indonesian", "vi": "Vietnamese",
    "th": "Thai", "he": "Hebrew", "uk": "Ukrainian", "ms": "Malay",
    "sw": "Swahili", "ro": "Romanian", "el": "Greek", "cs": "Czech",
}

_ENGLISH_CODES = {"en", "eng", "english", "en-us", "en-gb", "en-in"}


def language_name(code: Optional[str]) -> str:
    if not code:
        return "Unknown"
    c = str(code).strip().lower()
    return LANGUAGE_NAMES.get(c, LANGUAGE_NAMES.get(c.split("-")[0], code))


_CODES_BY_NAME: Dict[str, str] = {name.lower(): code for code, name in LANGUAGE_NAMES.items()}


def code_for_reported_language(reported: Optional[str]) -> Optional[str]:
    """ISO code for a language an STT engine REPORTED (Groq's verbose_json says ``"english"``, others ``"en"``).

    None when nothing usable was reported; an unrecognised value is kept only if it looks like a code.
    """
    r = (reported or "").strip().lower()
    if not r or r in _NOT_A_LANGUAGE:
        return None
    if r in _CODES_BY_NAME:
        return _CODES_BY_NAME[r]
    return r if _CODE_RE.match(r) else None


def is_english_code(code: Optional[str]) -> bool:
    return bool(code) and str(code).strip().lower() in _ENGLISH_CODES


_CODE_RE = re.compile(r"^[a-z]{2,3}(-[a-z]{2,4})?$")
_NOT_A_LANGUAGE = {"", "auto", "any", "detect", "multi"}


def normalize_language_code(code: Optional[str]) -> Optional[str]:
    """Lower-cased ISO code (`hi`, `te`, `en-in` ...) or None for blank / "auto"-style values.

    Raises ValueError for something that is not a language code at all (so an upload typo is rejected).
    """
    c = (code or "").strip().lower()
    if c in _NOT_A_LANGUAGE:
        return None
    if not _CODE_RE.match(c):
        raise ValueError(f"'{code}' is not a language code (use an ISO 639-1 code such as hi, te, ml, ta, en)")
    return c


def resolve_spoken_language(requested: Optional[str], configured: Optional[str]) -> Tuple[Optional[str], str]:
    """(code, source) for AIRAWAT Whisper: the meeting's own language wins over the configured default.

    `source` is "upload", "configured" or "none" (code None: the caller must fail with a clear error).
    """
    try:
        req = normalize_language_code(requested)
    except ValueError:
        req = None
    if req:
        return req, "upload"
    try:
        conf = normalize_language_code(configured)
    except ValueError:
        conf = None
    if conf:
        return conf, "configured"
    return None, "none"
