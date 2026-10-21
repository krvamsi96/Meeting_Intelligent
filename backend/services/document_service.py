import html as _html_escape
import json
import logging
import os
import re
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Hostile-input limits ──────────────────────────────────────────────────────
# Uploads are capped by compressed size only, so a tiny .docx/.xlsx/.pptx can expand to many GB
# (zip bomb) and a spreadsheet can declare billions of cells; either exhausts RAM on a ~1 GB
# server before any of the code below runs. Bound what we are willing to read.
MAX_ARCHIVE_UNCOMPRESSED_BYTES = int(os.getenv("DOC_MAX_UNCOMPRESSED_MB", "500")) * 1024 * 1024
MAX_PDF_PAGES = int(os.getenv("DOC_MAX_PDF_PAGES", "3000"))
MAX_SPREADSHEET_ROWS = int(os.getenv("DOC_MAX_SPREADSHEET_ROWS", "200000"))
MAX_EXTRACTED_CHARS = int(os.getenv("DOC_MAX_EXTRACTED_CHARS", "5000000"))


def _assert_safe_archive(filepath: str) -> None:
    """Refuse OOXML containers (docx/pptx/xlsx) whose declared expanded size is absurd."""
    if not zipfile.is_zipfile(filepath):
        return
    try:
        with zipfile.ZipFile(filepath) as zf:
            total = sum(info.file_size for info in zf.infolist())
    except zipfile.BadZipFile:
        return  # let the format-specific reader report the corruption
    if total > MAX_ARCHIVE_UNCOMPRESSED_BYTES:
        raise ValueError(
            f"Document expands to {total / (1024 * 1024):.0f} MB when unpacked "
            f"(limit {MAX_ARCHIVE_UNCOMPRESSED_BYTES // (1024 * 1024)} MB); refusing to process it."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic Configuration — All values overridable via DOCSERVICE_* env vars
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class DocumentServiceConfig:
    """Configurable parameters for document text extraction and HTML preview.

    Override any value by setting DOCSERVICE_<FIELD_NAME> as env var.
    """

    # ── LLM extraction limits ──
    LLM_METADATA_MAX_CHARS: int = 12000
    LLM_TITLE_MAX_CHARS: int = 10000
    LLM_PREVIEW_MAX_CHARS: int = 8000

    # ── Document segmentation ──
    SEGMENT_WORD_CHUNK_SIZE: int = 100
    SEGMENT_INTERVAL_SECONDS: int = 2

    # ── PDF heading detection ──
    PDF_HEADING_MAX_LENGTH: int = 80
    PDF_HEADING_MIN_LENGTH: int = 3

    # ── HTML conversion ──
    HTML_FALLBACK_CHARS: int = 8000

    @classmethod
    def from_env(cls):
        cfg = cls()
        prefix = "DOCSERVICE_"
        for field in cls.__dataclass_fields__:
            env_key = f"{prefix}{field}"
            val = os.getenv(env_key)
            if val is not None:
                f_type = cls.__dataclass_fields__[field].type
                try:
                    if f_type == int or f_type == Optional[int]:
                        setattr(cfg, field, int(val))
                    elif f_type == float or f_type == Optional[float]:
                        setattr(cfg, field, float(val))
                    elif f_type == str or f_type == Optional[str]:
                        setattr(cfg, field, val)
                    elif f_type == bool or f_type == Optional[bool]:
                        setattr(cfg, field, val.lower() in ("true", "1", "yes"))
                except (ValueError, TypeError):
                    logger.warning(f"Invalid env var {env_key}={val}, keeping default")
        return cfg


doc_cfg = DocumentServiceConfig.from_env()


def extract_text_from_pdf(filepath: str) -> str:
    """Extract text from PDF using pypdf."""
    text_content = []
    try:
        import pypdf

        reader = pypdf.PdfReader(filepath)
        for page_idx, page in enumerate(reader.pages):
            if page_idx >= MAX_PDF_PAGES:
                logger.warning(f"PDF {filepath} has more than {MAX_PDF_PAGES} pages; extracting the first {MAX_PDF_PAGES} only.")
                break
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_content.append(f"--- Page {page_idx + 1} ---\n{page_text.strip()}")
    except Exception as e:
        logger.warning(f"pypdf extraction failed for {filepath}: {e}")

    return "\n\n".join(text_content)


def extract_text_from_docx(filepath: str) -> str:
    """Extract text from DOCX using python-docx, with raw string fallback for .doc."""
    text_content = []
    try:
        import docx

        doc = docx.Document(filepath)
        for p in doc.paragraphs:
            if p.text.strip():
                text_content.append(p.text.strip())
        for table in doc.tables:
            for row in table.rows:
                row_str = " | ".join(
                    [cell.text.strip() for cell in row.cells if cell.text.strip()]
                )
                if row_str:
                    text_content.append(row_str)
    except Exception as e:
        logger.warning(f"python-docx extraction failed for {filepath}: {e}")

    res = "\n".join(text_content)
    if not res.strip():
        res = extract_text_from_plain_text(filepath)
    return res


def extract_text_from_pptx(filepath: str) -> str:
    """Extract text from PPTX using python-pptx, with fallback."""
    text_content = []
    try:
        from pptx import Presentation

        prs = Presentation(filepath)
        for idx, slide in enumerate(prs.slides):
            slide_lines = [f"--- Slide {idx + 1} ---"]
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_lines.append(shape.text.strip())
            if len(slide_lines) > 1:
                text_content.append("\n".join(slide_lines))
    except Exception as e:
        logger.warning(f"python-pptx extraction failed for {filepath}: {e}")

    res = "\n\n".join(text_content)
    if not res.strip():
        res = extract_text_from_plain_text(filepath)
    return res


def extract_text_from_xlsx(filepath: str) -> str:
    """Extract text from XLSX/XLS using openpyxl, with fallback."""
    text_content = []
    try:
        import openpyxl

        # read_only streams rows instead of materialising the whole workbook in memory
        wb = openpyxl.load_workbook(filepath, data_only=True, read_only=True)
        rows_seen = 0
        for sheet in wb.sheetnames:
            ws = wb[sheet]
            sheet_lines = [f"--- Sheet: {sheet} ---"]
            for row in ws.iter_rows(values_only=True):
                rows_seen += 1
                if rows_seen > MAX_SPREADSHEET_ROWS:
                    logger.warning(f"Spreadsheet {filepath} exceeds {MAX_SPREADSHEET_ROWS} rows; truncating.")
                    break
                row_vals = [
                    str(v).strip()
                    for v in row
                    if v is not None and str(v).strip() != ""
                ]
                if row_vals:
                    sheet_lines.append(" | ".join(row_vals))
            if len(sheet_lines) > 1:
                text_content.append("\n".join(sheet_lines))
            if rows_seen > MAX_SPREADSHEET_ROWS:
                break
        wb.close()
    except Exception as e:
        logger.warning(f"openpyxl extraction failed for {filepath}: {e}")

    res = "\n\n".join(text_content)
    if not res.strip():
        res = extract_text_from_plain_text(filepath)
    return res


def extract_text_from_html(filepath: str) -> str:
    """Extract clean text from HTML file using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.extract()
            text = soup.get_text(separator="\n")
            lines = (line.strip() for line in text.splitlines())
            return "\n".join(line for line in lines if line)
    except Exception as e:
        logger.warning(f"BeautifulSoup HTML extraction warning for {filepath}: {e}")
        return extract_text_from_plain_text(filepath)


# "let us begin" / "var" as an ordinary word at the start of a hard-wrapped line is prose, not
# code: only an actual declaration (`let x =`, `const y=`, `var z =`) is JavaScript noise.
_JS_DECLARATION_RE = re.compile(r"^(?:var|const|let)\s+[A-Za-z_$][\w$]*\s*=")
_JS_HINT_RE = re.compile(r"function\s*\w*\s*\(|window\[|\b(?:var|const|let)\s+[A-Za-z_$][\w$]*\s*=|String\.prototype|Array\.prototype|Object\.defineProperty")


def clean_document_text(text: str) -> str:
    """
    Sanitizes extracted document text by stripping HTML tags, JavaScript blocks,
    script/style tags, CSS declarations, and minified web code, leaving clean plain text.
    """
    if not text or not text.strip():
        return ""

    raw_text = text

    # 1. If HTML tags or DOCTYPE or scripts are present, parse with BeautifulSoup first
    if any(
        k in raw_text.lower()
        for k in ["<html", "<!doctype", "<script", "<div", "<head"]
    ):
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(raw_text, "html.parser")
            # Extract and remove noise tags
            for tag in soup(
                [
                    "script",
                    "style",
                    "head",
                    "nav",
                    "footer",
                    "iframe",
                    "noscript",
                    "meta",
                    "link",
                ]
            ):
                tag.extract()
            extracted = soup.get_text(separator="\n")
            if extracted.strip():
                raw_text = extracted
        except Exception as e:
            logger.warning(f"BeautifulSoup text cleaning fallback: {e}")

    # 2. Regex cleaning for residual HTML tags, JS variables, and minified code blocks
    raw_text = re.sub(
        r"<script[^>]*>[\s\S]*?</script>", " ", raw_text, flags=re.IGNORECASE
    )
    raw_text = re.sub(
        r"<style[^>]*>[\s\S]*?</style>", " ", raw_text, flags=re.IGNORECASE
    )
    raw_text = re.sub(r"<!DOCTYPE[^>]*>", " ", raw_text, flags=re.IGNORECASE)
    # Strip remaining HTML tags. A tag must start with a letter, "/" or "!": the old `<[^>]+>`
    # also swallowed everything between a "<" and the next ">" in ordinary text ("if x < 5 and
    # y > 3" lost " 5 and y ").
    raw_text = re.sub(r"</?[A-Za-z!][^>]*>", " ", raw_text)

    # Remove Javascript code patterns
    js_patterns = [
        r"window\[.*?\]\s*=\s*\{.*?\};?",
        r"function\s*\w*\s*\(.*?\)\s*\{.*?\}",
        r"var\s+\w+\s*=\s*.*?;",
        r"const\s+\w+\s*=\s*.*?;",
        r"let\s+\w+\s*=\s*.*?;",
        r"\(function\s*\(.*?\)\s*\{.*?\}\)\s*\(.*?\);?",
        r'\{.*?productName.*?\}"',
        r"deleteIsEnforced:.*?,",
        r"sealIsEnforced:.*?,",
        r"heartbeatRate:.*?,",
        r"disableAllReporting:.*?,",
        r"String\.prototype\.\w+",
        r"Array\.prototype\.\w+",
        r"Object\.defineProperty\s*\(.*?\)",
    ]
    # These DOTALL patterns scan forward to the next ';' / '}' from every hit, so run them only on
    # text that actually looks like scraped web code.
    if _JS_HINT_RE.search(raw_text):
        for pattern in js_patterns:
            raw_text = re.sub(pattern, " ", raw_text, flags=re.IGNORECASE | re.DOTALL)

    # 3. Line-by-line filtering to filter out residual JS/CSS code noise
    clean_lines = []
    for line in raw_text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue

        # Skip JS / CSS code noise lines
        if (
            "function(" in line_str
            or "window[" in line_str
            or "typeof Symbol" in line_str
            or "TypeError(" in line_str
            or "encodeURIComponent" in line_str
            or "JSON.stringify" in line_str
            or "csp.withgoogle.com" in line_str
            or "ppConfig" in line_str
            or _JS_DECLARATION_RE.match(line_str)
            or line_str.endswith("};")
            or (line_str.count("{") > 2 and line_str.count("}") > 2)
            or ("margin:" in line_str and "padding:" in line_str)
            or line_str.startswith("/*")
        ):
            continue

        # Keep valid text line
        clean_lines.append(line_str)

    cleaned_result = "\n".join(clean_lines)

    # 4. Collapse multiple blank lines
    cleaned_result = re.sub(r"\n{3,}", "\n\n", cleaned_result).strip()

    return (
        cleaned_result
        if cleaned_result
        else "Clean text could not be extracted from the document."
    )


def extract_text_from_plain_text(filepath: str) -> str:
    """Extract plain text files (txt, csv, md, rtf, log, json, html) or extract printable strings."""
    encodings = ["utf-8", "latin-1", "cp1252", "utf-16"]
    for enc in encodings:
        try:
            with open(filepath, "r", encoding=enc, errors="ignore") as f:
                content = f.read()

                # If the content looks like HTML, use BeautifulSoup to extract clean text
                if any(
                    tag in content[:2000].lower()
                    for tag in ["<html", "<!doctype", "<head", "<body", "<div"]
                ):
                    try:
                        from bs4 import BeautifulSoup

                        soup = BeautifulSoup(content, "html.parser")
                        for tag in soup(
                            [
                                "script",
                                "style",
                                "head",
                                "nav",
                                "footer",
                                "iframe",
                                "noscript",
                                "meta",
                                "link",
                            ]
                        ):
                            tag.extract()
                        text = soup.get_text(separator="\n")
                        lines = [
                            line.strip() for line in text.splitlines() if line.strip()
                        ]
                        if lines:
                            return "\n".join(lines)
                    except Exception:
                        pass

                # If printable ASCII/Unicode characters dominate
                clean_lines = []
                for line in content.splitlines():
                    printable = "".join(
                        [c for c in line if c.isprintable() or c in "\t\n\r"]
                    )
                    if len(printable.strip()) > 3:
                        clean_lines.append(printable.strip())
                if clean_lines:
                    return "\n".join(clean_lines)
        except Exception:
            continue
    return ""


def extract_document_text(filepath: str) -> str:
    """
    Main entry point for extracting raw text from any document format.
    Supports PDF, DOC, DOCX, PPT, PPTX, XLS, XLSX, TXT, CSV, RTF, MD, JSON, LOG, HTML, etc.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in (".docx", ".pptx", ".xlsx"):
        # Outside the per-format try/except blocks on purpose: their broad handlers would swallow
        # this and fall back to "printable strings" of the raw binary.
        _assert_safe_archive(filepath)

    if ext == ".pdf":
        text = extract_text_from_pdf(filepath)
    elif ext in [".docx", ".doc"]:
        text = extract_text_from_docx(filepath)
    elif ext in [".pptx", ".ppt"]:
        text = extract_text_from_pptx(filepath)
    elif ext in [".xlsx", ".xls", ".csv"]:
        text = extract_text_from_xlsx(filepath)
    elif ext in [".html", ".htm"]:
        text = extract_text_from_html(filepath)
    else:
        text = extract_text_from_plain_text(filepath)

    if not text.strip():
        text = extract_text_from_plain_text(filepath)

    if len(text) > MAX_EXTRACTED_CHARS:
        logger.warning(f"Extracted text ({len(text)} chars) truncated to {MAX_EXTRACTED_CHARS} chars.")
        text = text[:MAX_EXTRACTED_CHARS]

    # Sanitize and strip HTML/JS code noise
    cleaned_text = clean_document_text(text)
    return cleaned_text


def extract_document_metadata_with_llm(doc_text: str, filename: str) -> Dict[str, Any]:
    """
    Extracts structured meeting metadata from document text using the AIRAWAT Enterprise AI provider:
    - Meeting Title
    - Agenda
    - Participants (list of names)
    - Meeting Date
    - Meeting Time
    - Formatted Transcript Segments
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from backend.services.llm_workflow import get_llm

    truncated_text = doc_text[: doc_cfg.LLM_METADATA_MAX_CHARS]

    system_prompt = (
        "You are an expert executive meeting assistant. Analyze the provided document text "
        "and extract meeting metadata into valid JSON. Do not include any markdown formatting, "
        "only raw JSON text."
    )

    user_prompt = f"""
Filename: {filename}
Document Text Snippet:
{truncated_text}

Extract the following meeting details into valid JSON with these exact keys:
{{
  "title": "Meeting Title (string, max 10 words, fall back to filename if unknown)",
  "agenda": "Meeting agenda summary (string)",
  "participants": ["Name 1", "Name 2"],
  "meeting_date": "YYYY-MM-DD (string, or 'Not Specified')",
  "meeting_time": "HH:MM AM/PM (string, or 'Not Specified')"
}}
"""

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        res_text = response.content.strip()

        # Strip code blocks if present
        if res_text.startswith("```"):
            res_text = re.sub(r"^```[a-zA-Z]*\n?", "", res_text)
            res_text = re.sub(r"\n?```$", "", res_text)

        data = json.loads(res_text.strip())
        if not isinstance(data, dict):
            raise ValueError("LLM metadata response was not a JSON object")

        def _as_text(value: Any, default: str) -> str:
            return str(value).strip() if value not in (None, "", [], {}) else default

        participants = data.get("participants")
        if isinstance(participants, list):
            # elements can be numbers / objects — one non-string used to raise and throw away the
            # ENTIRE (otherwise fine) LLM answer in favour of the generic filename fallback
            participants_text = ", ".join(str(x).strip() for x in participants if str(x).strip())
        else:
            participants_text = str(participants or "")
        return {
            "title": _as_text(data.get("title"), filename),
            "agenda": _as_text(data.get("agenda"), ""),          # nothing stated -> empty, never a made-up agenda
            "participants": participants_text,
            "meeting_date": _as_text(data.get("meeting_date"), "Not Specified"),
            "meeting_time": _as_text(data.get("meeting_time"), "Not Specified"),
        }
    except Exception as e:
        logger.warning(f"LLM document metadata extraction failed: {e}")
        # Return fallback metadata based on filename
        clean_title = (
            os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
        )
        return {
            "title": clean_title,
            "agenda": "",                                        # LLM unavailable: no agenda is made up
            "participants": "Document Author / Presenters",
            "meeting_date": "Not Specified",
            "meeting_time": "Not Specified",
        }


def create_document_segments(doc_text: str) -> List[Dict[str, str]]:
    """
    Splits document text into readable paragraphs or section chunks
    formatted as pseudo-transcript segments.
    """
    paragraphs = [p.strip() for p in doc_text.split("\n") if p.strip()]
    segments = []

    current_chunk = []
    chunk_word_count = 0
    seg_idx = 1

    for p in paragraphs:
        current_chunk.append(p)
        chunk_word_count += len(p.split())

        if (
            chunk_word_count >= doc_cfg.SEGMENT_WORD_CHUNK_SIZE
            or p.startswith("---")
            or p.startswith("#")
        ):
            start_min = (seg_idx - 1) * doc_cfg.SEGMENT_INTERVAL_SECONDS
            end_min = seg_idx * doc_cfg.SEGMENT_INTERVAL_SECONDS
            start_str = f"{start_min:02d}:00"
            end_str = f"{end_min:02d}:00"

            # Determine speaker/section title
            speaker = f"Document Section {seg_idx}"
            if (
                current_chunk[0].startswith("--- Page")
                or current_chunk[0].startswith("--- Slide")
                or current_chunk[0].startswith("--- Sheet")
            ):
                speaker = current_chunk[0].replace("-", "").strip()

            segments.append(
                {
                    "speaker": speaker,
                    "start_time": start_str,
                    "end_time": end_str,
                    "text": "\n".join(current_chunk),
                }
            )
            seg_idx += 1
            current_chunk = []
            chunk_word_count = 0

    if current_chunk:
        start_min = (seg_idx - 1) * doc_cfg.SEGMENT_INTERVAL_SECONDS
        end_min = seg_idx * doc_cfg.SEGMENT_INTERVAL_SECONDS
        segments.append(
            {
                "speaker": f"Document Section {seg_idx}",
                "start_time": f"{start_min:02d}:00",
                "end_time": f"{end_min:02d}:00",
                "text": "\n".join(current_chunk),
            }
        )

    return segments


def generate_document_title_with_llm(doc_text: str, filename: str) -> Dict[str, str]:
    """
    Requirement 4: Automatically generates a meaningful, professional document name
    and 2-sentence summary based on the document's content and context.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from backend.services.llm_workflow import get_llm

    truncated_text = doc_text[: doc_cfg.LLM_TITLE_MAX_CHARS].strip()
    clean_fallback_title = (
        os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
    )

    if not truncated_text:
        return {
            "display_name": clean_fallback_title,
            "summary": "Empty or non-textual document file.",
        }

    system_prompt = (
        "You are an expert enterprise document intelligence analyzer. "
        "Analyze the provided document text and output a JSON object with a highly descriptive, "
        "professional document title (max 8-10 words) and a concise 2-sentence executive summary. "
        "Do not include markdown or additional text, only valid JSON."
    )

    user_prompt = f"""
Original Filename: {filename}
Document Content Snippet:
{truncated_text}

Return JSON with exact structure:
{{
  "display_name": "Concise & Meaningful Document Title",
  "summary": "Brief 2-sentence summary of document content, key themes, and purpose."
}}
"""

    try:
        llm = get_llm()
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        response = llm.invoke(messages)
        res_text = response.content.strip()

        if res_text.startswith("```"):
            res_text = re.sub(r"^```[a-zA-Z]*\n?", "", res_text)
            res_text = re.sub(r"\n?```$", "", res_text)

        data = json.loads(res_text.strip())
        display_name = data.get("display_name") or clean_fallback_title
        summary = data.get("summary") or f"Document summary for {filename}."

        return {"display_name": display_name.strip(), "summary": summary.strip()}
    except Exception as e:
        logger.warning(f"LLM document title generation failed: {e}")
        return {
            "display_name": clean_fallback_title,
            "summary": f"Uploaded document ({filename}). Text content extracted and vector indexed.",
        }


# ─────────────────────────────────────────────────────────────────────────────
# HTML Preview Generation — Preserves Document Formatting for Frontend Render
# ─────────────────────────────────────────────────────────────────────────────

_HTML_WRAP = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: 'Calibri', 'Segoe UI', Arial, sans-serif; font-size: 12pt; line-height: 1.5;
         color: #1e293b; padding: 24px 32px; margin: 0; background: #fff; max-width: 100%; }}
  h1 {{ font-size: 18pt; font-weight: 700; color: #0f172a; margin: 18pt 0 6pt; border-bottom: 2px solid #1e293b; padding-bottom: 4pt; page-break-after: avoid; }}
  h2 {{ font-size: 16pt; font-weight: 600; color: #1e3a8a; margin: 16pt 0 4pt; page-break-after: avoid; }}
  h3 {{ font-size: 14pt; font-weight: 600; color: #2563eb; margin: 14pt 0 4pt; page-break-after: avoid; }}
  h4 {{ font-size: 12pt; font-weight: 600; color: #334155; margin: 12pt 0 3pt; page-break-after: avoid; }}
  h5, h6 {{ font-size: 11pt; font-weight: 600; color: #475569; margin: 10pt 0 3pt; page-break-after: avoid; }}
  p {{ margin: 4pt 0 8pt; text-align: justify; }}
  b, strong {{ font-weight: 700; }}
  i, em {{ font-style: italic; }}
  u {{ text-decoration: underline; }}
  s, strike {{ text-decoration: line-through; }}
  sub {{ font-size: 75%; vertical-align: sub; }}
  sup {{ font-size: 75%; vertical-align: super; }}
  ul, ol {{ margin: 4pt 0 8pt 24pt; padding: 0; }}
  li {{ margin-bottom: 2pt; }}
  li > ul, li > ol {{ margin-top: 2pt; margin-bottom: 2pt; }}
  table {{ border-collapse: collapse; width: 100%; margin: 10pt 0; font-size: 11pt; }}
  th {{ background: #f1f5f9; padding: 6pt 10pt; border: 1px solid #94a3b8; text-align: left; font-weight: 600; vertical-align: top; }}
  td {{ padding: 5pt 10pt; border: 1px solid #cbd5e1; vertical-align: top; }}
  tr:hover td {{ background: #eff6ff; }}
  img {{ max-width: 100%; height: auto; margin: 8pt 0; display: block; border: 1px solid #e2e8f0; border-radius: 4px; }}
  blockquote {{ margin: 8pt 0 8pt 16pt; padding: 6pt 12pt; border-left: 4px solid #3b82f6; background: #f8fafc; color: #334155; font-style: italic; }}
  hr {{ border: none; border-top: 1px solid #cbd5e1; margin: 12pt 0; }}
  .page-break {{ border-top: 2px dashed #94a3b8; margin: 18pt 0 12pt; padding-top: 6pt; color: #64748b; font-size: 9pt; font-weight: 600; letter-spacing: 0.05em; text-align: center; }}
  .slide-break {{ border-top: 2px solid #6366f1; margin: 14pt 0 10pt; padding-top: 6pt; color: #6366f1; font-size: 9pt; font-weight: 700; text-align: center; }}
  .sheet-break {{ border-top: 2px solid #10b981; margin: 16pt 0 10pt; padding-top: 6pt; color: #10b981; font-size: 9pt; font-weight: 700; text-align: center; }}
  code, pre {{ font-family: 'Consolas', 'Courier New', 'Cascadia Code', monospace; background: #f1f5f9; padding: 2px 5px; border-radius: 3px; font-size: 10pt; }}
  pre {{ display: block; padding: 10pt; border: 1px solid #e2e8f0; border-radius: 6px; overflow-x: auto; white-space: pre-wrap; word-break: break-word; }}
  .docx-table {{ overflow-x: auto; }}
  @media print {{ body {{ padding: 0; }} h1, h2, h3, h4 {{ page-break-after: avoid; }} }}
</style>
</head>
<body>{content}</body>
</html>"""


def _extract_docx_images_as_data_uris(doc) -> dict:
    """Extract all images from a DOCX document and return {rId: base64 data URI}."""
    import base64

    images = {}
    for rel in doc.part.rels.values():
        reltype = rel.reltype or ""
        if "image" in reltype:
            try:
                image_data = rel.target_part.blob
                content_type = rel.target_part.content_type or "image/png"
                b64 = base64.b64encode(image_data).decode("utf-8")
                images[rel.rId] = f"data:{content_type};base64,{b64}"
            except Exception:
                continue
    return images


def _parse_docx_paragraph_xml(para, images: dict) -> str:
    """
    Parse a DOCX paragraph's raw XML to produce HTML.
    Handles:
    - Text runs with font, size, bold, italic, underline, strikethrough, color, highlight
    - Inline images (drawing elements)
    - Hyperlinks
    """
    from docx.oxml.ns import qn

    elements = []
    for child in para._p.iterchildren():
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag

        if tag == "r":
            # Text run
            texts = []
            font_name = None
            font_size = None
            bold = False
            italic = False
            underline = False
            strikethrough = False
            color = None
            highlight = None

            # Extract run properties
            rpr = child.find(qn("w:rPr"))
            if rpr is not None:
                sz = rpr.find(qn("w:sz"))
                if sz is not None:
                    val = sz.get(qn("w:val"))
                    if val:
                        font_size = int(val) // 2
                    else:
                        sz_cs = rpr.find(qn("w:szCs"))
                        if sz_cs is not None:
                            val_cs = sz_cs.get(qn("w:val"))
                            if val_cs:
                                font_size = int(val_cs) // 2

                rfonts = rpr.find(qn("w:rFonts"))
                if rfonts is not None:
                    font_name = rfonts.get(qn("w:ascii")) or rfonts.get(qn("w:hAnsi"))

                bold_elem = rpr.find(qn("w:b"))
                if bold_elem is not None:
                    val = bold_elem.get(qn("w:val"))
                    bold = val is None or val.lower() in ("true", "1", "on")

                italic_elem = rpr.find(qn("w:i"))
                if italic_elem is not None:
                    val = italic_elem.get(qn("w:val"))
                    italic = val is None or val.lower() in ("true", "1", "on")

                u_elem = rpr.find(qn("w:u"))
                if u_elem is not None:
                    u_val = u_elem.get(qn("w:val"), "single")
                    underline = u_val not in ("none", None)

                strike_elem = rpr.find(qn("w:strike"))
                if strike_elem is not None:
                    val = strike_elem.get(qn("w:val"))
                    strikethrough = val is None or val.lower() in ("true", "1", "on")

                color_elem = rpr.find(qn("w:color"))
                if color_elem is not None:
                    color = color_elem.get(qn("w:val"))

                hl_elem = rpr.find(qn("w:highlight"))
                if hl_elem is not None:
                    highlight = hl_elem.get(qn("w:val"))

            # Extract text from run
            for t in child.iterchildren():
                t_tag = t.tag.split("}")[-1] if "}" in t.tag else t.tag
                if t_tag == "t":
                    texts.append(t.text or "")
                elif t_tag == "tab":
                    texts.append(" ")
                elif t_tag == "br":
                    texts.append("<br/>")

            raw_text = "".join(texts)
            if not raw_text.strip():
                continue

            escaped = _html_escape.escape(raw_text)
            style_parts = []
            if font_name:
                style_parts.append(f"font-family:'{font_name}'")
            if font_size:
                style_parts.append(f"font-size:{font_size}pt")
            if color and color != "auto":
                style_parts.append(f"color:#{color}")
            if highlight and highlight != "none":
                hl_colors = {
                    "yellow": "#fff3cd",
                    "green": "#d4edda",
                    "cyan": "#d1ecf1",
                    "magenta": "#f8d7da",
                    "red": "#f8d7da",
                    "blue": "#cfe2ff",
                    "darkBlue": "#cfe2ff",
                    "darkYellow": "#fff3cd",
                    "darkGray": "#dee2e6",
                    "lightGray": "#f8f9fa",
                    "black": "#212529",
                }
                bg = hl_colors.get(highlight, f"#{highlight}")
                style_parts.append(f"background-color:{bg}")

            style_attr = f' style="{"; ".join(style_parts)}"' if style_parts else ""

            text = escaped
            if bold:
                text = f"<strong>{text}</strong>"
            if italic:
                text = f"<em>{text}</em>"
            if underline:
                text = f"<u>{text}</u>"
            if strikethrough:
                text = f"<s>{text}</s>"

            elements.append(f"<span{style_attr}>{text}</span>")

        elif tag == "drawing" or tag == "graphic":
            # Inline image
            try:
                blip = child.find(".//" + qn("a:blip"))
                if blip is not None:
                    embed_id = blip.get(qn("r:embed"))
                    if embed_id and embed_id in images:
                        # Get image dimensions from extent
                        cx_elem = child.find(".//" + qn("wp:extent"))
                        width = None
                        if cx_elem is not None:
                            cx_val = cx_elem.get("cx")
                            if cx_val:
                                width = int(int(cx_val) / 914400 * 96)  # EMU to pixels
                        style = f' style="max-width:{width}px"' if width else ""
                        elements.append(
                            f'<img src="{images[embed_id]}" alt="document_image"{style}/>'
                        )
            except Exception:
                pass

        elif tag == "hyperlink":
            # Process hyperlink runs
            link_texts = []
            for r in child.iterchildren():
                r_tag = r.tag.split("}")[-1] if "}" in r.tag else r.tag
                if r_tag == "r":
                    for t in r.iterchildren():
                        t_tag = t.tag.split("}")[-1] if "}" in t.tag else t.tag
                        if t_tag == "t":
                            link_texts.append(t.text or "")
            if link_texts:
                link_text = "".join(link_texts).strip()
                if link_text:
                    elements.append(
                        f'<a href="{_html_escape.escape(link_text)}" target="_blank" rel="noopener">'
                        f"{_html_escape.escape(link_text)}</a>"
                    )

    return "".join(elements)


def extract_formatted_html_from_docx(filepath: str) -> str:
    """
    Converts a DOCX file to HTML preserving:
    - Heading levels (h1–h6) from paragraph styles
    - Bold, italic, underline, strikethrough, font, size, color, highlight
    - Ordered and unordered bullet lists
    - Tables with header rows
    - Inline images embedded as base64 data URIs
    - Hyperlinks
    Falls back to plain text if python-docx fails.
    """
    try:
        import docx
        from docx.oxml.ns import qn

        doc = docx.Document(filepath)

        # Extract all images as data URIs (rId → data URI)
        images = _extract_docx_images_as_data_uris(doc)

        html_parts = []

        def para_to_html(para) -> str:
            style_name = (para.style.name if para.style else "").lower()
            inner = _parse_docx_paragraph_xml(para, images)
            if not inner.strip():
                return ""

            # Map heading styles
            if "heading 1" in style_name:
                return f"<h1>{inner}</h1>"
            elif "heading 2" in style_name:
                return f"<h2>{inner}</h2>"
            elif "heading 3" in style_name:
                return f"<h3>{inner}</h3>"
            elif "heading 4" in style_name:
                return f"<h4>{inner}</h4>"
            elif "heading 5" in style_name or "heading 6" in style_name:
                return f"<h5>{inner}</h5>"
            elif "title" in style_name:
                return f"<h1><strong>{inner}</strong></h1>"
            elif "subtitle" in style_name:
                return f"<h2>{inner}</h2>"
            elif "list paragraph" in style_name or "list" in style_name:
                num_pr = para._p.find(qn("w:numPr"))
                if num_pr is not None:
                    return f"<li>{inner}</li>"
                return f"<p>{inner}</p>"
            else:
                num_pr = para._p.find(qn("w:numPr"))
                if num_pr is not None:
                    return f"<li>{inner}</li>"
                return f"<p>{inner}</p>"

        # Process paragraphs and tables in document order
        body = doc.element.body
        in_list = False
        list_type = "ul"  # default to unordered
        for child in body:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "p":
                for para in doc.paragraphs:
                    if para._p is child:
                        ph = para_to_html(para)
                        if ph:
                            is_li = ph.startswith("<li>")
                            if is_li and not in_list:
                                num_pr = para._p.find(qn("w:numPr"))
                                list_type = "ul"  # default
                                if num_pr is not None:
                                    num_id_elem = num_pr.find(qn("w:numId"))
                                    if num_id_elem is not None:
                                        num_id = num_id_elem.get(qn("w:val"))
                                        try:
                                            # Check numbering definition for ordered vs unordered
                                            num_elem = (
                                                doc.part.numbering_part.element.find(
                                                    f".//"
                                                    + qn("w:num")
                                                    + f'[@w:numId="{num_id}"]'
                                                )
                                            )
                                            if num_elem is not None:
                                                abs_num_id_elem = num_elem.find(
                                                    qn("w:abstractNumId")
                                                )
                                                if abs_num_id_elem is not None:
                                                    abs_num_id = abs_num_id_elem.get(
                                                        qn("w:val")
                                                    )
                                                    abs_num_elem = doc.part.numbering_part.element.find(
                                                        f".//"
                                                        + qn("w:abstractNum")
                                                        + f'[@w:abstractNumId="{abs_num_id}"]'
                                                    )
                                                    if abs_num_elem is not None:
                                                        # Check for ordered list pattern (decimal, 1, etc.)
                                                        lvl = abs_num_elem.find(
                                                            qn("w:lvl")
                                                            + f'[@w:ilvl="0"]'
                                                        )
                                                        if lvl is not None:
                                                            fmt_elem = lvl.find(
                                                                qn("w:numFmt")
                                                            )
                                                            if fmt_elem is not None:
                                                                fmt_val = fmt_elem.get(
                                                                    qn("w:val")
                                                                )
                                                                if fmt_val in (
                                                                    "decimal",
                                                                    "upperRoman",
                                                                    "lowerRoman",
                                                                    "upperLetter",
                                                                    "lowerLetter",
                                                                ):
                                                                    list_type = "ol"
                                        except Exception:
                                            pass
                                html_parts.append(f"<{list_type}>")
                                in_list = True
                            elif not is_li and in_list:
                                html_parts.append(f"</{list_type}>")
                                in_list = False
                            html_parts.append(ph)
                        break
            elif tag == "tbl":
                if in_list:
                    html_parts.append(f"</{list_type}>")
                    in_list = False
                for tbl in doc.tables:
                    if tbl._tbl is child:
                        tbl_html = ['<div class="docx-table"><table>']
                        for row_idx, row in enumerate(tbl.rows):
                            row_html = ["<tr>"]
                            for cell_idx, cell in enumerate(row.cells):
                                # Extract formatted content from each cell (bold, italic, etc.)
                                cell_parts = []
                                for para in cell.paragraphs:
                                    cell_inner = _parse_docx_paragraph_xml(para, images)
                                    if cell_inner.strip():
                                        cell_parts.append(f"<p>{cell_inner}</p>")
                                cell_html = (
                                    "".join(cell_parts)
                                    if cell_parts
                                    else _html_escape.escape(cell.text.strip())
                                )
                                tag_name = "th" if row_idx == 0 else "td"
                                row_html.append(f"<{tag_name}>{cell_html}</{tag_name}>")
                            row_html.append("</tr>")
                            tbl_html.append("".join(row_html))
                        tbl_html.append("</table></div>")
                        html_parts.append("".join(tbl_html))
                        break

        if in_list:
            html_parts.append(f"</{list_type}>")

        content = "\n".join(html_parts)
        return _HTML_WRAP.format(content=content) if content.strip() else ""
    except Exception as e:
        logger.warning(f"DOCX to HTML conversion failed for {filepath}: {e}")
        return ""


def _extract_pdf_images_as_data_uris(page) -> dict:
    """
    Extract images from a PDF page and return as {name: data_uri}.
    Uses pypdf's image extraction API.
    """
    import base64

    images = {}
    try:
        for image in getattr(page, "images", []) or []:
            try:
                img_data = image.data
                img_name = getattr(image, "name", "img") or "img"
                if not img_data:
                    continue
                ext = os.path.splitext(img_name)[1].lower().lstrip(".") or "png"
                mime_map = {
                    "png": "image/png",
                    "jpg": "image/jpeg",
                    "jpeg": "image/jpeg",
                    "gif": "image/gif",
                    "bmp": "image/bmp",
                    "tiff": "image/tiff",
                    "tif": "image/tiff",
                    "webp": "image/webp",
                }
                content_type = mime_map.get(ext, "image/png")
                b64 = base64.b64encode(img_data).decode("utf-8")
                images[img_name] = f"data:{content_type};base64,{b64}"
            except Exception:
                continue
    except Exception:
        pass
    return images


def extract_formatted_html_from_pdf(filepath: str) -> str:
    """
    Converts PDF pages to structured HTML with page separators.
    Preserves paragraph spacing and extracts embedded images.
    Falls back if pypdf fails.
    """
    try:
        import pypdf

        reader = pypdf.PdfReader(filepath)
        html_parts = []
        for page_idx, page in enumerate(reader.pages):
            # Extract images from this page
            page_images = _extract_pdf_images_as_data_uris(page)

            page_text = page.extract_text() or ""
            if not page_text.strip() and not page_images:
                continue

            # Page header
            html_parts.append(
                f'<div class="page-break">&mdash; Page {page_idx + 1} &mdash;</div>'
            )

            # Embed images found on this page
            for img_name, img_uri in page_images.items():
                html_parts.append(
                    f'<img src="{img_uri}" alt="Page {page_idx + 1} image" />'
                )

            if page_text.strip():
                raw_paras = re.split(r"\n{2,}", page_text.strip())
                for para in raw_paras:
                    lines = [l.strip() for l in para.splitlines() if l.strip()]
                    if not lines:
                        continue
                    merged = " ".join(lines)
                    escaped = _html_escape.escape(merged)
                    if (
                        len(merged) < doc_cfg.PDF_HEADING_MAX_LENGTH
                        and merged == merged.upper()
                        and len(merged) > doc_cfg.PDF_HEADING_MIN_LENGTH
                    ):
                        html_parts.append(f"<h2>{escaped}</h2>")
                    else:
                        html_parts.append(f"<p>{escaped}</p>")

        content = "\n".join(html_parts)
        return _HTML_WRAP.format(content=content) if content.strip() else ""
    except Exception as e:
        logger.warning(f"PDF to HTML conversion failed for {filepath}: {e}")
        return ""


def convert_text_to_html(text: str, ext: str = ".txt") -> str:
    """
    Converts plain text / markdown-like content to basic HTML.
    Handles # headings, **bold**, *italic*, - bullet lists, tables (|), code blocks.
    """
    if not text or not text.strip():
        return ""

    html_parts = []
    lines = text.splitlines()
    in_list = False
    list_type = "ul"
    in_code = False

    for line in lines:
        stripped = line.strip()

        # Code blocks (triple backtick)
        if stripped.startswith("```"):
            if in_code:
                html_parts.append("</pre>")
                in_code = False
            else:
                if in_list:
                    html_parts.append(f"</{list_type}>")
                    in_list = False
                html_parts.append("<pre>")
                in_code = True
            continue
        if in_code:
            html_parts.append(_html_escape.escape(line))
            continue

        if not stripped:
            if in_list:
                html_parts.append(f"</{list_type}>")
                in_list = False
            html_parts.append("")
            continue

        # Markdown headings
        m = re.match(r"^(#{1,6})\s+(.*)", stripped)
        if m:
            level = min(len(m.group(1)), 6)
            txt = _html_escape.escape(m.group(2))
            if in_list:
                html_parts.append(f"</{list_type}>")
                in_list = False
            html_parts.append(f"<h{level}>{txt}</h{level}>")
            continue

        # Horizontal rules
        if re.match(r"^[-*_]{3,}$", stripped):
            html_parts.append("<hr/>")
            continue

        # Table rows (detect | separators) — only for actual markdown tables
        # Check that this looks like a real table row, not just text with pipes
        if "|" in stripped and stripped.count("|") >= 2:
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            # Skip if cells contain HTML tags, URLs, or very long content (likely not a table)
            is_real_table = (
                cells
                and all(len(c) < 200 for c in cells)
                and not any(
                    re.search(r"<[a-z]+|https?://|javascript:", c) for c in cells
                )
            )
            if is_real_table:
                if in_list:
                    html_parts.append(f"</{list_type}>")
                    in_list = False
                row_html = "".join(f"<td>{_html_escape.escape(c)}</td>" for c in cells)
                html_parts.append(f"<table><tr>{row_html}</tr></table>")
            else:
                # Not a real table — render as paragraph
                if in_list:
                    html_parts.append(f"</{list_type}>")
                    in_list = False
                para_text = _apply_inline_formatting(stripped)
                html_parts.append(f"<p>{para_text}</p>")
            continue

        # Bullet list items
        if re.match(r"^[-*+]\s+", stripped):
            inner = stripped[2:].strip()
            inner = _apply_inline_formatting(inner)
            if not in_list:
                html_parts.append("<ul>")
                list_type = "ul"
                in_list = True
            html_parts.append(f"<li>{inner}</li>")
            continue

        # Numbered list items
        if re.match(r"^\d+\.\s+", stripped):
            inner = re.sub(r"^\d+\.\s+", "", stripped)
            inner = _apply_inline_formatting(inner)
            if not in_list:
                html_parts.append("<ol>")
                list_type = "ol"
                in_list = True
            html_parts.append(f"<li>{inner}</li>")
            continue

        # Page markers
        if re.match(r"^---\s*Page\s*\d+", stripped, re.IGNORECASE):
            if in_list:
                html_parts.append(f"</{list_type}>")
                in_list = False
            label = _html_escape.escape(stripped.replace("---", "").strip())
            html_parts.append(f'<div class="page-break">&mdash; {label} &mdash;</div>')
            continue

        # Regular paragraph
        if in_list:
            html_parts.append(f"</{list_type}>")
            in_list = False
        para_text = _apply_inline_formatting(stripped)
        html_parts.append(f"<p>{para_text}</p>")
    if in_code:
        html_parts.append("</pre>")

    content = "\n".join(html_parts)
    return _HTML_WRAP.format(content=content) if content.strip() else ""


def _apply_inline_formatting(text: str) -> str:
    """Applies inline markdown-like formatting: bold, italic, code, escape."""
    # Escape HTML special chars first
    text = _html_escape.escape(text)
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    # Italic: *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"_(.+?)_", r"<em>\1</em>", text)
    # Inline code: `text`
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text


def extract_document_html(filepath: str) -> str:
    """
    Main entry point for generating rich formatted HTML from any document.
    Returns an HTML string ready to embed in a sandboxed iframe, or empty string on failure.

    Supports: DOCX (full formatting), PDF (page structure), PPTX (slide structure),
    XLSX/CSV (table layout), TXT/MD (markdown parsing).
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext in [".docx", ".doc"]:
        html = extract_formatted_html_from_docx(filepath)
        if html:
            return html

    if ext == ".pdf":
        html = extract_formatted_html_from_pdf(filepath)
        if html:
            return html

    if ext in [".pptx", ".ppt"]:
        # PPTX: extract slide-by-slide text and render as sections
        try:
            from pptx import Presentation

            prs = Presentation(filepath)
            html_parts = []
            for idx, slide in enumerate(prs.slides):
                html_parts.append(f'<div class="slide-break">— Slide {idx + 1} —</div>')
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        text = shape.text.strip()
                        escaped = _html_escape.escape(text)
                        if len(text) < 60:
                            html_parts.append(f"<h3>{escaped}</h3>")
                        else:
                            for line in text.splitlines():
                                l = line.strip()
                                if l:
                                    html_parts.append(
                                        f"<p>{_html_escape.escape(l)}</p>"
                                    )
            content = "\n".join(html_parts)
            if content.strip():
                return _HTML_WRAP.format(content=content)
        except Exception as e:
            logger.warning(f"PPTX to HTML failed for {filepath}: {e}")

    if ext in [".xlsx", ".xls", ".csv"]:
        # Spreadsheet: render as HTML tables
        try:
            if ext == ".csv":
                import csv

                html_parts = ["<table>"]
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    reader = csv.reader(f)
                    for row_idx, row in enumerate(reader):
                        tag = "th" if row_idx == 0 else "td"
                        cells = "".join(
                            f"<{tag}>{_html_escape.escape(str(c))}</{tag}>" for c in row
                        )
                        html_parts.append(f"<tr>{cells}</tr>")
                html_parts.append("</table>")
                content = "\n".join(html_parts)
                if content.strip():
                    return _HTML_WRAP.format(content=content)
            else:
                import openpyxl

                wb = openpyxl.load_workbook(filepath, data_only=True)
                html_parts = []
                for sheet_name in wb.sheetnames:
                    ws = wb[sheet_name]
                    html_parts.append(
                        f'<div class="sheet-break">— Sheet: {_html_escape.escape(sheet_name)} —</div>'
                    )
                    html_parts.append("<table>")
                    for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
                        vals = [str(v) if v is not None else "" for v in row]
                        if not any(v.strip() for v in vals):
                            continue
                        tag = "th" if row_idx == 0 else "td"
                        cells = "".join(
                            f"<{tag}>{_html_escape.escape(v)}</{tag}>" for v in vals
                        )
                        html_parts.append(f"<tr>{cells}</tr>")
                    html_parts.append("</table>")
                content = "\n".join(html_parts)
                if content.strip():
                    return _HTML_WRAP.format(content=content)
        except Exception as e:
            logger.warning(f"Spreadsheet to HTML failed for {filepath}: {e}")

    # Fallback: plain text with markdown-like formatting
    try:
        plain = extract_text_from_plain_text(filepath)
        if plain.strip():
            return convert_text_to_html(plain, ext)
    except Exception as e:
        logger.warning(f"Plaintext fallback to HTML failed for {filepath}: {e}")

    return ""
