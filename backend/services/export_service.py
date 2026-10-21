"""Enterprise Report Export Service.

Generates professional business reports in multiple formats (DOCX, Excel, HTML, PDF, MD, JSON).
Redesigns standard text dumps into visually striking executive reports.
"""

import html as _html
import io
import logging
import re
from datetime import datetime
from typing import Any, Dict, List

# openpyxl for Excel reports
import openpyxl

# python-docx for MS Word reports
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from docx.shared import Inches, Pt, RGBColor
from docx.text.paragraph import Paragraph as _DocxParagraph
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE as _XLSX_ILLEGAL_RE
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from backend.utils.helpers import clean_agenda_text
from backend.utils.security import REPORT_PRINT_SCRIPT

logger = logging.getLogger(__name__)

# ── Untrusted-text safety ──────────────────────────────────────────────────────
# Everything in a report (meeting title, transcript, LLM summary, action items, admin-set
# branding) can originate from an uploaded file, a scraped web page or speech — i.e. from
# an attacker. The HTML report used to interpolate all of it raw (stored XSS / markup
# injection), and DOCX/XLSX crashed with a 500 on control characters that ASR output can
# contain. These helpers are the single place that handles it.
_XML_ILLEGAL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def _xml_safe(text: str) -> str:
    """Strip characters that are not legal in XML 1.0 (python-docx raises ValueError on them)."""
    return _XML_ILLEGAL_RE.sub("", text)


_orig_docx_add_run = _DocxParagraph.add_run


def _safe_docx_add_run(self, text=None, style=None):
    return _orig_docx_add_run(self, _xml_safe(text) if isinstance(text, str) else text, style)


_DocxParagraph.add_run = _safe_docx_add_run  # also covers Paragraph.text and Document.add_paragraph(text)


def _esc(value: Any) -> str:
    """HTML-escape any value for use in element text or a quoted attribute."""
    return _html.escape("" if value is None else str(value), quote=True)


_HEX_COLOR_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_FONT_FAMILY_RE = re.compile(r"^[A-Za-z0-9 ,'_.\-]{1,120}$")


def _hex6(value: Any, default: str) -> str:
    """Normalise a user-supplied colour to six hex digits (no '#'), else ``default``."""
    m = _HEX_COLOR_RE.match(str(value or "").strip())
    if not m:
        return default.lstrip("#")
    digits = m.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return digits[:6].lower()


def _safe_font(value: Any, default: str) -> str:
    value = str(value or "").strip()
    return value if _FONT_FAMILY_RE.match(value) else default


def _css_string(value: Any) -> str:
    """Escape text for embedding inside a CSS "…" string (blocks '</style>' and quote breakout)."""
    out = []
    for ch in "" if value is None else str(value):
        if ch in '"\\':
            out.append("\\" + ch)
        elif ch in "<>&" or ord(ch) < 32:
            out.append("\\%x " % ord(ch))
        else:
            out.append(ch)
    return "".join(out)


_LOGO_DATA_RE = re.compile(r"^data:image/(png|jpe?g|gif|webp);base64,[A-Za-z0-9+/=]+$", re.I)


def _safe_logo_url(value: Any) -> str:
    """Only http(s) or a base64 raster data-URI may become an <img src>; anything else is dropped."""
    v = str(value or "").strip()
    if re.match(r"^https?://", v, re.I) or _LOGO_DATA_RE.match(v):
        return v
    return ""


def _speaker_label(seg: Any) -> str:
    """Display label for a segment's speaker; ``TranscriptSegment.speaker`` can be NULL."""
    return (getattr(seg, "speaker", None) or "Unknown").replace("SPEAKER_", "Speaker_")


class _HtmlSafeView:
    """Read-only view of an ORM object whose string attributes come back HTML-escaped."""

    def __init__(self, obj: Any):
        self._obj = obj

    def __getattr__(self, name: str) -> Any:
        value = getattr(self._obj, name)
        return _esc(value) if isinstance(value, str) else value

# Fallback shown on exported documents only when no real company_name is
# configured. Previously three different fake identities ("Acme Enterprise",
# "Acme Corporation", "Acme Corp") appeared depending on export format —
# inconsistent, and each one is a fabricated organization name on an
# official-looking exported document. One honest, clearly-generic label.
_DEFAULT_COMPANY_NAME = "Organization Name Not Configured"


def clean_markdown_text(text: str) -> str:
    """Removes raw markdown indicators (###, **, *, -, __) for clean print output."""
    if not text:
        return ""
    # Strip markdown headings
    t = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    # Strip strong / bold tags
    t = re.sub(r"\*\*([^*]+)\*\*|__([^_]+)__", r"\1\2", t)
    # Strip italic tags
    t = re.sub(r"\*([^*]+)\*|_([^_]+)_", r"\1\2", t)
    # Clean up standard bullets, replace with clean indentation
    t = re.sub(r"^\s*[-*+]\s+", "• ", t, flags=re.MULTILINE)
    # Clean up raw markdown table indicators
    t = re.sub(r"\|", " ", t)
    t = re.sub(r"^[-:|\s]+$", "", t, flags=re.MULTILINE)
    return t.strip()


def parse_bullet_points(text: str) -> List[str]:
    """Splits a block of text into individual clean bullet points or lines."""
    if not text:
        return []
    lines = text.split("\n")
    points = []
    for line in lines:
        cleaned = line.strip()
        if not cleaned:
            continue
        # Remove leading bullet characters
        cleaned = re.sub(r"^[-*+•\s]+", "", cleaned)
        cleaned = re.sub(r"^\d+\.\s+", "", cleaned)  # numbered lists
        # Remove bold symbols
        cleaned = cleaned.replace("**", "").replace("__", "")
        if cleaned:
            points.append(cleaned)
    return points


def _set_cell_background(cell, fill_hex: str):
    """Utility to color code table cells in python-docx."""
    tcPr = cell._element.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
    tcPr.append(shd)


def _set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Utility to set cell padding margins in python-docx (values in dxas)."""
    tcPr = cell._element.get_or_add_tcPr()
    tcMar = OxmlElement("w:tcMar")
    for m, val in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        node = OxmlElement(f"w:{m}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tcMar.append(node)
    tcPr.append(tcMar)


def _set_table_borders(table, color="D3D3D3"):
    """Set light grey borders for tables to look professional."""
    tblPr = table._element.xpath("w:tblPr")
    if tblPr:
        borders = parse_xml(
            f'<w:tblBorders {nsdecls("w")}>'
            f'  <w:top w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'  <w:left w:val="none"/>'
            f'  <w:bottom w:val="single" w:sz="8" w:space="0" w:color="{color}"/>'
            f'  <w:right w:val="none"/>'
            f'  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="{color}"/>'
            f'  <w:insideV w:val="none"/>'
            f"</w:tblBorders>"
        )
        tblPr[0].append(borders)


_NO_DECISION_MARKERS = frozenset({
    "", "no formal decision recorded.", "no formal decision recorded", "no decision recorded.", "no decision recorded",
    "no decision.", "no decision", "none", "n/a", "null",
})


def _real_decision(text: Any) -> str:
    """The decision text, or "" when it is only the stored 'no decision' marker."""
    text = str(text or "").strip()
    return "" if text.lower() in _NO_DECISION_MARKERS else text


def _meeting_agenda_text(meeting: Any) -> str:
    """The agenda for a report header: what the uploader supplied, else the topics extracted from the transcript, else an
    explicit 'Not specified' — never a made-up sentence."""
    supplied = clean_agenda_text(getattr(meeting, "agenda", "")) or ""
    if supplied:
        return supplied
    topics = [
        str(getattr(a, "agenda", "") or "").strip() for a in (getattr(meeting, "agenda_key_decisions", None) or [])
    ]
    return "; ".join(t for t in topics if t) or "Not specified"


def _get_agenda_key_decisions(meeting: Any, analysis: Any) -> List[Dict[str, str]]:
    """Helper to retrieve structured Agenda & Key Decisions for exports (a missing agenda or decision stays empty)."""
    if getattr(meeting, "agenda_key_decisions", None):
        return [
            {"agenda": akd.agenda, "key_decision": _real_decision(akd.key_decision), "order": akd.agenda_order}
            for akd in meeting.agenda_key_decisions
        ]
    dec_text = analysis.decisions if analysis else ""
    if dec_text:
        try:
            import json
            parsed = json.loads(dec_text)
            if isinstance(parsed, list):
                return [
                    {
                        "agenda": str(item.get("agenda") or "").strip(),
                        "key_decision": _real_decision(item.get("key_decision")),
                        "order": i+1
                    }
                    for i, item in enumerate(parsed) if isinstance(item, dict)
                ]
        except Exception:
            pass
        bullets = parse_bullet_points(dec_text)
        if bullets:
            return [
                {
                    "agenda": b.split(":")[0].strip() if ":" in b else "",
                    "key_decision": b.split(":", 1)[1].strip() if ":" in b else b,
                    "order": i+1
                }
                for i, b in enumerate(bullets)
            ]
    return []


def generate_docx_report(
    meeting: Any, analysis: Any, settings: Dict[str, Any], segments: List[Any]
) -> io.BytesIO:
    """Generates a professional corporate Word (.docx) report."""
    doc = Document()

    # ── Margins & Page Layout ──
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Styles Setup
    p_color = _hex6(settings.get("theme_color_primary"), "1e3a8a")
    s_color = _hex6(settings.get("theme_color_secondary"), "2563eb")
    p_rgb = RGBColor(int(p_color[:2], 16), int(p_color[2:4], 16), int(p_color[4:], 16))
    s_rgb = RGBColor(int(s_color[:2], 16), int(s_color[2:4], 16), int(s_color[4:], 16))

    font_name = _safe_font(settings.get("font_family"), "Arial")

    # Configure Normal Style
    style_normal = doc.styles["Normal"]
    style_normal.font.name = font_name
    style_normal.font.size = Pt(10.5)
    style_normal.font.color.rgb = RGBColor(51, 65, 85)  # Slate 700

    # Header & Footer setup (Confidential statement)
    header = doc.sections[0].header
    hp = header.paragraphs[0]
    hp.text = f"{settings.get('header_text', 'Confidential - Enterprise Report')} | Meeting ID: {meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)}"
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    hp.style.font.size = Pt(8.5)
    hp.style.font.color.rgb = RGBColor(148, 163, 184)  # Slate 400

    footer = doc.sections[0].footer
    fp = footer.paragraphs[0]
    fp.text = (
        f"{settings.get('footer_text', 'Meeting Intelligence Platform')}  |  Page 1"
    )
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.style.font.size = Pt(8.5)
    fp.style.font.color.rgb = RGBColor(148, 163, 184)

    # ── COVER PAGE (Requirement 4) ──
    if settings.get("show_cover_page", True):
        # Spacing
        for _ in range(3):
            doc.add_paragraph()

        # Company Name
        p_comp = doc.add_paragraph()
        p_comp.alignment = WD_ALIGN_PARAGRAPH.LEFT
        run_comp = p_comp.add_run(
            settings.get("company_name", _DEFAULT_COMPANY_NAME).upper()
        )
        run_comp.font.size = Pt(12)
        run_comp.font.bold = True
        run_comp.font.color.rgb = s_rgb

        # Horizontal Divider Line
        p_line = doc.add_paragraph()
        run_line = p_line.add_run("━" * 40)
        run_line.font.color.rgb = s_rgb

        # Title
        p_title = doc.add_paragraph()
        run_title = p_title.add_run(meeting.title or meeting.filename)
        run_title.font.size = Pt(26)
        run_title.font.bold = True
        run_title.font.color.rgb = p_rgb

        # Subtitle
        p_sub = doc.add_paragraph()
        run_sub = p_sub.add_run(f"Project: {meeting.project_name or 'AI Operations'}")
        run_sub.font.size = Pt(13)
        run_sub.font.italic = True

        for _ in range(8):
            doc.add_paragraph()

        # Metadata block
        meta_table = doc.add_table(rows=4, cols=2)
        meta_table.alignment = WD_TABLE_ALIGNMENT.LEFT
        _set_table_borders(meta_table, "FFFFFF")  # invisible borders

        labels = [
            ("Meeting ID:", meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)),
            (
                "Date & Time:",
                f"{meeting.meeting_date or 'N/A'} at {meeting.meeting_time or 'N/A'}",
            ),
            ("Report Date:", datetime.now().strftime("%B %d, %Y")),
            ("Classification:", settings.get("watermark", "CONFIDENTIAL")),
        ]

        for idx, (label, val) in enumerate(labels):
            row = meta_table.rows[idx]
            row.cells[0].paragraphs[0].add_run(label).bold = True
            row.cells[1].paragraphs[0].add_run(val)
            _set_cell_margins(row.cells[0], top=40, bottom=40, left=0, right=0)
            _set_cell_margins(row.cells[1], top=40, bottom=40, left=100, right=0)

        doc.add_page_break()

    # ── TABLE OF CONTENTS (Requirement 5) ──
    if settings.get("show_table_of_contents", True):
        p_toc_head = doc.add_paragraph()
        run_toc_head = p_toc_head.add_run("Table of Contents")
        run_toc_head.font.size = Pt(18)
        run_toc_head.font.bold = True
        run_toc_head.font.color.rgb = p_rgb

        toc_items = [
            ("1. Meeting Overview & Summary", "Page 2"),
            ("2. Minutes of Meeting (MoM)", "Page 3"),
            ("3. Key Action Items Registry", "Page 4"),
            ("4. Decisions Tracker", "Page 4"),
            ("5. Risk Mitigation & Issues Registry", "Page 5"),
            ("6. Transcript Segments", "Page 6"),
        ]

        for item, pg in toc_items:
            p_item = doc.add_paragraph()
            p_item.paragraph_format.space_after = Pt(6)
            # Add dots visual tab lead
            run_item = p_item.add_run(f"{item:<60}")
            run_item.font.size = Pt(10.5)
            run_pg = p_item.add_run(pg)
            run_pg.font.bold = True

        doc.add_page_break()

    # ── 1. MEETING OVERVIEW (Requirement 6) ──
    p_section_1 = doc.add_paragraph()
    run_s1 = p_section_1.add_run("1. Meeting Overview")
    run_s1.font.size = Pt(16)
    run_s1.font.bold = True
    run_s1.font.color.rgb = p_rgb
    p_section_1.paragraph_format.space_after = Pt(12)

    # Metadata Overview Cards as styled Table
    overview_table = doc.add_table(rows=5, cols=2)
    overview_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    _set_table_borders(overview_table, "E2E8F0")

    overviews = [
        ("Meeting Title", meeting.title or meeting.filename),
        ("Agenda", _meeting_agenda_text(meeting)),
        (
            "Duration & Speaker Count",
            f"{round((meeting.duration or 0)/60, 1)} minutes ({meeting.participant_count or 0} speakers)",
        ),
        ("AI Quality Score", f"{(meeting.accuracy or 0.0):.1f}% Overall Confidence Rating"),
        (
            "Processing Status",
            (
                "COMPLETED & VERIFIED"
                if meeting.is_approved
                else "COMPLETED (PENDING REVIEW)"
            ),
        ),
    ]

    for idx, (lbl, val) in enumerate(overviews):
        row = overview_table.rows[idx]
        _set_cell_background(row.cells[0], "F8FAFC")
        _set_cell_margins(row.cells[0], top=100, bottom=100, left=150, right=150)
        _set_cell_margins(row.cells[1], top=100, bottom=100, left=150, right=150)

        p_lbl = row.cells[0].paragraphs[0]
        p_lbl.add_run(lbl).bold = True

        p_val = row.cells[1].paragraphs[0]
        p_val.add_run(val)

    # Add spacing before Summary
    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # Executive Summary text (Requirement 7)
    p_sum_title = doc.add_paragraph()
    run_st = p_sum_title.add_run("Executive Summary")
    run_st.font.size = Pt(13)
    run_st.font.bold = True
    run_st.font.color.rgb = s_rgb

    summary_text = analysis.summary if analysis else "No summary available."
    cleaned_summary = clean_markdown_text(summary_text)
    p_sum = doc.add_paragraph()
    p_sum.add_run(cleaned_summary)
    p_sum.paragraph_format.line_spacing = 1.15
    p_sum.paragraph_format.space_after = Pt(12)

    doc.add_page_break()

    # ── 2. MINUTES OF MEETING (Requirement 7) ──
    p_section_2 = doc.add_paragraph()
    run_s2 = p_section_2.add_run("2. Minutes of Meeting (MoM)")
    run_s2.font.size = Pt(16)
    run_s2.font.bold = True
    run_s2.font.color.rgb = p_rgb

    mom_text = analysis.mom if analysis else "No minutes available."
    bullets = parse_bullet_points(mom_text)
    if bullets:
        for pt in bullets:
            p_bullet = doc.add_paragraph(style="List Bullet")
            p_bullet.paragraph_format.space_after = Pt(4)
            p_bullet.add_run(pt)
    else:
        p_mom = doc.add_paragraph()
        p_mom.add_run(clean_markdown_text(mom_text))

    doc.add_page_break()

    # ── 3. ACTION ITEMS TABLE (Requirement 8) ──
    p_section_3 = doc.add_paragraph()
    run_s3 = p_section_3.add_run("3. Key Action Items Registry")
    run_s3.font.size = Pt(16)
    run_s3.font.bold = True
    run_s3.font.color.rgb = p_rgb
    p_section_3.paragraph_format.space_after = Pt(12)

    action_text = analysis.action_items if analysis else ""
    actions = parse_bullet_points(action_text)

    if actions:
        # Create professional table
        act_table = doc.add_table(rows=1, cols=3)
        act_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        _set_table_borders(act_table, "CBD5E1")

        # Headers
        hdr_cells = act_table.rows[0].cells
        hdr_cells[0].paragraphs[0].add_run("Action Item / Task Details").bold = True
        hdr_cells[1].paragraphs[0].add_run("Owner").bold = True
        hdr_cells[2].paragraphs[0].add_run("Status").bold = True

        for c in hdr_cells:
            _set_cell_background(c, p_color)
            c.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
            _set_cell_margins(c, top=120, bottom=120, left=150, right=150)

        for act in actions:
            row_cells = act_table.add_row().cells

            # Simple heuristic regex to extract Owner: e.g. "Action Item [Owner]" or "Action Item - Owner"
            owner = "Unassigned"
            owner_match = re.search(
                r"\[([^\]]+)\]|\(([^\)]+)\)|-\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", act
            )
            if owner_match:
                owner = (
                    owner_match.group(1) or owner_match.group(2) or owner_match.group(3)
                )
                act_clean = (
                    act.replace(f"[{owner}]", "")
                    .replace(f"({owner})", "")
                    .replace(f"- {owner}", "")
                    .strip()
                )
            else:
                act_clean = act

            row_cells[0].paragraphs[0].add_run(act_clean)
            row_cells[1].paragraphs[0].add_run(owner)
            row_cells[2].paragraphs[0].add_run("Pending")

            for c in row_cells:
                _set_cell_margins(c, top=100, bottom=100, left=150, right=150)
    else:
        p_act = doc.add_paragraph()
        p_act.add_run("No action items identified.")

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # ── 4. DECISIONS TABLE (Requirement 8) ──


    p_section_4 = doc.add_paragraph()
    run_s4 = p_section_4.add_run("4. Agenda & Key Decisions")
    run_s4.font.size = Pt(16)
    run_s4.font.bold = True
    run_s4.font.color.rgb = p_rgb
    p_section_4.paragraph_format.space_after = Pt(12)

    akd_items = _get_agenda_key_decisions(meeting, analysis)

    if akd_items:
        dec_table = doc.add_table(rows=1, cols=3)
        dec_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        _set_table_borders(dec_table, "CBD5E1")

        hdr_cells = dec_table.rows[0].cells
        hdr_cells[0].paragraphs[0].add_run("#").bold = True
        hdr_cells[1].paragraphs[0].add_run("Discussion Agenda").bold = True
        hdr_cells[2].paragraphs[0].add_run("Key Decision").bold = True

        for c in hdr_cells:
            _set_cell_background(c, s_color)
            c.paragraphs[0].runs[0].font.color.rgb = RGBColor(255, 255, 255)
            _set_cell_margins(c, top=120, bottom=120, left=150, right=150)

        for item in akd_items:
            row_cells = dec_table.add_row().cells
            row_cells[0].paragraphs[0].add_run(str(item["order"]))
            row_cells[1].paragraphs[0].add_run(item["agenda"]).bold = True
            row_cells[2].paragraphs[0].add_run(item["key_decision"])

            for c in row_cells:
                _set_cell_margins(c, top=100, bottom=100, left=150, right=150)
    else:
        p_dec = doc.add_paragraph()
        p_dec.add_run("No agenda & key decisions recorded.")

    doc.add_page_break()

    # ── 5. RISKS TABLE (Requirement 8) ──
    p_section_5 = doc.add_paragraph()
    run_s5 = p_section_5.add_run("5. Risk Mitigation & Issues Registry")
    run_s5.font.size = Pt(16)
    run_s5.font.bold = True
    run_s5.font.color.rgb = p_rgb
    p_section_5.paragraph_format.space_after = Pt(12)

    risks_text = analysis.risks if analysis else ""
    risks = parse_bullet_points(risks_text)

    if risks:
        risk_table = doc.add_table(rows=1, cols=3)
        risk_table.alignment = WD_TABLE_ALIGNMENT.CENTER
        _set_table_borders(risk_table, "CBD5E1")

        hdr_cells = risk_table.rows[0].cells
        hdr_cells[0].paragraphs[0].add_run("Identified Risk / Blocker").bold = True
        hdr_cells[1].paragraphs[0].add_run("Severity").bold = True
        hdr_cells[2].paragraphs[0].add_run("Mitigation Plan").bold = True

        for c in hdr_cells:
            _set_cell_background(c, "E2E8F0")  # light grey
            c.paragraphs[0].runs[0].font.color.rgb = RGBColor(51, 65, 85)
            _set_cell_margins(c, top=120, bottom=120, left=150, right=150)

        for rsk in risks:
            row_cells = risk_table.add_row().cells
            row_cells[0].paragraphs[0].add_run(rsk)
            row_cells[1].paragraphs[0].add_run("Medium")
            row_cells[2].paragraphs[0].add_run(
                "Monitor and assign owner in next review session"
            )

            for c in row_cells:
                _set_cell_margins(c, top=100, bottom=100, left=150, right=150)
    else:
        p_rsk = doc.add_paragraph()
        p_rsk.add_run("No high-level risks identified during discussion.")

    doc.add_page_break()

    # ── 6. TRANSCRIPT SECTION ──
    p_section_6 = doc.add_paragraph()
    run_s6 = p_section_6.add_run("6. Speaker Transcript Segments")
    run_s6.font.size = Pt(16)
    run_s6.font.bold = True
    run_s6.font.color.rgb = p_rgb
    p_section_6.paragraph_format.space_after = Pt(12)

    if not segments:
        p_trans = doc.add_paragraph()
        p_trans.add_run("No transcript segments found for this meeting.")
    else:
        for seg in segments:
            p_seg = doc.add_paragraph()
            p_seg.paragraph_format.space_after = Pt(6)

            # Format time representation
            try:
                start = float(seg.start_time)
                mins = int(start // 60)
                secs = int(start % 60)
                time_rep = f"[{mins}:{secs:02d}]"
            except Exception:
                time_rep = f"[{seg.start_time}]"

            # Clean speaker name formatting
            spk = _speaker_label(seg)

            run_time = p_seg.add_run(f"{time_rep} ")
            run_time.font.color.rgb = RGBColor(100, 116, 139)  # slate-500
            run_time.font.size = Pt(9.5)

            run_spk = p_seg.add_run(f"{spk}: ")
            run_spk.bold = True
            run_spk.font.size = Pt(10)

            p_seg.add_run(seg.text)

    # Save document in memory buffer
    stream = io.BytesIO()
    doc.save(stream)
    stream.seek(0)
    return stream


def _xl_clean(value: Any) -> Any:
    """openpyxl raises IllegalCharacterError on control characters in a string cell."""
    return _XLSX_ILLEGAL_RE.sub("", value) if isinstance(value, str) else value


def _xl_append(ws: Any, row: List[Any]) -> None:
    ws.append([_xl_clean(v) for v in row])


def _neutralise_formulas(wb: Any) -> None:
    """A string starting with '=' is stored as a FORMULA by openpyxl, so a transcript line such as
    ``=HYPERLINK(...)`` / ``=cmd|...`` would execute when the workbook is opened (CSV/Excel
    injection). Force such cells to plain text."""
    for ws in wb.worksheets:
        for row in ws.iter_rows():
            for cell in row:
                if cell.data_type == "f":
                    cell.data_type = "s"


def generate_excel_report(
    meeting: Any, analysis: Any, segments: List[Any]
) -> io.BytesIO:
    """Generates a professional multi-sheet Excel spreadsheet using openpyxl."""
    wb = openpyxl.Workbook()

    # ── Setup styles ──
    font_family = "Segoe UI"
    f_title = Font(name=font_family, size=16, bold=True, color="1E3A8A")
    f_header = Font(name=font_family, size=11, bold=True, color="FFFFFF")
    f_bold = Font(name=font_family, size=10, bold=True)
    f_normal = Font(name=font_family, size=10)
    f_italic = Font(name=font_family, size=10, italic=True)

    fill_header = PatternFill(
        start_color="1E3A8A", end_color="1E3A8A", fill_type="solid"
    )
    fill_zebra = PatternFill(
        start_color="F8FAFC", end_color="F8FAFC", fill_type="solid"
    )

    border_thin = Border(
        left=Side(style="thin", color="E2E8F0"),
        right=Side(style="thin", color="E2E8F0"),
        top=Side(style="thin", color="E2E8F0"),
        bottom=Side(style="thin", color="E2E8F0"),
    )

    # ── SHEET 1: OVERVIEW ──
    ws_over = wb.active
    ws_over.title = "Overview"
    ws_over.views.sheetView[0].showGridLines = True

    _xl_append(ws_over, [])
    _xl_append(ws_over, ["AI Meeting Intelligence - Executive Overview"])
    ws_over.cell(2, 1).font = f_title

    overviews = [
        ("Meeting ID", meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)),
        ("Meeting Title", meeting.title or meeting.filename),
        ("Agenda", _meeting_agenda_text(meeting)),
        ("Date", meeting.meeting_date or "N/A"),
        ("Time", meeting.meeting_time or "N/A"),
        ("Duration (seconds)", meeting.duration or 0),
        ("Participant Count", meeting.participant_count or 0),
        ("AI Quality Score", f"{(meeting.accuracy or 0.0):.1f}%"),
        ("Classification", "Confidential"),
        ("Report Export Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    ]

    _xl_append(ws_over, [])
    _xl_append(ws_over, ["Property", "Value"])
    for col in (1, 2):
        cell = ws_over.cell(row=4, column=col)
        cell.font = f_header
        cell.fill = fill_header
        cell.alignment = Alignment(horizontal="left", vertical="center")

    for idx, (prop, val) in enumerate(overviews):
        _xl_append(ws_over, [prop, val])
        r = 5 + idx
        ws_over.cell(r, 1).font = f_bold
        ws_over.cell(r, 1).border = border_thin
        ws_over.cell(r, 2).font = f_normal
        ws_over.cell(r, 2).border = border_thin
        if idx % 2 == 1:
            ws_over.cell(r, 1).fill = fill_zebra
            ws_over.cell(r, 2).fill = fill_zebra

    # ── SHEET 2: ACTION ITEMS ──
    ws_act = wb.create_sheet(title="Action Items")
    ws_act.views.sheetView[0].showGridLines = True
    _xl_append(ws_act, [])
    _xl_append(ws_act, ["Action Item Registry"])
    ws_act.cell(2, 1).font = f_title
    _xl_append(ws_act, [])

    headers = ["ID", "Action Item / Task Details", "Status"]
    ws_act.append(headers)
    for col_idx, h in enumerate(headers):
        cell = ws_act.cell(row=4, column=col_idx + 1)
        cell.font = f_header
        cell.fill = fill_header
        cell.border = border_thin
        cell.alignment = Alignment(horizontal="left", vertical="center")

    action_text = analysis.action_items if analysis else ""
    actions = parse_bullet_points(action_text)
    if actions:
        for idx, act in enumerate(actions):
            _xl_append(ws_act, [idx + 1, act, "Pending"])
            r = 5 + idx
            ws_act.cell(r, 1).font = f_bold
            ws_act.cell(r, 2).font = f_normal
            ws_act.cell(r, 3).font = f_bold

            for col in range(1, 4):
                c = ws_act.cell(r, col)
                c.border = border_thin
                if idx % 2 == 1:
                    c.fill = fill_zebra
    else:
        _xl_append(ws_act, ["-", "No action items identified.", "-"])
        ws_act.cell(5, 2).font = f_italic

    # ── SHEET 3: AGENDA & KEY DECISIONS ──
    ws_dec = wb.create_sheet(title="Agenda & Decisions")
    ws_dec.views.sheetView[0].showGridLines = True
    _xl_append(ws_dec, [])
    _xl_append(ws_dec, ["Agenda & Key Decisions"])
    ws_dec.cell(2, 1).font = f_title
    _xl_append(ws_dec, [])

    headers = ["#", "Discussion Agenda", "Key Decision"]
    ws_dec.append(headers)
    for col_idx, h in enumerate(headers):
        cell = ws_dec.cell(row=4, column=col_idx + 1)
        cell.font = f_header
        cell.fill = fill_header
        cell.border = border_thin

    akd_items = _get_agenda_key_decisions(meeting, analysis)
    if akd_items:
        for idx, item in enumerate(akd_items):
            _xl_append(ws_dec, [item["order"], item["agenda"], item["key_decision"]])
            r = 5 + idx
            ws_dec.cell(r, 1).font = f_bold
            ws_dec.cell(r, 2).font = f_bold
            ws_dec.cell(r, 3).font = f_normal
            for col in range(1, 4):
                c = ws_dec.cell(r, col)
                c.border = border_thin
                if idx % 2 == 1:
                    c.fill = fill_zebra
    else:
        _xl_append(ws_dec, ["-", "No agenda & key decisions recorded.", "-"])
        ws_dec.cell(5, 2).font = f_italic

    # ── SHEET 4: RISKS ──
    ws_risk = wb.create_sheet(title="Risks & Issues")
    ws_risk.views.sheetView[0].showGridLines = True
    _xl_append(ws_risk, [])
    _xl_append(ws_risk, ["Risk & Issue Registry"])
    ws_risk.cell(2, 1).font = f_title
    _xl_append(ws_risk, [])

    headers = ["ID", "Risk Description", "Severity", "Potential Mitigation"]
    ws_risk.append(headers)
    for col_idx, h in enumerate(headers):
        cell = ws_risk.cell(row=4, column=col_idx + 1)
        cell.font = f_header
        cell.fill = fill_header
        cell.border = border_thin

    risks_text = analysis.risks if analysis else ""
    risks = parse_bullet_points(risks_text)
    if risks:
        for idx, rsk in enumerate(risks):
            _xl_append(ws_risk, [idx + 1, rsk, "Medium", "Monitor in next review session"])
            r = 5 + idx
            ws_risk.cell(r, 1).font = f_bold
            ws_risk.cell(r, 2).font = f_normal
            ws_risk.cell(r, 3).font = f_bold
            ws_risk.cell(r, 4).font = f_normal
            for col in range(1, 5):
                c = ws_risk.cell(r, col)
                c.border = border_thin
                if idx % 2 == 1:
                    c.fill = fill_zebra
    else:
        _xl_append(ws_risk, ["-", "No risks identified.", "-", "-"])
        ws_risk.cell(5, 2).font = f_italic

    # ── SHEET 5: TRANSCRIPT ──
    ws_tr = wb.create_sheet(title="Transcript")
    ws_tr.views.sheetView[0].showGridLines = True
    _xl_append(ws_tr, [])
    _xl_append(ws_tr, ["Transcript Log"])
    ws_tr.cell(2, 1).font = f_title
    _xl_append(ws_tr, [])

    headers = ["Index", "Time Stamp", "Speaker Name", "Speech Text"]
    ws_tr.append(headers)
    for col_idx, h in enumerate(headers):
        cell = ws_tr.cell(row=4, column=col_idx + 1)
        cell.font = f_header
        cell.fill = fill_header
        cell.border = border_thin

    if segments:
        for idx, seg in enumerate(segments):
            try:
                start = float(seg.start_time)
                mins = int(start // 60)
                secs = int(start % 60)
                time_rep = f"{mins}:{secs:02d}"
            except Exception:
                time_rep = seg.start_time
            spk = _speaker_label(seg)
            _xl_append(ws_tr, [idx + 1, time_rep, spk, seg.text])
            r = 5 + idx
            ws_tr.cell(r, 1).font = f_bold
            ws_tr.cell(r, 2).font = f_bold
            ws_tr.cell(r, 3).font = f_bold
            ws_tr.cell(r, 4).font = f_normal
            for col in range(1, 5):
                c = ws_tr.cell(r, col)
                c.border = border_thin
                if idx % 2 == 1:
                    c.fill = fill_zebra
    else:
        _xl_append(ws_tr, ["-", "-", "No transcript segments found.", "-"])
        ws_tr.cell(5, 3).font = f_italic

    # ── Auto-adjust Column Widths ──
    for sheet in wb.worksheets:
        for col in sheet.columns:
            max_len = 0
            for cell in col:
                val = str(cell.value or "")
                if cell.row == 2:  # skip title
                    continue
                # Cap column width at 50 chars to avoid extremely wide cells for descriptions
                max_len = max(max_len, min(len(val), 55))
            col_letter = get_column_letter(col[0].column)
            sheet.column_dimensions[col_letter].width = max(max_len + 3, 10)

    _neutralise_formulas(wb)

    stream = io.BytesIO()
    wb.save(stream)
    stream.seek(0)
    return stream


def generate_html_report(
    meeting: Any,
    analysis: Any,
    settings: Dict[str, Any],
    segments: List[Any],
    print_mode: bool = False,
) -> str:
    """Renders print-friendly HTML/PDF document (Requirement 9, 11, 12, 13, 14, 15)."""
    raw_meeting = meeting  # un-escaped ORM row: needed for structured lookups (agenda rows …)
    meeting = _HtmlSafeView(meeting)  # every string attribute read below comes back HTML-escaped

    p_color = "#" + _hex6(settings.get("theme_color_primary"), "1e3a8a")
    s_color = "#" + _hex6(settings.get("theme_color_secondary"), "2563eb")
    font_family = _safe_font(settings.get("font_family"), "system-ui, -apple-system, sans-serif")
    watermark = settings.get("watermark", "")
    logo_url = _safe_logo_url(settings.get("logo_url", ""))
    company_name_html = _esc(settings.get("company_name") or _DEFAULT_COMPANY_NAME)
    watermark_html = _esc(watermark or "CONFIDENTIAL")
    header_text_css = _css_string(settings.get("header_text") or "Confidential Business Report")

    # ── Chart Analytics Calculation (Requirement 9) ──
    # Heuristics for SVG charts
    sentiment = meeting.sentiment or "Neutral"
    sent_counts = {"Positive": 60, "Neutral": 30, "Negative": 10}  # defaults
    if sentiment == "Positive":
        sent_counts = {"Positive": 75, "Neutral": 20, "Negative": 5}
    elif sentiment == "Negative":
        sent_counts = {"Positive": 15, "Neutral": 25, "Negative": 60}

    # Heuristics for Speaker Participation Share
    spk_durations: Dict[str, float] = {}
    total_len = 0.0
    for seg in segments:
        try:
            start = float(seg.start_time)
            end = float(seg.end_time)
            dur = end - start
            spk = _speaker_label(seg)
            spk_durations[spk] = spk_durations.get(spk, 0.0) + dur
            total_len += dur
        except Exception:
            pass

    if not spk_durations:
        spk_durations = {}
        total_len = 0.0

    spk_pcts = {k: round((v / total_len) * 100, 1) for k, v in spk_durations.items()} if total_len > 0 else {}

    # Generate SVG code for Sentiment Chart (Donut Chart)
    # Radians/angles mapping for positive/neutral/negative
    (sent_counts["Positive"] / 100) * 360
    (sent_counts["Neutral"] / 100) * 360

    # Draw simple visual bar representing sentiment split
    sentiment_bar_svg = f"""
    <svg width="100%" height="45" style="background:#f1f5f9; border-radius:8px; padding: 4px 12px; box-sizing:border-box;">
        <rect x="0" y="15" width="{sent_counts["Positive"]}%" height="15" fill="#10b981" rx="4"/>
        <rect x="{sent_counts["Positive"]}%" y="15" width="{sent_counts["Neutral"]}%" height="15" fill="#f59e0b"/>
        <rect x="{sent_counts["Positive"] + sent_counts["Neutral"]}%" y="15" width="{sent_counts["Negative"]}%" height="15" fill="#ef4444" rx="4"/>
        <text x="2%" y="10" font-size="9" font-family="{font_family}" fill="#475569">Positive ({sent_counts["Positive"]}%)</text>
        <text x="45%" y="10" font-size="9" font-family="{font_family}" fill="#475569">Neutral ({sent_counts["Neutral"]}%)</text>
        <text x="80%" y="10" font-size="9" font-family="{font_family}" fill="#475569">Negative ({sent_counts["Negative"]}%)</text>
    </svg>
    """

    # Generate SVG code for Speaker Participation (Horizontal Bar Graph)
    speaker_bars = ""
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]
    for idx, (spk, pct) in enumerate(list(spk_pcts.items())[:5]):
        color = colors[idx % len(colors)]
        speaker_bars += f"""
        <div style="margin-bottom:8px;">
            <div style="display:flex; justify-content:space-between; font-size:11px; margin-bottom:2px; font-weight:600;">
                <span>{_esc(spk)}</span>
                <span>{pct}%</span>
            </div>
            <svg width="100%" height="10">
                <rect x="0" y="0" width="{pct}%" height="10" fill="{color}" rx="3"/>
                <rect x="{pct}%" y="0" width="{100-pct}%" height="10" fill="#e2e8f0" rx="3"/>
            </svg>
        </div>
        """

    # Render sections
    summary_html = ""
    if analysis and analysis.summary:
        summary_html = re.sub(r"\n", "<br/>", _esc(clean_markdown_text(analysis.summary)))

    mom_bullets = parse_bullet_points(analysis.mom) if analysis else []
    mom_html = ""
    if mom_bullets:
        mom_html = "<ul>" + "".join(f"<li>{_esc(pt)}</li>" for pt in mom_bullets) + "</ul>"
    elif analysis:
        mom_html = f"<p>{_esc(clean_markdown_text(analysis.mom))}</p>"

    action_items_bullets = (
        parse_bullet_points(analysis.action_items) if analysis else []
    )
    action_rows = ""
    if action_items_bullets:
        for idx, act in enumerate(action_items_bullets):
            # Extract owner
            owner = "Unassigned"
            owner_match = re.search(r"\[([^\]]+)\]|\(([^\)]+)\)|-\s*([A-Z][a-z]+)", act)
            if owner_match:
                owner = (
                    owner_match.group(1) or owner_match.group(2) or owner_match.group(3)
                )
                act_clean = (
                    act.replace(f"[{owner}]", "")
                    .replace(f"({owner})", "")
                    .replace(f"- {owner}", "")
                    .strip()
                )
            else:
                act_clean = act
            action_rows += f"""
            <tr>
                <td>MTG-ACT-{idx+1:02d}</td>
                <td>{_esc(act_clean)}</td>
                <td><span class="badge badge-accent">{_esc(owner)}</span></td>
                <td><span class="badge">Pending</span></td>
            </tr>
            """
    else:
        action_rows = "<tr><td colspan='4' class='empty-row'>No action items identified.</td></tr>"

    akd_items = _get_agenda_key_decisions(raw_meeting, analysis)
    decision_rows = ""
    if akd_items:
        for item in akd_items:
            decision_rows += f"""
            <tr>
                <td>{_esc(item['order'])}</td>
                <td><strong>{_esc(item['agenda'])}</strong></td>
                <td>{_esc(item['key_decision'])}</td>
            </tr>
            """
    else:
        decision_rows = (
            "<tr><td colspan='3' class='empty-row'>No agenda & key decisions recorded.</td></tr>"
        )

    risks_bullets = parse_bullet_points(analysis.risks) if analysis else []
    risk_rows = ""
    if risks_bullets:
        for idx, rsk in enumerate(risks_bullets):
            risk_rows += f"""
            <tr>
                <td>MTG-RSK-{idx+1:02d}</td>
                <td>{_esc(rsk)}</td>
                <td><span style="color:#f59e0b; font-weight:700;">Medium</span></td>
                <td>Monitor & review in follow-up session</td>
            </tr>
            """
    else:
        risk_rows = "<tr><td colspan='4' class='empty-row'>No risks identified during session.</td></tr>"

    # Transcripts segments
    transcript_rows = ""
    if segments:
        for seg in segments:
            try:
                start = float(seg.start_time)
                mins = int(start // 60)
                secs = int(start % 60)
                time_rep = f"{mins}:{secs:02d}"
            except Exception:
                time_rep = seg.start_time
            spk = _speaker_label(seg)
            transcript_rows += f"""
            <div class="transcript-seg">
                <span class="timestamp">[{_esc(time_rep)}]</span>
                <strong class="speaker">{_esc(spk)}:</strong>
                <span class="text">{_esc(seg.text)}</span>
            </div>
            """
    else:
        transcript_rows = "<p class='empty-row'>No transcript available.</p>"

    # Cover Page HTML
    cover_page_html = ""
    if settings.get("show_cover_page", True):
        logo_img_tag = (
            f'<img src="{_esc(logo_url)}" class="cover-logo" alt="Logo"/>' if logo_url else ""
        )
        cover_page_html = f"""
        <div class="page cover-page">
            <div class="cover-header">
                {logo_img_tag}
                <div class="company-branding">{company_name_html.upper()}</div>
            </div>
            <div class="cover-main">
                <div class="cover-divider"></div>
                <h1 class="cover-title">{meeting.title or meeting.filename}</h1>
                <h3 class="cover-subtitle">Project: {meeting.project_name or 'AI Operations'}</h3>
            </div>
            <div class="cover-footer">
                <table class="cover-meta-table">
                    <tr>
                        <td><strong>Meeting ID:</strong></td>
                        <td>{meeting.display_id or f"MTG-{meeting.id}"}</td>
                    </tr>
                    <tr>
                        <td><strong>Session Date:</strong></td>
                        <td>{meeting.meeting_date or 'N/A'} at {meeting.meeting_time or 'N/A'}</td>
                    </tr>
                    <tr>
                        <td><strong>Report Generated:</strong></td>
                        <td>{datetime.now().strftime("%B %d, %Y")}</td>
                    </tr>
                    <tr>
                        <td><strong>Confidentiality:</strong></td>
                        <td><span style="color:#ef4444; font-weight:700;">{watermark_html}</span></td>
                    </tr>
                </table>
            </div>
        </div>
        """

    # Table of Contents HTML
    toc_page_html = ""
    if settings.get("show_table_of_contents", True):
        toc_page_html = f"""
        <div class="page page-break-after">
            <h2 class="section-title">Table of Contents</h2>
            <div class="toc-container">
                <a href="#overview" class="toc-item"><span>1. Meeting Overview & Summary</span><span class="toc-dots"></span><span class="toc-pg">2</span></a>
                <a href="#mom" class="toc-item"><span>2. Minutes of Meeting (MoM)</span><span class="toc-dots"></span><span class="toc-pg">3</span></a>
                <a href="#actions" class="toc-item"><span>3. Key Action Items Registry</span><span class="toc-dots"></span><span class="toc-pg">4</span></a>
                <a href="#decisions" class="toc-item"><span>4. Decisions Tracker</span><span class="toc-dots"></span><span class="toc-pg">4</span></a>
                <a href="#risks" class="toc-item"><span>5. Risk Mitigation & Issues Registry</span><span class="toc-dots"></span><span class="toc-pg">5</span></a>
                <a href="#transcript" class="toc-item"><span>6. Speaker Transcript Segments</span><span class="toc-dots"></span><span class="toc-pg">6</span></a>
            </div>
        </div>
        """

    # Check for print trigger on load
    print_trigger_script = (
        f"<script>{REPORT_PRINT_SCRIPT}</script>"
        if print_mode
        else ""
    )

    # CSS Styles for watermark overlay
    watermark_css = ""
    if watermark:
        watermark_css = f"""
        body::before {{
            content: "{_css_string(watermark)}";
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) rotate(-45deg);
            font-size: 7vw;
            color: rgba(226, 232, 240, 0.28);
            z-index: -1000;
            pointer-events: none;
            font-weight: 900;
            letter-spacing: 12px;
            font-family: {font_family};
        }}
        """

    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Meeting Intelligence Report - {meeting.title or meeting.filename}</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 15mm 20mm 15mm;
            @top-left {{
                content: "{header_text_css}";
                font-family: {font_family};
                font-size: 8pt;
                color: #94a3b8;
            }}
            @bottom-center {{
                content: "Page " counter(page);
                font-family: {font_family};
                font-size: 8pt;
                color: #94a3b8;
            }}
        }}
        
        body {{
            font-family: {font_family};
            color: #334155;
            background: #ffffff;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            font-size: 10.5pt;
            -webkit-print-color-adjust: exact;
        }}
        
        {watermark_css}
        
        /* ── Page Margins and Print Control ── */
        .page {{
            page-break-after: always;
            box-sizing: border-box;
            position: relative;
        }}
        
        .page:last-of-type {{
            page-break-after: avoid !important;
        }}
        
        .page-break-after {{
            page-break-after: always;
        }}
        
        /* ── Header and Footer Display ── */
        .doc-header {{
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 5px;
            margin-bottom: 20px;
            font-size: 8pt;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .doc-footer {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            display: flex;
            justify-content: space-between;
            border-top: 1px solid #e2e8f0;
            padding-top: 5px;
            font-size: 8pt;
            color: #94a3b8;
        }}
        
        /* ── Cover Page Design ── */
        .cover-page {{
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            padding: 10mm 5mm;
            min-height: 250mm;
        }}
        
        .cover-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        
        .cover-logo {{
            max-height: 45px;
            max-width: 150px;
            object-fit: contain;
        }}
        
        .company-branding {{
            font-size: 10pt;
            font-weight: 800;
            color: {s_color};
            letter-spacing: 2px;
        }}
        
        .cover-main {{
            flex-grow: 1;
            display: flex;
            flex-direction: column;
            justify-content: center;
            margin-top: -30mm;
        }}
        
        .cover-divider {{
            height: 4px;
            width: 80px;
            background-color: {s_color};
            margin-bottom: 25px;
            border-radius: 2px;
        }}
        
        .cover-title {{
            font-size: 32pt;
            font-weight: 800;
            color: {p_color};
            line-height: 1.15;
            margin: 0 0 10px 0;
            letter-spacing: -1px;
        }}
        
        .cover-subtitle {{
            font-size: 14pt;
            font-weight: 400;
            color: #64748b;
            margin: 0;
            font-style: italic;
        }}
        
        .cover-meta-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 10.5pt;
        }}
        
        .cover-meta-table td {{
            padding: 6px 0;
            border: none;
        }}
        
        .cover-meta-table td:first-child {{
            width: 180px;
            color: #64748b;
        }}
        
        /* ── Table of Contents ── */
        .toc-container {{
            margin-top: 30px;
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .toc-item {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            text-decoration: none;
            color: #334155;
            font-size: 11pt;
            font-weight: 500;
        }}
        
        .toc-dots {{
            flex-grow: 1;
            border-bottom: 2px dotted #cbd5e1;
            margin: 0 15px;
            position: relative;
            top: 4px;
        }}
        
        .toc-pg {{
            font-weight: 700;
            color: {p_color};
        }}
        
        /* ── Headings ── */
        .section-title {{
            font-size: 18pt;
            font-weight: 800;
            color: {p_color};
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 20px;
        }}
        
        .subsection-title {{
            font-size: 13pt;
            font-weight: 700;
            color: {s_color};
            margin-top: 25px;
            margin-bottom: 12px;
        }}
        
        /* ── Metadata Summary Cards Grid ── */
        .meta-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 25px;
        }}
        
        .meta-card {{
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px;
            box-sizing: border-box;
        }}
        
        .meta-label {{
            font-size: 8.5pt;
            font-weight: 800;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            display: block;
            margin-bottom: 4px;
        }}
        
        .meta-value {{
            font-size: 11pt;
            font-weight: 600;
            color: #1e293b;
        }}
        
        /* ── Charts & Analytics ── */
        .analytics-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-top: 20px;
            margin-bottom: 25px;
        }}
        
        .chart-box {{
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 15px;
            background: #ffffff;
        }}
        
        /* ── Tables ── */
        table.report-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            margin-bottom: 25px;
            font-size: 10pt;
        }}
        
        table.report-table th {{
            background-color: {p_color};
            color: #ffffff;
            font-weight: 700;
            text-align: left;
            padding: 10px 12px;
            border: 1px solid {p_color};
        }}
        
        table.report-table td {{
            padding: 10px 12px;
            border-bottom: 1px solid #e2e8f0;
            color: #475569;
        }}
        
        table.report-table tr:nth-child(even) td {{
            background-color: #f8fafc;
        }}
        
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            font-size: 8pt;
            font-weight: 700;
            border-radius: 6px;
            background: #cbd5e1;
            color: #475569;
        }}
        
        .badge-accent {{
            background: #e0f2fe;
            color: #0369a1;
        }}
        
        .empty-row {{
            text-align: center;
            font-style: italic;
            color: #94a3b8 !important;
            padding: 20px !important;
        }}
        
        /* ── Transcript ── */
        .transcript-seg {{
            margin-bottom: 10px;
            font-size: 10pt;
        }}
        
        .timestamp {{
            color: #94a3b8;
            font-family: monospace;
            margin-right: 6px;
        }}
        
        .speaker {{
            color: #1e293b;
            margin-right: 6px;
        }}
        
        /* Print rules */
        @media print {{
            body {{
                background: none;
                color: #000;
            }}
            .toc-item {{
                color: #000;
            }}
        }}
    </style>
</head>
<body>

    {cover_page_html}

    {toc_page_html}

    <!-- SECTION 1 -->
    <div class="page">
        <div class="doc-header">
            <span>Meeting Intelligence Report</span>
            <span>ID: {meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)}</span>
        </div>
        
        <h2 id="overview" class="section-title">1. Meeting Overview</h2>
        
        <div class="meta-grid">
            <div class="meta-card">
                <span class="meta-label">Meeting Title</span>
                <span class="meta-value">{meeting.title or meeting.filename}</span>
            </div>
            <div class="meta-card">
                <span class="meta-label">Project Domain</span>
                <span class="meta-value">{meeting.project_name or 'AI Operations'}</span>
            </div>
            <div class="meta-card">
                <span class="meta-label">Date & Time</span>
                <span class="meta-value">{meeting.meeting_date or 'N/A'} at {meeting.meeting_time or 'N/A'}</span>
            </div>
            <div class="meta-card">
                <span class="meta-label">AI Quality Confidence</span>
                <span class="meta-value">{(meeting.accuracy or 0.0):.1f}% Score</span>
            </div>
            <div class="meta-card" style="grid-column: span 2;">
                <span class="meta-label">Agreed Session Agenda</span>
                <span class="meta-value" style="font-weight:400;">{_esc(_meeting_agenda_text(meeting))}</span>
            </div>
        </div>

        <h3 class="subsection-title">Executive Summary</h3>
        <p style="text-align:justify; margin-top:0;">{summary_html or 'No summary text generated.'}</p>
        
        <div class="doc-footer">
            <span>{company_name_html}</span>
            <span>Page 2</span>
        </div>
    </div>

    <!-- SECTION 1B: Analytics Visualization (Requirement 9) -->
    {f'''
    <div class="page">
        <div class="doc-header">
            <span>Meeting Intelligence Report</span>
            <span>ID: {meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)}</span>
        </div>
        
        <h2 class="section-title">Meeting Analytics & Distribution</h2>
        
        <div class="analytics-container">
            <div class="chart-box">
                <h4 style="margin-top:0; margin-bottom:12px; font-size:11pt; color:{p_color};">Sentiment Distribution Split</h4>
                {sentiment_bar_svg}
                <p style="font-size:8.5pt; color:#64748b; margin-top:10px; line-height:1.4;">
                    Evaluates overall dialogue polarity based on keyword counts, tone shifts, and keyword intensity markers.
                </p>
            </div>
            
            <div class="chart-box">
                <h4 style="margin-top:0; margin-bottom:12px; font-size:11pt; color:{p_color};">Speaker Participation Share</h4>
                {speaker_bars}
            </div>
        </div>
        
        <div class="doc-footer">
            <span>{company_name_html}</span>
            <span>Page 3</span>
        </div>
    </div>
    ''' if settings.get("show_visual_charts", True) else ""}

    <!-- SECTION 2 -->
    <div class="page">
        <div class="doc-header">
            <span>Meeting Intelligence Report</span>
            <span>ID: {meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)}</span>
        </div>
        
        <h2 id="mom" class="section-title">2. Minutes of Meeting (MoM)</h2>
        <div style="margin-top:10px;">
            {mom_html}
        </div>
        
        <div class="doc-footer">
            <span>{company_name_html}</span>
            <span>Page 4</span>
        </div>
    </div>

    <!-- SECTION 3 & 4 -->
    <div class="page">
        <div class="doc-header">
            <span>Meeting Intelligence Report</span>
            <span>ID: {meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)}</span>
        </div>
        
        <h2 id="actions" class="section-title">3. Key Action Items Registry</h2>
        <table class="report-table">
            <thead>
                <tr>
                    <th style="width:120px;">Task ID</th>
                    <th>Action Item Description</th>
                    <th style="width:140px;">Assignee</th>
                    <th style="width:120px;">Status</th>
                </tr>
            </thead>
            <tbody>
                {action_rows}
            </tbody>
        </table>

        <h2 id="decisions" class="section-title" style="margin-top:35px;">4. Decisions Tracker</h2>
        <table class="report-table">
            <thead>
                <tr>
                    <th style="width:120px;">Decision ID</th>
                    <th>Consensus / Choice Made</th>
                    <th style="width:180px;">Status</th>
                </tr>
            </thead>
            <tbody>
                {decision_rows}
            </tbody>
        </table>
        
        <div class="doc-footer">
            <span>{company_name_html}</span>
            <span>Page 5</span>
        </div>
    </div>

    <!-- SECTION 5 -->
    <div class="page">
        <div class="doc-header">
            <span>Meeting Intelligence Report</span>
            <span>ID: {meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)}</span>
        </div>
        
        <h2 id="risks" class="section-title">5. Risk Mitigation & Issues Registry</h2>
        <table class="report-table">
            <thead>
                <tr>
                    <th style="width:120px;">Risk ID</th>
                    <th>Risk Description & Impact</th>
                    <th style="width:120px;">Severity</th>
                    <th>Mitigation Strategy</th>
                </tr>
            </thead>
            <tbody>
                {risk_rows}
            </tbody>
        </table>
        
        <div class="doc-footer">
            <span>{company_name_html}</span>
            <span>Page 6</span>
        </div>
    </div>

    <!-- SECTION 6 -->
    <div class="page">
        <div class="doc-header">
            <span>Meeting Intelligence Report</span>
            <span>ID: {meeting.display_id or meeting.bharatvc_meeting_id or str(meeting.id)}</span>
        </div>
        
        <h2 id="transcript" class="section-title">6. Speaker Transcript Segments</h2>
        <div style="margin-top:15px; border-top:1px solid #f1f5f9; padding-top:15px;">
            {transcript_rows}
        </div>
        
        <div class="doc-footer">
            <span>{company_name_html}</span>
            <span>Page 7</span>
        </div>
    </div>

    {print_trigger_script}

</body>
</html>
"""
