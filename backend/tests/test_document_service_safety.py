"""Document ingestion: prose must survive cleaning, hostile archives must be refused."""

import zipfile

import pytest

from backend.services import document_service as ds


def test_prose_lines_starting_with_let_or_var_are_not_dropped_as_javascript():
    text = "The agenda is short.\nlet us begin with the budget\nvar is a common abbreviation for variance\nconst\nDone."
    cleaned = ds.clean_document_text(text)
    assert "let us begin with the budget" in cleaned
    assert "var is a common abbreviation for variance" in cleaned


def test_real_javascript_declarations_are_still_removed():
    text = "Meeting notes here.\nvar config = {a: 1};\nconst x = 5;\nlet y = 6;\nSecond real sentence."
    cleaned = ds.clean_document_text(text)
    assert "Meeting notes here." in cleaned and "Second real sentence." in cleaned
    assert "var config" not in cleaned and "const x" not in cleaned and "let y" not in cleaned


def test_comparison_operators_in_plain_text_are_preserved():
    cleaned = ds.clean_document_text("Approve if cost < 5 lakh and risk > low for the vendor.")
    assert "cost < 5 lakh and risk > low" in cleaned


def test_html_documents_are_still_reduced_to_text():
    cleaned = ds.clean_document_text("<html><head><title>x</title></head><body><script>var a=1;</script><p>Hello budget</p></body></html>")
    assert cleaned == "Hello budget"


def test_trailing_segment_uses_the_configured_interval(monkeypatch):
    monkeypatch.setattr(ds.doc_cfg, "SEGMENT_INTERVAL_SECONDS", 5)
    monkeypatch.setattr(ds.doc_cfg, "SEGMENT_WORD_CHUNK_SIZE", 3)
    segs = ds.create_document_segments("one two three\nfour five six\nseven")
    assert [s["start_time"] for s in segs] == ["00:00", "05:00", "10:00"]
    assert [s["end_time"] for s in segs] == ["05:00", "10:00", "15:00"], "the last chunk used a hardcoded 2"


def _fake_llm(monkeypatch, content):
    from types import SimpleNamespace

    import backend.services.llm_workflow as wf

    monkeypatch.setattr(wf, "get_llm", lambda *a, **k: SimpleNamespace(invoke=lambda msgs: SimpleNamespace(content=content)))


def test_llm_metadata_with_non_string_participants_is_kept_not_discarded(monkeypatch):
    _fake_llm(monkeypatch, '{"title": "Budget Review", "agenda": "Q3", "participants": ["Rahul", 42, {"n": 1}], "meeting_date": 20260925}')
    meta = ds.extract_document_metadata_with_llm("some text", "notes.txt")
    assert meta["title"] == "Budget Review", "a single odd element must not throw away the whole answer"
    assert meta["participants"].startswith("Rahul, 42")
    assert meta["meeting_date"] == "20260925" and isinstance(meta["meeting_date"], str)


def test_llm_metadata_that_is_not_an_object_falls_back_to_the_filename(monkeypatch):
    _fake_llm(monkeypatch, '["not", "an", "object"]')
    meta = ds.extract_document_metadata_with_llm("some text", "weekly_sync-notes.txt")
    assert meta["title"] == "Weekly Sync Notes"


def test_archive_that_expands_beyond_the_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MAX_ARCHIVE_UNCOMPRESSED_BYTES", 1024 * 1024)  # 1 MB for the test
    bomb = tmp_path / "bomb.docx"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("word/document.xml", "A" * (5 * 1024 * 1024))  # compresses to a few KB
    assert bomb.stat().st_size < 100 * 1024
    with pytest.raises(ValueError, match="expands to"):
        ds.extract_document_text(str(bomb))


def test_normal_text_file_still_extracts(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("Rahul: approve the budget.\nPriya: send quotes tomorrow.\n", encoding="utf-8")
    assert "approve the budget" in ds.extract_document_text(str(f))


def test_extracted_text_is_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "MAX_EXTRACTED_CHARS", 2000)
    f = tmp_path / "big.txt"
    f.write_text("\n".join(f"line number {i} with some words" for i in range(5000)), encoding="utf-8")
    assert len(ds.extract_document_text(str(f))) <= 2000


def test_xlsx_is_read_in_streaming_mode(tmp_path):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Task", "Owner"])
    ws.append(["Send quotes", "Priya"])
    path = tmp_path / "plan.xlsx"
    wb.save(path)
    text = ds.extract_document_text(str(path))
    assert "Send quotes | Priya" in text
