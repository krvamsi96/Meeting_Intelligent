"""Regression tests for SSRF / upload / header-safety fixes in backend.utils.security and url_service."""

import socket
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from backend.utils import security
from backend.utils.security import (
    assert_public_host,
    content_disposition,
    is_public_ip,
    safe_float,
    validate_file_type,
    validate_time_range,
    validate_url,
)


def _fake_getaddrinfo(*addresses):
    """socket.getaddrinfo stand-in returning the given IPs (v4 or v6) for any host."""

    def _impl(host, port, *args, **kwargs):
        out = []
        for ip in addresses:
            family = socket.AF_INET6 if ":" in ip else socket.AF_INET
            sockaddr = (ip, port or 0, 0, 0) if ":" in ip else (ip, port or 0)
            out.append((family, socket.SOCK_STREAM, 6, "", sockaddr))
        return out

    return _impl


# ── SSRF ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ip",
    [
        "127.0.0.1",
        "10.192.110.24",          # the DB server from the leaked config
        "172.16.5.4",
        "192.168.1.10",
        "169.254.169.254",        # cloud metadata endpoint
        "100.64.0.1",             # CGNAT — not covered by the old hand-written range list
        "0.0.0.0",
        "::1",
        "fe80::1",                # IPv6 link-local — also missed before
        "fc00::1",
        "::ffff:10.0.0.1",        # IPv4-mapped IPv6 must not smuggle a private IPv4 past the check
        "224.0.0.1",
    ],
)
def test_non_public_addresses_are_rejected(ip):
    import ipaddress

    assert is_public_ip(ipaddress.ip_address(ip)) is False


@pytest.mark.parametrize("ip", ["8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"])
def test_public_addresses_are_allowed(ip):
    import ipaddress

    assert is_public_ip(ipaddress.ip_address(ip)) is True


def test_a_hostname_with_one_private_address_among_public_ones_is_rejected(monkeypatch):
    # A dual-stack / multi-A-record host: gethostbyname() only ever looked at ONE address.
    monkeypatch.setattr(security.socket, "getaddrinfo", _fake_getaddrinfo("8.8.8.8", "10.0.0.5"))
    with pytest.raises(HTTPException) as exc:
        assert_public_host("evil.example")
    assert exc.value.status_code == 400


def test_validate_url_accepts_a_public_host(monkeypatch):
    monkeypatch.setattr(security.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34"))
    assert validate_url("https://example.com/a.mp4") == "https://example.com/a.mp4"


@pytest.mark.parametrize(
    "url",
    ["file:///etc/passwd", "ftp://example.com/x", "gopher://example.com", "https://", "https://user:pw@example.com/"],
)
def test_validate_url_rejects_bad_schemes_and_embedded_credentials(url, monkeypatch):
    monkeypatch.setattr(security.socket, "getaddrinfo", _fake_getaddrinfo("93.184.216.34"))
    with pytest.raises(HTTPException) as exc:
        validate_url(url)
    assert exc.value.status_code == 400


def test_unresolvable_host_is_a_400_not_a_500(monkeypatch):
    def _boom(*a, **k):
        raise socket.gaierror("no such host")

    monkeypatch.setattr(security.socket, "getaddrinfo", _boom)
    with pytest.raises(HTTPException) as exc:
        validate_url("https://does-not-exist.invalid/")
    assert exc.value.status_code == 400


def test_safe_request_revalidates_every_redirect_hop(monkeypatch):
    """A public URL that 302s to the metadata service must be blocked at the SECOND hop."""
    from backend.services import url_service

    resolved = {"public.example": "93.184.216.34", "169.254.169.254": "169.254.169.254"}
    monkeypatch.setattr(
        security.socket,
        "getaddrinfo",
        lambda host, port, *a, **k: _fake_getaddrinfo(resolved.get(host, "8.8.8.8"))(host, port),
    )

    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((url, kwargs.get("allow_redirects")))
        resp = MagicMock()
        resp.status_code = 302
        resp.headers = {"Location": "http://169.254.169.254/latest/meta-data/"}
        return resp

    monkeypatch.setattr(url_service.requests, "request", fake_request)
    with pytest.raises(HTTPException) as exc:
        url_service.safe_request("GET", "https://public.example/video", timeout=5)
    assert exc.value.status_code == 400
    assert calls == [("https://public.example/video", False)], "the internal target must never be requested"


def test_platform_detection_matches_the_hostname_not_a_substring():
    from backend.services.url_service import detect_url_content_type

    # Extension check is by path; host trick must not be classified as a YouTube video
    assert detect_url_content_type("https://youtube.com.evil.example/report.pdf") == "document"
    assert detect_url_content_type("https://www.youtube.com/watch?v=abc") == "video"
    assert detect_url_content_type("https://youtu.be/abc") == "video"


# ── upload content validation ─────────────────────────────────────────────────


def test_validate_file_type_rejects_wrong_content_type_from_magic(monkeypatch, tmp_path):
    """Regression: the HTTPException used to be swallowed by a blanket `except Exception`."""
    f = tmp_path / "notes.mp4"
    f.write_bytes(b"MZ\x90\x00 not really a video")

    fake_magic = MagicMock()
    fake_magic.Magic.return_value.from_file.return_value = "application/x-dosexec"
    monkeypatch.setitem(__import__("sys").modules, "magic", fake_magic)

    with pytest.raises(HTTPException) as exc:
        validate_file_type(str(f), "media")
    assert exc.value.status_code == 400
    assert "application/x-dosexec" in exc.value.detail


@pytest.mark.parametrize("name", ["a.mov", "a.mkv", "a.flv", "a.wma", "a.mp3", "a.wav", "a.m4a", "a.mp4", "a.webm"])
def test_every_supported_media_extension_passes_the_extension_fallback(name, tmp_path, monkeypatch):
    """.mov/.mkv/.flv/.wma are supported ingest types but were rejected with HTTP 400."""
    monkeypatch.setitem(__import__("sys").modules, "magic", None)  # force the mimetypes fallback
    f = tmp_path / name
    f.write_bytes(b"\x00" * 16)
    validate_file_type(str(f), "media")  # must not raise


@pytest.mark.parametrize("name", ["a.pdf", "a.docx", "a.xlsx", "a.pptx", "a.txt", "a.csv", "a.md", "a.json", "a.doc", "a.xls", "a.ppt", "a.rtf", "a.log"])
def test_every_supported_document_extension_passes_the_extension_fallback(name, tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "magic", None)
    f = tmp_path / name
    f.write_bytes(b"x")
    validate_file_type(str(f), "document")


# ── numbers / headers ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["nan", "inf", "-inf", "NaN", "-1", "abc"])
def test_safe_float_rejects_non_finite_and_negative(bad):
    with pytest.raises(HTTPException):
        safe_float(bad, "start")


def test_validate_time_range_rejects_nan_bounds():
    with pytest.raises(HTTPException):
        validate_time_range("nan", "10")
    with pytest.raises(HTTPException):
        validate_time_range("0", "inf")


def test_content_disposition_survives_unicode_titles():
    header = content_disposition("बैठक रिपोर्ट.docx")
    header.encode("latin-1")  # Starlette encodes headers as latin-1: this used to raise -> HTTP 500
    assert header.startswith("attachment; ")
    assert "filename*=UTF-8''" in header
    assert 'filename="download.docx"' in header


def test_content_disposition_blocks_header_injection():
    header = content_disposition('evil"; x=y\r\nSet-Cookie: pwned=1\r\n.docx')
    assert "\r" not in header and "\n" not in header
    assert header.count('"') == 2, "only the two quotes we add around the fallback filename"


def test_content_disposition_strips_path_components():
    header = content_disposition("../../etc/passwd.md")
    assert "/" not in header.split("filename=")[1].split(";")[0]


# ── generated meeting ids ─────────────────────────────────────────────────────


def test_generated_meeting_ids_are_unique_even_within_one_millisecond():
    from backend.utils.helpers import generate_numeric_meeting_id

    ids = [generate_numeric_meeting_id() for _ in range(5000)]  # far faster than 1 id / ms
    assert len(set(ids)) == len(ids)
    assert all(i.isdigit() and len(i) == 16 for i in ids)


def test_generated_ids_are_unique_across_threads():
    import threading

    from backend.utils.helpers import generate_numeric_meeting_id

    out, lock = [], threading.Lock()

    def work():
        local = [generate_numeric_meeting_id() for _ in range(500)]
        with lock:
            out.extend(local)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(out)) == len(out) == 4000


def test_supplied_ids_are_still_sanitised_to_digits():
    from backend.utils.helpers import generate_numeric_meeting_id

    assert generate_numeric_meeting_id("BVC-1785956313139-501") == "1785956313139501"
    assert generate_numeric_meeting_id("MTG_42") == "42"
