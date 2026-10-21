import logging
import mimetypes
import os
import re
import uuid
from typing import Iterable, Optional, Tuple
from urllib.parse import unquote, urljoin, urlparse

import requests

from backend.config.config import settings
from backend.utils.security import validate_url

logger = logging.getLogger(__name__)

# Hosts handled by yt-dlp. Matching is on the parsed HOSTNAME (exact or subdomain), never a
# substring of the URL: "notyoutube.com.evil.example" must not count as YouTube.
VIDEO_DOMAINS = [
    "youtube.com",
    "youtu.be",
    "vimeo.com",
    "dailymotion.com",
    "twitch.tv",
    "loom.com",
    "wistia.com",
    "tiktok.com",
]

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".mp3",
    ".wav",
    ".m4a",
    ".webm",
    ".flv",
    ".aac",
    ".ogg",
    ".wma",
    ".3gp",
}

DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".txt",
    ".csv",
    ".rtf",
    ".md",
    ".json",
    ".log",
    ".html",
    ".htm",
}

_USER_AGENT = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_MAX_REDIRECTS = 5
_MAX_HTML_BYTES = 10 * 1024 * 1024  # a web page is text; never buffer more than this in RAM
_MAX_VIDEO_BYTES = 500 * 1024 * 1024
_MAX_DOC_BYTES = 100 * 1024 * 1024


def _host_matches(host: str, domains: Iterable[str]) -> bool:
    host = (host or "").lower().rstrip(".")
    return any(host == d or host.endswith("." + d) for d in domains)


def _safe_extension(name: str, default: str) -> str:
    """Return a filesystem-safe extension (``.abc123``) taken from ``name``, else ``default``."""
    ext = os.path.splitext(name or "")[1].lower()
    return ext if re.fullmatch(r"\.[a-z0-9]{1,8}", ext) else default


def safe_request(
    method: str,
    url: str,
    *,
    timeout: float,
    stream: bool = False,
    headers: Optional[dict] = None,
) -> requests.Response:
    """HTTP request that re-validates EVERY redirect hop against the SSRF rules.

    ``requests`` follows redirects on its own, so a public URL answering ``302 → http://
    169.254.169.254/…`` would sail past a check that only looked at the first URL. Here
    redirects are followed manually and each target is validated before it is fetched.
    Raises ``fastapi.HTTPException(400)`` for a blocked hop (the same error ``validate_url``
    gives the caller) and ``ValueError`` for a redirect loop.
    """
    current = url
    for _ in range(_MAX_REDIRECTS + 1):
        validate_url(current)
        resp = requests.request(
            method,
            current,
            headers=headers or _USER_AGENT,
            stream=stream,
            timeout=timeout,
            allow_redirects=False,
        )
        location = resp.headers.get("Location")
        if resp.status_code in (301, 302, 303, 307, 308) and location:
            resp.close()
            current = urljoin(current, location)
            if resp.status_code == 303:
                method = "GET"
            continue
        return resp
    raise ValueError(f"Too many redirects (more than {_MAX_REDIRECTS}) fetching URL")


def detect_url_content_type(url: str) -> str:
    """
    Auto-detects whether a URL points to a video or a document/webpage.
    Returns: 'video' or 'document'
    """
    parsed = urlparse(url.strip())
    host = parsed.hostname or ""

    # 1. Check known video streaming platforms
    if _host_matches(host, VIDEO_DOMAINS):
        return "video"

    # 2. Check path extension
    path_ext = os.path.splitext(parsed.path)[1].lower()
    if path_ext in VIDEO_EXTENSIONS:
        return "video"
    if path_ext in DOCUMENT_EXTENSIONS:
        return "document"

    # 3. Perform lightweight HTTP HEAD request to inspect Content-Type header
    try:
        resp = safe_request("HEAD", url.strip(), timeout=5)
        try:
            ct = resp.headers.get("Content-Type", "").lower()
        finally:
            resp.close()
        if any(v in ct for v in ["video/", "audio/", "octet-stream", "media"]):
            return "video"
        if any(
            d in ct
            for d in [
                "application/pdf",
                "application/msword",
                "officedocument",
                "text/",
                "application/json",
            ]
        ):
            return "document"
    except Exception as e:
        logger.warning(f"HTTP HEAD content-type inspection warning for {url}: {e}")

    # Fallback default: Treat HTML web pages as documents
    return "document"


def _stream_to_file(resp: requests.Response, local_path: str, max_bytes: int, label: str) -> None:
    """Stream ``resp`` to ``local_path``; delete the partial file on ANY failure."""
    total_size = 0
    try:
        with open(local_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                if not chunk:
                    continue
                total_size += len(chunk)
                if total_size > max_bytes:
                    raise ValueError(f"{label} download exceeds maximum limit of {max_bytes // (1024 * 1024)}MB")
                f.write(chunk)
    except BaseException:
        try:
            os.remove(local_path)
        except OSError:
            pass
        raise


def _download_with_ytdlp(clean_url: str, output_dir: str) -> Optional[Tuple[str, str]]:
    """Try yt-dlp (audio extracted to mp3). Returns (path, title) or None if it produced nothing."""
    import yt_dlp

    unique_id = uuid.uuid4().hex[:10]
    out_template = os.path.join(output_dir, f"video_{unique_id}.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": out_template,
        "quiet": True,
        "no_warnings": True,
        # A playlist / channel URL would otherwise queue hundreds of downloads.
        "noplaylist": True,
        "playlist_items": "1",
        "socket_timeout": 30,
        "max_filesize": 2 * 1024 * 1024 * 1024,  # 2 GB max
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean_url, download=True)
        if info and info.get("_type") == "playlist":
            entries = [e for e in (info.get("entries") or []) if e]
            info = entries[0] if entries else None
        if not info:
            return None
        filename = ydl.prepare_filename(info)
        mp3_path = os.path.splitext(filename)[0] + ".mp3"

        orig_title = info.get("title") or "video_recording"
        orig_title = re.sub(r"[^\w\s-]", "", orig_title).strip() or "video_recording"
        orig_title += ".mp3"

        if os.path.exists(mp3_path):
            return mp3_path, orig_title
        if os.path.exists(filename):
            return filename, orig_title
    return None


def download_video_from_url(url: str, output_dir: str) -> Tuple[str, str]:
    """
    Downloads video or audio from YouTube, Vimeo, Drive (yt-dlp) or a direct media file URL.
    Returns: (local_file_path, original_filename)
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_url = url.strip()
    host = urlparse(clean_url).hostname or ""
    is_platform = _host_matches(host, VIDEO_DOMAINS)

    # yt-dlp fetches the URL (and anything the page points at) itself, following redirects
    # internally where we cannot validate each hop. So it is only used for known video
    # platforms unless the operator explicitly opts in with YTDLP_ALLOW_ANY_HOST=true.
    if is_platform or settings.YTDLP_ALLOW_ANY_HOST:
        try:
            result = _download_with_ytdlp(clean_url, output_dir)
            if result:
                return result
        except Exception as yt_err:
            logger.warning(f"yt-dlp extraction failed for {url}: {yt_err}")
            if is_platform:
                # A platform page fetched directly is just HTML — saving it as a "video"
                # only produces a confusing failure later in the pipeline.
                raise Exception(f"Failed to download video from URL '{url}': {yt_err}") from yt_err

    # Streaming HTTP download for direct media file URLs
    try:
        resp = safe_request("GET", clean_url, timeout=30, stream=True)
        try:
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "").lower()
            if "text/html" in content_type:
                raise ValueError("URL returned a web page, not an audio/video file")

            orig_filename = os.path.basename(unquote(urlparse(resp.url or clean_url).path)) or "downloaded_video.mp4"
            ext = _safe_extension(orig_filename, ".mp4")

            unique_filename = f"video_{uuid.uuid4()}{ext}"
            local_path = os.path.join(output_dir, unique_filename)
            _stream_to_file(resp, local_path, _MAX_VIDEO_BYTES, "Video")
        finally:
            resp.close()
        return local_path, orig_filename
    except Exception as http_err:
        raise Exception(f"Failed to download video from URL '{url}': {http_err}") from http_err


def _read_capped(resp: requests.Response, max_bytes: int) -> bytes:
    """Read a streamed body but refuse to buffer more than ``max_bytes`` in memory."""
    buf = bytearray()
    for chunk in resp.iter_content(chunk_size=64 * 1024):
        if not chunk:
            continue
        buf.extend(chunk)
        if len(buf) > max_bytes:
            raise ValueError(f"Web page exceeds maximum size of {max_bytes // (1024 * 1024)}MB")
    return bytes(buf)


def download_document_from_url(url: str, output_dir: str) -> Tuple[str, str, str]:
    """
    Downloads document file (PDF, DOCX, PPTX, XLSX, TXT) or fetches HTML web page text.
    Returns: (local_file_path, original_filename, extracted_text)
    """
    os.makedirs(output_dir, exist_ok=True)
    clean_url = url.strip()

    # Handle Google Drive share links (match on the real hostname, not a substring of the URL)
    if _host_matches(urlparse(clean_url).hostname or "", ["drive.google.com"]):
        match = re.search(r"/d/([a-zA-Z0-9_-]+)", clean_url)
        if match:
            file_id = match.group(1)
            # Use confirm=1 to bypass the virus scan warning page for smaller files
            clean_url = (
                f"https://drive.google.com/uc?export=download&id={file_id}&confirm=1"
            )

    resp = safe_request("GET", clean_url, timeout=20, stream=True)
    try:
        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "").lower()
        parsed = urlparse(resp.url or clean_url)
        orig_filename = os.path.basename(unquote(parsed.path)) or "downloaded_document.txt"
        url_ext = _safe_extension(orig_filename, "")

        # HTML page: the server says so, or says nothing and the URL has no file extension.
        # (A PDF served from an extension-less URL such as /download?id=7 is NOT html.)
        is_html = "html" in ct or (not ct and not url_ext)
        if is_html:
            try:
                from bs4 import BeautifulSoup

                soup = BeautifulSoup(_read_capped(resp, _MAX_HTML_BYTES), "html.parser")

                # Extract title
                page_title = (
                    soup.title.string.strip()
                    if soup.title and soup.title.string
                    else parsed.netloc
                )

                # Remove scripts, styles
                for script in soup(["script", "style", "header", "footer", "nav"]):
                    script.extract()

                text = soup.get_text(separator="\n")
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                clean_text = "\n".join(chunk for chunk in chunks if chunk)

                unique_filename = f"doc_{uuid.uuid4()}.txt"
                local_path = os.path.join(output_dir, unique_filename)

                with open(local_path, "w", encoding="utf-8") as f:
                    f.write(clean_text)

                return local_path, f"{page_title}.txt", clean_text
            except Exception as html_err:
                logger.warning(f"HTML parsing warning for {clean_url}: {html_err}")
                raise

        # Binary or text document file download. Prefer the URL's extension; otherwise infer
        # it from the Content-Type so an extension-less download keeps a type the document
        # pipeline can dispatch on.
        ext = url_ext
        if not ext:
            guessed = _safe_extension("x" + (mimetypes.guess_extension(ct.split(";")[0].strip()) or ""), "")
            ext = guessed if guessed in DOCUMENT_EXTENSIONS else ".txt"
        if not os.path.splitext(orig_filename)[1]:
            orig_filename += ext
        unique_filename = f"doc_{uuid.uuid4()}{ext}"
        local_path = os.path.join(output_dir, unique_filename)
        _stream_to_file(resp, local_path, _MAX_DOC_BYTES, "Document")
    finally:
        resp.close()

    return local_path, orig_filename, ""
