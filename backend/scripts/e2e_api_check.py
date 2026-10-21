"""End-to-end API verification against a RUNNING server (your laptop, the Docker stack, or the Linux server).

    # 1. upload recordings, wait for processing, verify every endpoint and the content of every result
    python -m backend.scripts.e2e_api_check --url http://SERVER:8000 --file a.mp4 --file b.mp4 --state e2e_state.json

    # 2. restart the server, then prove nothing was lost or mixed up
    python -m backend.scripts.e2e_api_check --url http://SERVER:8000 --verify-state e2e_state.json

Upload mode checks:
  * validation / error paths: unsupported type -> 400, corrupt media -> 400, unknown meeting -> 404, bad chat body -> 4xx
  * concurrent uploads each get their own meeting (no overwrite), then every read endpoint answers 200
  * content: transcript present, English summary that is a real summary (not the transcript, no "Speaker_1" labels),
    English agenda / decisions / action items, no duplicates, nothing invented (owners / deadlines / evidence must trace
    back to the transcript), the source-language transcript preserved separately
  * metrics: reference-free (no ground truth => no accuracy percentages), the six quality scores are present and explained
  * RAG: a question is answered from this meeting only (every source carries this meeting's id), history persisted
Verify mode (after a restart): the same outputs come back byte-for-byte from PostgreSQL, RAG still answers from ChromaDB.

Exit code 0 = no failures (warnings allowed), 1 = failures. Nothing secret is sent or printed; pass --admin-key only if
the server protects admin routes (it is not needed by this script).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

import requests

_SPEAKER_LABEL = re.compile(r"\b(?:Speaker[_ ]?\d+|SPEAKER_\d+)\b")
_TIMESTAMP = re.compile(r"\[?\b\d{1,2}:\d{2}(?::\d{2})?\b\]?")
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_GENERIC_OWNERS = {"", "none", "n/a", "na", "unassigned", "not specified", "unknown", "tbd", "team", "everyone", "all"}
QUALITY_KEYS = ("transcript_quality", "summary_quality", "rag_groundedness", "action_item_quality", "decision_quality",
                "agenda_coverage")


class Checker:
    def __init__(self) -> None:
        self.counts = {"PASS": 0, "WARN": 0, "FAIL": 0}
        self.rows: List[Dict[str, str]] = []

    def ok(self, name: str, detail: str = "") -> bool:
        return self._add("PASS", name, detail)

    def warn(self, name: str, detail: str = "") -> bool:
        return self._add("WARN", name, detail)

    def fail(self, name: str, detail: str = "") -> bool:
        return self._add("FAIL", name, detail)

    def check(self, condition: bool, name: str, detail: str = "", warn_only: bool = False) -> bool:
        if condition:
            return self.ok(name, detail if not warn_only else "")
        return self.warn(name, detail) if warn_only else self.fail(name, detail)

    def _add(self, status: str, name: str, detail: str) -> bool:
        self.counts[status] += 1
        self.rows.append({"status": status, "check": name, "detail": detail})
        print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""), flush=True)
        return status == "PASS"


class Client:
    def __init__(self, base: str, timeout: float):
        self.base = base.rstrip("/") + "/api/v1"
        self.timeout = timeout
        self.session = requests.Session()
        # uvicorn closes an idle keep-alive connection after 5 s; polling at that same interval over a reused
        # connection then hits a just-closed socket ("Remote end closed connection without response"). One connection
        # per request avoids the race, and idempotent GETs are retried once more on a transport error.
        self.session.headers["Connection"] = "close"

    def request(self, method: str, path: str, **kw) -> requests.Response:
        kw.setdefault("timeout", self.timeout)
        attempts = 3 if method.upper() == "GET" else 1
        for attempt in range(1, attempts + 1):
            try:
                return self.session.request(method, self.base + path, **kw)
            except (requests.exceptions.ConnectionError, requests.exceptions.ChunkedEncodingError):
                if attempt >= attempts:
                    raise
                time.sleep(1.0 * attempt)
        raise RuntimeError("unreachable")

    def get(self, path: str, **kw) -> requests.Response:
        return self.request("GET", path, **kw)

    def post(self, path: str, **kw) -> requests.Response:
        return self.request("POST", path, **kw)


# ─────────────────────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────────────────────

def words(text: Any) -> List[str]:
    return [w.lower() for w in _WORD.findall(str(text or ""))]


def overlap(needle: Any, haystack_words: set) -> float:
    tokens = [w for w in words(needle) if len(w) > 2]
    return sum(1 for w in tokens if w in haystack_words) / len(tokens) if tokens else 1.0


def non_ascii_letter_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if ord(c) > 127) / len(letters) if letters else 0.0


def normalise(text: Any) -> str:
    return " ".join(words(text))


def sha(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()[:16]


def json_or_none(resp: requests.Response) -> Any:
    try:
        return resp.json()
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────────────────────────────────────
# phases
# ─────────────────────────────────────────────────────────────────────────────────────────────

def phase_health(api: Client, c: Checker) -> None:
    print("\n== health ==")
    r = api.get("/system/health")
    body = json_or_none(r) or {}
    c.check(r.status_code == 200, "GET /system/health -> 200", f"HTTP {r.status_code}")
    services = body.get("services", {})
    c.check(bool(services.get("database")), "PostgreSQL reachable", str(body.get("database_target", {}).get("environment", "")))
    target = body.get("database_target") or {}
    if target:
        c.check(not target.get("fallback_active"), "connected to the configured (non-fallback) database",
                target.get("warning", ""))
    c.check(bool(services.get("vector_database")), "ChromaDB available", json.dumps(body.get("vector_database", {})))
    c.check(bool(services.get("models")), "AI models & embeddings healthy", "still loading? re-run in a minute", warn_only=True)
    r = api.get("/system/health/models")
    c.check(r.status_code == 200, "GET /system/health/models -> 200")
    r = api.get("/system/metrics")
    c.check(r.status_code == 200 and "meetings" in (json_or_none(r) or {}), "GET /system/metrics -> 200")


def phase_negative(api: Client, c: Checker) -> None:
    print("\n== validation and error responses ==")
    tmp = tempfile.mkdtemp(prefix="e2e_neg_")
    exe, bad = os.path.join(tmp, "run.exe"), os.path.join(tmp, "corrupt.mp4")
    open(exe, "wb").write(b"MZ\x90\x00")
    open(bad, "wb").write(os.urandom(4096))
    with open(exe, "rb") as fh:
        r = api.post("/upload", files={"file": ("run.exe", fh, "application/octet-stream")})
    c.check(r.status_code == 400, "POST /upload unsupported extension -> 400", f"HTTP {r.status_code}")
    with open(bad, "rb") as fh:
        r = api.post("/upload", files={"file": ("corrupt.mp4", fh, "video/mp4")}, data={"meeting_id": "1"})
    c.check(r.status_code == 400, "POST /upload corrupt media -> 400 (rejected at upload, no meeting created)",
            f"HTTP {r.status_code} {r.text[:120]}")
    r = api.post("/upload")
    c.check(r.status_code in (400, 422), "POST /upload without a file -> 4xx", f"HTTP {r.status_code}")
    for path in ("/meetings/987654321987", "/meetings/987654321987/final", "/meetings/987654321987/action-items",
                 "/meetings/987654321987/agenda-key-decisions", "/meetings/987654321987/metrics",
                 "/meetings/987654321987/chat/history"):
        r = api.get(path)
        c.check(r.status_code == 404, f"GET {path} unknown meeting -> 404", f"HTTP {r.status_code}")
    r = api.post("/meetings/987654321987/chat", json={"message": "hello"})
    c.check(r.status_code == 404, "POST chat on an unknown meeting -> 404", f"HTTP {r.status_code}")
    c.check(json_or_none(api.get("/meetings/987654321987")) is not None, "error bodies are JSON")


def phase_upload(api: Client, c: Checker, files: List[str]) -> List[Dict[str, Any]]:
    print("\n== upload (concurrent) ==")
    base = int(time.time() * 1000) % 10**12
    plan = [{"file": f, "meeting_id": str(base + i)} for i, f in enumerate(files)]

    def send(item):
        with open(item["file"], "rb") as fh:
            return api.post("/upload", files={"file": (os.path.basename(item["file"]), fh, "video/mp4")},
                            data={"meeting_id": item["meeting_id"]}, timeout=max(api.timeout, 600))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(plan))) as pool:
        responses = list(pool.map(send, plan))
    uploaded = []
    for item, r in zip(plan, responses):
        body = json_or_none(r) or {}
        good = r.status_code == 200 and body.get("status") == "processing" and body.get("meeting_id") == item["meeting_id"]
        c.check(good, f"POST /upload {os.path.basename(item['file'])} -> 200 processing (meeting_id preserved)",
                f"HTTP {r.status_code} {str(body)[:160]}")
        if good:
            uploaded.append({**item, "pk": body["id"], "upload": body})
    c.check(len({u["pk"] for u in uploaded}) == len(uploaded), "each upload got its own meeting")
    return uploaded


def phase_wait(api: Client, c: Checker, uploaded: List[Dict[str, Any]], timeout: float) -> List[Dict[str, Any]]:
    print("\n== processing ==")
    t0 = time.time()
    pending = {u["pk"]: u for u in uploaded}
    last: Dict[int, Any] = {}
    done: List[Dict[str, Any]] = []
    while pending and time.time() - t0 < timeout:
        for pk in list(pending):
            r = api.get(f"/meetings/{pk}")
            body = json_or_none(r) or {}
            status = (body.get("meeting") or {}).get("status") or body.get("status")
            if status != last.get(pk):
                print(f"  meeting {pk}: {status} ({time.time() - t0:.0f}s)", flush=True)
                last[pk] = status
            if status in ("completed", "error"):
                pending.pop(pk)
                done.append({**uploaded[[u["pk"] for u in uploaded].index(pk)], "status": status, "detail": body})
        time.sleep(5)
    for pk, u in pending.items():
        c.fail(f"meeting {pk} finished within {timeout:.0f}s", f"still {last.get(pk)}")
    for d in done:
        if d["status"] == "completed":
            c.ok(f"meeting {d['pk']} completed")
        else:
            c.fail(f"meeting {d['pk']} completed", f"status=error: {d['detail'].get('error_message')}")
    return [d for d in done if d["status"] == "completed"]


def phase_endpoints(api: Client, c: Checker, pk: int) -> Dict[str, Any]:
    print(f"\n== endpoints, meeting {pk} ==")
    out: Dict[str, Any] = {}
    routes = {
        "meeting": f"/meetings/{pk}", "final": f"/meetings/{pk}/final", "action_items": f"/meetings/{pk}/action-items",
        "agenda": f"/meetings/{pk}/agenda-key-decisions", "metrics": f"/meetings/{pk}/metrics",
        "diarization": f"/meetings/{pk}/diarization", "speaker_analytics": f"/meetings/{pk}/speaker-analytics",
        "similar": f"/meetings/{pk}/similar", "duplicate": f"/meetings/{pk}/duplicate-check",
        "export_json": f"/meetings/{pk}/export/json", "activity_logs": f"/meetings/{pk}/activity-logs",
        "live_metrics": f"/evaluation/{pk}/live-metrics", "verify": f"/meetings/{pk}/verify",
        "chat_history": f"/meetings/{pk}/chat/history",
    }
    for name, path in routes.items():
        r = api.get(path)
        c.check(r.status_code == 200, f"GET {path} -> 200", f"HTTP {r.status_code}")
        out[name] = json_or_none(r)
    r = api.get("/meetings")
    c.check(r.status_code == 200 and any(str(m.get("id")) == str(pk) for m in (json_or_none(r) or [])),
            "GET /meetings lists it")

    # GET /meetings/{id} re-aggregates and upserts the consolidated record on every call: parallel readers must all
    # get the same, complete answer (no 500 from a write race, no half-built record).
    def read(path: str) -> tuple:
        resp = api.get(path)
        return resp.status_code, (json_or_none(resp) or {})

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        replies = list(pool.map(read, [f"/meetings/{pk}", f"/meetings/{pk}/final"] * 4))
    statuses = {code for code, _ in replies}
    summaries = {sha((body.get("analysis") or {}).get("summary") if "analysis" in body else body.get("summary"))
                 for _, body in replies}
    c.check(statuses == {200}, "8 parallel reads of the same meeting all answer 200", f"statuses={sorted(statuses)}")
    c.check(len(summaries) == 1, "parallel reads return identical summaries", f"{len(summaries)} distinct")
    return out


def phase_content(c: Checker, pk: int, data: Dict[str, Any]) -> Dict[str, Any]:
    print(f"\n== content quality, meeting {pk} ==")
    final = data.get("final") or {}
    meeting = data.get("meeting") or {}
    tag = f"meeting {pk}: "
    c.check(final.get("processing_status") == "COMPLETED", tag + "final record COMPLETED", str(final.get("processing_status")))

    transcript = final.get("transcript") or []
    text = " ".join(str(s.get("text", "")) for s in transcript).strip()
    tw = set(words(text))
    c.check(bool(text), tag + "English transcript present", f"{len(transcript)} segment(s), {len(words(text))} words")
    starts = [float(s.get("start_time") or 0) for s in transcript]
    c.check(starts == sorted(starts) and all(float(s.get("end_time") or 0) >= float(s.get("start_time") or 0) for s in transcript),
            tag + "segments ordered, end >= start")

    # language + original transcript preserved separately
    segs = meeting.get("segments") or []
    lang = meeting.get("source_language")
    c.check(bool(lang), tag + "source language detected", f"{lang} ({meeting.get('source_language_name')})")
    c.check(any(s.get("raw_text") for s in segs), tag + "source-language transcript kept separately (raw_text)")
    c.check(meeting.get("output_language", "en") == "en", tag + "output language is English", str(meeting.get("output_language")))

    # speakers
    speakers = final.get("speakers") or []
    labels = {s.get("speaker") for s in transcript}
    count = (final.get("meeting") or {}).get("speaker_count")
    c.check(len(speakers) >= 1 and count == len(speakers), tag + "speaker_count matches the speaker list",
            f"speaker_count={count}, listed={len(speakers)}")
    c.check(labels <= {s.get("speaker_id") for s in speakers} | {s.get("label") for s in speakers},
            tag + "every transcript speaker is in the speaker list")
    diar = (data.get("diarization") or {}).get("diarization") or {}
    if diar.get("embedding_fidelity") == "acoustic_fallback":
        c.warn(tag + "speaker count came from the WEAK fallback embedder", diar.get("warning", ""))
    else:
        c.ok(tag + "speaker embedder", str(diar.get("embedding_fidelity")))

    # summary
    summary = str(final.get("summary") or "").strip()
    sw = words(summary)
    c.check(bool(summary), tag + "summary present", f"{len(sw)} words")
    if summary:
        c.check(len(words(text)) < 40 or len(sw) <= 0.6 * len(words(text)), tag + "summary is a summary, not the transcript",
                f"{len(sw)} of {len(words(text))} words")
        c.check(normalise(summary) != normalise(text), tag + "summary differs from the transcript")
        c.check(not _SPEAKER_LABEL.search(summary), tag + "summary has no Speaker_N labels")
        c.check(not _TIMESTAMP.search(summary), tag + "summary has no timestamps")
        c.check(summary[-1] in ".!?\"')", tag + "summary ends like a sentence", repr(summary[-30:]), warn_only=True)
        sentences = [normalise(s) for s in re.split(r"(?<=[.!?])\s+", summary) if s.strip()]
        c.check(len(sentences) == len(set(sentences)), tag + "no repeated sentences in the summary")
        c.check(overlap(summary, tw) >= 0.5, tag + "summary vocabulary comes from the transcript",
                f"{overlap(summary, tw):.0%}", warn_only=True)

    # English-only outputs
    agenda = final.get("agenda") or []
    decisions = final.get("decisions") or []
    actions = final.get("action_items") or []
    english_blob = " ".join([summary, str(final.get("mom") or ""), *[str(a.get("title", "")) for a in agenda],
                             *[str(d.get("decision", "")) for d in decisions], *[str(a.get("task", "")) for a in actions]])
    c.check(non_ascii_letter_share(english_blob) < 0.03, tag + "summary / MoM / agenda / decisions / actions are English",
            f"{non_ascii_letter_share(english_blob):.1%} non-Latin letters")
    mom = str(final.get("mom") or "")
    c.check(bool(mom.strip()), tag + "MoM present", f"{len(words(mom))} words")
    if "Duration:" in mom and (final.get("meeting") or {}).get("duration_seconds"):
        c.check("Duration:** Not Specified" not in mom, tag + "MoM states the known duration")

    # agenda / decisions
    titles = [normalise(a.get("title")) for a in agenda]
    c.check(len(titles) == len(set(titles)), tag + "agenda items are unique", f"{len(agenda)} item(s)")
    dnorm = [normalise(d.get("decision")) for d in decisions]
    c.check(len(dnorm) == len(set(dnorm)), tag + "decisions are unique", f"{len(decisions)} decision(s)")
    c.check(all(str(d.get("decision", "")).strip() for d in decisions), tag + "no empty decisions")
    for d in decisions:
        c.check(overlap(d.get("decision"), tw) >= 0.4 or bool(d.get("evidence")), tag + "decision is grounded in the transcript",
                str(d.get("decision"))[:80], warn_only=True)
    ids = {a.get("agenda_id") for a in agenda}
    for m in final.get("agenda_decision_mapping") or []:
        if m.get("has_decision"):
            c.check(bool(str(m.get("decision", "")).strip()) and m.get("agenda_id") in ids,
                    tag + "agenda->decision mapping points at a real agenda item and decision")
        else:
            c.check(not str(m.get("decision", "")).strip(), tag + "unmapped agenda item shows no decision")

    # action items: not invented
    tasks = [normalise(a.get("task")) for a in actions]
    c.check(len(tasks) == len(set(tasks)), tag + "action items are unique", f"{len(actions)} item(s)")
    speaker_words = {w for s in speakers for w in words(s.get("label"))}
    for a in actions:
        owner = str(a.get("owner") or "").strip().lower()
        if owner not in _GENERIC_OWNERS:
            c.check(all(w in tw or w in speaker_words for w in words(owner)), tag + "action-item owner appears in the meeting",
                    owner, warn_only=True)
        if a.get("evidence"):
            c.check(overlap(a["evidence"], tw) >= 0.6, tag + "action-item evidence comes from the transcript",
                    str(a["evidence"])[:70], warn_only=True)
        if a.get("deadline") and str(a.get("deadline_type") or "") not in ("", "unresolvable"):
            c.check(overlap(a.get("deadline"), tw) >= 0.5 or bool(a.get("due_date")), tag + "deadline is stated in the meeting",
                    str(a.get("deadline"))[:50], warn_only=True)
    ai_endpoint = data.get("action_items") or {}
    c.check(ai_endpoint.get("total_items") == len(ai_endpoint.get("action_items") or []) or not ai_endpoint,
            tag + "action-items endpoint is self-consistent")

    # metrics
    m = meeting.get("metrics") or {}
    gt = bool(m.get("ground_truth_available"))
    c.check(m.get("evaluation_type") == "reference_free" or gt, tag + "metrics are reference-free without ground truth",
            str(m.get("evaluation_type")))
    if not gt:
        c.check(all(meeting.get(k) is None for k in ("accuracy", "speaker_accuracy", "transcription_accuracy")),
                tag + "no accuracy percentage is reported without ground truth")
    for key in QUALITY_KEYS:
        block = m.get(key)
        good = isinstance(block, dict) and ("score" in block) and (block.get("score") is None and bool(block.get("reason") or block.get("issues"))
                                                                    or isinstance(block.get("score"), (int, float)) and 0 <= block["score"] <= 100)
        c.check(good, tag + f"quality score '{key}' present and explained",
                json.dumps({k: block.get(k) for k in ("score", "label")}) if isinstance(block, dict) else "missing")
    for name in ("export_json", "similar", "duplicate", "verify"):
        c.check(data.get(name) is not None, tag + f"{name} returned JSON")

    return {
        "transcript_sha": sha(text), "raw_sha": sha(" ".join(str(s.get("raw_text") or "") for s in segs)),
        "summary": summary, "mom_sha": sha(mom), "language": lang, "speaker_count": count,
        "agenda": [a.get("title") for a in agenda], "decisions": [d.get("decision") for d in decisions],
        "actions": [a.get("task") for a in actions],
        "scores": {k: (m.get(k) or {}).get("score") for k in QUALITY_KEYS if k != "rag_groundedness"},
    }


def phase_rag(api: Client, c: Checker, pk: int, snap: Dict[str, Any], other_pks: List[int]) -> None:
    print(f"\n== RAG, meeting {pk} ==")
    topic = (snap["agenda"] or ["this meeting"])[0]
    r = api.post(f"/meetings/{pk}/chat", json={"message": f"What was discussed about {topic}?"}, timeout=max(api.timeout, 300))
    body = json_or_none(r) or {}
    c.check(r.status_code == 200 and str(body.get("content", "")).strip() != "", f"POST /meetings/{pk}/chat -> 200 with an answer",
            f"HTTP {r.status_code}")
    sources = body.get("sources") or []
    c.check(len(sources) >= 1, "answer cites retrieved chunks (ChromaDB)", f"{len(sources)} source(s)")
    foreign = [s for s in sources if str(s.get("meeting_id")) not in {str(pk), str((body or {}).get("meeting_id"))}
               and str(s.get("meeting_id")) in {str(o) for o in other_pks}]
    c.check(not foreign, "no source comes from another meeting (isolation)", f"{len(foreign)} foreign source(s)")
    c.check(all(str(s.get("meeting_id")) == str(pk) for s in sources) or not sources,
            "every retrieved chunk belongs to this meeting", str({s.get("meeting_id") for s in sources}))
    c.check(isinstance(body.get("rag_groundedness"), dict), "answer carries a RAG groundedness score",
            str((body.get("rag_groundedness") or {}).get("score")))
    hist = json_or_none(api.get(f"/meetings/{pk}/chat/history")) or []
    c.check(len(hist) >= 2, "chat history persisted (question + answer)", f"{len(hist)} message(s)")


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def phase_isolation(api: Client, c: Checker, snaps: Dict[int, Dict[str, Any]], file_hash: Dict[int, str]) -> None:
    if len(snaps) < 2:
        return
    print("\n== isolation between meetings ==")
    listing = json_or_none(api.get("/meetings")) or []
    ids = {str(m.get("id")) for m in listing}
    c.check(all(str(pk) in ids for pk in snaps), "all meetings are listed independently", f"{len(snaps)} of {len(listing)}")
    pks = list(snaps)
    for i, a in enumerate(pks):
        for b in pks[i + 1:]:
            if file_hash[a] != file_hash[b]:      # different recordings must never share results
                c.check(snaps[a]["transcript_sha"] != snaps[b]["transcript_sha"], f"meetings {a} and {b}: different transcripts")
                c.check(snaps[a]["summary"] != snaps[b]["summary"], f"meetings {a} and {b}: different summaries")
            else:                                  # the same file uploaded twice is two independent meetings
                c.check(True, f"meetings {a} and {b} are the same file, processed as two independent meetings")
                sim = json_or_none(api.get(f"/meetings/{a}/similar")) or {}
                listed = {str(m.get("meeting_id") or m.get("id")) for m in sim.get("meetings", [])}
                found = bool({str(b), str(snaps_meeting_code(snaps, b))} & listed) or bool(sim.get("duplicate_detected"))
                c.check(found, f"identical recording {b} is found as similar to {a} (ChromaDB similar-meeting search)",
                        f"similar={sorted(listed)}", warn_only=True)
    for pk in pks:
        sim = json_or_none(api.get(f"/meetings/{pk}/similar")) or {}
        listed = {str(m.get("id")) for m in sim.get("meetings", [])}
        c.check(str(pk) not in listed, f"meeting {pk} is not listed as similar to itself")


def snaps_meeting_code(snaps: Dict[int, Dict[str, Any]], pk: int) -> str:
    return str(snaps[pk].get("meeting_code", pk))


def save_state(path: str, base: str, uploaded, snaps: Dict[int, Dict[str, Any]]) -> None:
    payload = {"url": base, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "meetings": {str(pk): {**snaps[pk], "meeting_id": next(u["meeting_id"] for u in uploaded if u["pk"] == pk)}
                            for pk in snaps}}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nState saved to {path} - restart the server, then run with --verify-state {path}")


def verify_state(api: Client, c: Checker, path: str) -> None:
    print("\n== persistence after restart ==")
    state = json.load(open(path, encoding="utf-8"))
    pks = [int(p) for p in state["meetings"]]
    for pk in pks:
        saved = state["meetings"][str(pk)]
        data = phase_endpoints(api, c, pk)
        now = phase_content(c, pk, data)
        for key in ("transcript_sha", "raw_sha", "summary", "mom_sha", "language", "speaker_count", "agenda", "decisions",
                    "actions", "scores"):
            c.check(now[key] == saved[key], f"meeting {pk}: {key} identical after restart",
                    "" if now[key] == saved[key] else f"before={str(saved[key])[:80]} after={str(now[key])[:80]}")
        r = api.get(f"/meetings/{saved['meeting_id']}")
        c.check(r.status_code == 200, f"meeting {pk} retrievable by its public id {saved['meeting_id']}")
        phase_rag(api, c, pk, now, [p for p in pks if p != pk])


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--file", action="append", default=[], help="recording to upload (repeat for several)")
    parser.add_argument("--state", help="write meeting ids + result fingerprints here (for --verify-state)")
    parser.add_argument("--verify-state", help="verify a previous run after a server restart")
    parser.add_argument("--timeout", type=float, default=1800, help="seconds to wait for processing (default 1800)")
    parser.add_argument("--request-timeout", type=float, default=60)
    parser.add_argument("--skip-negative", action="store_true")
    args = parser.parse_args(argv)
    if not args.file and not args.verify_state:
        parser.error("give --file (upload mode) or --verify-state (after a restart)")

    api, c = Client(args.url, args.request_timeout), Checker()
    print(f"API verification against {api.base}")
    try:
        phase_health(api, c)
        if args.verify_state:
            verify_state(api, c, args.verify_state)
        else:
            if not args.skip_negative:
                phase_negative(api, c)
            uploaded = phase_upload(api, c, args.file)
            completed = phase_wait(api, c, uploaded, args.timeout)
            snaps: Dict[int, Dict[str, Any]] = {}
            for u in completed:
                data = phase_endpoints(api, c, u["pk"])
                snaps[u["pk"]] = phase_content(c, u["pk"], data)
            for pk in snaps:
                phase_rag(api, c, pk, snaps[pk], [p for p in snaps if p != pk])
            phase_isolation(api, c, snaps, {u["pk"]: file_sha256(u["file"]) for u in uploaded})
            if args.state and snaps:
                save_state(args.state, args.url, uploaded, snaps)
    except requests.exceptions.ConnectionError as exc:
        c.fail("server reachable", f"{type(exc).__name__}: {str(exc)[:120]}")

    print(f"\nRESULT: {c.counts['PASS']} passed, {c.counts['WARN']} warning(s), {c.counts['FAIL']} failed")
    return 1 if c.counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
