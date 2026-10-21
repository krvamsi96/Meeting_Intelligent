"""Make sure the REAL embedding model is on disk before the API starts - and fetch it through Git LFS when possible.

The weights (``models/embeddings/<name>/model.safetensors``) are a Git LFS object in the OpenForge repository. A checkout
made without git-lfs holds a ~130-byte text pointer under the same name. This module:

  1. inspects the configured model directory (required files present, no LFS pointer),
  2. if the weights are a pointer / missing and this is a git checkout with git-lfs installed, runs
     ``git lfs install --local`` + ``git lfs pull --include=<model dir>`` (the project's own LFS remote - nothing else),
  3. verifies the weights byte-for-byte: size and SHA-256 must equal the oid the repository's pointer pins.

Standard library only, so launch scripts can run it before the virtual environment exists:
``python3 -m backend.scripts.ensure_embedding_model``. It never downloads from any other source and never creates
substitute files: when the real model cannot be obtained it says exactly why and how to fix it.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
WEIGHTS_FILE = "model.safetensors"
# Pinned by the repository's LFS pointer for models/embeddings/all-MiniLM-L6-v2/model.safetensors. Used only when neither
# the working-tree pointer nor git history can be read (e.g. an archive deployment without .git).
PINNED_WEIGHTS = {DEFAULT_MODEL_NAME: ("53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db", 90868376)}
# Small files a sentence-transformers directory needs besides the weights.
REQUIRED_FILES = ("config.json", "modules.json", "sentence_bert_config.json", "tokenizer.json", "vocab.txt",
                  os.path.join("1_Pooling", "config.json"))
LFS_POINTER_PREFIX = b"version https://git-lfs"
GIT_LFS_TIMEOUT_SEC = int(os.getenv("EMBEDDING_LFS_PULL_TIMEOUT_SEC", "900"))


@dataclass
class ModelCheck:
    ok: bool
    model_dir: str
    weights_bytes: int = 0
    expected_sha256: Optional[str] = None
    expected_bytes: Optional[int] = None
    sha256_verified: bool = False
    fetched_with_git_lfs: bool = False
    problem: str = ""
    fix: str = ""
    steps: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ok:
            return (f"embedding model OK - {self.model_dir} ({self.weights_bytes} bytes, SHA-256 verified"
                    f"{', fetched with git lfs pull' if self.fetched_with_git_lfs else ''})")
        return f"embedding model NOT usable - {self.problem}. FIX: {self.fix}"

    def as_dict(self) -> Dict[str, object]:
        return {k: getattr(self, k) for k in ("ok", "model_dir", "weights_bytes", "expected_bytes", "sha256_verified",
                                              "fetched_with_git_lfs", "problem", "fix")}


def model_name() -> str:
    return os.getenv("EMBEDDING_MODEL_NAME") or DEFAULT_MODEL_NAME


def model_dir() -> str:
    """The configured directory (EMBEDDING_MODEL_PATH, relative paths resolve against the project root)."""
    configured = os.getenv("EMBEDDING_MODEL_PATH", "").strip()
    if configured:
        return os.path.abspath(configured if os.path.isabs(configured) else os.path.join(PROJECT_ROOT, configured))
    return os.path.join(PROJECT_ROOT, "models", "embeddings", model_name())


def read_lfs_pointer(path: str) -> Optional[Tuple[str, int]]:
    """(sha256, size) when ``path`` is a Git LFS pointer file, else None."""
    try:
        if os.path.getsize(path) > 1024:
            return None
        with open(path, "rb") as fh:
            data = fh.read(1024)
    except OSError:
        return None
    return _parse_pointer(data)


def _parse_pointer(data: bytes) -> Optional[Tuple[str, int]]:
    if not data.startswith(LFS_POINTER_PREFIX):
        return None
    text = data.decode("utf-8", "replace")
    oid = re.search(r"^oid sha256:([0-9a-f]{64})\s*$", text, re.M)
    size = re.search(r"^size (\d+)\s*$", text, re.M)
    return (oid.group(1), int(size.group(1))) if oid and size else None


def _git(args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="never")
    return subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout, env=env)


def _is_git_checkout() -> bool:
    return os.path.exists(os.path.join(PROJECT_ROOT, ".git")) and shutil.which("git") is not None


def _git_lfs_available() -> bool:
    try:
        return _is_git_checkout() and _git(["lfs", "version"], timeout=20).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _relpath(path: str) -> str:
    return os.path.relpath(path, PROJECT_ROOT).replace(os.sep, "/")


def _expected_from_git(weights: str) -> Optional[Tuple[str, int]]:
    """The pointer committed in HEAD for the weights file (what `git lfs pull` will fetch)."""
    if not _is_git_checkout():
        return None
    try:
        size = subprocess.run(["git", "cat-file", "-s", f"HEAD:{_relpath(weights)}"], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, timeout=30)
        if size.returncode != 0 or not size.stdout.strip().isdigit() or int(size.stdout) > 1024:
            return None                                   # committed as the real file (not a pointer): use the pin
        out = subprocess.run(["git", "show", f"HEAD:{_relpath(weights)}"], cwd=PROJECT_ROOT, capture_output=True,
                             timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return _parse_pointer(out.stdout) if out.returncode == 0 else None


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect(check: ModelCheck) -> None:
    """Fill ``check`` from the files on disk (no fetching)."""
    d = check.model_dir
    weights = os.path.join(d, WEIGHTS_FILE)
    check.ok, check.sha256_verified, check.weights_bytes = False, False, 0
    if not os.path.isdir(d):
        check.problem = f"the model directory {d} does not exist"
        return
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(d, f))]
    if missing:
        check.problem = f"{d} is missing {', '.join(missing)}"
        return
    if not os.path.isfile(weights):
        check.problem = f"{weights} does not exist"
        return
    check.weights_bytes = os.path.getsize(weights)
    pointer = read_lfs_pointer(weights)
    if pointer:
        check.expected_sha256, check.expected_bytes = pointer
        check.problem = (f"{weights} is a Git LFS pointer ({check.weights_bytes} bytes) instead of the real "
                         f"{pointer[1]}-byte model")
        return
    if not check.expected_sha256:
        pinned = _expected_from_git(weights) or PINNED_WEIGHTS.get(model_name())
        if pinned:
            check.expected_sha256, check.expected_bytes = pinned
    if check.expected_bytes and check.weights_bytes != check.expected_bytes:
        check.problem = (f"{weights} is {check.weights_bytes} bytes, expected {check.expected_bytes} "
                         f"(truncated / interrupted copy or a different file)")
        return
    if check.expected_sha256:
        if _sha256(weights) != check.expected_sha256:
            check.problem = f"{weights} has the right size but its SHA-256 does not match the repository's LFS oid (corrupt)"
            return
        check.sha256_verified = True
    elif check.weights_bytes < 1_000_000:
        check.problem = f"{weights} is only {check.weights_bytes} bytes - not a real model"
        return
    check.ok, check.problem, check.fix = True, "", ""


def _git_lfs_pull(check: ModelCheck) -> Tuple[bool, str]:
    include = _relpath(check.model_dir) + "/**"
    try:
        _git(["lfs", "install", "--local"], timeout=60)
        res = _git(["lfs", "pull", f"--include={include}"], timeout=GIT_LFS_TIMEOUT_SEC)
    except subprocess.TimeoutExpired:
        return False, f"git lfs pull timed out after {GIT_LFS_TIMEOUT_SEC}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return False, f"git lfs pull could not run ({exc})"
    tail = " | ".join(line.strip() for line in (res.stderr or res.stdout or "").strip().splitlines()[-3:])
    return res.returncode == 0, tail


def _fix_for(check: ModelCheck, attempted: bool, git_error: str) -> str:
    rel = _relpath(check.model_dir)
    copy_hint = (f"or copy {rel}/ from a machine that has it: bash deploy_model.sh <user> <host> <deploy_path>, "
                 f"then restart")
    if not _is_git_checkout():
        return (f"{PROJECT_ROOT} is not a git checkout (deployed as an archive) and the archive carried the Git LFS "
                f"pointer. The repository now stores the real weights as a normal file: redeploy the current commit "
                f"(any checkout or archive of it contains the 91 MB model), {copy_hint}")
    if not _git_lfs_available():
        return ("git-lfs is not installed on this machine. Install it (Ubuntu/Debian: sudo apt-get install -y git-lfs; "
                "RHEL: sudo dnf install -y git-lfs), then restart - the startup check runs "
                f"`git lfs pull --include={rel}/**` automatically, {copy_hint}")
    if attempted:
        return (f"`git lfs pull` from the repository's LFS remote failed ({git_error or 'no output'}). Check network / "
                f"proxy (https_proxy) / credentials for the OpenForge remote (`git lfs env` shows the endpoint), "
                f"then restart, {copy_hint}")
    return f"run `git lfs install && git lfs pull --include={rel}/**` in {PROJECT_ROOT}, {copy_hint}"


LAST_CHECK: Optional[ModelCheck] = None     # result of the most recent check in this process (for /system/health)


def ensure_embedding_model(auto_fetch: bool = True, log=None) -> ModelCheck:
    """Verify the model on disk; if its weights are an LFS pointer / missing / wrong, try ``git lfs pull`` once."""
    global LAST_CHECK
    check = LAST_CHECK = _ensure(auto_fetch, log or (lambda _msg: None))
    return check


def _ensure(auto_fetch: bool, say) -> ModelCheck:
    check = ModelCheck(ok=False, model_dir=model_dir())
    _inspect(check)
    if check.ok:
        check.steps.append("verified")
        return check

    attempted, git_error = False, ""
    fetchable = WEIGHTS_FILE in check.problem          # only the weights are LFS objects; the small files come with git
    if auto_fetch and fetchable and _git_lfs_available():
        say(f"{check.problem} - fetching it with git lfs pull from the repository's LFS remote...")
        attempted = True
        pulled, git_error = _git_lfs_pull(check)
        check.steps.append("git lfs pull " + ("ok" if pulled else f"failed: {git_error}"))
        _inspect(check)
        if check.ok:
            check.fetched_with_git_lfs = True
            return check
    check.fix = _fix_for(check, attempted, git_error)
    return check
