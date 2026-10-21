import base64
import binascii
import hashlib
import logging
import os
import threading

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)

# Master key file location
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
# Persist key inside mounted docker volume if available
if os.path.exists("/app/data") and os.path.isdir("/app/data"):
    KEY_FILE = "/app/data/.secret_key"
else:
    KEY_FILE = os.path.join(project_root, ".secret_key")

_fernet_instance = None
_fernet_lock = threading.Lock()


def _is_valid_fernet_key(raw: bytes) -> bool:
    """A Fernet key is exactly 32 bytes, url-safe base64 encoded."""
    try:
        return len(base64.urlsafe_b64decode(raw)) == 32
    except (binascii.Error, ValueError):
        return False


def _derive_fernet_key(secret: bytes) -> bytes:
    """Deterministically turn an arbitrary passphrase into a valid Fernet key.

    ``SECRET_KEY`` in .env is commonly a free-form string (e.g. ``openssl rand -hex 32``),
    which Fernet rejects outright. Deriving via SHA-256 lets any non-empty secret work
    while a properly generated Fernet key is still used verbatim (no change for
    deployments whose stored ciphertext was produced with such a key).
    """
    return base64.urlsafe_b64encode(hashlib.sha256(secret).digest())


def _load_or_create_key_file() -> bytes:
    """Read the persisted master key, creating it atomically (0600) on first use."""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, "rb") as f:
            key = f.read().strip()
        if not key:
            raise ValueError("Key file is empty")
        return key

    logger.info("Generating a new master key for API key encryption...")
    key = Fernet.generate_key()
    try:
        # O_EXCL: if two workers race, exactly one creates the file; the loser re-reads it
        # instead of overwriting the winner's key (which would orphan already-encrypted data).
        fd = os.open(KEY_FILE, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
    except FileExistsError:
        with open(KEY_FILE, "rb") as f:
            existing = f.read().strip()
        if not existing:
            raise ValueError("Key file is empty")
        return existing
    with os.fdopen(fd, "wb") as f:
        f.write(key)
    return key


def _get_fernet() -> Fernet:
    global _fernet_instance
    if _fernet_instance is not None:
        return _fernet_instance

    with _fernet_lock:
        if _fernet_instance is not None:
            return _fernet_instance
        try:
            env_key = os.getenv("SECRET_KEY")
            if env_key and env_key.strip():
                raw = env_key.encode("utf-8").strip()
                key = raw if _is_valid_fernet_key(raw) else _derive_fernet_key(raw)
            else:
                key = _load_or_create_key_file()
                if not _is_valid_fernet_key(key):
                    raise ValueError("Key file does not contain a valid Fernet key")

            _fernet_instance = Fernet(key)
            return _fernet_instance
        except Exception as e:
            logger.error(f"Failed to initialize encryption key: {e}")
            raise RuntimeError(
                "Encryption key could not be loaded and no secure fallback is permitted. "
                "Ensure .secret_key exists or SECRET_KEY is set in the environment."
            ) from e


def encrypt(data: str) -> str:
    """Encrypt a string using the master key.

    Fails CLOSED: if encryption is impossible this raises rather than handing the
    plaintext back — callers persist the return value as ``encrypted_config``, so
    returning the input would silently store API keys unencrypted.
    """
    if not data:
        return ""
    try:
        f = _get_fernet()
        token = f.encrypt(data.encode("utf-8"))
        return token.decode("utf-8")
    except Exception as e:
        logger.error(f"Encryption failed: {e}")
        raise RuntimeError("Encryption failed; refusing to store data unencrypted.") from e


def decrypt(token: str) -> str:
    """Decrypt a token using the master key."""
    if not token:
        return ""
    try:
        f = _get_fernet()
        data = f.decrypt(token.encode("utf-8"))
        return data.decode("utf-8")
    except Exception as e:
        err_detail = str(e) or type(e).__name__
        logger.error(f"Decryption failed: {err_detail}")
        return ""
