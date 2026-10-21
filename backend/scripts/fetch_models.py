"""Provision the offline model files once, on a machine with internet access (then copy ``models/`` to the server).

    python -m backend.scripts.fetch_models --embedding     # all-MiniLM-L6-v2 -> models/embeddings/  (no Git LFS needed)
    python -m backend.scripts.fetch_models --diarization   # NeMo TitaNet speaker model -> models/diarization/*.nemo
    python -m backend.scripts.fetch_models --all

Why it exists: the API runs fully offline (OFFLINE_MODE=true) and these directories are not in the repository
(``models/diarization`` is git-ignored; the embedding weights are committed to git as a normal file). Without them:
speaker counts come from a weak fallback embedder, and there are no embeddings for RAG. Speech-to-text is handled
via the configured AI speech-to-text engine.

Every download is verified (size + SHA-256 for the embedding weights, which the repository's own LFS pointer pins).
The script contacts Hugging Face only; nothing is sent to it but the download requests. Honours HTTPS_PROXY.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

# The repository's Git LFS pointer for models/embeddings/all-MiniLM-L6-v2/model.safetensors pins these.
EMBEDDING_REPO = "sentence-transformers/all-MiniLM-L6-v2"
from backend.services.embedding_model_guard import DEFAULT_MODEL_NAME, PINNED_WEIGHTS  # noqa: E402

EMBEDDING_WEIGHTS_SHA256, EMBEDDING_WEIGHTS_SIZE = PINNED_WEIGHTS[DEFAULT_MODEL_NAME]
EMBEDDING_FILES = [
    "config.json", "config_sentence_transformers.json", "data_config.json", "model.safetensors", "modules.json",
    "sentence_bert_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt",
    "1_Pooling/config.json",
]


def _allow_network() -> None:
    """The backend forces HF offline mode at import; this tool is the one place that must go online."""
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    import huggingface_hub.constants as hub_constants

    hub_constants.HF_HUB_OFFLINE = False


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_embedding(target: str) -> bool:
    from huggingface_hub import snapshot_download

    print(f"[embedding] downloading {EMBEDDING_REPO} -> {target}")
    os.makedirs(target, exist_ok=True)
    snapshot_download(repo_id=EMBEDDING_REPO, local_dir=target, allow_patterns=EMBEDDING_FILES)
    weights = os.path.join(target, "model.safetensors")
    size = os.path.getsize(weights)
    if size != EMBEDDING_WEIGHTS_SIZE:
        print(f"[embedding] FAILED: model.safetensors is {size} bytes, expected {EMBEDDING_WEIGHTS_SIZE}.")
        return False
    if _sha256(weights) != EMBEDDING_WEIGHTS_SHA256:
        print("[embedding] FAILED: model.safetensors SHA-256 does not match the checksum pinned by the repository.")
        return False
    print(f"[embedding] OK - {size} bytes, SHA-256 verified.")
    return True


def fetch_diarization(target: str, model: str) -> bool:
    try:
        import nemo.collections.asr as nemo_asr
    except Exception as exc:  # noqa: BLE001
        print(f"[diarization] NeMo is not installed here ({type(exc).__name__}). Run this on a Linux machine with "
              "`pip install -r requirements.txt` (nemo_toolkit[asr]), or copy an existing titanet_large.nemo.")
        return False
    os.makedirs(target, exist_ok=True)
    out = os.path.join(target, f"{model}.nemo")
    print(f"[diarization] downloading NeMo '{model}' -> {out}")
    nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(model_name=model).save_to(out)
    if not os.path.isfile(out) or os.path.getsize(out) < 1_000_000:
        print("[diarization] FAILED: the saved .nemo file is missing or too small.")
        return False
    print(f"[diarization] OK - {os.path.getsize(out)} bytes.")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--models-dir", default=None, help="override the models directory (default: from configuration)")
    args = parser.parse_args(argv)
    if not (args.embedding or args.diarization or args.all):
        parser.print_help()
        return 2

    _allow_network()
    from backend.config.config import settings

    base = args.models_dir
    embedding_dir = os.path.join(base, "embeddings", settings.EMBEDDING_MODEL_NAME) if base else settings.EMBEDDING_MODEL_PATH
    diarization_dir = os.path.join(base, "diarization") if base else settings.DIARIZATION_MODEL_PATH

    results = {}
    if args.embedding or args.all:
        results["embedding"] = fetch_embedding(embedding_dir)
    if args.diarization or args.all:
        results["diarization"] = fetch_diarization(diarization_dir, settings.NEMO_SPEAKER_MODEL)

    print("\n" + ", ".join(f"{k}: {'OK' if v else 'FAILED'}" for k, v in results.items()))
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
