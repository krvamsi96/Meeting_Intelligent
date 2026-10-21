#!/bin/bash
# deploy_model.sh - copy the local embedding model to a Linux server (rsync over SSH).
#
# Usage: bash deploy_model.sh <server_user> <server_host> [deploy_path]
# Example: bash deploy_model.sh ubuntu my-server.example.org /opt/ai_meeting
#
# Run it from the project root of a machine whose models/embeddings/all-MiniLM-L6-v2/model.safetensors is the REAL
# ~91 MB file. There is deliberately no default host: a wrong default would silently overwrite another server.

if [ -z "$1" ] || [ -z "$2" ]; then
  echo "Usage: bash deploy_model.sh <server_user> <server_host> [deploy_path]"
  exit 2
fi
SERVER_USER="$1"
SERVER_IP="$2"
DEPLOY_PATH="${3:-/app}"
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" || exit 1
MODEL_SRC="./models/embeddings/all-MiniLM-L6-v2"
MODEL_DST="$DEPLOY_PATH/models/embeddings/all-MiniLM-L6-v2"

echo "====================================================="
echo " AI Meeting Intelligence — Model Deployment"
echo "====================================================="
echo " Source : $MODEL_SRC"
echo " Target : $SERVER_USER@$SERVER_IP:$MODEL_DST"
echo ""

# The weights are tracked with Git LFS. A checkout made without `git lfs pull` holds a ~130-byte text pointer with the
# same file name - uploading it would break the server ("Error while deserializing header: header too large").
WEIGHTS_SIZE=$(wc -c < "$MODEL_SRC/model.safetensors" 2>/dev/null | tr -d ' ')
if [ -z "$WEIGHTS_SIZE" ] || [ "$WEIGHTS_SIZE" -lt 1000000 ]; then
  echo "[ERROR] $MODEL_SRC/model.safetensors is missing or only ${WEIGHTS_SIZE:-0} bytes (a Git LFS pointer, not the ~91 MB model)."
  echo "        Run 'git lfs install && git lfs pull' here first, then re-run this script."
  exit 1
fi

LOCAL_SHA=$( (sha256sum "$MODEL_SRC/model.safetensors" 2>/dev/null || shasum -a 256 "$MODEL_SRC/model.safetensors") | cut -d' ' -f1)
echo " SHA-256: $LOCAL_SHA"

# Create directory on server
echo "[1/3] Creating target directory on server..."
ssh "$SERVER_USER@$SERVER_IP" "mkdir -p $MODEL_DST"

# Upload model files (rsync for efficiency)
echo "[2/3] Uploading model files..."
rsync -avz --progress \
  "$MODEL_SRC/" \
  "$SERVER_USER@$SERVER_IP:$MODEL_DST/"

# Verify on server
echo "[3/3] Verifying upload..."
ssh "$SERVER_USER@$SERVER_IP" "
  echo 'Files in model directory:'
  ls -lh $MODEL_DST/
  echo ''
  echo 'Checking critical files:'
  REMOTE_SHA=\$(sha256sum $MODEL_DST/model.safetensors 2>/dev/null | cut -d' ' -f1)
  test \"\$REMOTE_SHA\" = \"$LOCAL_SHA\" && echo 'model.safetensors: OK (SHA-256 matches the source)' || echo 'model.safetensors: MISMATCH - the copy is incomplete or not the real weights'
  test -f $MODEL_DST/config.json && echo 'config.json: OK' || echo 'config.json: MISSING'
  test -f $MODEL_DST/tokenizer.json && echo 'tokenizer.json: OK' || echo 'tokenizer.json: MISSING'
  echo ''
  echo 'Model deployment complete!'
"
