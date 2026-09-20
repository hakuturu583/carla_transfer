#!/usr/bin/env bash
# Log in to Hugging Face when a token is provided, then exec the server.
set -euo pipefail
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
cd /opt/flashdreams/integrations/omnidreams
exec "$@"
