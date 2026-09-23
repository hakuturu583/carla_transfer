#!/usr/bin/env bash
# Log in to Hugging Face when a token is provided, then exec the server.
set -euo pipefail
if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
# flashdreams main keeps OmniDreams under integrations_v2/, older branches
# under integrations/ (manifest paths resolve relative to it there).
for dir in /opt/flashdreams/integrations_v2/omnidreams /opt/flashdreams/integrations/omnidreams; do
  if [[ -d "$dir" ]]; then cd "$dir"; break; fi
done
exec "$@"
