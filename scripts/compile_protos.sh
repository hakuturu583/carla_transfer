#!/usr/bin/env bash
# Regenerate the vendored OmniDreams gRPC stubs.
#
# The .proto files are copied verbatim from
# flashdreams/integrations/omnidreams/omnidreams/grpc/protos (Apache-2.0),
# with only the import paths rewritten so they live under this package.
# The proto *package* names are untouched, so the stubs stay wire-compatible
# with the upstream WorldModelService server.
set -euo pipefail
cd "$(dirname "$(realpath "$0")")/.."
uv run --extra dev python -m grpc_tools.protoc \
    -I src \
    --python_out=src \
    --grpc_python_out=src \
    --pyi_out=src \
    src/carla_transfer/protos/omnidreams/*.proto
