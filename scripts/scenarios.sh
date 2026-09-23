#!/usr/bin/env bash
# Town10 scenario set: record traffic in CARLA, bake it into a scene archive
# with its own Cosmos3 first image, then replay it while OmniDreams renders.
#
#   scripts/scenarios.sh CONFIG STAGE [NAME...]
#
# STAGE: record  pass 1 — Traffic Manager drive, every actor logged
#                (outputs/scenarios/<name>/tracks.json) → archive.usdz
#        image   Cosmos3 first image for every archive in one batch (stops the
#                model server: the raster-only server needs its port)
#        run     model server up; replay each log with run --record-compare
#        all     record + image + run
# NAME defaults to every scenario below. Needs CARLA up (carla-transfer carla up).
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=$1
STAGE=$2
shift 2
CT=${CT:-.venv/bin/carla-transfer}
BASE=${BASE_ARCHIVE:-outputs/scenes/town10.usdz}
OUT=outputs/scenarios
# Extra carla-transfer-cosmos3 args; on a 96 GB card nothing needs offloading.
COSMOS3_ARGS=${COSMOS3_ARGS:---offload none}

# name -> "SECONDS|`carla-transfer drive` flags" for pass 1. The hero always
# starts at the archive's initial pose (what the first image shows); from
# there the road bends left for 168 m before the first junction (junction 23,
# which has no right turn: TM ignores --route Right there) and its crosswalk,
# so the junction scenarios drive faster and longer to reach it.
declare -A SCENARIOS=(
  [dense]="10|--traffic 30 --seed 7"
  [walkers]="10|--walkers 6 --walkers-side both --traffic 6 --seed 2"
  [fast]="10|--target-speed 50 --traffic 8 --seed 9"
  [junction_straight]="28|--target-speed 45 --route Straight --traffic 10 --seed 11"
  [left_turn]="28|--target-speed 45 --route Left --traffic 16 --seed 5"
  [dart_traffic]="25|--target-speed 45 --dart-crosswalk --traffic 8 --seed 4"
)
secs() { echo "${SCENARIOS[$1]%%|*}"; }
flags() { echo "${SCENARIOS[$1]#*|}"; }
NAMES=("$@")
[[ ${#NAMES[@]} -eq 0 ]] && NAMES=(dense walkers fast junction_straight left_turn dart_traffic)

# Per-scenario config: its archive, sent inline (1 MB) so one model server
# serves every scenario without a restart.
scenario_config() {
  local name=$1 dir=$OUT/$1
  python3 - "$CONFIG" "$dir/config.yaml" "$dir/archive.usdz" <<'PY'
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
c.setdefault("omnidreams", {}).update(scene_archive=sys.argv[3], scene_transfer="inline")
yaml.safe_dump(c, open(sys.argv[2], "w"), sort_keys=False)
PY
}

record() {
  local name=$1 dir=$OUT/$1
  mkdir -p "$dir"
  cp "$BASE" "$dir/archive.usdz"
  scenario_config "$name"
  echo "=== record $name ($(secs "$name") s): $(flags "$name")"
  # shellcheck disable=SC2046
  $CT -c "$dir/config.yaml" drive --spawn-at-archive-start $(flags "$name") \
    --record-tracks "$dir/tracks.json" --sim-seconds $(($(secs "$name") + 2)) --no-realtime --destroy \
    > "$dir/record.log" 2>&1
  $CT add-obstacles "$dir/archive.usdz" "$dir/tracks.json"
}

image() {
  # One raster server and one Cosmos3 load for every named scenario.
  local archives=()
  for n in "$@"; do archives+=("$OUT/$n/archive.usdz"); done
  echo "=== image $*"
  # shellcheck disable=SC2086
  $CT -c "$CONFIG" first-image "${archives[@]}" --cosmos3-args $COSMOS3_ARGS \
    > "$OUT/image.log" 2>&1
}

run() {
  local name=$1 dir=$OUT/$1
  echo "=== run $name"
  rm -f "$dir/go"
  $CT -c "$dir/config.yaml" drive --replay-tracks "$dir/tracks.json" --wait-for "$dir/go" \
    --destroy > "$dir/replay.log" 2>&1 &
  local driver=$!
  sleep 5
  timeout 900 $CT -c "$dir/config.yaml" run --follow --go-file "$dir/go" --sim-seconds "$(secs "$name")" \
    --record "$dir/generated.mp4" --record-compare "$dir/compare.mp4" > "$dir/run.log" 2>&1 \
    || echo "run $name failed (see $dir/run.log)"
  wait $driver || true
  grep "chunks gen" "$dir/run.log" | tail -1 || true
}

case $STAGE in
  record) for n in "${NAMES[@]}"; do record "$n"; done ;;
  image)
    $CT -c "$CONFIG" down
    image "${NAMES[@]}" ;;
  run)
    $CT -c "$CONFIG" up
    for n in "${NAMES[@]}"; do run "$n"; done ;;
  all)
    for n in "${NAMES[@]}"; do record "$n"; done
    $CT -c "$CONFIG" down
    image "${NAMES[@]}"
    $CT -c "$CONFIG" up
    for n in "${NAMES[@]}"; do run "$n"; done ;;
  *) echo "unknown stage $STAGE" >&2; exit 2 ;;
esac
echo "SCENARIOS_DONE $STAGE"
