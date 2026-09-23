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
# traffic_light layer: "carla" (signal heads recorded from CARLA) or "map"
# (the OpenDRIVE export's lights), see `carla-transfer add-obstacles -h`.
TRAFFIC_LIGHTS=${TRAFFIC_LIGHTS:-carla}
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
  # Junction 23 from 45 m out (base town10_tl, below). red_stop_go: the
  # hero's light is red for 10 s, the hero (obeying lights) stops at the
  # line, then goes when it turns green.
  [tl_left]="15|--route Left --traffic 8 --seed 21"
  [tl_straight]="15|--route Straight --traffic 8 --seed 22"
  [red_stop_go]="20|--ignore-lights-pct 0 --red-light-for 10 --route Straight --traffic 6 --seed 31"
  # The same without traffic: nothing ahead of the hero moves off first, so
  # a colour change in the video can only come from the light / prompt.
  [red_stop_go_solo]="20|--ignore-lights-pct 0 --red-light-for 10 --route Straight --seed 31"
  # A long red: the hero waits 15 s while the crossing roads (green for them)
  # stream past in front of it.
  [red_cross_traffic]="22|--ignore-lights-pct 0 --red-light-for 15 --route Straight --traffic 14 --seed 41"
)
# Scenarios on another base archive than BASE_ARCHIVE. town10_tl starts 45 m
# before junction 23 (Town10's signalized junction on the default route):
#   carla-transfer -c CONFIG export-map --xodr Town10HD_Opt.xodr \
#     --out outputs/scenes/town10_tl.usdz --clip-id town10 \
#     --route lane/17/0,lane/7/5,lane/24/0,lane/22/0
TL_BASE=${TL_BASE_ARCHIVE:-outputs/scenes/town10_tl.usdz}
declare -A BASES=(
  [tl_left]=$TL_BASE
  [tl_straight]=$TL_BASE
  [red_stop_go]=$TL_BASE
  [red_stop_go_solo]=$TL_BASE
  [red_cross_traffic]=$TL_BASE
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

# Anything a failed or killed drive/run left in CARLA (a leftover hero blocks
# the next replay's spawn), and a synchronous world nobody ticks.
cleanup() {
  $CT carla cleanup > /dev/null 2>&1 || echo "carla cleanup failed" >&2
}

record() {
  local name=$1 dir=$OUT/$1
  cleanup
  local base=${BASES[$name]:-$BASE}
  if [[ ! -f $base ]]; then
    echo "$name: base archive $base missing (see BASES above for how to export it)" >&2
    return 1
  fi
  mkdir -p "$dir"
  cp "$base" "$dir/archive.usdz"
  scenario_config "$name"
  echo "=== record $name ($(secs "$name") s): $(flags "$name")"
  # shellcheck disable=SC2046
  $CT -c "$dir/config.yaml" drive --spawn-at-archive-start $(flags "$name") \
    --record-tracks "$dir/tracks.json" --sim-seconds $(($(secs "$name") + 2)) --no-realtime --destroy \
    > "$dir/record.log" 2>&1
  $CT add-obstacles "$dir/archive.usdz" "$dir/tracks.json" --traffic-lights "$TRAFFIC_LIGHTS"
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
  cleanup
  rm -f "$dir/go"
  $CT -c "$dir/config.yaml" drive --replay-tracks "$dir/tracks.json" --wait-for "$dir/go" \
    --destroy > "$dir/replay.log" 2>&1 &
  local driver=$!
  sleep 5
  timeout 900 $CT -c "$dir/config.yaml" run --follow --go-file "$dir/go" --sim-seconds "$(secs "$name")" \
    --record "$dir/generated.mp4" --record-compare "$dir/compare.mp4" \
    --dump-trajectory "$dir/trajectory.json" > "$dir/run.log" 2>&1 \
    || { echo "run $name failed (see $dir/run.log)"; kill "$driver" 2>/dev/null; }
  wait $driver || true
  cleanup
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
