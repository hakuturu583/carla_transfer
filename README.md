# carla_transfer

Real-time sim-to-real video generation for [CARLA](https://carla.org/):
CARLA's front camera goes into a world model running in Docker
(**OmniDreams** via NVIDIA flashdreams, or **Cosmos3-Nano** via vLLM-Omni) and
photorealistic frames come back, at the simulator's frame rate.

```
CARLA (sync mode) ──RGB+pose──▶ StreamPipeline ──gRPC/HTTP──▶ [Docker: world model on GPU]
                                     │
                                     └──▶ window / MJPEG / mp4
```

The model never runs in this process. `carla_transfer` is the host-side
client, container manager, and real-time loop.

## Layout

```
src/carla_transfer/
  config.py          TransferConfig (YAML) – backend, resolution, fps, docker, per-backend settings
  frames.py          Frame / Pose containers + image codecs
  pipeline.py        StreamPipeline: chunking, worker thread, stale-chunk dropping, paced output
  sinks.py           WindowSink (cv2), MjpegSink (http), Mp4Sink (imageio)
  backends/
    base.py          WorldModelBackend: open(initial) / generate(chunk) / close()
    omnidreams.py    gRPC client for the OmniDreams WorldModelService (5-frame first chunk, then 8)
    cosmos3.py       HTTP client for vLLM-Omni /v1/videos/sync (video-to-video)
    mock.py          echo backend for tests and CARLA-side bring-up
  docker/manager.py  build / up / down / logs via the docker CLI
  carla/source.py    CarlaCameraSource: sensor.camera.rgb → Frame (+ ego Pose)
  hdmap/             CARLA .xodr → OmniDreams scene archive (roadgen), HD-map control
                     video, prepare-scene orchestration
  server/            run INSIDE the containers:
    omnidreams_server.py  WorldModelService over flashdreams' interactive-drive world
                          model (RTX 3090 manifest, --offload-text-encoder); --raster-only
    cosmos3_transfer.py   Cosmos3 transfer (world-scenario hint) → first_image.png
  docker/carla.py    CARLA server container, off-screen at -quality-level=Low
  protos/omnidreams/ vendored OmniDreams .proto + generated stubs (scripts/compile_protos.sh)
docker/              Dockerfile.omnidreams, Dockerfile.cosmos3, docker-compose.yaml
configs/             omnidreams_rtx3090.yaml, cosmos3_nano.yaml, mock.yaml
```

## Install

```bash
uv sync --extra dev            # library + tests
uv sync --extra display --extra video   # cv2 window + mp4 recording
uv sync --extra carla          # CARLA python client (needs a CARLA 0.9.16 server)
```

## Run

```bash
export HF_TOKEN=hf_...   # read access to nvidia/omni-dreams-models, nvidia/Cosmos3-Nano
C=configs/omnidreams_rtx3090.yaml

carla-transfer -c $C build omnidreams cosmos3        # docker build (fork flashdreams; vllm-omni + diffusers)

# CARLA map → scene archive with a Cosmos-Transfer2.5-rendered first frame
carla-transfer -c $C prepare-scene --xodr Town10HD_Opt.xodr --out scenes/town10.usdz --clip-id town10
# then set omnidreams.scene_archive: scenes/town10.usdz in the YAML

carla-transfer -c $C carla up                        # CARLA off-screen, -quality-level=Low
carla-transfer -c $C up                              # world model container + wait for gRPC
carla-transfer -c $C run --mjpeg :8080 --record outputs/run.mp4
carla-transfer -c $C down && carla-transfer -c $C carla down
```

`prepare-scene` = `export-map` (roadgen reads the OpenDRIVE, writes the ClipGT
layers, packed into a 240 KB archive) → **Cosmos-Transfer2.5-2B auto-multiview**
renders its world-scenario control video from the same ClipGT layers and
generates a photoreal drive along the route (front camera only, one GPU) → its
first frame is written into the archive as `first_image.png`. With
`omnidreams.initial_image: archive` (default) OmniDreams starts from that
image instead of CARLA's rendering. `--first-image none` stops after the
archive; `--first-image cosmos3` is the alternative path through the
OmniDreams raster-only server + Cosmos3 transfer (needs a bigger GPU).

`run` connects to CARLA at `--carla-host/--carla-port`, puts the world in
synchronous mode at `fps`, attaches a camera to the vehicle whose
`role_name` is `hero` (spawning one on autopilot if absent), and streams.

No GPU / no CARLA smoke test:

```bash
carla-transfer -c configs/mock.yaml selftest --mjpeg :8080
```

## Library use

```python
from carla_transfer import StreamPipeline, TransferConfig
from carla_transfer.sinks import MjpegSink

cfg = TransferConfig.from_yaml("configs/omnidreams_rtx3090.yaml")
with StreamPipeline(cfg, sinks=[MjpegSink(port=8080)]) as pipe:
    pipe.start(first_frame)  # Frame(rgb HxWx3 uint8, timestamp_us, pose)
    for frame in carla_frames:  # one per world.tick()
        pipe.push(frame)  # non-blocking; chunks go to the model thread
```

## Backends

| backend | server in container | transport | real-time on RTX 3090 |
|---|---|---|---|
| `omnidreams` | `carla-transfer-server` over flashdreams' interactive-drive world model | gRPC `WorldModelService` | yes – 512x288 measured 29 fps, 16 GiB peak (bf16, LightVAE, torch.compile, text encoder offloaded) |
| `cosmos3` | `vllm serve nvidia/Cosmos3-Nano --omni` | HTTP `/v1/videos/sync` | no – diffusion, seconds per chunk; offline-quality path |
| `mock` | – | – | – |

## How the RTX 3090 path works

flashdreams' stock `omnidreams.grpc.server` only knows the 480p/720p/704p
pipeline configs and keeps the ~15 GB Cosmos-Reason1 text encoder resident,
so it needs ~48 GB. The fork branch
[`hakuturu583/flashdreams@claude/omnidreams-rtx3090-ahncny`](https://github.com/hakuturu583/flashdreams/tree/claude/omnidreams-rtx3090-ahncny)
adds `example_world_model_rtx3090.yaml` (bf16 PyTorch path, LightVAE +
LightTAE, torch.compile + CUDA graphs, attention sinks, 512x288) for the
`interactive-drive` demo, driven with `--offload-text-encoder`. Measured there:

| `resolution_wh` | chunk (8 frames) | effective fps | VRAM peak |
|---|---|---|---|
| 512x288 | 275 ms | 29.1 | 16.3 GiB (init), ~12 GiB steady |
| 640x352 | 404 ms | 19.8 | 16.3 GiB |

`carla_transfer.server.omnidreams_server` wraps that demo's
`WorldModelRenderBackend` and exposes it behind the upstream
`WorldModelService` proto, so the host client is the same one that would talk
to the stock server. `docker/Dockerfile.omnidreams` clones the fork branch and
installs this package into its venv. `width`/`height`/`fps` in the YAML are
passed through as `--resolution`/`--fps` overrides of the manifest.

Session protocol: `start_session` takes the clipgt scene archive (as bytes, or
as `scene://<file>` resolved against the container's `/scenes` mount — the
official archives are 2.3 GB) plus the CARLA frame as the initial image and
the prompt; the first `render_video_chunk` must carry 5 poses, later ones 8.

## RTX PRO 6000 Blackwell (96 GB)

`configs/omnidreams_blackwell.yaml` uses `server/manifests/world_model_blackwell.yaml`:
the OmniDreams single-view native path (FP8 KV cache + cuDNN attention, SM 12.0
only), the text encoder kept resident, CARLA 0.10 at Low with UE's off-screen
60 fps cap lifted (`carla.launcher_args`). Measured with CARLA on the same card
(`drive --realtime` + `run --follow`, Town10):

| resolution | model ms / 8-frame chunk | end-to-end fps |
|---|---|---|
| **640x352** (default) | 126 | 29.8 |
| 896x496 | 274 | 22.9 |
| 1024x560 | 356 | 19.4 |
| 1280x704 | 705 | 10.0 |

The first start JIT-builds the native extension into
`~/.cache/carla_transfer/omnidreams_native`; `MAX_JOBS=4` keeps nvcc from
running out of host RAM on a 31 GB machine.

## CARLA map → scene archive

[roadgen](https://pypi.org/project/roadgen/) reads OpenDRIVE and writes the
ClipGT parquet layers the ludus rasterizer consumes (same naming as
cosmos-transfer2.5's `clipgt_loader`). `hdmap/xodr.py` first rewrites the
RoadRunner quirks in CARLA's Town maps that roadgen's strict OpenDRIVE 1.7
reader rejects (`<header><userData>` without `code`, `type="curb"` road marks
without `color`, `<object type="-1">`, outline corners without `height`); no
geometry is touched. `hdmap/clipgt.py` drives a route (successor lanes from
`--start-lane`, or `--route`) with one f-theta front camera and packs
`clipgt/*.parquet` + `metadata.yaml` + `rig_trajectories.json` + prompt +
placeholder image. Verified on the 3090: Town10HD_Opt.xodr → 240 KB archive,
loads in omnidreams' `load_scene_bundle` and ludus, 101 condition frames in 2 s.

## Cosmos-Transfer2.5 on the 3090

The first image comes from `nvidia/Cosmos-Transfer2.5-2B` `auto/multiview`,
whose control input is the HD map (world-scenario videos rendered from ClipGT
parquet by `scripts/generate_control_videos.py` in the
[cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)
checkout, `transfer25.repo`). `hdmap/transfer25.py` drives that checkout's
venv as subprocesses: render the front camera, write a one-view
`examples/multiview.py` spec (`num_conditional_frames: 0`), run it with
`torchrun --nproc_per_node=1`, take frame 0. Notes:

- roadgen writes no `obstacle.parquet`; the native renderer requires it, so an
  empty table with the documented schema is added.
- The checkout carries local (uncommitted) patches: `COSMOS_EVICT_TEXT_ENCODER=1`
  keeps the 7B Cosmos-Reason1 text encoder and the 2B denoiser from sharing
  the card (in `transfer2` and, added here, `transfer2_multiview` inference),
  and `world_scenario/utils/pcd_utils.py` triangulates polygons on their own
  plane — synthetic maps have exactly flat ground, and Delaunay on RDF x-y
  was degenerate for every intersection area.
- Cosmos3-Nano was dropped for this step: 30 GB bf16 / 16.5 GB fp8, and no
  FP8 GEMM on Ampere.

## Status / open points

- **Not yet run on the GPU** from this repo: the docker images and the
  world-model `up`/`run`. Verified pieces: the OmniDreams server against a
  real archive with the model stubbed; the raster-only server and control
  video on the real rasterizer (flashdreams venv, RTX 3090).
- **Dynamic actors**: the rasterizer reads vehicle tracks from the archive
  at load time (`clipgt/obstacle.parquet`); live CARLA vehicles are not
  rendered yet. The upstream proto has `DynamicWorldState` per chunk; the
  flashdreams side needs an API to update ludus cubes per chunk.
- First container start pays the model download and a ~7 min
  torch.compile; the Inductor cache is persisted under
  `~/.cache/carla_transfer/torchinductor` so later starts take ~30 s.

## 10 s scenario with traffic (two passes)

OmniDreams renders dynamic objects from the archive's `clipgt/obstacle.parquet`
(timestamped cuboid tracks, loaded once per scene), so live actors are
recorded first and replayed while the model runs:

```bash
# pass 1: Traffic Manager drives hero + traffic near it; every actor pose is logged
carla-transfer -c $C drive --spawn-at-archive-start --traffic 12 --seed 3 \
    --record-tracks scenes/town10_tracks.json --sim-seconds 12 --no-realtime --destroy
carla-transfer add-obstacles scenes/town10.usdz scenes/town10_tracks.json

# pass 2: model session opens, the driver holds every actor still until the
# first chunk exists (go file), then replays the log kinematically for 10 s
carla-transfer -c $C run --follow --go-file go --sim-seconds 10 \
    --record out.mp4 --record-input carla.mp4 &
carla-transfer -c $C drive --replay-tracks scenes/town10_tracks.json --wait-for go --tick-fps 18 --destroy
```

`run --go-file` rebases frame timestamps to the release, matching the track
log (t=0 at the first moving tick); standstill frames during the model's
warm-up are dropped, everything after the release is queued without loss.
Live per-chunk actors (the proto's `DynamicWorldState`) need an update API in
ludus/flashdreams and are the next step.

## Scenario set

`scripts/scenarios.sh CONFIG STAGE [NAME...]` runs the two-pass flow above for
a set of Town10 scenarios (dense traffic, crossing pedestrians, a fast hero,
straight through / left at the first junction, a crosswalk dart-out), each
under `outputs/scenarios/<name>/`:

- `record`: Traffic Manager drive with every actor logged, baked into a copy of
  the base archive (`add-obstacles`). Spawned actors are first held braked
  until they rest on the ground: spawns are lifted 0.3-1 m, and recording
  them in mid-air put floating cars into the first image.
- `image`: `carla-transfer first-image` renders each archive's control video and
  regenerates its Cosmos3 first image, so the cars at t=0 are this scenario's.
  All archives share one raster server and one Cosmos3 load, and only a
  9-frame / 20-step clip is generated (the first frame is all that is used):
  six scenarios take 3.5 min on the Blackwell card instead of 19.
- `run`: replays each log with `run --record-compare`, which writes
  [CARLA camera | HD-map condition | generated] side by side.
