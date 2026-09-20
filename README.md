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
  server/            carla-transfer-server: runs INSIDE the container on top of the
                     flashdreams interactive-drive world model (RTX 3090 manifest,
                     --offload-text-encoder), speaking the same WorldModelService proto
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
export HF_TOKEN=hf_...   # read access to nvidia/omni-dreams-models + omni-dreams-scenes
# point configs/omnidreams_rtx3090.yaml `omnidreams.scene_archive` at a clipgt .usdz

carla-transfer -c configs/omnidreams_rtx3090.yaml build      # docker build
carla-transfer -c configs/omnidreams_rtx3090.yaml up         # docker run + wait for gRPC
carla-transfer -c configs/omnidreams_rtx3090.yaml run --mjpeg :8080 --record outputs/run.mp4
carla-transfer -c configs/omnidreams_rtx3090.yaml down
```

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

## Status / open points

- **Not yet run on the GPU** from this repo: `carla-transfer build` (clones
  the fork + `uv sync`, several GB) and `up` have not been executed. The server
  was exercised end to end against a real `omni-dreams-scenes` archive with the
  world model stubbed (scene load, initial-frame/prompt override, 5-then-8
  chunking, close/reset).
- **HD map from CARLA**: the scene archive is still an
  `nvidia/omni-dreams-scenes` one. The conditioning data inside is small
  (`clipgt/lane.parquet` 0.9 MB, `obstacle.parquet` 4.5 MB, calibration,
  `metadata.yaml`); the 2.3 GB is meshes/checkpoint the rasterizer does not
  need. Exporting CARLA's OpenDRIVE map into that layout is the next step
  (`synthetic_scene.py` in the fork is the template).
- **Dynamic actors**: the rasterizer reads vehicle tracks from the archive at
  load time, so live CARLA vehicles are not rendered yet.
- First container start pays the model download and a ~7 min
  torch.compile; the Inductor cache is persisted under
  `~/.cache/carla_transfer/torchinductor` so later starts take ~30 s.
