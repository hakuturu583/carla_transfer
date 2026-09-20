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
    omnidreams.py    gRPC client for flashdreams' WorldModelService
    cosmos3.py       HTTP client for vLLM-Omni /v1/videos/sync (video-to-video)
    mock.py          echo backend for tests and CARLA-side bring-up
  docker/manager.py  build / up / down / logs via the docker CLI
  carla/source.py    CarlaCameraSource: sensor.camera.rgb → Frame (+ ego Pose)
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
export HF_TOKEN=hf_...   # read access to nvidia/omni-dreams-models

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
| `omnidreams` | `python -m omnidreams.grpc.server` (flashdreams) | gRPC `WorldModelService` | yes – 512x288 measured 29 fps in the interactive demo (bf16, LightVAE, torch.compile) |
| `cosmos3` | `vllm serve nvidia/Cosmos3-Nano --omni` | HTTP `/v1/videos/sync` | no – diffusion, seconds per chunk; offline-quality path |
| `mock` | – | – | – |

## Status / open points

- **HD map conditioning** (OmniDreams): `OmniDreamsBackend.hdmap_parquets`
  is empty. CARLA's map has to be exported to the parquet layout the server
  expects. Dynamic actors (`DynamicWorldState`) are likewise not populated yet.
- The OmniDreams gRPC server accepts `--resolution 480p|720p|704p`; the
  512x288 RTX 3090 manifest is only wired into the `interactive-drive`
  demo, so the gRPC path on a 24 GB card is unverified.
- First container start pays the model download and a ~7 min
  torch.compile; the Inductor cache is persisted under
  `~/.cache/carla_transfer/torchinductor` so later starts are fast.
