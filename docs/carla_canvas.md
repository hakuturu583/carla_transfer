# carla_transfer as a carla_canvas renderer

[carla_canvas](https://github.com/hakuturu583/carla_canvas) runs a
[typesafe_carla](https://github.com/hakuturu583/typesafe_carla) program
**unchanged** and has its CARLA cameras rendered by a renderer service over
gRPC. `carla_transfer.carla_canvas.CarlaTransferRenderer` makes carla_transfer's
world model (OmniDreams) that renderer: the program's `sensor.camera.rgb`
delivers generated, photorealistic frames instead of CARLA's.

```
typesafe_carla program ── carla_canvas plugin ──gRPC──▶ carla_canvas service ──gRPC──▶ OmniDreams
   (unchanged)               (claims the camera)          ├─ ROS 2 topics            (Docker, GPU:
                                                          └─ CarlaTransferRenderer    carla-transfer-server)
                                                               camera tick → rig pose
                                                               f-theta → CARLA pinhole
```

carla_canvas owns everything CARLA-specific (sensor attributes, measurement
layout, `listen()` / `enable_for_ros()`), so `sensor.enable_for_ros()` in the
program gets CARLA's topics with the generated images.

## What it renders

| | |
|---|---|
| sensors | `sensor.camera.rgb`, one camera (one model session on one GPU). Depth, segmentation and LiDARs stay CARLA's (list only the camera under `sensors:`) |
| poses | `pose: camera` (default): the rig is placed so that the archive's camera (`archive_camera`, the clipgt calibration: 1.7 m ahead, 1.45 m up, 120° f-theta) is exactly at the CARLA camera's pose. `pose: vehicle`: the camera's parent vehicle is the rig, the camera is the archive's mount (what `carla-transfer run` does) |
| frame | the sensor's (or vehicle's) CARLA world transform from carla_canvas's snapshot, converted by carla_canvas into the OpenDRIVE frame (y negated) the archive is in, since the renderer declares `coordinate_frame = "right_handed"`: archives from `carla-transfer prepare-scene` / `export-map` of the same map. Only the rig offset (`archive_camera`) is computed here |
| images | `camera_model: pinhole` (default): the model's f-theta image resampled into the CARLA camera's fov and size (a camera wider than the model's coverage gets black borders, logged); `ftheta`: resized only |
| timing | timestamps are sim time since the session's first tick; set `fixed_delta_seconds` to `1 / fps` of the model (30 Hz) |

### Delay

OmniDreams renders a chunk of poses at once (5, then 8), so a tick cannot be
answered with its own frame: tick *k* delivers the model's frame of tick
*k − 7* (the measurement's `frame`, timestamp and transform are tick *k*'s).
Until then the camera shows the archive's `first_image.png`. With
`pipelined_chunks: n` the model generates behind the ticks instead of blocking
the tick that completes a chunk (8·n more ticks of delay) — for asynchronous
mode or a program ticking faster than the model.

## Use

```sh
uv sync --extra carla_canvas        # (or carla_canvas_ros for the ROS 2 output)
carla-transfer -c configs/omnidreams_rtx3090.yaml up    # the world model container
carla-canvas serve --config configs/canvas.yaml --endpoint 0.0.0.0:50052
# on the client: CARLA_CANVAS_CONFIG=canvas.yaml typesafe-codon run main.py
```

```yaml
renderer:
  backend: carla_transfer.carla_canvas:CarlaTransferRenderer
  endpoint: "gpu-host:50052"
  timeout: 120                     # a Render call may wait for a chunk
  startup_timeout: 300             # LoadWorld opens the session (scene, prompt embedding)
  assets:
    maps:
      "Town10HD*": {scene: scenes/town10.usdz}   # glob on the map name -> scene archive
  options:
    world_model: "127.0.0.1:50051" # the OmniDreams server (default: $CARLA_TRANSFER_WORLD_MODEL)
    transfer_config: omnidreams_rtx3090.yaml     # optional carla_transfer YAML: width, height, fps, omnidreams.*
    omnidreams: {positive_prompt: "..."}         # over its omnidreams section
    pose: camera                   # camera | vehicle
    camera_model: pinhole          # pinhole | ftheta
    pipelined_chunks: 0
    world_model_wait: 30           # seconds LoadWorld waits for the model server
    archive_camera: {position: [1.7, 0.0, 1.45], roll_pitch_yaw: [0, 0, 0], fov: 120}
    # backend: mock                # bring-up without a GPU (tinted black frames)
sensors:
  - sensor.camera.rgb
```

The archive goes to the model with the session (`omnidreams.scene_transfer:
inline`, CARLA archives are a few hundred KB), so the model container needs no
mount. A new episode (`load_world`) or a new camera opens a new session.
[configs/canvas.yaml](../configs/canvas.yaml) is a complete example.

## Docker

`docker/Dockerfile.canvas` is the carla_canvas service (no GPU, ~python:slim);
the compose profile `canvas` runs it in front of the OmniDreams container:

```sh
carla-transfer build omnidreams canvas
# CANVAS_DIR: canvas.yaml (endpoint aside, the same file as the client's) and its archives
CANVAS_DIR=$PWD/canvas docker compose -f docker/docker-compose.yaml --profile canvas up
# the client: renderer.endpoint: "<host>:50052" (CANVAS_PORT), no renderer.command
```

The service reaches the model at `omnidreams:50051` (`CARLA_TRANSFER_WORLD_MODEL`);
asset paths in the file are relative to it (`/config` in the container). For
the ROS 2 topics, run the canvas service with `network_mode: host` (DDS
discovery) and `CARLA_TRANSFER_WORLD_MODEL=127.0.0.1:50051`. Without compose:

```sh
docker run --rm -p 50052:50051 -v $PWD/canvas:/config \
  -e CARLA_TRANSFER_WORLD_MODEL=<model-host>:50051 carla-transfer/canvas:latest
```

## Limits

* The delay above; frames between chunks are the model's, not CARLA's.
* The model starts from the archive's start pose and first image: build the
  archive with a route starting where the program's ego spawns
  (`prepare-scene --start-lane` / `--route`), or the first chunks catch up.
* No live actors and no traffic-light colours: the carla_canvas protocol
  carries actor transforms but OmniDreams takes dynamic objects from the
  archive only (`add-obstacles`), and light states are not in the snapshots.
* One RGB camera per service.
