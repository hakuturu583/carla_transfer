"""carla_transfer as a carla_canvas renderer: pose/camera math, and the whole
chain (carla_canvas service → CarlaTransferRenderer → OmniDreams client →
a fake WorldModelService enforcing the 5-then-8 chunks)."""

from __future__ import annotations

import math
import socket
import zipfile
from concurrent import futures
from types import SimpleNamespace

import grpc
import numpy as np
import pytest

pytest.importorskip("carla_canvas")

from carla_canvas.config import CanvasConfig, RosConfig  # noqa: E402
from carla_canvas.geometry import Transform, quaternion_xyzw  # noqa: E402
from carla_canvas.server import start  # noqa: E402
from carla_canvas.v1 import renderer_pb2 as pb  # noqa: E402
from carla_canvas.v1 import renderer_pb2_grpc as pb_grpc  # noqa: E402

from carla_transfer.carla.source import pose_from_transform  # noqa: E402
from carla_transfer.carla_canvas.frames import (  # noqa: E402
    ArchiveCamera,
    FThetaToPinhole,
    delivery_delay,
)
from carla_transfer.carla_canvas.renderer import CarlaTransferRenderer  # noqa: E402
from carla_transfer.frames import encode_image  # noqa: E402
from carla_transfer.protos.omnidreams import common_pb2, video_model_pb2  # noqa: E402
from carla_transfer.protos.omnidreams import video_model_pb2_grpc as vm_grpc  # noqa: E402
from carla_transfer.server.geometry import quat_xyzw_to_matrix  # noqa: E402

# --- frames -----------------------------------------------------------------------


@pytest.mark.parametrize("rot", [(0, 0, 0), (10, 30, -5), (-20, 135, 40), (5, -90, 0)])
def test_canvas_pose_matches_pose_from_transform(rot):
    """carla_canvas's CARLA → right-handed conversion is the one `run` uses."""
    loc = (12.0, -3.5, 1.2)
    t = Transform(loc, rot)
    fake = SimpleNamespace(
        location=SimpleNamespace(x=loc[0], y=loc[1], z=loc[2]),
        rotation=SimpleNamespace(pitch=rot[0], yaw=rot[1], roll=rot[2]),
    )
    expected = pose_from_transform(fake)
    m = t.matrix("right_handed")
    assert np.allclose(m[:3, 3], expected.xyz)
    assert np.allclose(m[:3, :3], quat_xyzw_to_matrix(expected.quat_xyzw), atol=1e-6)
    assert np.isclose(np.linalg.det(m[:3, :3]), 1.0)
    q = quaternion_xyzw(m[:3, :3])
    assert np.allclose(quat_xyzw_to_matrix(q), m[:3, :3], atol=1e-6)


def test_rig_from_camera_puts_the_archive_camera_on_the_carla_camera():
    cam = ArchiveCamera(position=(1.7, 0.2, 1.45), roll_pitch_yaw=(0, -5, 3))
    carla_cam = Transform((30.0, 4.0, 2.1), (-8.0, 60.0, 0.0)).matrix("right_handed")
    rig = cam.rig_from_camera(carla_cam)
    assert np.allclose(rig @ cam.camera_to_rig(), carla_cam)
    # a level camera on the default mount: the rig is 1.45 m below it
    rig = ArchiveCamera().rig_from_camera(Transform((0, 0, 1.45)).matrix("right_handed"))
    assert np.allclose(rig[:3, 3], (-1.7, 0.0, 0.0))


def test_ftheta_to_pinhole_maps_angles():
    sw, sh, fov = 640, 352, 120.0
    f_src = (sw / 2) / math.radians(fov / 2)
    # each f-theta column holds its angle off axis in degrees (+90 offset)
    u = np.arange(sw) + 0.5 - sw / 2
    angle = np.degrees(u / f_src)
    src = np.repeat(np.clip(angle + 90, 0, 255).astype(np.uint8)[None, :, None], sh, 0)
    src = np.repeat(src, 3, 2)
    remap = FThetaToPinhole(sw, sh, fov, 800, 450, 90.0)  # 16:9 like the model
    out = remap(src)
    assert remap.coverage == 1.0
    w = 800
    f_dst = w / (2 * math.tan(math.radians(45)))
    for col in (0, 150, 399, 650, 799):
        expected = math.degrees(math.atan((col + 0.5 - w / 2) / f_dst)) + 90
        assert abs(int(out[225, col, 0]) - expected) <= 1.5  # uint8 steps
    # a pinhole wider than the model's image is partly black
    wide = FThetaToPinhole(sw, sh, fov, 400, 400, 140.0)
    assert wide.coverage < 1.0 and wide(src)[0, 0].tolist() == [0, 0, 0]


def test_delivery_delay():
    assert delivery_delay(5, 8) == 7
    assert delivery_delay(5, 8, 1) == 15
    assert delivery_delay(10, 8) == 9
    assert delivery_delay(1, 1) == 0
    with pytest.raises(ValueError):
        delivery_delay(5, 0)


# --- the whole chain --------------------------------------------------------------


def _level(i: int) -> int:
    """The grey level the fake model paints frame i with (JPEG-safe)."""
    return 20 + (i * 6) % 220


class FakeWorldModel(vm_grpc.WorldModelServiceServicer):
    W, H = 64, 36

    def __init__(self):
        self.sessions = []
        self.chunks: list[list] = []
        self.closed = []

    def get_version(self, request, context):
        return common_pb2.VersionId()

    def start_session(self, request, context):
        self.sessions.append(request)
        self.chunks = []
        return video_model_pb2.SessionId(session_id=f"s{len(self.sessions)}")

    def render_video_chunk(self, request, context):
        poses = list(request.rig_trajectory.poses)
        expected = 5 if not self.chunks else 8
        if len(poses) != expected:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"{len(poses)} poses")
        first = sum(len(c) for c in self.chunks)
        self.chunks.append(poses)
        frames = [
            video_model_pb2.Image(
                data=encode_image(np.full((self.H, self.W, 3), _level(first + i), np.uint8)),
                format=video_model_pb2.JPEG,
            )
            for i in range(len(poses))
        ]
        return video_model_pb2.VideoChunkReturn(
            camera_outputs=[
                video_model_pb2.CameraOutput(camera_logical_id="front", rgb_frames=frames)
            ]
        )

    def close_session(self, request, context):
        self.closed.append(request.session_id)
        return common_pb2.Empty()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _actor(actor_id, type_id, parent_id=0, **attributes) -> pb.Actor:
    a = pb.Actor(id=actor_id, type_id=type_id, parent_id=parent_id)
    for k, v in attributes.items():
        a.attributes.add(id=k, value=str(v))
    return a


VEHICLE, CAMERA = 10, 11


def _request(frame: int, x: float, new=()) -> pb.RenderRequest:
    r = pb.RenderRequest()
    r.snapshot.id = 1
    r.snapshot.timestamp.frame = frame
    r.snapshot.timestamp.elapsed_seconds = frame / 30.0
    r.snapshot.timestamp.delta_seconds = 1 / 30.0
    for actor_id, t in {
        VEHICLE: Transform((x, 2.0, 0.0), (0.0, 90.0, 0.0)),
        CAMERA: Transform((x, 2.0, 1.45), (0.0, 90.0, 0.0)),
    }.items():
        t.to_proto(r.snapshot.actors.add(id=actor_id).transform)
    r.new_actors.extend(new)
    r.sensors.add(sensor_id=CAMERA, deliver=True)
    return r


@pytest.fixture
def world_model():
    svc = FakeWorldModel()
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    vm_grpc.add_WorldModelServiceServicer_to_server(svc, srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    srv.start()
    yield svc, port
    srv.stop(0)


@pytest.fixture
def canvas(tmp_path, world_model):
    """(stub, fake world model, renderer) for a carla_canvas service running
    CarlaTransferRenderer with the given options."""
    svc, port = world_model
    archive = tmp_path / "town10.usdz"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("prompt1.txt", "a sunny town")
        zf.writestr("first_image.png", encode_image(np.full((36, 64, 3), 3, np.uint8), "PNG"))
    started = []

    def make(**options):
        endpoint = f"127.0.0.1:{_free_port()}"
        renderer_config = {
            "assets": {"maps": {"Town10HD*": {"scene": "town10.usdz"}}},
            "options": {"world_model": f"127.0.0.1:{port}", "width": 64, "height": 36,
                        **options},
        }  # fmt: skip
        config = CanvasConfig(
            endpoint=endpoint,
            renderer=renderer_config,
            ros=RosConfig(backend="none"),
            path=tmp_path / "canvas.yaml",
        )
        renderer = CarlaTransferRenderer()
        server, _, _ = start(renderer, endpoint, config)
        channel = grpc.insecure_channel(endpoint)
        started.append((server, channel, renderer))
        return pb_grpc.RendererStub(channel), svc, renderer

    yield make
    for server, channel, renderer in started:
        channel.close()
        server.stop(None)
        renderer.close()


def _image(data) -> np.ndarray:
    img = data.image
    return np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)


@pytest.mark.parametrize("pipelined", [0, 1])
def test_service_delivers_model_frames_with_a_fixed_delay(canvas, pipelined):
    stub, model, _ = canvas(camera_model="ftheta", pipelined_chunks=pipelined)
    stub.LoadWorld(pb.LoadWorldRequest(map_name="Carla/Maps/Town10HD_Opt", episode_id=1))
    assert len(model.sessions) == 1  # opened by LoadWorld
    session = model.sessions[0]
    assert session.static_world_map.hdmap_parquets.startswith(b"PK")  # inline archive
    assert session.text_prompt.positive == "a sunny town"
    assert not session.initial_frames  # the archive's first image
    stub.SpawnSensor(
        pb.SpawnSensorRequest(
            sensor=_actor(
                CAMERA, "sensor.camera.rgb", VEHICLE, image_size_x=64, image_size_y=36, fov=120
            )
        )
    )
    delay = 7 + 8 * pipelined
    new = [_actor(VEHICLE, "vehicle.ego", role_name="hero")]
    for k in range(40):
        r = stub.Render(_request(100 + k, x=0.5 * k, new=new if k == 0 else ()))
        (data,) = r.measurements
        assert data.frame == 100 + k
        bgra = _image(data)
        if k < delay:
            assert abs(int(bgra[18, 32, 2]) - 3) <= 2  # first_image.png
        else:
            assert abs(int(bgra[18, 32, 2]) - _level(k - delay)) <= 3, k
    # 5, then 8 per chunk, as consecutive poses with timestamps from 0
    sizes = [len(c) for c in model.chunks]
    assert sizes[0] == 5 and set(sizes[1:]) == {8}
    poses = [p for c in model.chunks for p in c]
    assert [p.timestamp_us for p in poses[:3]] == [0, 33333, 66667]
    # pose: camera (default) puts the archive camera (1.7 m ahead, 1.45 m up)
    # on the CARLA camera: CARLA yaw 90 at y = 2 is OpenDRIVE heading -90 at y = -2
    p = poses[2].pose
    assert p.vec.x == pytest.approx(1.0) and p.vec.y == pytest.approx(-2.0 + 1.7)
    assert p.vec.z == pytest.approx(0.0, abs=1e-9)
    assert (p.quat.z, p.quat.w) == pytest.approx((-math.sqrt(0.5), math.sqrt(0.5)))


def test_vehicle_pose_and_pinhole_camera(canvas):
    stub, model, _ = canvas(pose="vehicle")
    stub.LoadWorld(pb.LoadWorldRequest(map_name="Carla/Maps/Town10HD_Opt", episode_id=1))
    stub.SpawnSensor(
        pb.SpawnSensorRequest(
            sensor=_actor(
                CAMERA, "sensor.camera.rgb", VEHICLE, image_size_x=200, image_size_y=100, fov=90
            )
        )
    )
    new = [_actor(VEHICLE, "vehicle.ego")]
    for k in range(13):
        r = stub.Render(_request(k, x=float(k), new=new if k == 0 else ()))
    bgra = _image(r.measurements[0])
    assert bgra.shape == (100, 200, 4)  # resampled to the CARLA camera
    assert abs(int(bgra[50, 100, 2]) - _level(12 - 7)) <= 3
    p = model.chunks[1][0].pose  # frame 5: the vehicle at x = 5
    assert (p.vec.x, p.vec.y, p.vec.z) == pytest.approx((5.0, -2.0, 0.0))


def test_new_episode_and_new_camera_restart_the_session(canvas):
    stub, model, _ = canvas()
    stub.LoadWorld(pb.LoadWorldRequest(map_name="Carla/Maps/Town10HD_Opt", episode_id=1))
    stub.SpawnSensor(
        pb.SpawnSensorRequest(
            sensor=_actor(
                CAMERA, "sensor.camera.rgb", VEHICLE, image_size_x=64, image_size_y=36, fov=90
            )
        )
    )
    stub.Render(_request(1, 0.0, new=[_actor(VEHICLE, "vehicle.ego")]))
    stub.LoadWorld(pb.LoadWorldRequest(map_name="Carla/Maps/Town10HD_Opt", episode_id=2))
    assert model.closed == ["s1"] and len(model.sessions) == 2
    with pytest.raises(grpc.RpcError) as e:
        stub.LoadWorld(pb.LoadWorldRequest(map_name="Carla/Maps/Town01", episode_id=3))
    assert e.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert "no scene archive" in e.value.details()


def test_mock_backend_needs_no_archive(tmp_path):
    from carla_canvas.geometry import PinholeCamera
    from carla_canvas.renderer import CameraView, WorldInfo
    from carla_canvas.scene import Scene

    r = CarlaTransferRenderer({"backend": "mock", "width": 32, "height": 18})
    r.load_world(WorldInfo("Carla/Maps/Town01", 1, {}, tmp_path))
    scene = Scene()
    for k in range(9):
        scene.frame, scene.elapsed_seconds = k, k / 30
        frame = r.render_camera(CameraView(PinholeCamera(32, 18, 90.0), scene,
                                           frozenset({"rgb"}), sensor_id=5))  # fmt: skip
    assert frame.rgb.shape == (18, 32, 3)
    assert frame.rgb[9, 16].tolist() == [40, 0, 0]  # the mock's tint, from frame 1 on
    r.close()


def test_example_config_resolves():
    from pathlib import Path

    from carla_canvas.config import load_config
    from carla_canvas.renderer import WorldInfo

    path = Path(__file__).resolve().parents[1] / "configs" / "canvas.yaml"
    config = load_config(path)
    assert config.backend == "carla_transfer.carla_canvas:CarlaTransferRenderer"
    r = CarlaTransferRenderer(config.options)
    world = WorldInfo("Carla/Maps/Town10HD_Opt", 1, config.renderer, path.parent)
    cfg = r._transfer_config(world)
    assert cfg.resolution == (512, 288)  # from omnidreams_rtx3090.yaml
    archive = path.parents[1] / "outputs" / "scenes" / "town10.usdz"
    assert cfg.omnidreams.scene_archive.resolve() == archive
    assert cfg.omnidreams.scene_transfer == "inline"


def test_one_camera_at_a_time(tmp_path):
    from carla_canvas.geometry import PinholeCamera
    from carla_canvas.renderer import CameraView, WorldInfo
    from carla_canvas.scene import Scene

    r = CarlaTransferRenderer({"backend": "mock", "width": 32, "height": 18})
    r.load_world(WorldInfo("Carla/Maps/Town01", 1, {}, tmp_path))
    scene = Scene()

    def render(sensor_id, frame):
        scene.frame, scene.elapsed_seconds = frame, frame / 30
        cam = PinholeCamera(32, 18, 90.0)
        return r.render_camera(CameraView(cam, scene, frozenset({"rgb"}), sensor_id=sensor_id))

    render(5, 1)
    with pytest.raises(ValueError, match="one camera"):
        render(6, 1)  # both alive
    session = r._session
    render(6, 10)  # 5 is gone: a new session for 6
    assert r._session is not session and r._session.sensor_id == 6
    r.close()


def test_transfer_config_port_is_kept_when_only_the_host_defaults(tmp_path, monkeypatch):
    from carla_canvas.renderer import WorldInfo

    monkeypatch.delenv("CARLA_TRANSFER_WORLD_MODEL", raising=False)
    (tmp_path / "t.yaml").write_text("backend: mock\nomnidreams: {port: 50099}\n")
    r = CarlaTransferRenderer({"transfer_config": "t.yaml"})
    cfg = r._transfer_config(WorldInfo("Carla/Maps/Town01", 1, {}, tmp_path))
    assert (cfg.omnidreams.host, cfg.omnidreams.port) == ("127.0.0.1", 50099)
    r.options["world_model"] = "model:6000"  # an explicit endpoint wins
    cfg = r._transfer_config(WorldInfo("Carla/Maps/Town01", 1, {}, tmp_path))
    assert (cfg.omnidreams.host, cfg.omnidreams.port) == ("model", 6000)


class _HangingBackend:
    """generate() blocks until close() (like a gRPC call whose channel closes),
    or forever with stubborn=True."""

    frames_per_chunk, first_chunk_frames = 8, 5

    def __init__(self, stubborn=False):
        import threading

        self.closed = threading.Event()
        self.stubborn = stubborn
        self.release = threading.Event()  # lets a stubborn generate() end at teardown

    def chunk_size(self, index):
        return self.first_chunk_frames if index == 0 else self.frames_per_chunk

    def wait_ready(self, timeout_s):
        pass

    def open(self, frame):
        pass

    def generate(self, frames):
        (self.release if self.stubborn else self.closed).wait()
        raise RuntimeError("cancelled")

    def close(self):
        self.closed.set()


@pytest.mark.parametrize("stubborn", [False, True])
def test_closing_a_session_does_not_wait_forever_for_a_generation(tmp_path, stubborn):
    import time

    from carla_canvas.geometry import PinholeCamera
    from carla_canvas.renderer import CameraView, WorldInfo
    from carla_canvas.scene import Scene

    backends = []

    def factory(config):
        backends.append(_HangingBackend(stubborn))
        return backends[-1]

    r = CarlaTransferRenderer(
        {"backend": "mock", "width": 32, "height": 18, "generate_timeout": 0.2,
         "close_timeout": 0.5}
    )  # fmt: skip
    r.backend_factory = factory
    r.load_world(WorldInfo("Carla/Maps/Town01", 1, {}, tmp_path))
    scene = Scene()
    try:
        # tick 7 waits for the first chunk (concurrent.futures' TimeoutError: py3.10)
        with pytest.raises(futures.TimeoutError):
            for k in range(8):
                scene.frame, scene.elapsed_seconds = k, k / 30
                r.render_camera(
                    CameraView(PinholeCamera(32, 18, 90.0), scene, frozenset({"rgb"}), 5)
                )
        t0 = time.monotonic()
        r.load_world(WorldInfo("Carla/Maps/Town01", 2, {}, tmp_path))  # closes the session
        assert time.monotonic() - t0 < 3.0
        assert backends[0].closed.is_set() and len(backends) == 2
        r.close()
    finally:
        for b in backends:
            b.release.set()
