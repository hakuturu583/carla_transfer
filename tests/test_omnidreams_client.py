"""Exercise the OmniDreams gRPC client against an in-process fake server built
from the same vendored stubs, so request/response shapes are checked end to end."""

from concurrent import futures

import grpc
import numpy as np
import pytest

from carla_transfer.backends.omnidreams import OmniDreamsBackend
from carla_transfer.config import BackendKind, TransferConfig
from carla_transfer.frames import Frame, Pose, decode_image, encode_image
from carla_transfer.protos.omnidreams import common_pb2, video_model_pb2
from carla_transfer.protos.omnidreams import video_model_pb2_grpc as vm_grpc


class FakeService(vm_grpc.WorldModelServiceServicer):
    def __init__(self):
        self.session_req = None
        self.chunk_reqs = []
        self.closed = []

    def get_version(self, request, context):
        return common_pb2.VersionId()

    def start_session(self, request, context):
        self.session_req = request
        return video_model_pb2.SessionId(session_id="s1")

    def render_video_chunk(self, request, context):
        self.chunk_reqs.append(request)
        n = len(request.rig_trajectory.poses)
        spec = self.session_req.camera_specs[0]
        w, h = spec.resolution_w, spec.resolution_h
        frames = [
            video_model_pb2.Image(
                data=encode_image(np.full((h, w, 3), 7, np.uint8)), format=video_model_pb2.JPEG
            )
            for _ in range(n)
        ]
        return video_model_pb2.VideoChunkReturn(
            camera_outputs=[
                video_model_pb2.CameraOutput(camera_logical_id="front", rgb_frames=frames)
            ]
        )

    def close_session(self, request, context):
        self.closed.append(request.session_id)
        return common_pb2.Empty()


@pytest.fixture
def server():
    svc = FakeService()
    srv = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    vm_grpc.add_WorldModelServiceServicer_to_server(svc, srv)
    port = srv.add_insecure_port("127.0.0.1:0")
    srv.start()
    yield svc, port
    srv.stop(0)


def _cfg(port, tmp_path):
    cfg = TransferConfig(backend=BackendKind.OMNIDREAMS, width=64, height=32)
    cfg.omnidreams.port = port
    cfg.omnidreams.positive_prompt = "sunny"
    cfg.omnidreams.initial_image = "carla"
    scene = tmp_path / "scene.usdz"
    scene.write_bytes(b"PK-fake-clipgt")
    cfg.omnidreams.scene_archive = scene
    return cfg


def test_round_trip(server, tmp_path):
    svc, port = server
    be = OmniDreamsBackend(_cfg(port, tmp_path))
    be.wait_ready(5)
    init = Frame(np.zeros((32, 64, 3), np.uint8), 0)
    with be:
        be.open(init)
        chunk = [
            Frame(np.zeros((32, 64, 3), np.uint8), i * 1000, Pose((i, 0, 0), (0, 0, 0, 1)))
            for i in range(8)
        ]
        out = be.generate(chunk)

    assert svc.session_req.static_world_map.hdmap_parquets == b"scene://scene.usdz"
    assert svc.session_req.text_prompt.positive == "sunny"
    assert svc.session_req.camera_specs[0].logical_id == "camera_front_wide_120fov"
    assert svc.session_req.camera_specs[0].resolution_w == 64
    assert decode_image(svc.session_req.initial_frames[0].data).shape == (32, 64, 3)
    poses = svc.chunk_reqs[0].rig_trajectory.poses
    assert [p.timestamp_us for p in poses] == [i * 1000 for i in range(8)]
    assert poses[3].pose.vec.x == 3
    assert len(out) == 8 and out[0].rgb[0, 0, 0] == 7 and out[5].timestamp_us == 5000
    assert svc.closed == ["s1"]


def test_inline_scene_transfer_sends_bytes(server, tmp_path):
    svc, port = server
    cfg = _cfg(port, tmp_path)
    cfg.omnidreams.scene_transfer = "inline"
    with OmniDreamsBackend(cfg) as be:
        be.open(Frame(np.zeros((32, 64, 3), np.uint8), 0))
    assert svc.session_req.static_world_map.hdmap_parquets == b"PK-fake-clipgt"


def test_initial_image_archive_sends_no_frame(server, tmp_path):
    svc, port = server
    cfg = _cfg(port, tmp_path)
    cfg.omnidreams.initial_image = "archive"
    with OmniDreamsBackend(cfg) as be:
        be.open(Frame(np.zeros((32, 64, 3), np.uint8), 0))
    assert len(svc.session_req.initial_frames) == 0


def test_open_without_scene_archive_fails(server, tmp_path):
    _, port = server
    cfg = _cfg(port, tmp_path)
    cfg.omnidreams.scene_archive = None
    be = OmniDreamsBackend(cfg)
    with pytest.raises(RuntimeError, match="scene_archive"):
        be.open(Frame(np.zeros((32, 64, 3), np.uint8), 0))


def test_pipeline_sends_5_then_8_poses(server, tmp_path):
    from carla_transfer.pipeline import StreamPipeline

    svc, port = server
    cfg = _cfg(port, tmp_path)
    cfg.fps = 1000
    got = []
    backend = OmniDreamsBackend(cfg)
    with StreamPipeline(cfg, backend=backend, sinks=[got.append], drop_stale=False) as pipe:
        pipe.start(Frame(np.zeros((32, 64, 3), np.uint8), 0))
        for i in range(5 + 8 + 8):
            pipe.push(Frame(np.zeros((32, 64, 3), np.uint8), i * 1000))
    assert [len(r.rig_trajectory.poses) for r in svc.chunk_reqs] == [5, 8, 8]
    assert len(got) == 21
