import time

import numpy as np
import pytest

from carla_transfer import Frame, StreamPipeline, TransferConfig
from carla_transfer.backends.mock import MockBackend
from carla_transfer.config import BackendKind


def _frame(i: int, w: int = 64, h: int = 32) -> Frame:
    return Frame(np.full((h, w, 3), i % 255, dtype=np.uint8), timestamp_us=i * 33_333)


def _cfg(**kw) -> TransferConfig:
    cfg = TransferConfig(backend=BackendKind.MOCK, width=64, height=32, fps=1000)
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def test_pipeline_emits_every_frame_when_model_keeps_up():
    cfg = _cfg()
    got: list[Frame] = []
    backend = MockBackend(cfg)
    with StreamPipeline(cfg, backend=backend, sinks=[got.append]) as pipe:
        pipe.start(_frame(0))
        for i in range(16):
            pipe.push(_frame(i))
    assert len(got) == 16
    assert all(f.meta.get("generated") for f in got)
    assert backend.calls == 2
    assert pipe.stats.chunks_dropped == 0
    assert pipe.stats.frames_out == 16
    # tint applied and resolution enforced
    assert got[0].rgb.shape == (32, 64, 3)
    assert got[0].rgb[0, 0, 0] == 40


def test_output_pacing_does_not_stall_generation():
    # 8 frames at 20 fps take 0.4 s to present; generation must not wait for
    # that, so 4 chunks at 0.1 s each finish in ~0.4 s, not ~2 s.
    cfg = _cfg(fps=20)
    backend = MockBackend(cfg, latency_s=0.1)
    got: list[Frame] = []
    pipe = StreamPipeline(cfg, backend=backend, sinks=[got.append], drop_stale=False,
                          max_pending_chunks=100)  # fmt: skip
    pipe.start(_frame(0))
    t0 = time.perf_counter()
    for i in range(8 * 4):
        pipe.push(_frame(i))
    while pipe.stats.chunks_generated < 4 and time.perf_counter() - t0 < 5:
        time.sleep(0.01)
    assert time.perf_counter() - t0 < 1.0
    pipe.stop()
    assert len(got) == 32


def test_pipeline_drops_stale_chunks_when_model_is_slow():
    cfg = _cfg()
    got: list[Frame] = []
    backend = MockBackend(cfg, latency_s=0.05)
    pipe = StreamPipeline(cfg, backend=backend, sinks=[got.append], max_pending_chunks=1)
    pipe.start(_frame(0))
    for i in range(8 * 10):  # 10 chunks pushed instantly
        pipe.push(_frame(i))
    pipe.stop()
    assert pipe.stats.chunks_dropped > 0
    assert pipe.stats.chunks_generated + pipe.stats.chunks_dropped == 10
    assert len(got) == 8 * pipe.stats.chunks_generated


def test_pipeline_resizes_input_to_configured_resolution():
    cfg = _cfg()
    got: list[Frame] = []
    with StreamPipeline(cfg, backend=MockBackend(cfg, tint=False), sinks=[got.append]) as pipe:
        pipe.start(_frame(0, 640, 360))
        for i in range(8):
            pipe.push(_frame(i, 640, 360))
    assert got[0].rgb.shape == (32, 64, 3)


def test_sink_exception_does_not_kill_worker():
    cfg = _cfg()
    got: list[Frame] = []

    def bad(_):
        raise RuntimeError("boom")

    with StreamPipeline(cfg, backend=MockBackend(cfg), sinks=[bad, got.append]) as pipe:
        pipe.start(_frame(0))
        for i in range(8):
            pipe.push(_frame(i))
        time.sleep(0.05)
    assert len(got) == 8


def test_first_chunk_is_never_dropped():
    cfg = _cfg()
    backend = MockBackend(cfg, latency_s=0.05)
    backend.first_chunk_frames = 5
    seen: list[int] = []
    orig = backend.generate

    def spy(chunk):
        seen.append(len(chunk))
        return orig(chunk)

    backend.generate = spy
    pipe = StreamPipeline(cfg, backend=backend, sinks=[], max_pending_chunks=1)
    pipe.start(_frame(0))
    for i in range(5 + 8 * 6):
        pipe.push(_frame(i))
    pipe.stop()
    assert seen[0] == 5
    assert all(n == 8 for n in seen[1:])
    assert pipe.stats.chunks_dropped > 0


def test_compare_sink_pairs_input_by_timestamp(monkeypatch, tmp_path):
    from carla_transfer import sinks as S

    written: list[np.ndarray] = []
    monkeypatch.setattr(S.Mp4Sink, "__init__", lambda self, path, fps: None)
    monkeypatch.setattr(S.Mp4Sink, "__call__", lambda self, f: written.append(f.rgb))
    sink = S.CompareSink(tmp_path / "c.mp4", 30)
    sink.add_input(Frame(np.full((96, 64, 3), 10, np.uint8), timestamp_us=100))
    sink.add_input(Frame(np.full((96, 64, 3), 20, np.uint8), timestamp_us=200))
    hd = np.full((96, 64, 3), 30, np.uint8)
    sink(Frame(np.full((96, 64, 3), 40, np.uint8), 200, meta={"hdmap": hd}))
    (out,) = written
    assert out.shape == (96, 192, 3)
    assert out[48, 32, 0] == 20 and out[48, 96, 0] == 30 and out[48, 160, 0] == 40
    assert 100 not in sink._inputs


def test_discard_pending_drops_queued_and_partial_chunks():
    cfg = _cfg()
    backend = MockBackend(cfg, latency_s=0.2)
    got: list[Frame] = []
    pipe = StreamPipeline(cfg, backend=backend, sinks=[got.append], max_pending_chunks=1)
    pipe.start(_frame(0))
    for i in range(8):
        pipe.push(_frame(i))
    deadline = time.perf_counter() + 2
    while pipe._pending and time.perf_counter() < deadline:  # chunk 0 is generating
        time.sleep(0.005)
    for i in range(8, 8 * 3 + 3):  # one chunk queued (the newest), 3 frames partial
        pipe.push(_frame(i))
    assert pipe.discard_pending() == 1
    for i in range(100, 108):
        pipe.push(_frame(i))
    pipe.stop()
    # chunk 0 and the post-discard chunk, nothing from before the discard
    assert pipe.stats.chunks_generated == 2
    assert [f.timestamp_us for f in got[8:]] == [i * 33_333 for i in range(100, 108)]


@pytest.mark.parametrize("use_cv2", [True, False])
def test_ftheta_remap_puts_a_ray_at_its_ftheta_radius(monkeypatch, use_cv2):
    import builtins
    import math

    from carla_transfer.carla.source import FThetaRemap

    if not use_cv2:
        real_import = builtins.__import__

        def no_cv2(name, *a, **k):
            if name == "cv2":
                raise ImportError(name)
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_cv2)
    else:
        pytest.importorskip("cv2")
    remap = FThetaRemap(640, 352, 120.0)
    f_ftheta = 320 / math.radians(60)
    f_pin = (remap.pinhole_width / 2) / math.tan(math.radians(remap.pinhole_fov_deg / 2))
    # A bright ray 40 deg to the right of the axis, at the horizon.
    pin = np.zeros((remap.pinhole_height, remap.pinhole_width, 3), np.uint8)
    px = round(remap.pinhole_width / 2 + f_pin * math.tan(math.radians(40)))
    pin[remap.pinhole_height // 2 - 2 : remap.pinhole_height // 2 + 2, px - 2 : px + 2] = 255
    out = remap(pin)
    assert out.shape == (352, 640, 3)
    col = int(np.argmax(out[176, :, 0]))
    assert abs(col - (320 + f_ftheta * math.radians(40))) <= 2
