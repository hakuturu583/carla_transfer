"""HTTP client for a vLLM-Omni server hosting ``nvidia/Cosmos3-Nano``.

Uses the video-to-video mode of ``POST /v1/videos/sync``: a chunk of CARLA
frames is muxed to MP4, uploaded as ``input_reference``, and the returned MP4
is demuxed back into frames. Cosmos3 is a diffusion model, not a streaming
one, so this backend is offline-quality / non-real-time on a single 3090; it
exists so both models can be driven through the same pipeline.
"""

from __future__ import annotations

import io
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import httpx
import numpy as np

from carla_transfer.backends.base import WorldModelBackend
from carla_transfer.config import TransferConfig
from carla_transfer.frames import Frame

log = logging.getLogger(__name__)


class Cosmos3Backend(WorldModelBackend):
    def __init__(self, config: TransferConfig):
        self.config = config
        self.c3 = config.cosmos3
        self.frames_per_chunk = self.c3.num_frames
        self._client = httpx.Client(base_url=self.c3.base_url, timeout=self.c3.request_timeout_s)
        self._opened = False

    def wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                r = self._client.get("/health", timeout=5)
                if r.status_code == 200:
                    log.info("Cosmos3 server ready")
                    return
            except httpx.HTTPError:
                pass
            if time.monotonic() > deadline:
                raise TimeoutError("Cosmos3 server did not come up")
            time.sleep(2)

    def open(self, initial_frame: Frame) -> None:
        self._opened = True

    def generate(self, sim_frames: Sequence[Frame]) -> list[Frame]:
        if not self._opened:
            raise RuntimeError("generate() before open()")
        w, h = self.config.resolution
        frames = [f.resized(w, h) for f in sim_frames]
        mp4 = _encode_mp4([f.rgb for f in frames], self.config.fps)
        extra = {
            "use_resolution_template": False,
            "use_duration_template": False,
            # Keep every uploaded frame as clean conditioning: the model
            # re-renders the CARLA content rather than continuing it.
            "condition_frame_indexes_vision": list(range(len(frames))),
            "condition_video_keep": True,
        }
        data = {
            "prompt": self.c3.prompt,
            "negative_prompt": self.c3.negative_prompt,
            "size": f"{w}x{h}",
            "num_frames": str(len(frames)),
            "fps": str(self.config.fps),
            "num_inference_steps": str(self.c3.num_inference_steps),
            "guidance_scale": str(self.c3.guidance_scale),
            "extra_params": json.dumps(extra),
        }
        r = self._client.post(
            "/v1/videos/sync",
            data=data,
            files={"input_reference": ("chunk.mp4", mp4, "video/mp4")},
        )
        r.raise_for_status()
        out_rgb = _decode_mp4(r.content)
        out: list[Frame] = []
        for i, rgb in enumerate(out_rgb):
            ts = frames[min(i, len(frames) - 1)].timestamp_us
            out.append(Frame(rgb, ts, None, {"generated": True}))
        return out

    def close(self) -> None:
        self._opened = False
        self._client.close()


def _encode_mp4(rgbs: list[np.ndarray], fps: int) -> bytes:
    import tempfile

    from carla_transfer.hdmap.control_video import write_video

    with tempfile.TemporaryDirectory() as tmp:
        return write_video(rgbs, Path(tmp) / "chunk.mp4", fps).read_bytes()


def _decode_mp4(data: bytes) -> list[np.ndarray]:
    import imageio.v3 as iio

    return [np.asarray(f) for f in iio.imread(io.BytesIO(data), extension=".mp4", plugin="pyav")]
