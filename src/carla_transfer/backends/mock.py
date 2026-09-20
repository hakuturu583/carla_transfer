"""Backend that returns the input frames (optionally tinted). Used by tests
and for bringing up the CARLA side without a GPU."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np

from carla_transfer.backends.base import WorldModelBackend
from carla_transfer.config import TransferConfig
from carla_transfer.frames import Frame


class MockBackend(WorldModelBackend):
    def __init__(self, config: TransferConfig, latency_s: float = 0.0, tint: bool = True):
        self.config = config
        self.latency_s = latency_s
        self.tint = tint
        self.frames_per_chunk = config.omnidreams.frames_per_chunk
        self.opened = False
        self.calls = 0

    def open(self, initial_frame: Frame) -> None:
        self.opened = True

    def generate(self, sim_frames: Sequence[Frame]) -> list[Frame]:
        if not self.opened:
            raise RuntimeError("generate() before open()")
        self.calls += 1
        if self.latency_s:
            time.sleep(self.latency_s)
        out = []
        for f in sim_frames:
            f = f.resized(*self.config.resolution)
            rgb = f.rgb
            if self.tint:
                rgb = rgb.astype(np.uint16)
                rgb[:, :, 0] = np.minimum(rgb[:, :, 0] + 40, 255)
                rgb = rgb.astype(np.uint8)
            out.append(Frame(rgb, f.timestamp_us, f.pose, {"generated": True}))
        return out

    def close(self) -> None:
        self.opened = False
