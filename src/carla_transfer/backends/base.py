"""Backend interface shared by OmniDreams, Cosmos3, and the mock."""

from __future__ import annotations

import abc
from collections.abc import Sequence

from carla_transfer.frames import Frame


class WorldModelBackend(abc.ABC):
    """A session-oriented, chunked video generator.

    Lifecycle: ``open(initial_frame)`` once, then ``generate(sim_frames)`` per
    chunk, then ``close()``. ``generate`` is blocking; the pipeline runs it on a
    worker thread so the CARLA tick loop keeps its cadence.
    """

    #: Number of simulator frames the backend wants per ``generate`` call.
    frames_per_chunk: int = 1

    @abc.abstractmethod
    def open(self, initial_frame: Frame) -> None: ...

    @abc.abstractmethod
    def generate(self, sim_frames: Sequence[Frame]) -> list[Frame]:
        """Turn one chunk of simulator frames into generated frames."""

    @abc.abstractmethod
    def close(self) -> None: ...

    def wait_ready(self, timeout_s: float) -> None:
        """Block until the server answers, or raise TimeoutError."""
        return None

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()
