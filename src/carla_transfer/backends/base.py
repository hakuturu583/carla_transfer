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
    #: Size of the very first chunk after ``open`` (OmniDreams: 5, then 8).
    first_chunk_frames: int | None = None
    #: Whether a short trailing chunk may be sent on shutdown.
    accepts_partial_chunk: bool = True

    def chunk_size(self, chunk_index: int) -> int:
        if chunk_index == 0 and self.first_chunk_frames is not None:
            return self.first_chunk_frames
        return self.frames_per_chunk

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
