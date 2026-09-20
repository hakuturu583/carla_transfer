"""Real-time loop: simulator frames in, generated frames out.

Frames are pushed with :meth:`StreamPipeline.push`. They are grouped into
chunks of ``backend.frames_per_chunk`` and handed to a worker thread that
calls ``backend.generate``. Generated frames are delivered to sinks at the
configured fps; if the model falls behind, the newest chunk wins and older
queued chunks are dropped so latency stays bounded (``drop_stale``).
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from carla_transfer.backends import WorldModelBackend, make_backend
from carla_transfer.config import TransferConfig
from carla_transfer.frames import Frame

log = logging.getLogger(__name__)

Sink = Callable[[Frame], None]


@dataclass
class PipelineStats:
    chunks_submitted: int = 0
    chunks_generated: int = 0
    chunks_dropped: int = 0
    frames_out: int = 0
    generate_times_s: list[float] = field(default_factory=list)

    @property
    def mean_generate_s(self) -> float:
        t = self.generate_times_s[-50:]
        return sum(t) / len(t) if t else 0.0

    @property
    def effective_fps(self) -> float:
        return (
            0.0
            if not self.generate_times_s
            else (self._frames_per_chunk / self.mean_generate_s if self.mean_generate_s else 0.0)
        )

    _frames_per_chunk: int = 1


class StreamPipeline:
    def __init__(
        self,
        config: TransferConfig,
        backend: WorldModelBackend | None = None,
        sinks: Iterable[Sink] = (),
        drop_stale: bool = True,
        max_pending_chunks: int = 2,
    ):
        self.config = config
        self.backend = backend or make_backend(config)
        self.sinks: list[Sink] = list(sinks)
        self.drop_stale = drop_stale
        self.stats = PipelineStats(_frames_per_chunk=self.backend.frames_per_chunk)
        self._pending: collections.deque[tuple[int, Sequence[Frame]] | None] = collections.deque()
        self._max_pending = max_pending_chunks
        self._cv = threading.Condition()
        self._chunk: list[Frame] = []
        self._worker: threading.Thread | None = None
        self._opened = False
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def start(self, initial_frame: Frame) -> None:
        self.backend.open(initial_frame.resized(*self.config.resolution))
        self._opened = True
        self._worker = threading.Thread(target=self._run, name="carla-transfer-gen", daemon=True)
        self._worker.start()

    def stop(self, flush: bool = True) -> None:
        if not self._opened:
            return
        if flush and self._chunk:
            self._submit(self._chunk)
            self._chunk = []
        self._enqueue(None)
        if self._worker:
            self._worker.join()
        self.backend.close()
        self._opened = False

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.stop(flush=exc[0] is None)

    # -- input ---------------------------------------------------------------

    def push(self, frame: Frame) -> None:
        """Feed one simulator frame. Non-blocking."""
        if not self._opened:
            raise RuntimeError("push() before start()")
        self._chunk.append(frame.resized(*self.config.resolution))
        if len(self._chunk) >= self.backend.chunk_size(self.stats.chunks_submitted):
            self._submit(self._chunk)
            self._chunk = []

    def _submit(self, chunk: list[Frame]) -> None:
        index = self.stats.chunks_submitted
        self.stats.chunks_submitted += 1
        self._enqueue((index, chunk))

    def _enqueue(self, item: tuple[int, Sequence[Frame]] | None) -> None:
        with self._cv:
            if item is not None and self.drop_stale:
                while len(self._pending) >= self._max_pending:
                    # The first chunk defines the session (OmniDreams' 5-frame
                    # initial chunk); never drop it. Drop the newcomer instead.
                    if self._pending[0] is not None and self._pending[0][0] == 0:
                        self.stats.chunks_dropped += 1
                        return
                    self._pending.popleft()
                    self.stats.chunks_dropped += 1
            elif item is not None:
                self._cv.wait_for(lambda: len(self._pending) < self._max_pending)
            self._pending.append(item)
            self._cv.notify_all()

    def _dequeue(self) -> tuple[int, Sequence[Frame]] | None:
        with self._cv:
            self._cv.wait_for(lambda: len(self._pending) > 0)
            item = self._pending.popleft()
            self._cv.notify_all()
            return item

    # -- worker --------------------------------------------------------------

    def _run(self) -> None:
        period = 1.0 / self.config.fps if self.config.fps else 0.0
        while True:
            item = self._dequeue()
            if item is None:
                return
            _, chunk = item
            t0 = time.perf_counter()
            try:
                out = self.backend.generate(chunk)
            except Exception:
                log.exception("generate failed; dropping chunk")
                continue
            self.stats.generate_times_s.append(time.perf_counter() - t0)
            self.stats.chunks_generated += 1
            next_t = time.perf_counter()
            for f in out:
                self._emit(f)
                next_t += period
                delay = next_t - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

    def _emit(self, frame: Frame) -> None:
        self.stats.frames_out += 1
        for sink in self.sinks:
            try:
                sink(frame)
            except Exception:
                log.exception("sink %r failed", sink)
