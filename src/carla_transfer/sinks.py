"""Places generated frames can go: a window, an MJPEG stream, an mp4 file, a callback."""

from __future__ import annotations

import http.server
import logging
import threading
from pathlib import Path

import numpy as np

from carla_transfer.frames import Frame, encode_image

log = logging.getLogger(__name__)


class WindowSink:
    """OpenCV window. Needs the ``display`` extra."""

    def __init__(self, title: str = "carla_transfer"):
        import cv2

        self.cv2 = cv2
        self.title = title

    def __call__(self, frame: Frame) -> None:
        self.cv2.imshow(self.title, frame.rgb[:, :, ::-1])
        self.cv2.waitKey(1)


class Mp4Sink:
    """Appends frames to an H.264 mp4 (PyAV directly; the stream is sized by
    the first frame)."""

    def __init__(self, path: str | Path, fps: int):
        import av

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._av = av
        self._container = av.open(str(path), mode="w")
        self._stream = self._container.add_stream("libx264", rate=fps)
        self._stream.pix_fmt = "yuv420p"
        self._frames = 0

    def __call__(self, frame: Frame) -> None:
        if self._frames == 0:
            self._stream.width = frame.width
            self._stream.height = frame.height
        vf = self._av.VideoFrame.from_ndarray(np.ascontiguousarray(frame.rgb), format="rgb24")
        for packet in self._stream.encode(vf):
            self._container.mux(packet)
        self._frames += 1

    def close(self) -> None:
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        log.info("mp4 closed after %d frames", self._frames)


class OverlaySink:
    """mp4 of the generated frame with the HD-map condition (Frame.meta["hdmap"])
    blended on top — the quickest way to see whether the rollout drifts off
    the map."""

    def __init__(self, path: str | Path, fps: int, alpha: float = 0.55):
        self.inner = Mp4Sink(path, fps)
        self.alpha = alpha

    def __call__(self, frame: Frame) -> None:
        hd = frame.meta.get("hdmap")
        rgb = frame.rgb
        if hd is not None:
            if hd.shape != rgb.shape:
                from PIL import Image

                hd = np.asarray(Image.fromarray(hd).resize((frame.width, frame.height)))
            mask = hd.max(axis=2, keepdims=True) > 40
            blended = (1 - self.alpha) * rgb.astype(np.float32) + self.alpha * hd.astype(np.float32)
            rgb = np.where(mask, blended, rgb).astype(np.uint8)
        self.inner(Frame(rgb, frame.timestamp_us, frame.pose, frame.meta))

    def close(self) -> None:
        self.inner.close()


class CompareSink:
    """mp4 of [CARLA camera | HD-map condition | generated] side by side, for
    checking what the model was conditioned on. Generated frames carry their
    input frame's timestamp; the run loop hands inputs in via add_input()."""

    LABELS = ("CARLA", "HD-map condition", "generated")

    def __init__(self, path: str | Path, fps: int):
        self.inner = Mp4Sink(path, fps)
        self._inputs: dict[int, Frame] = {}
        self._lock = threading.Lock()

    def add_input(self, frame: Frame) -> None:
        with self._lock:
            # Standstill frames all have timestamp 0; the latest one wins.
            self._inputs[frame.timestamp_us] = frame

    def __call__(self, frame: Frame) -> None:
        from PIL import Image, ImageDraw

        h, w = frame.rgb.shape[:2]
        with self._lock:
            source = self._inputs.get(frame.timestamp_us)
            carla_rgb = None if source is None else source.rgb
            # Inputs older than this frame are never shown again.
            for ts in [t for t in self._inputs if 0 < t < frame.timestamp_us]:
                del self._inputs[ts]
        blank = np.zeros((h, w, 3), np.uint8)
        panels = []
        for img in (carla_rgb, frame.meta.get("hdmap"), frame.rgb):
            if img is None:
                img = blank
            elif img.shape[:2] != (h, w):
                img = np.asarray(Image.fromarray(img).resize((w, h)))
            panels.append(img)
        canvas = Image.fromarray(np.concatenate(panels, axis=1))
        draw = ImageDraw.Draw(canvas)
        for i, label in enumerate(self.LABELS):
            draw.text((i * w + 8, 6), label, fill=(255, 255, 0))
        draw.text((8, h - 18), f"t={frame.timestamp_us / 1e6:.2f}s", fill=(255, 255, 0))
        light = None if source is None else source.meta.get("traffic_light")
        if light:
            colour = {"Red": (255, 60, 60), "Yellow": (255, 220, 0), "Green": (60, 255, 60)}
            draw.text(
                (90, h - 18), f"light ahead: {light}", fill=colour.get(light, (255, 255, 255))
            )
        self.inner(Frame(np.asarray(canvas), frame.timestamp_us, frame.pose, frame.meta))

    def close(self) -> None:
        self.inner.close()


class MjpegSink:
    """Serves the latest frame as multipart MJPEG on http://host:port/ — the
    quickest way to watch output from another machine (e.g. over Tailscale)."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080, quality: int = 85):
        self.quality = quality
        self._latest: bytes | None = None
        self._cond = threading.Condition()
        sink = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):  # quiet
                pass

            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                seen = None
                try:
                    while True:
                        with sink._cond:
                            sink._cond.wait_for(lambda: sink._latest is not seen, timeout=5)  # noqa: B023
                            data = sink._latest
                            seen = data
                        if data is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = http.server.ThreadingHTTPServer((host, port), Handler)
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        log.info("MJPEG stream on http://%s:%d/", host, port)

    def __call__(self, frame: Frame) -> None:
        data = encode_image(frame.rgb, "JPEG", self.quality)
        with self._cond:
            self._latest = data
            self._cond.notify_all()

    def close(self) -> None:
        self.server.shutdown()
