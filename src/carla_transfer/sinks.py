"""Places generated frames can go: a window, an MJPEG stream, an mp4 file, a callback."""

from __future__ import annotations

import http.server
import logging
import threading
from pathlib import Path

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
    """Appends frames to an mp4 with imageio-ffmpeg. Needs the ``video`` extra."""

    def __init__(self, path: str | Path, fps: int):
        import imageio.v2 as iio

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.writer = iio.get_writer(str(path), fps=fps, macro_block_size=1)

    def __call__(self, frame: Frame) -> None:
        self.writer.append_data(frame.rgb)

    def close(self) -> None:
        self.writer.close()


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
