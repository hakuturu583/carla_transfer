"""carla_transfer: real-time sim-to-real video generation for CARLA.

CARLA renders a frame → a world model (OmniDreams or Cosmos3-Nano) running in a
Docker container turns it into a photorealistic frame → the frame is displayed,
streamed, or written to disk. The GPU model never runs in this process; this
package is the host-side client, container manager, and real-time loop.
"""

from carla_transfer.config import BackendKind, TransferConfig
from carla_transfer.frames import Frame
from carla_transfer.pipeline import StreamPipeline

__all__ = ["BackendKind", "Frame", "StreamPipeline", "TransferConfig"]
__version__ = "0.1.2"
