"""``carla-transfer`` command line.

carla-transfer build            build the model image for the configured backend
carla-transfer up               start the model container and wait until it serves
carla-transfer down             stop it
carla-transfer logs             tail container logs
carla-transfer run              connect to CARLA, stream through the model
carla-transfer selftest         run the pipeline with the mock backend (no GPU/CARLA)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

from carla_transfer.config import BackendKind, TransferConfig
from carla_transfer.frames import Frame

log = logging.getLogger("carla_transfer")


def _load_config(args: argparse.Namespace) -> TransferConfig:
    cfg = TransferConfig.from_yaml(args.config) if args.config else TransferConfig()
    if args.backend:
        cfg.backend = BackendKind(args.backend)
    return cfg


def cmd_build(args: argparse.Namespace) -> int:
    from carla_transfer.docker import ContainerManager

    ContainerManager(_load_config(args)).build(tag=args.tag)
    return 0


def cmd_up(args: argparse.Namespace) -> int:
    from carla_transfer.backends import make_backend
    from carla_transfer.docker import ContainerManager

    cfg = _load_config(args)
    mgr = ContainerManager(cfg)
    if args.dry_run:
        print(" ".join(mgr.run_args()))
        return 0
    mgr.up()
    if not args.no_wait:
        log.info("waiting up to %.0fs for the server", cfg.docker.startup_timeout_s)
        make_backend(cfg).wait_ready(cfg.docker.startup_timeout_s)
    return 0


def cmd_down(args: argparse.Namespace) -> int:
    from carla_transfer.docker import ContainerManager

    ContainerManager(_load_config(args)).down()
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    from carla_transfer.docker import ContainerManager

    sys.stdout.write(ContainerManager(_load_config(args)).logs(tail=args.tail))
    return 0


def _make_sinks(cfg: TransferConfig, args: argparse.Namespace) -> list:
    from carla_transfer import sinks as S

    out = []
    if args.window:
        out.append(S.WindowSink())
    if args.mjpeg:
        host, _, port = args.mjpeg.rpartition(":")
        out.append(S.MjpegSink(host or "0.0.0.0", int(port)))
    if args.record:
        out.append(S.Mp4Sink(Path(args.record), cfg.fps))
    return out


def _close_sinks(sinks: list) -> None:
    for s in sinks:
        close = getattr(s, "close", None)
        if close:
            close()


def cmd_run(args: argparse.Namespace) -> int:
    import carla

    from carla_transfer.carla import CarlaCameraSource
    from carla_transfer.pipeline import StreamPipeline

    cfg = _load_config(args)
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(20.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / cfg.fps
    world.apply_settings(settings)

    vehicle = _find_or_spawn_vehicle(world, args.role_name)
    src = CarlaCameraSource(world, vehicle, cfg.width, cfg.height)
    sinks = _make_sinks(cfg, args)
    pipe = StreamPipeline(cfg, sinks=sinks)
    try:
        world.tick()
        pipe.start(src.get())
        last_report = time.monotonic()
        while True:
            world.tick()
            pipe.push(src.get())
            if time.monotonic() - last_report > 5:
                s = pipe.stats
                log.info(
                    "chunks gen=%d drop=%d  gen %.0f ms/chunk  ~%.1f fps",
                    s.chunks_generated,
                    s.chunks_dropped,
                    s.mean_generate_s * 1e3,
                    s.effective_fps,
                )
                last_report = time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()
        src.destroy()
        _close_sinks(sinks)
        settings.synchronous_mode = False
        world.apply_settings(settings)
    return 0


def _find_or_spawn_vehicle(world, role_name: str):
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name") == role_name:
            return a
    import random

    bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    bp.set_attribute("role_name", role_name)
    v = world.try_spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
    if v is None:
        raise RuntimeError("could not spawn ego vehicle")
    v.set_autopilot(True)
    return v


def cmd_selftest(args: argparse.Namespace) -> int:
    from carla_transfer.backends.mock import MockBackend
    from carla_transfer.pipeline import StreamPipeline

    cfg = _load_config(args)
    cfg.backend = BackendKind.MOCK
    sinks = _make_sinks(cfg, args)
    pipe = StreamPipeline(cfg, backend=MockBackend(cfg, latency_s=0.02), sinks=sinks)
    rng = np.random.default_rng(0)
    n = args.frames
    frames = [
        Frame(
            rng.integers(0, 255, (cfg.height, cfg.width, 3), dtype=np.uint8),
            int(i * 1e6 / cfg.fps),
        )
        for i in range(n)
    ]
    with pipe:
        pipe.start(frames[0])
        for f in frames:
            pipe.push(f)
            time.sleep(1 / cfg.fps)
    _close_sinks(sinks)
    s = pipe.stats
    print(f"frames_out={s.frames_out} chunks={s.chunks_generated} dropped={s.chunks_dropped}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="carla-transfer", description=__doc__.split("\n\n")[0])
    p.add_argument("-c", "--config", help="YAML config (see configs/)")
    p.add_argument("--backend", choices=[b.value for b in BackendKind])
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--tag")
    b.set_defaults(fn=cmd_build)

    u = sub.add_parser("up")
    u.add_argument("--dry-run", action="store_true", help="print the docker run command")
    u.add_argument("--no-wait", action="store_true")
    u.set_defaults(fn=cmd_up)

    sub.add_parser("down").set_defaults(fn=cmd_down)

    lg = sub.add_parser("logs")
    lg.add_argument("--tail", type=int, default=100)
    lg.set_defaults(fn=cmd_logs)

    def add_sink_args(sp):
        sp.add_argument("--window", action="store_true", help="show output in an OpenCV window")
        sp.add_argument("--mjpeg", metavar="[HOST]:PORT", help="serve output as MJPEG")
        sp.add_argument("--record", metavar="FILE.mp4", help="write output to mp4")

    r = sub.add_parser("run")
    r.add_argument("--carla-host", default="127.0.0.1")
    r.add_argument("--carla-port", type=int, default=2000)
    r.add_argument("--role-name", default="hero")
    add_sink_args(r)
    r.set_defaults(fn=cmd_run)

    st = sub.add_parser("selftest")
    st.add_argument("--frames", type=int, default=64)
    add_sink_args(st)
    st.set_defaults(fn=cmd_selftest)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
