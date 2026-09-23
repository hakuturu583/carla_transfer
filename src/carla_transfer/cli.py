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
import math
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

    mgr = ContainerManager(_load_config(args))
    for target in args.targets or [None]:
        mgr.build(target=target, tag=args.tag if len(args.targets or []) <= 1 else None)
    return 0


def _scene_spec(args: argparse.Namespace):
    from carla_transfer.hdmap import SceneSpec

    spec = SceneSpec(clip_id=args.clip_id, fps=args.fps, speed_mps=args.speed)
    if args.route:
        spec.route = args.route.split(",")
    if args.start_lane:
        spec.start_lane = args.start_lane
    if args.prompt:
        spec.prompt = args.prompt
    return spec


def cmd_export_map(args: argparse.Namespace) -> int:
    from carla_transfer.hdmap import export_scene_archive

    cfg = _load_config(args)
    spec = _scene_spec(args)
    spec.camera.width, spec.camera.height = cfg.resolution
    out = export_scene_archive(args.xodr, args.out, spec)
    print(out)
    return 0


def cmd_prepare_scene(args: argparse.Namespace) -> int:
    from carla_transfer.hdmap.prepare import prepare_scene

    cfg = _load_config(args)
    out = prepare_scene(
        cfg,
        Path(args.xodr),
        Path(args.out),
        _scene_spec(args),
        work_dir=Path(args.work_dir) if args.work_dir else None,
        first_image=args.first_image,
        local=args.local,
        cosmos3_args=args.cosmos3_args,
    )
    print(out)
    return 0


def cmd_first_image(args: argparse.Namespace) -> int:
    """Regenerate archives' first_image.png (e.g. after add-obstacles changed
    what stands in front of the ego at t=0). Several archives share one raster
    server and one Cosmos3 load."""
    from carla_transfer.hdmap.prepare import cosmos3_first_images, replace_in_archive

    cfg = _load_config(args)
    archives = [Path(a) for a in args.archives]
    if args.work_dir and len(archives) > 1:
        raise SystemExit("--work-dir takes a single archive")
    work_dirs = [Path(args.work_dir)] if args.work_dir else [
        a.parent / f"{a.stem}_first_image" for a in archives
    ]  # fmt: skip
    firsts = cosmos3_first_images(cfg, archives, work_dirs, args.local, args.cosmos3_args)
    for archive, first in zip(archives, firsts, strict=True):
        replace_in_archive(archive, "first_image.png", first.read_bytes())
        print(f"{archive}: first_image.png from {first}")
    return 0


def cmd_add_obstacles(args: argparse.Namespace) -> int:
    """Write a `drive --record-tracks` log into the archive as clipgt/obstacle.parquet."""
    import zipfile

    from carla_transfer.carla.tracks import load_tracks, obstacle_parquet_bytes
    from carla_transfer.hdmap.prepare import replace_in_archive

    archive = Path(args.archive)
    with zipfile.ZipFile(archive) as zf:
        import yaml

        clip_id = str(yaml.safe_load(zf.read("metadata.yaml"))["scene_id"])
    data = obstacle_parquet_bytes(load_tracks(args.tracks), clip_id)
    replace_in_archive(archive, "clipgt/obstacle.parquet", data)
    print(f"{archive}: clipgt/obstacle.parquet ({len(data)} bytes)")
    return 0


def cmd_carla(args: argparse.Namespace) -> int:
    from carla_transfer.docker.carla import CarlaServerManager

    mgr = CarlaServerManager(_load_config(args).carla)
    if args.action == "up":
        if args.dry_run:
            print(" ".join(mgr.run_args()))
        else:
            mgr.up()
    elif args.action == "down":
        mgr.down()
    else:
        sys.stdout.write(mgr.logs())
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
    if getattr(args, "record_overlay", None):
        cfg.omnidreams.return_hdmap = True
        out.append(S.OverlaySink(Path(args.record_overlay), cfg.fps))
    if getattr(args, "record_compare", None):
        cfg.omnidreams.return_hdmap = True
        out.append(S.CompareSink(Path(args.record_compare), cfg.fps))
    return out


def _close_sinks(sinks: list) -> None:
    for s in sinks:
        close = getattr(s, "close", None)
        if close:
            close()


def cmd_run(args: argparse.Namespace) -> int:
    import carla

    from carla_transfer.carla import CarlaCameraSource
    from carla_transfer.hdmap.clipgt import CameraSpec
    from carla_transfer.pipeline import StreamPipeline

    cfg = _load_config(args)
    sinks = _make_sinks(cfg, args)
    # OmniDreams' KV cache needs consecutive poses, and the first chunk's
    # compile would otherwise skip tens of seconds of driving, so by default
    # nothing is dropped:
    #   --sync-to-model  this process ticks; push() blocks while the model is
    #                    behind, so the world is paced at the model's rate.
    #   --follow         another process ticks (carla-transfer drive); chunks
    #                    queue up unboundedly and the model works through the
    #                    backlog, so keep the driver's --tick-fps below the
    #                    model's throughput.
    #   --drop-stale     real-time at any cost: skip chunks when behind.
    pipe = StreamPipeline(
        cfg,
        sinks=sinks,
        drop_stale=args.drop_stale,
        max_pending_chunks=1 if args.sync_to_model else 100_000,
    )

    # With initial_image=archive the model session does not need a CARLA
    # frame, so open it *before* CARLA takes its share of the GPU: the
    # OmniDreams init peak (text encoder resident, ~16 GB) plus CARLA
    # (~6.5 GB) does not fit a 24 GB card, the steady state (~12 GB) does.
    session_first = cfg.omnidreams.initial_image == "archive"
    carla_mgr = None
    if args.carla_up:
        from carla_transfer.docker.carla import CarlaServerManager

        carla_mgr = CarlaServerManager(cfg.carla)
        if session_first:
            carla_mgr.down()
    if session_first:
        placeholder = Frame(np.zeros((cfg.height, cfg.width, 3), np.uint8), 0)
        pipe.start(placeholder)
        log.info("model session open")
    if carla_mgr is not None:
        carla_mgr.up()
        _wait_for_carla(args.carla_host, args.carla_port, timeout_s=600)

    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(120.0)
    world = client.get_world()
    settings = world.get_settings()
    if args.follow and args.sync_to_model:
        raise SystemExit("--sync-to-model ticks and drives itself; drop --follow")
    follow = args.follow
    if not follow:
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1.0 / cfg.fps
        world.apply_settings(settings)

    def step():
        # `drive` (or another client) owns the tick in --follow mode.
        if follow:
            world.wait_for_tick()
        else:
            world.tick()

    spawn = None
    if args.spawn_at_archive_start and cfg.omnidreams.scene_archive:
        from carla_transfer.carla.source import transform_from_rig_to_world
        from carla_transfer.hdmap.control_video import archive_trajectory

        poses, _ = archive_trajectory(Path(cfg.omnidreams.scene_archive))
        spawn = transform_from_rig_to_world(poses[0])
        log.info("spawning ego at the archive's initial pose: %s", spawn)
    if follow:
        vehicle = _wait_for_vehicle(world, args.role_name, step)
    else:
        vehicle = _find_or_spawn_vehicle(world, args.role_name, spawn)
    # Same intrinsics/mount as the archive camera the model renders with
    # (clipgt calibration), so CARLA frames and generated frames line up.
    cam = CameraSpec()
    src = CarlaCameraSource(
        world,
        vehicle,
        cfg.width,
        cfg.height,
        fov_deg=cam.fov_deg,
        transform=carla.Transform(
            carla.Location(x=cam.position[0], y=-cam.position[1], z=cam.position[2])
        ),
    )
    from carla_transfer.sinks import CompareSink

    compare_sinks = [s for s in sinks if isinstance(s, CompareSink)]
    input_sink = None
    if args.record_input:
        from carla_transfer.sinks import Mp4Sink

        input_sink = Mp4Sink(Path(args.record_input), cfg.fps)
    # Until the model has produced its first chunk the driver holds still, so
    # dropping the (identical) frames that pile up meanwhile is harmless and
    # keeps the model from working through a backlog of standstill.
    if args.go_file:
        pipe.set_backpressure(1, True)
    released = False
    n_pushed = 0
    n_go = 0
    try:
        first = _first_frame(step, src)
        if not session_first:
            pipe.start(first)
        last_report = time.monotonic()
        t_start = time.monotonic()
        while not args.duration or time.monotonic() - t_start < args.duration:
            step()
            frame = src.get()
            if args.go_file and not released and pipe.stats.chunks_generated >= 1:
                pipe.set_backpressure(1 if args.sync_to_model else 100_000, args.drop_stale)
                Path(args.go_file).touch()
                released = True
                n_go = n_pushed
                log.info("first chunk done; released the driver via %s", args.go_file)
            # Timestamps count from the release, matching the driver's track
            # log (t=0 at the first moving tick); standstill frames stay at 0.
            frame.timestamp_us = (
                max(0, n_pushed - n_go) * (1_000_000 // cfg.fps)
                if args.go_file
                else frame.timestamp_us
            )
            n_pushed += 1
            if input_sink is not None:
                input_sink(frame.resized(cfg.width, cfg.height))
            for c in compare_sinks:
                c.add_input(frame.resized(cfg.width, cfg.height))
            if args.sim_seconds and released and n_pushed - n_go >= int(args.sim_seconds * cfg.fps):
                break
            pipe.push(frame)
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
        if input_sink is not None:
            input_sink.close()
        if args.go_file and Path(args.go_file).exists():
            Path(args.go_file).unlink()
        if not follow:
            settings.synchronous_mode = False
            world.apply_settings(settings)
        if carla_mgr is not None and args.carla_down:
            carla_mgr.down()
    return 0


def _wait_for_carla(host: str, port: int, timeout_s: float) -> None:
    import carla

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            c = carla.Client(host, port)
            c.set_timeout(10.0)
            c.get_world().get_map()
            return
        except RuntimeError as e:
            if time.monotonic() > deadline:
                raise TimeoutError("CARLA did not come up") from e
            time.sleep(5)


def _get_traffic_manager(client, port: int, attempts: int = 10):
    """Right after CARLA starts, TM creation can throw while the map loads."""
    for i in range(attempts):
        try:
            return client.get_trafficmanager(port)
        except RuntimeError:
            if i == attempts - 1:
                raise
            time.sleep(3)


def _wait_for_vehicle(world, role_name: str, step, max_ticks: int = 3000):
    for _ in range(max_ticks):
        for a in world.get_actors().filter("vehicle.*"):
            if a.attributes.get("role_name") == role_name:
                return a
        try:
            step()
        except RuntimeError:
            # synchronous world with nobody ticking yet (driver not started)
            time.sleep(1)
    raise RuntimeError(f"no vehicle with role_name={role_name!r} appeared")


def _first_frame(step, src, max_ticks: int = 60):
    """Tick until the camera delivers (sensors come up a few ticks after spawn)."""
    import queue

    for _ in range(max_ticks):
        step()
        try:
            return src.get(timeout=1.0)
        except queue.Empty:
            continue
    raise RuntimeError("camera delivered no frame")


def cmd_drive(args: argparse.Namespace) -> int:
    """Traffic-Manager driver: owns the synchronous tick; `run --follow` reads.

    --wait-for FILE holds every actor still (ticking, so cameras deliver)
    until FILE exists — `run --go-file FILE` creates it once the model has
    produced its first chunk. --record-tracks logs all actors for
    `add-obstacles`; --replay-tracks plays such a log back kinematically.
    """
    import carla

    from carla_transfer.carla.tracks import (
        TrackRecorder,
        TrackReplayer,
        find_crosswalk_ahead,
        load_tracks,
        settle_actors,
        spawn_dart_walker,
        spawn_traffic,
        spawn_walkers,
        step_walkers,
    )

    cfg = _load_config(args)
    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(120.0)
    world = client.get_world()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / cfg.fps
    world.apply_settings(settings)
    tm = _get_traffic_manager(client, args.tm_port)
    tm.set_synchronous_mode(True)
    if args.seed is not None:
        tm.set_random_device_seed(args.seed)

    spawn = None
    if args.spawn_at_archive_start and cfg.omnidreams.scene_archive:
        from carla_transfer.carla.source import transform_from_rig_to_world
        from carla_transfer.hdmap.control_video import archive_trajectory

        poses, _ = archive_trajectory(Path(cfg.omnidreams.scene_archive))
        spawn = transform_from_rig_to_world(poses[0])
        if args.spawn_back_m:
            # previous() branches at junctions; a spawn inside one confuses the
            # Traffic Manager's route, so take a non-junction candidate.
            cands = world.get_map().get_waypoint(spawn.location).previous(args.spawn_back_m)
            wp = next((w for w in cands if not w.is_junction), cands[0])
            spawn = wp.transform
            spawn.location.z += 0.3
        if args.spawn_ahead_m:
            from carla_transfer.carla.tracks import advance_along_lane

            wp = advance_along_lane(
                world.get_map().get_waypoint(spawn.location), args.spawn_ahead_m
            )
            if wp is None:
                raise SystemExit("lane ends before --spawn-ahead-m")
            spawn = wp.transform
            spawn.location.z += 0.3

    replayer = None
    traffic: list = []
    walkers: list = []
    dart: list = []  # [(walker, ctrl, tf)] released when the hero nears the crosswalk
    dart_cw = None
    if args.replay_tracks:
        replayer = TrackReplayer(world, load_tracks(args.replay_tracks))
        replayer.apply(0)
        vehicle = replayer.hero(args.role_name)
        if vehicle is None:
            raise SystemExit("replay log has no hero")
    else:
        vehicle = _find_or_spawn_vehicle(world, args.role_name, spawn, autopilot=False)
        if args.traffic:
            world.tick()  # settle the hero so its location is valid
            traffic = spawn_traffic(
                world,
                client,
                args.traffic,
                args.tm_port,
                args.seed or 0,
                near=vehicle.get_location(),
            )
        if args.walkers:
            world.tick()
            lo, hi = (float(v) for v in args.walkers_ahead.split(","))
            walkers = spawn_walkers(
                world,
                args.walkers,
                args.seed or 0,
                near=vehicle.get_location(),
                ahead_m=(lo, hi),
                side=args.walkers_side,
            )
            world.tick()  # let the spawns land before any transform/physics call
        if args.dart_crosswalk:
            world.tick()
            dart_cw = find_crosswalk_ahead(world.get_map(), vehicle.get_transform())
            if dart_cw is None:
                raise SystemExit(
                    "no crosswalk 20-200 m ahead of the hero; use --spawn-ahead-m/--spawn-back-m"
                )
            dart = [
                spawn_dart_walker(
                    world,
                    dart_cw,
                    vehicle.get_transform(),
                    args.seed or 0,
                    side=args.walkers_side if args.walkers_side != "both" else "right",
                    speed=args.dart_speed,
                )
            ]
            world.tick()
        # Let everything land before anything is recorded or driven.
        settle_actors(world, [vehicle, *traffic], walkers + dart)
        for a in traffic:
            a.set_autopilot(True, args.tm_port)
    log.info("hero %s at %s", vehicle.id, vehicle.get_transform())

    period = 1.0 / (args.tick_fps or cfg.fps)
    recorder = TrackRecorder(world, cfg.fps) if args.record_tracks else None
    if recorder is not None and not args.no_parked:
        recorder.add_static_vehicles(vehicle.get_location())

    def paced_tick():
        t0 = time.monotonic()
        world.tick()
        if args.realtime:
            delay = period - (time.monotonic() - t0)
            if delay > 0:
                time.sleep(delay)

    try:
        # Hold phase: everything still until the go file appears.
        if args.wait_for:
            log.info("holding until %s exists", args.wait_for)
            while not Path(args.wait_for).exists():
                paced_tick()
            log.info("released")
        if replayer is None:
            vehicle.set_autopilot(True, args.tm_port)
            if args.route:
                world.tick()  # TM registers the vehicle on a tick; set_route before that is lost
                tm.set_route(vehicle, args.route.split(","))
            if args.speed_pct:
                tm.vehicle_percentage_speed_difference(vehicle, args.speed_pct)
            if args.target_speed:
                tm.set_desired_speed(vehicle, args.target_speed)
            tm.ignore_lights_percentage(vehicle, args.ignore_lights_pct)
            for a in traffic:
                tm.ignore_lights_percentage(a, args.ignore_lights_pct)
            for w, *_ in walkers + dart:
                w.set_simulate_physics(False)
        n_ticks = int(args.sim_seconds * cfg.fps) if args.sim_seconds else None
        if replayer is not None and n_ticks is None:
            n_ticks = replayer.frames
        t_start = time.monotonic()
        frame = 0
        while True:
            if n_ticks is not None and frame >= n_ticks:
                break
            if args.duration and time.monotonic() - t_start >= args.duration:
                break
            if replayer is not None:
                replayer.apply(frame)
            else:
                if walkers:
                    step_walkers(walkers, 1.0 / cfg.fps)
                if dart and dart_cw is not None:
                    loc = vehicle.get_location()
                    gap = math.hypot(dart_cw["cx"] - loc.x, dart_cw["cy"] - loc.y)
                    if gap <= args.dart_trigger_m:
                        walkers += dart  # from now on it runs with the others
                        dart = []
                        log.info(
                            "dart released at frame %d (hero %.1f m from the crosswalk)", frame, gap
                        )
            paced_tick()
            if recorder is not None:
                recorder.sample()
            frame += 1
        log.info("drove %d ticks (%.1f s sim)", frame, frame / cfg.fps)
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.save(args.record_tracks)
        if replayer is None:
            vehicle.set_autopilot(False, args.tm_port)
            for a in traffic:
                a.set_autopilot(False, args.tm_port)
            world.tick()
            if args.destroy:
                for a in traffic:
                    a.destroy()
                for w, *_ in walkers + dart:
                    w.destroy()
                vehicle.destroy()
                world.tick()
        elif args.destroy:
            replayer.destroy()
            world.tick()
        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
    return 0


def _find_or_spawn_vehicle(world, role_name: str, spawn=None, autopilot: bool = True):
    for a in world.get_actors().filter("vehicle.*"):
        if a.attributes.get("role_name") == role_name:
            return a
    import random

    bps = world.get_blueprint_library().filter("vehicle.*")
    bp = next((b for b in bps if "model3" in b.id or "mini" in b.id), bps[0])
    bp.set_attribute("role_name", role_name)
    if spawn is None:
        spawn = random.choice(world.get_map().get_spawn_points())
    v = world.try_spawn_actor(bp, spawn)
    if v is None:
        raise RuntimeError(f"could not spawn ego vehicle at {spawn}")
    if autopilot:
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
    b.add_argument("targets", nargs="*", help="omnidreams / cosmos3 (default: config backend)")
    b.add_argument("--tag")
    b.set_defaults(fn=cmd_build)

    def add_map_args(sp):
        sp.add_argument("--xodr", required=True, help="CARLA OpenDRIVE file")
        sp.add_argument("--out", required=True, help="output scene archive (.usdz)")
        sp.add_argument("--clip-id", default="carla")
        sp.add_argument("--start-lane", help="roadgen lane id to start from (default: first)")
        sp.add_argument("--route", help="comma-separated roadgen lane ids")
        sp.add_argument("--speed", type=float, default=8.0, help="ego speed m/s")
        sp.add_argument("--fps", type=float, default=30.0)
        sp.add_argument("--prompt", help="default scene prompt stored in the archive")

    em = sub.add_parser("export-map")
    add_map_args(em)
    em.set_defaults(fn=cmd_export_map)

    ps = sub.add_parser("prepare-scene")
    add_map_args(ps)
    ps.add_argument("--work-dir", help="where the control video and Cosmos3 output go")
    ps.add_argument(
        "--first-image",
        choices=["cosmos3", "transfer25", "none"],
        default="cosmos3",
        help="how the archive's first image is generated (default: Cosmos3-Nano transfer)",
    )
    ps.add_argument(
        "--local", action="store_true", help="cosmos3 path: run raster + Cosmos3 in this env"
    )
    ps.add_argument(
        "--cosmos3-args", nargs=argparse.REMAINDER, default=[],
        help="extra args for carla-transfer-cosmos3 (e.g. --steps 20 --num-frames 33)",
    )  # fmt: skip
    ps.set_defaults(fn=cmd_prepare_scene)

    fi = sub.add_parser("first-image", help="regenerate archives' first images (Cosmos3)")
    fi.add_argument("archives", nargs="+")
    fi.add_argument(
        "--work-dir",
        help="control video + Cosmos3 output (one archive only; "
        "default: <archive stem>_first_image next to it)",
    )
    fi.add_argument("--local", action="store_true", help="run raster + Cosmos3 in this env")
    fi.add_argument(
        "--cosmos3-args", nargs=argparse.REMAINDER, default=[],
        help="extra args for carla-transfer-cosmos3 (e.g. --revision fp8 --offload none)",
    )  # fmt: skip
    fi.set_defaults(fn=cmd_first_image)

    ao = sub.add_parser("add-obstacles")
    ao.add_argument("archive")
    ao.add_argument("tracks")
    ao.set_defaults(fn=cmd_add_obstacles)

    ca = sub.add_parser("carla")
    ca.add_argument("action", choices=["up", "down", "logs"])
    ca.add_argument("--dry-run", action="store_true")
    ca.set_defaults(fn=cmd_carla)

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
    r.add_argument(
        "--spawn-at-archive-start",
        action="store_true",
        help="spawn the ego at the scene archive's initial pose (rig_trajectories.json)",
    )
    r.add_argument("--duration", type=float, default=0.0, help="seconds to run (0 = until Ctrl-C)")
    r.add_argument(
        "--carla-up",
        action="store_true",
        help="start the CARLA container (after the model session, to fit the GPU)",
    )
    r.add_argument("--carla-down", action="store_true", help="stop the CARLA container at the end")
    r.add_argument(
        "--follow",
        action="store_true",
        help="do not spawn: attach to the hero driven by `carla-transfer drive`",
    )
    r.add_argument(
        "--sync-to-model",
        action="store_true",
        help="spawn + drive the hero here and pace the world tick at the model's rate",
    )
    r.add_argument(
        "--drop-stale",
        action="store_true",
        help="skip chunks when the model falls behind (breaks rollout continuity)",
    )
    r.add_argument("--go-file", help="create this file once the first chunk is generated "
                   "(`drive --wait-for` releases on it); timestamps restart there")  # fmt: skip
    r.add_argument("--sim-seconds", type=float, default=0.0,
                   help="stop this many sim seconds after the release")  # fmt: skip
    r.add_argument("--record-input", metavar="FILE.mp4", help="also record the CARLA camera")
    r.add_argument(
        "--record-overlay",
        metavar="FILE.mp4",
        help="record the generated frames with the HD-map condition blended on top",
    )
    r.add_argument(
        "--record-compare",
        metavar="FILE.mp4",
        help="record [CARLA camera | HD-map condition | generated] side by side",
    )
    add_sink_args(r)
    r.set_defaults(fn=cmd_run)

    d = sub.add_parser("drive")
    d.add_argument("--carla-host", default="127.0.0.1")
    d.add_argument("--carla-port", type=int, default=2000)
    d.add_argument("--tm-port", type=int, default=8000)
    d.add_argument("--role-name", default="hero")
    d.add_argument("--spawn-at-archive-start", action="store_true")
    d.add_argument("--duration", type=float, default=0.0)
    d.add_argument("--speed-pct", type=float, default=0.0,
                   help="Traffic Manager speed difference %% (negative = faster)")  # fmt: skip
    d.add_argument("--target-speed", type=float, default=0.0,
                   help="hero cruise speed km/h, ignores the limit (0 = off)")  # fmt: skip
    d.add_argument("--ignore-lights-pct", type=float, default=100.0)
    d.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True,
                   help="pace ticks at the configured fps")  # fmt: skip
    d.add_argument("--destroy", action="store_true", help="destroy the hero on exit")
    d.add_argument("--traffic", type=int, default=0, help="background TM vehicles to spawn")
    d.add_argument(
        "--walkers",
        type=int,
        default=0,
        help="pedestrians on the sidewalks ahead that cross the road on release",
    )
    d.add_argument("--walkers-ahead", default="15,45", help="MIN,MAX metres ahead of the hero")
    d.add_argument("--walkers-side", choices=["left", "right", "both"], default="both")
    d.add_argument(
        "--dart-crosswalk",
        action="store_true",
        help="a pedestrian waits at the next crosswalk and runs out when the hero gets close",
    )
    d.add_argument(
        "--dart-trigger-m", type=float, default=14.0, help="hero distance that releases it"
    )
    d.add_argument("--dart-speed", type=float, default=3.0, help="m/s")
    d.add_argument(
        "--no-parked", action="store_true", help="do not record the map's parked vehicles"
    )
    d.add_argument("--route", help="TM junction choices for the hero, e.g. Straight,Straight,Right")
    d.add_argument(
        "--spawn-back-m",
        type=float,
        default=0.0,
        help="spawn the hero this far back along its lane",
    )
    d.add_argument(
        "--spawn-ahead-m",
        type=float,
        default=0.0,
        help="spawn the hero this far ahead along its lane (straightest path through junctions)",
    )
    d.add_argument("--seed", type=int, default=None, help="Traffic Manager / spawn seed")
    d.add_argument("--sim-seconds", type=float, default=0.0, help="stop after this much sim time")
    d.add_argument("--wait-for", help="hold all actors still until this file exists")
    d.add_argument("--record-tracks", help="write every actor's per-tick pose to this JSON")
    d.add_argument("--replay-tracks", help="replay a --record-tracks log instead of driving")
    d.add_argument(
        "--tick-fps",
        type=float,
        default=None,
        help="wall-clock tick rate (default: the configured fps). Sim time still advances "
        "1/fps per tick; run this below the model's throughput so `run --follow --no-drop` "
        "never falls behind",
    )
    d.set_defaults(fn=cmd_drive)

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
