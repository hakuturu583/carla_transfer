"""Record and replay CARLA actor tracks; export them as ClipGT obstacles.

OmniDreams renders dynamic objects from the scene archive's
``clipgt/obstacle.parquet`` — timestamped cuboid tracks loaded once per
scene — so live actors cannot be injected per chunk (yet). The workaround
is two passes over the same 10 s scenario:

    pass 1  `carla-transfer drive --traffic 20 --record-tracks tracks.json`
            Traffic Manager drives the hero and the background traffic; every
            tick, every vehicle/walker pose is logged (CARLA frame).
    ---     `carla-transfer add-obstacles archive tracks.json`
            The log becomes ``clipgt/obstacle.parquet`` (FLU frame, the
            official schema's fields) inside the archive.
    pass 2  `carla-transfer drive --replay-tracks tracks.json --wait-for go`
            + `carla-transfer run --follow --go-file go`
            The recorded actors are replayed kinematically (physics off,
            transform per tick), held at t=0 until the model has produced its
            first chunk, then played; the model sees the same obstacles the
            archive carries, at the same timestamps.

Timestamps in the log are sim seconds since the hold was released (t=0 at
the first moving tick), which is what `run --go-file` rebases its frame
timestamps to.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# CARLA blueprint prefix → ClipGT category
_CATEGORY = {"vehicle": "automobile", "walker": "person"}


@dataclass
class ActorTrack:
    actor_id: int
    blueprint: str
    role_name: str
    category: str
    size_lwh: tuple[float, float, float]
    # bounding-box centre offset in the actor frame (CARLA units)
    bbox_offset: tuple[float, float, float]
    # CARLA-frame samples: [t_s, x, y, z, roll, pitch, yaw]
    samples: list[list[float]] = field(default_factory=list)


class TrackRecorder:
    def __init__(self, world: Any, fps: int):
        self.world = world
        self.fps = fps
        self.tracks: dict[int, ActorTrack] = {}
        self.frame = 0

    def _track_for(self, actor: Any) -> ActorTrack:
        t = self.tracks.get(actor.id)
        if t is None:
            bb = actor.bounding_box
            kind = actor.type_id.split(".")[0]
            t = ActorTrack(
                actor_id=actor.id,
                blueprint=actor.type_id,
                role_name=actor.attributes.get("role_name", ""),
                category=_CATEGORY.get(kind, "other_vehicle"),
                size_lwh=(2 * bb.extent.x, 2 * bb.extent.y, 2 * bb.extent.z),
                bbox_offset=(bb.location.x, bb.location.y, bb.location.z),
            )
            self.tracks[actor.id] = t
        return t

    def sample(self) -> None:
        """Call once per tick after the world advanced."""
        t_s = self.frame / self.fps
        actors = list(self.world.get_actors().filter("vehicle.*")) + list(
            self.world.get_actors().filter("walker.pedestrian.*")
        )
        for a in actors:
            tr = a.get_transform()
            self._track_for(a).samples.append(
                [
                    t_s,
                    tr.location.x,
                    tr.location.y,
                    tr.location.z,
                    tr.rotation.roll,
                    tr.rotation.pitch,
                    tr.rotation.yaw,
                ]
            )
        self.frame += 1

    def save(self, path: str | Path) -> Path:
        doc = {
            "fps": self.fps,
            "frames": self.frame,
            "tracks": [
                {
                    "actor_id": t.actor_id,
                    "blueprint": t.blueprint,
                    "role_name": t.role_name,
                    "category": t.category,
                    "size_lwh": list(t.size_lwh),
                    "bbox_offset": list(t.bbox_offset),
                    "samples": t.samples,
                }
                for t in self.tracks.values()
            ],
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(doc))
        log.info("recorded %d tracks x %d frames → %s", len(doc["tracks"]), self.frame, path)
        return path


def load_tracks(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


# -- ClipGT export -----------------------------------------------------------


def _yaw_quat_flu(roll_deg: float, pitch_deg: float, yaw_deg: float):
    """CARLA rotation (left-handed, degrees) → FLU quaternion (x, y, z, w)."""
    roll, pitch, yaw = math.radians(roll_deg), math.radians(-pitch_deg), math.radians(-yaw_deg)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def tracks_to_obstacle_table(doc: dict, clip_id: str, exclude_roles: tuple[str, ...] = ("hero",)):
    """Recorded tracks → the ClipGT obstacle table (pyarrow), FLU world frame."""
    import pyarrow as pa

    rows = []
    for t in doc["tracks"]:
        if t["role_name"] in exclude_roles:
            continue
        ox, oy, oz = t["bbox_offset"]
        for s in t["samples"]:
            t_s, x, y, z, roll, pitch, yaw = s
            # bbox centre: actor origin + offset rotated by yaw (flat offset)
            cy, sy = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
            cx = x + ox * cy - oy * sy
            cy_ = y + ox * sy + oy * cy
            cz = z + oz
            qx, qy, qz, qw = _yaw_quat_flu(roll, pitch, yaw)
            rows.append(
                {
                    "key": {
                        "clip_id": clip_id,
                        "timestamp_micros": int(round(t_s * 1e6)),
                        "label_class_id": "carla_transfer:tracks:v1",
                    },
                    "obstacle": {
                        "trackline_id": str(t["actor_id"]),
                        "center": {"x": cx, "y": -cy_, "z": cz},
                        "size": {
                            "x": t["size_lwh"][0],
                            "y": t["size_lwh"][1],
                            "z": t["size_lwh"][2],
                        },
                        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw},
                        "category": t["category"],
                    },
                    "version": 0,
                }
            )
    xyz = pa.struct([("x", pa.float64()), ("y", pa.float64()), ("z", pa.float64())])
    schema = pa.schema(
        [
            (
                "key",
                pa.struct(
                    [
                        ("clip_id", pa.string()),
                        ("timestamp_micros", pa.int64()),
                        ("label_class_id", pa.string()),
                    ]
                ),
            ),
            (
                "obstacle",
                pa.struct(
                    [
                        ("trackline_id", pa.string()),
                        ("center", xyz),
                        ("size", xyz),
                        (
                            "orientation",
                            pa.struct(
                                [
                                    ("x", pa.float64()),
                                    ("y", pa.float64()),
                                    ("z", pa.float64()),
                                    ("w", pa.float64()),
                                ]
                            ),
                        ),
                        ("category", pa.string()),
                    ]
                ),
            ),
            ("version", pa.uint64()),
        ]
    )
    return pa.Table.from_pylist(rows, schema=schema)


def obstacle_parquet_bytes(doc: dict, clip_id: str) -> bytes:
    import io

    import pyarrow.parquet as pq

    buf = io.BytesIO()
    pq.write_table(tracks_to_obstacle_table(doc, clip_id), buf)
    return buf.getvalue()


# -- replay ------------------------------------------------------------------


class TrackReplayer:
    """Spawn the recorded actors (physics off) and drive them by transform."""

    def __init__(self, world: Any, doc: dict):
        import carla

        self.world = world
        self.doc = doc
        self.actors: list[tuple[Any, list[list[float]]]] = []
        lib = world.get_blueprint_library()
        for t in doc["tracks"]:
            if not t["samples"]:
                continue
            bps = lib.filter(t["blueprint"]) or lib.filter("vehicle.*")
            bp = bps[0]
            if t["role_name"]:
                bp.set_attribute("role_name", t["role_name"])
            s0 = t["samples"][0]
            tf = carla.Transform(
                carla.Location(x=s0[1], y=s0[2], z=s0[3] + 0.05),
                carla.Rotation(roll=s0[4], pitch=s0[5], yaw=s0[6]),
            )
            actor = world.try_spawn_actor(bp, tf)
            if actor is None:
                log.warning("could not respawn %s at %s", t["blueprint"], tf)
                continue
            actor.set_simulate_physics(False)
            self.actors.append((actor, t["samples"]))
        log.info("replaying %d actors", len(self.actors))

    @property
    def frames(self) -> int:
        return int(self.doc["frames"])

    def apply(self, frame: int) -> None:
        import carla

        for actor, samples in self.actors:
            s = samples[min(frame, len(samples) - 1)]
            actor.set_transform(
                carla.Transform(
                    carla.Location(x=s[1], y=s[2], z=s[3]),
                    carla.Rotation(roll=s[4], pitch=s[5], yaw=s[6]),
                )
            )

    def hero(self, role_name: str = "hero"):
        for actor, _ in self.actors:
            if actor.attributes.get("role_name") == role_name:
                return actor
        return None

    def destroy(self) -> None:
        for actor, _ in self.actors:
            actor.destroy()


def spawn_traffic(
    world, client, n_vehicles: int, tm_port: int, seed: int = 0, near=None, radius_m: float = 80.0
) -> list:
    """Background vehicles on Traffic Manager autopilot.

    With ``near`` (a carla.Location, normally the hero) the vehicles are
    placed on lane waypoints ahead of and opposite to it, plus the nearest
    map spawn points, so they actually cross the camera in a 10 s rollout;
    otherwise random spawn points across the map."""
    import random

    rng = random.Random(seed)
    lib = [
        b
        for b in world.get_blueprint_library().filter("vehicle.*")
        if int(b.get_attribute("number_of_wheels")) == 4
    ]
    cmap = world.get_map()
    transforms = []
    if near is not None:
        import carla

        wp = cmap.get_waypoint(near)
        # Same direction, ahead: ~20 m spacing; opposite lane: oncoming cars.
        for d in (18.0, 36.0, 54.0, 72.0):
            nxt = wp.next(d)
            if nxt:
                transforms.append(nxt[0].transform)
        opp = wp.get_left_lane()
        while opp is not None and opp.lane_type != carla.LaneType.Driving:
            opp = opp.get_left_lane()
        if opp is not None and opp.lane_id * wp.lane_id < 0:
            for d in (25.0, 45.0, 65.0, 85.0):
                nxt = opp.next(d)
                if nxt:
                    transforms.append(nxt[0].transform)
        spawns = sorted(cmap.get_spawn_points(), key=lambda t: t.location.distance(near))
        transforms += [t for t in spawns if 12.0 < t.location.distance(near) < radius_m]
    else:
        spawns = cmap.get_spawn_points()
        rng.shuffle(spawns)
        transforms = spawns
    actors = []
    for tf in transforms:
        if len(actors) >= n_vehicles:
            break
        bp = rng.choice(lib)
        bp.set_attribute("role_name", "traffic")
        tf.location.z += 0.3
        a = world.try_spawn_actor(bp, tf)
        if a is not None:
            a.set_autopilot(True, tm_port)
            actors.append(a)
    log.info("spawned %d traffic vehicles", len(actors))
    return actors
