"""Build an OmniDreams scene archive from a CARLA OpenDRIVE map.

roadgen writes the ClipGT parquet layers (``{clip_id}.{layer}.parquet``) that
the ludus rasterizer reads. The OmniDreams interactive-drive loader wants
those inside a zip (``.usdz``) as ``clipgt/{layer}.parquet`` plus a few
sidecars; this module packs them:

    clipgt/<layer>.parquet   from roadgen (lane, lane_line, road_boundary,
                             crosswalk, ..., calibration_estimate,
                             egomotion_estimate)
    metadata.yaml            scene_id, sensor ids, time range
    rig_trajectories.json    ego poses as 4x4 rig-to-world + timestamps, from
                             egomotion_estimate; the loader takes its initial
                             pose/yaw/speed from here
    prompt1.txt              default prompt (overridable per session)
    first_image.png          placeholder; the session replaces it with the
                             CARLA frame or the Cosmos3-rendered image

The official archives also carry meshes and a checkpoint (2.3 GB); the
rasterizer does not need them, so a CARLA archive is a few hundred KB.
"""

from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from carla_transfer.hdmap.xodr import read_carla_xodr
from carla_transfer.server.geometry import quat_xyzw_to_matrix

# OmniDreams' front camera. Colon-separated like the official rigs so
# `camera_front_wide_120fov` (logical) ↔ `camera:front:wide:120fov` (clipgt)
# round-trips through omnidreams' normalize_camera_name.
DEFAULT_CAMERA = "camera:front:wide:120fov"


@dataclass
class CameraSpec:
    name: str = DEFAULT_CAMERA
    width: int = 512
    height: int = 288
    fov_deg: float = 120.0
    # Rig frame (FLU, metres / degrees). Roughly a windshield-mounted dashcam.
    position: tuple[float, float, float] = (1.7, 0.0, 1.45)
    roll_pitch_yaw: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def logical_name(self) -> str:
        return self.name.replace(":", "_")


@dataclass
class SceneSpec:
    clip_id: str = "carla"
    fps: float = 30.0
    speed_mps: float = 8.0
    # Either explicit lane ids in driving order, or a start lane to follow.
    route: list[str] = field(default_factory=list)
    start_lane: str | None = None
    max_route_lanes: int = 12
    camera: CameraSpec = field(default_factory=CameraSpec)
    prompt: str = (
        "Driving scene from a front-facing car camera. Urban environment with "
        "roads, vehicles, pedestrians, traffic signs, and buildings. Clear "
        "visibility, realistic lighting, photorealistic quality."
    )


def plan_route(m, spec: SceneSpec) -> list[str]:
    """Lane ids to drive: the explicit route, or successors from start_lane."""
    if spec.route:
        return list(spec.route)
    start = spec.start_lane or m.lane_ids()[0]
    route = [start]
    while len(route) < spec.max_route_lanes:
        nxt = m.successors(route[-1])
        if not nxt or nxt[0] in route:
            break
        route.append(nxt[0])
    return route


def export_clipgt_dir(m, spec: SceneSpec, out_dir: Path) -> str:
    """roadgen → ``out_dir/{clip_id}.{layer}.parquet``. Returns the clip id."""
    cam = spec.camera
    scenario = {
        "clip_id": spec.clip_id,
        "frame_rate": float(spec.fps),
        "speed": float(spec.speed_mps),
        "route": {"lanes": plan_route(m, spec)},
        "sensors": [
            {
                "name": cam.name,
                "position": list(cam.position),
                "roll_pitch_yaw": list(cam.roll_pitch_yaw),
                "width": cam.width,
                "height": cam.height,
                "fov_degrees": float(cam.fov_deg),
            }
        ],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = out_dir / "scenario.yaml"
    scenario_path.write_text(yaml.safe_dump(scenario, sort_keys=False))
    clip_id = m.export_clipgt(str(out_dir), scenario=str(scenario_path))
    normalize_lane_line_styles(out_dir, clip_id)
    return clip_id


# roadgen's lane_line style tokens → the ClipGT vocabulary the NVIDIA loaders
# know (config_color_geometry_laneline.json). Unknown tokens render as the
# generic "OTHER" colour instead of white/yellow dashed/double lines.
LANE_LINE_STYLE_MAP = {
    "DASHED_SINGLE": "LONG_DASHED_SINGLE",
    "SOLID_DOUBLE": "SOLID_GROUP",
    "DASHED_DOUBLE": "LONG_DASHED_SINGLE",
}


def normalize_lane_line_styles(clip_dir: Path, clip_id: str) -> int:
    """Rewrite ``{clip_id}.lane_line.parquet`` styles in place. Returns the
    number of rows changed."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = clip_dir / f"{clip_id}.lane_line.parquet"
    if not path.exists():
        return 0
    table = pq.read_table(path)
    rows = table.to_pylist()
    changed = 0
    for r in rows:
        styles = r["lane_line"].get("styles") or []
        new = [LANE_LINE_STYLE_MAP.get(s, s) for s in styles]
        if new != styles:
            r["lane_line"]["styles"] = new
            changed += 1
    if changed:
        pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    return changed


def rig_trajectory_from_egomotion(parquet_path: Path) -> dict:
    import pyarrow.parquet as pq

    rows = pq.read_table(parquet_path).to_pylist()
    rows.sort(key=lambda r: r["key"]["timestamp_micros"])
    poses, ts = [], []
    for r in rows:
        e = r["egomotion_estimate"]
        loc, q = e["location"], e["orientation"]
        m = np.eye(4)
        m[:3, :3] = quat_xyzw_to_matrix((q["x"], q["y"], q["z"], q["w"]))
        m[:3, 3] = (loc["x"], loc["y"], loc["z"])
        poses.append(m.tolist())
        ts.append(int(r["key"]["timestamp_micros"]))
    return {
        "rig_trajectories": [
            {"sequence_id": 0, "T_rig_worlds": poses, "T_rig_world_timestamps_us": ts}
        ]
    }


def pack_scene_archive(clip_dir: Path, clip_id: str, spec: SceneSpec, out_path: Path) -> Path:
    """Zip a roadgen clipgt directory into an OmniDreams scene archive."""
    prefix = f"{clip_id}."
    traj = rig_trajectory_from_egomotion(clip_dir / f"{prefix}egomotion_estimate.parquet")
    ts = traj["rig_trajectories"][0]["T_rig_world_timestamps_us"]
    cam = spec.camera
    metadata = {
        "scene_id": clip_id,
        "version_string": "carla_transfer",
        "sensors": {"camera_ids": [cam.logical_name], "lidar_ids": []},
        "time_range": {"start": ts[0], "end": ts[-1]},
    }
    placeholder = Image.fromarray(np.zeros((cam.height, cam.width, 3), np.uint8))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(clip_dir.iterdir()):
            if not f.name.startswith(prefix):
                continue
            rest = f.name[len(prefix) :]
            if f.suffix in (".parquet", ".json"):
                zf.write(f, f"clipgt/{rest}")
        zf.writestr("metadata.yaml", yaml.safe_dump(metadata, sort_keys=False))
        zf.writestr("rig_trajectories.json", json.dumps(traj))
        zf.writestr("prompt1.txt", spec.prompt.strip() + "\n")
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
            placeholder.save(tmp.name)
            zf.write(tmp.name, "first_image.png")
    return out_path


def export_scene_archive(
    xodr: str | Path, out_path: str | Path, spec: SceneSpec | None = None
) -> Path:
    """CARLA .xodr → OmniDreams scene archive (.usdz). Returns ``out_path``."""
    spec = spec or SceneSpec()
    m = read_carla_xodr(xodr)
    with tempfile.TemporaryDirectory() as tmp:
        clip_dir = Path(tmp) / "clip"
        clip_id = export_clipgt_dir(m, spec, clip_dir)
        return pack_scene_archive(clip_dir, clip_id, spec, Path(out_path))
