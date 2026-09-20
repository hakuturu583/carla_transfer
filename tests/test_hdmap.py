import json
import zipfile

import numpy as np
import pyarrow.parquet as pq
import pytest

from carla_transfer.hdmap import CameraSpec, SceneSpec, export_scene_archive, sanitize_carla_xodr
from carla_transfer.hdmap.control_video import archive_trajectory, resample_trajectory
from carla_transfer.hdmap.prepare import replace_in_archive


def test_sanitize_carla_xodr_rewrites_only_the_known_quirks():
    src = """<OpenDRIVE><header><userData><vectorScene program="RoadRunner"/></userData></header>
<roadMark sOffset="0" type="curb" material="standard" width="0.15" laneChange="none"/>
<roadMark sOffset="0" type="solid" color="white"/>
<object id="1" type="-1" orientation="+"/><object id="2" type="crosswalk"/>
<cornerLocal u="1" v="2" z="0"/><cornerLocal u="1" v="2" z="0" height="1"/></OpenDRIVE>"""
    out = sanitize_carla_xodr(src)
    assert "userData" not in out
    assert 'laneChange="none" color="standard"/>' in out
    assert '<roadMark sOffset="0" type="solid" color="white"/>' in out
    assert 'type="none" orientation="+"' in out and 'type="crosswalk"' in out
    assert out.count('height="0"') == 1 and 'height="1"' in out


@pytest.fixture(scope="module")
def tiny_xodr(tmp_path_factory):
    roadgen = pytest.importorskip("roadgen")
    m = roadgen.Map()
    a = m.add_road(
        start=(0.0, 0.0, 0.0),
        end=(120.0, 0.0, 0.0),
        lanes=[
            roadgen.Lane(width=3.5, direction="forward"),
            roadgen.Lane(width=3.5, direction="backward"),
        ],
    )
    b = m.add_road(
        start=(120.0, 0.0, 0.0),
        end=(200.0, 40.0, 0.0),
        lanes=[
            roadgen.Lane(width=3.5, direction="forward"),
            roadgen.Lane(width=3.5, direction="backward"),
        ],
    )
    m.connect(a, b)
    path = tmp_path_factory.mktemp("map") / "tiny.xodr"
    m.export_opendrive(str(path))
    return path


def test_export_scene_archive_layout(tiny_xodr, tmp_path):
    out = tmp_path / "tiny.usdz"
    spec = SceneSpec(clip_id="tiny", speed_mps=5.0, camera=CameraSpec(width=128, height=64))
    export_scene_archive(tiny_xodr, out, spec)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        for required in (
            "clipgt/lane.parquet",
            "clipgt/calibration_estimate.parquet",
            "clipgt/egomotion_estimate.parquet",
            "metadata.yaml",
            "rig_trajectories.json",
            "prompt1.txt",
            "first_image.png",
        ):
            assert required in names, required
        traj = json.loads(zf.read("rig_trajectories.json"))["rig_trajectories"][0]
        poses = np.asarray(traj["T_rig_worlds"])
        ts = traj["T_rig_world_timestamps_us"]
        assert poses.shape[1:] == (4, 4) and len(ts) == len(poses) > 30
        assert ts == sorted(ts)
        # ego moves forward along +x on the first road at ~5 m/s @ 30 fps
        assert 0.1 < poses[1, 0, 3] - poses[0, 0, 3] < 0.25
        assert zf.read("prompt1.txt").decode().startswith("Driving scene")
        with zf.open("clipgt/calibration_estimate.parquet") as f:
            rig = json.loads(pq.read_table(f).to_pylist()[0]["calibration_estimate"]["rig_json"])
        sensor = rig["rig"]["sensors"][0]
        assert sensor["name"] == "camera:front:wide:120fov"
        assert sensor["properties"]["width"] == "128"

    # trajectory helpers used by the control-video renderer
    poses, ts = archive_trajectory(out)
    p2, t2 = resample_trajectory(poses, ts, fps=10, num_frames=5)
    assert p2.shape == (5, 4, 4)
    assert np.all(np.diff(t2) >= 90_000)

    # archive member replacement keeps everything else intact
    replace_in_archive(out, "first_image.png", b"NEWPNG")
    with zipfile.ZipFile(out) as zf:
        assert zf.read("first_image.png") == b"NEWPNG"
        assert "clipgt/lane.parquet" in zf.namelist()


def test_transfer25_helpers(tmp_path):
    from carla_transfer.config import Transfer25Config
    from carla_transfer.hdmap.transfer25 import build_spec, write_empty_obstacle_parquet

    path = write_empty_obstacle_parquet(tmp_path, "clip")
    t = pq.read_table(path)
    assert t.num_rows == 0
    assert [f.name for f in t.schema] == ["key", "obstacle", "version"]
    obstacle = t.schema.field("obstacle").type
    assert {obstacle.field(i).name for i in range(obstacle.num_fields)} == {
        "trackline_id", "center", "size", "orientation", "category",
    }  # fmt: skip

    cfg = Transfer25Config(guidance=2.5, num_steps=20)
    spec = build_spec(cfg, tmp_path / "ws.mp4", tmp_path / "p.txt", "town")
    assert spec["num_conditional_frames"] == 0 and spec["guidance"] == 2.5
    assert spec["num_steps"] == 20 and set(spec) & {"rear", "cross_left"} == set()
    assert spec["front_wide"]["control_path"].endswith("ws.mp4")
