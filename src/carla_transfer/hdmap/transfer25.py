"""First image via Cosmos-Transfer2.5-2B auto-multiview (world-scenario control).

Cosmos-Transfer2.5's "Auto / Multiview" checkpoint is the model whose control
input *is* an HD map: it takes world-scenario videos rendered from ClipGT
parquet (the same layers roadgen writes) and generates photoreal driving
video. Only the front camera is active here, so it runs on one GPU.

Both steps are subprocesses of the cosmos-transfer2.5 checkout
(``Transfer25Config.repo``), which has its own uv venv:

    1. scripts/generate_control_videos.py  clipgt dir → ws_<camera>.mp4
    2. examples/multiview.py               spec.json  → <name>.mp4

On a 24 GB card the 7B text encoder and the 2B denoiser must not share the
card; the checkout carries a local patch enabling that with
``COSMOS_EVICT_TEXT_ENCODER=1`` (see README "Cosmos-Transfer2.5 on the 3090").
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from carla_transfer.config import Transfer25Config

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "The video captures a daytime urban driving scene from a front-facing dashcam "
    "inside a vehicle. The car drives forward along a clean asphalt road with white "
    "and yellow lane markings and concrete curbs. Mid-rise buildings with shop fronts, "
    "green street trees and lamp posts line both sides of the street, with a few "
    "parked cars at the kerb. The sky is clear and blue, the sunlight is bright with "
    "crisp shadows, and the colors are natural. The footage is sharp, high resolution "
    "and photorealistic."
)


def interpreter(cfg: Transfer25Config) -> Path:
    return cfg.python or (cfg.repo / ".venv" / "bin" / "python")


def write_empty_obstacle_parquet(clip_dir: Path, clip_id: str) -> Path:
    """The native renderer requires ``{clip_id}.obstacle.parquet``; roadgen
    writes no dynamic actors, so give it an empty table with the documented
    schema (docs/world_scenario_parquet.md)."""
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
    path = clip_dir / f"{clip_id}.obstacle.parquet"
    if not path.exists():
        pq.write_table(pa.table({"key": [], "obstacle": [], "version": []}, schema=schema), path)
    return path


def _env(cfg: Transfer25Config) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if cfg.evict_text_encoder:
        env["COSMOS_EVICT_TEXT_ENCODER"] = "1"
    env.update(cfg.env)
    return env


def render_world_scenario(
    cfg: Transfer25Config, clip_dir: Path, clip_id: str, out_dir: Path
) -> Path:
    write_empty_obstacle_parquet(clip_dir, clip_id)
    cmd = [
        str(interpreter(cfg)), "scripts/generate_control_videos.py",
        "-i", str(clip_dir), "-o", str(out_dir), "--cameras", cfg.camera,
    ]  # fmt: skip
    log.info("world scenario: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cfg.repo, env=_env(cfg), check=True)
    out = out_dir / clip_id / f"{clip_id}.{cfg.camera}.mp4"
    if not out.exists():
        raise FileNotFoundError(out)
    return out


def build_spec(cfg: Transfer25Config, control_mp4: Path, prompt_path: Path, name: str) -> dict:
    """examples/multiview.py input: front_wide only, no conditioning frames
    (control video only), so one GPU suffices."""
    spec = {
        "name": name,
        "prompt_path": str(prompt_path),
        "fps": cfg.fps,
        "num_conditional_frames": 0,
        "guidance": cfg.guidance,
        "seed": cfg.seed,
        "front_wide": {"control_path": str(control_mp4.resolve())},
    }
    if cfg.num_steps:
        spec["num_steps"] = cfg.num_steps
    return spec


def run_multiview(cfg: Transfer25Config, control_mp4: Path, work_dir: Path, name: str) -> Path:
    """Front-view-only auto-multiview inference. Returns the output mp4."""
    work_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = work_dir / "prompt.txt"
    prompt_path.write_text((cfg.prompt or DEFAULT_PROMPT).strip() + "\n")
    spec_path = work_dir / "spec.json"
    spec_path.write_text(json.dumps(build_spec(cfg, control_mp4, prompt_path, name), indent=2))
    out_dir = work_dir / "out"
    cmd = [
        str(interpreter(cfg)), "-m", "torch.distributed.run", "--nproc_per_node=1",
        "--master_port=12355", "examples/multiview.py",
        "-i", str(spec_path), "-o", str(out_dir),
    ]  # fmt: skip
    if cfg.disable_guardrails:
        cmd.append("--disable-guardrails")
    log.info("multiview: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=cfg.repo, env=_env(cfg), check=True)
    out = out_dir / f"{name}.mp4"
    if not out.exists():
        raise FileNotFoundError(out)
    return out


def first_frame_png(mp4: Path, png: Path) -> Path:
    import imageio.v3 as iio
    from PIL import Image

    frame = iio.imread(mp4, index=0, plugin="pyav")
    Image.fromarray(frame).save(png)
    return png


def first_image_from_clipgt(
    cfg: Transfer25Config, clip_dir: Path, clip_id: str, work_dir: Path
) -> Path:
    """clipgt dir → world-scenario video → Transfer2.5 video → first_image.png"""
    ws = render_world_scenario(cfg, clip_dir, clip_id, work_dir / "ws")
    video = run_multiview(cfg, ws, work_dir / "transfer25", clip_id)
    return first_frame_png(video, work_dir / "first_image.png")
