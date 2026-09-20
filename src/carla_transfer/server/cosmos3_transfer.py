"""``carla-transfer-cosmos3``: photoreal video from an HD-map control video.

Runs Cosmos3 *transfer* (structural control) with the world-scenario ("wsm")
hint through diffusers' ``Cosmos3OmniModularPipeline``: the control video is
the ludus rasterization of the CARLA map (see hdmap/control_video.py), the
output is a photoreal drive along the same road. Its first frame becomes
OmniDreams' initial image, so OmniDreams starts from a plausible photo of
*this* map rather than from CARLA's own rendering.

Meant to run inside the Cosmos3 container (docker/Dockerfile.cosmos3) but has
no container-specific assumptions: any environment with diffusers >= the
Cosmos3 modular pipeline, torch and a GPU works.

Defaults follow the diffusers transfer table for the WSM hint
(guidance 1.0, control_guidance 3.0, flow_shift 10, 101 frames @ 10 fps).
"""

from __future__ import annotations

import argparse
import json
import logging
from importlib import resources
from pathlib import Path

log = logging.getLogger("carla_transfer.cosmos3")

ASSETS = resources.files("carla_transfer") / "assets"


def load_prompt(path: Path | None, width: int, height: int, num_frames: int, fps: float) -> str:
    if path is None:
        doc = json.loads((ASSETS / "cosmos3_driving_prompt.json").read_text())
    else:
        doc = json.loads(Path(path).read_text())
    doc["resolution"] = {"W": width, "H": height}
    doc["fps"] = fps
    doc["duration"] = f"{round(num_frames / fps)}s"
    return json.dumps(doc)


def load_negative_prompt(path: Path | None) -> str:
    if path is None:
        return (ASSETS / "cosmos3_negative_prompt.json").read_text()
    return Path(path).read_text()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--control", type=Path, required=True, help="HD-map control video (mp4)")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--model", default="nvidia/Cosmos3-Nano")
    p.add_argument("--revision", default=None, help='e.g. "fp8" for the ModelOpt FP8 checkpoint')
    p.add_argument("--hint", default="wsm", help="control hint key (world scenario map)")
    p.add_argument("--prompt-json", type=Path)
    p.add_argument("--negative-prompt-json", type=Path)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num-frames", type=int, default=101)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--control-guidance", type=float, default=3.0)
    p.add_argument("--flow-shift", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--cpu-offload", action="store_true", help="model CPU offload (24 GB cards)")
    p.add_argument("--no-guardrails", action="store_true")
    return p


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch
    from diffusers import Cosmos3OmniModularPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video, load_video
    from PIL import Image

    kw = {"dtype": torch.bfloat16}
    if args.revision:
        kw["revision"] = args.revision
    log.info("loading %s %s", args.model, kw)
    pipe = Cosmos3OmniModularPipeline.from_pretrained(args.model, **kw)
    pipe.load_components(dtype=torch.bfloat16)
    if args.cpu_offload:
        for name in ("enable_model_cpu_offload", "enable_auto_cpu_offload"):
            fn = getattr(pipe, name, None)
            if fn is not None:
                log.info("using %s", name)
                fn()
                break
        else:
            log.warning("no CPU-offload method on this pipeline; loading fully on GPU")
            pipe.to("cuda")
    else:
        pipe.to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
    )

    control = load_video(str(args.control))
    if len(control) < args.num_frames:
        raise SystemExit(f"control video has {len(control)} frames, need {args.num_frames}")
    control = [f.resize((args.width, args.height)) for f in control[: args.num_frames]]

    call = dict(
        prompt=load_prompt(args.prompt_json, args.width, args.height, args.num_frames, args.fps),
        negative_prompt=load_negative_prompt(args.negative_prompt_json),
        control_videos={args.hint: control},
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        fps=args.fps,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        control_guidance=args.control_guidance,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        output="videos",
    )
    if args.no_guardrails:
        call["guardrails"] = False
    log.info(
        "generating %d frames at %dx%d, %d steps",
        args.num_frames,
        args.width,
        args.height,
        args.steps,
    )
    videos = pipe(**call)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = args.out_dir / "vision.mp4"
    export_to_video(videos, str(out_mp4), fps=int(args.fps), macro_block_size=1)
    first = videos[0] if not isinstance(videos[0], list) else videos[0][0]
    if not isinstance(first, Image.Image):
        first = Image.fromarray(np.asarray(first))
    first.save(args.out_dir / "first_image.png")
    log.info("wrote %s and first_image.png", out_mp4)
    return out_mp4


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
