"""``carla-transfer-cosmos3``: photoreal video from an HD-map control video.

Runs Cosmos3 *transfer* (structural control) with the world-scenario ("wsm")
hint through diffusers' ``Cosmos3OmniModularPipeline``: the control video is
the rasterization of the CARLA map (ludus via hdmap/control_video.py, or
cosmos-transfer2.5's world-scenario renderer), the output is a photoreal
drive along the same road. Its first frame becomes OmniDreams' initial image.

Fitting Cosmos3-Nano on a 24 GB Ampere card (measured on an RTX 3090:
320x192/33 frames/20 steps in 111 s, 4.8 GiB peak):

* ``revision="fp8"`` — the ModelOpt FP8 checkpoint (16.5 GB transformer).
* ``modelopt.torch.opt.enable_huggingface_checkpointing()`` before loading,
  and the transformer loaded from the *local snapshot directory*: ModelOpt's
  restore hook looks for ``modelopt_state.pth`` next to the config, and when
  given the hub id it looks at the repo root (revision main) and silently
  skips the restore — the model then holds raw float8 weights in plain
  ``nn.Linear`` and fails with a dtype mismatch.
* ``nvidia-modelopt==0.44.*`` — the version that produced the checkpoint;
  0.46 rejects its quantizer state ("Extra keys in quantizer state_dict").
* every denoising step in W8A16 (``mixed_precision_first_steps == last_steps
  == num_inference_steps``): Ampere has no FP8 GEMM, and the native W8A8
  forward raises on it. W8A16 dequantizes each layer's weight on the fly.
* the transformer group-offloaded to the host (block level, streamed), so
  the resident GPU set is activations + one block. 31 GB of host RAM holds
  the fp8 weights.
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
    p.add_argument("--revision", default="fp8", help='"fp8" (24 GB cards) or "main" (bf16)')
    p.add_argument("--hint", default="wsm", help="control hint key (world scenario map)")
    p.add_argument("--prompt-json", type=Path)
    p.add_argument("--negative-prompt-json", type=Path)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--num-frames", type=int, default=61)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--control-fps", type=float, default=None,
                   help="fps of the control video if it differs from --fps")  # fmt: skip
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--guidance-scale", type=float, default=1.0)
    p.add_argument("--control-guidance", type=float, default=4.0)
    p.add_argument("--flow-shift", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--offload", choices=["group", "none"], default="group",
                   help="group: stream transformer blocks from host RAM (24 GB cards)")  # fmt: skip
    p.add_argument("--w8a16-all-steps", action=argparse.BooleanOptionalAction, default=True,
                   help="dequant on every step (GPUs without FP8 GEMM, e.g. Ampere)")  # fmt: skip
    return p


def load_pipeline(model: str, revision: str | None, offload: str):
    import torch
    from huggingface_hub import snapshot_download
    from modelopt.torch.opt import enable_huggingface_checkpointing

    enable_huggingface_checkpointing()
    from diffusers import Cosmos3OmniModularPipeline, Cosmos3OmniTransformer

    snap = Path(snapshot_download(model, revision=revision, ignore_patterns=["sound_tokenizer/*"]))
    pipe = Cosmos3OmniModularPipeline.from_pretrained(str(snap), dtype=torch.bfloat16)
    others = [n for n in pipe.component_names if n not in ("transformer", "sound_tokenizer")]
    pipe.load_components(names=others, torch_dtype=torch.bfloat16)
    # Local directory so ModelOpt finds transformer/modelopt_state.pth (see module doc).
    tr = Cosmos3OmniTransformer.from_pretrained(
        str(snap / "transformer"), torch_dtype=torch.bfloat16
    )
    pipe.update_components(transformer=tr)
    n_q = sum(1 for _, m in tr.named_modules() if "Quant" in type(m).__name__)
    log.info("transformer loaded from %s; %d ModelOpt quantizer modules", snap, n_q)

    if offload == "group":
        tr.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type="block_level",
            num_blocks_per_group=1,
            use_stream=True,
        )
        for name, comp in pipe.components.items():
            if name != "transformer" and isinstance(comp, torch.nn.Module):
                comp.to("cuda")
    else:
        pipe.to("cuda")
    pipe.disable_safety_checker()
    return pipe


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    from diffusers.utils import export_to_video, load_video
    from PIL import Image

    pipe = load_pipeline(args.model, args.revision, args.offload)
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config, flow_shift=args.flow_shift, use_karras_sigmas=False
    )

    control = load_video(str(args.control))
    if args.control_fps and args.control_fps != args.fps:
        control = control[:: max(1, round(args.control_fps / args.fps))]
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
    if args.w8a16_all_steps:
        call.update(
            mixed_precision_format="fp8",
            mixed_precision_first_steps=args.steps,
            mixed_precision_last_steps=args.steps,
        )
    log.info(
        "generating %d frames at %dx%d, %d steps",
        args.num_frames,
        args.width,
        args.height,
        args.steps,
    )
    torch.cuda.reset_peak_memory_stats()
    videos = pipe(**call)
    log.info("peak GPU memory %.1f GiB", torch.cuda.max_memory_allocated() / 2**30)

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
