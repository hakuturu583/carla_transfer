# FastVideo experiments (Wan2.1-Fun-Control × Self-Forcing)

Scratch env: `/mnt/sata_ssd/carla_transfer_work/fvvenv` (FastVideo main @ 2026-09-18, torch 2.12 cu126).

- `run_fun_control.py` — bidirectional `IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers` with a CARLA-derived control
  video (Canny) + reference image. 512x288x49f: 1.4 s/step with CFG on the 3090 (10 steps ≈ 14 s DiT).
- `merge_sf_fun.py` — task-arithmetic merge `base + sf_scale*(SelfForcing-base) + alpha*(FunControl-base)`,
  patch embedding widened to 48 channels (noisy | control | zero pad). CLIP ref-image branch dropped.
- `wan_causal_control_pipeline.py` — drop into `fastvideo/pipelines/basic/wan/` of the FastVideo checkout;
  `WanCausalControlDMDPipeline` = causal DMD (4-step, KV cache) + control-latent concat per block.
- `run_causal_control.py <merged_model_dir> <control.mp4> <out> [frames] [dmd_steps]` — 57 frames = 15 latent
  frames = 5 causal blocks; 4 steps: 20 DiT calls at 3.7 it/s (≈5.4 s DiT for 3.6 s of video, no CFG).

Status 2026-09-21: control conditioning transfers to the causal model (geometry locked to CARLA) but the
merged weights render washed-out, mesh-textured frames; 8 DMD steps help, sf_scale=2 slightly. A short
LoRA distillation (teacher = Fun-Control) is the next step to recover quality.
