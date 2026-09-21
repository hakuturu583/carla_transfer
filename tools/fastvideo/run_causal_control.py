import sys, time
from fastvideo import VideoGenerator
from fastvideo.pipelines.basic.wan.wan_causal_control_pipeline import SelfForcingControlConfig

def main():
    model, ctrl, out = sys.argv[1], sys.argv[2], sys.argv[3]
    frames = int(sys.argv[4]) if len(sys.argv) > 4 else 57
    steps = [int(x) for x in sys.argv[5].split(',')] if len(sys.argv) > 5 else None
    cfg = SelfForcingControlConfig()
    if steps: cfg.dmd_denoising_steps = steps
    gen = VideoGenerator.from_pretrained(
        model, num_gpus=1, use_fsdp_inference=False, dit_cpu_offload=False, vae_cpu_offload=False,
        text_encoder_cpu_offload=True, pin_cpu_memory=False,
        override_pipeline_cls_name="WanCausalControlDMDPipeline", pipeline_config=cfg)
    prompt = ("Dashcam footage of a real city street on a sunny day, photorealistic, shot from a car driving down a wide "
              "multi-lane avenue with a double yellow center line, modern glass office buildings, a white bus and a black sedan "
              "ahead, sharp natural lighting, realistic asphalt and shadows, high quality, 4k.")
    for i in range(2):
        t = time.time()
        gen.generate_video(prompt, video_path=ctrl, height=288, width=512, num_frames=frames, fps=16, seed=0,
                           output_path=".", output_video_name=f"{out}_{i}", save_video=True)
        print("GEN_SECONDS", i, round(time.time() - t, 1), flush=True)

if __name__ == "__main__":
    main()
