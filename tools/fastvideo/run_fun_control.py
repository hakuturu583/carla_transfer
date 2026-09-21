import sys, time, os
from fastvideo import VideoGenerator
def main():
    ctrl, ref, out, steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    gen = VideoGenerator.from_pretrained(
        "IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers", num_gpus=1, use_fsdp_inference=False,
        dit_cpu_offload=False, vae_cpu_offload=False, text_encoder_cpu_offload=True, pin_cpu_memory=False)
    prompt = ("Dashcam footage of a real city street on a sunny day, photorealistic, shot from a car driving down a wide "
              "multi-lane avenue with a double yellow center line, modern glass office buildings, a white bus and a black sedan "
              "ahead, sharp natural lighting, realistic asphalt and shadows, high quality, 4k.")
    neg = "cartoon, cgi, video game, rendering, blurry, low quality, distorted, watermark, text"
    t = time.time()
    gen.generate_video(prompt, negative_prompt=neg, image_path=ref, video_path=ctrl, height=288, width=512, num_frames=49,
                       num_inference_steps=steps, guidance_scale=6.0, fps=16, seed=0,
                       output_path=".", output_video_name=out, save_video=True)
    print("GEN_SECONDS", round(time.time() - t, 1))

if __name__ == '__main__':
    main()
