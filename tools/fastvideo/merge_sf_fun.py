"""Task-arithmetic merge: Self-Forcing causal Wan2.1-1.3B + (Fun-Control - base Wan2.1-1.3B).

Output: a diffusers model dir usable by FastVideo's causal DMD pipeline with
in_channels=48 (noisy 16 | control 16 | zero pad 16). The CLIP reference-image
branch (add_k/add_v proj) is dropped: the causal cross-attention has no I2V path.
"""
import glob, json, os, shutil, sys
import torch
from safetensors.torch import load_file, save_file

HUB = "/mnt/sata_ssd/hf/cache/hub"
def snap(repo): return glob.glob(f"{HUB}/models--{repo.replace('/', '--')}/snapshots/*")[0]
SF, BASE, FUN = snap("wlsaidhi/SFWan2.1-T2V-1.3B-Diffusers"), snap("Wan-AI/Wan2.1-T2V-1.3B-Diffusers"), snap("IRMChen/Wan2.1-Fun-1.3B-Control-Diffusers")
OUT = sys.argv[1]; alpha = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0; sf_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

def load_dir(d):
    sd = {}
    for f in sorted(glob.glob(f"{d}/transformer/*.safetensors")): sd.update(load_file(f))
    return sd
sf, base, fun = load_dir(SF), load_dir(BASE), load_dir(FUN)
print(len(sf), len(base), len(fun))
merged, dropped, stats = {}, [], []
for k, v in sf.items():
    if k not in fun:  # SF-only key (shouldn't happen)
        merged[k] = v; print("sf-only", k); continue
    f, b = fun[k].float(), base[k].float()
    if k == "patch_embedding.weight":
        out = torch.empty(f.shape, dtype=torch.float32)
        out[:, :16] = b + sf_scale * (v.float() - b) + alpha * (f[:, :16] - b)
        out[:, 16:] = f[:, 16:]
        merged[k] = out.to(v.dtype); continue
    assert f.shape == v.shape == b.shape, (k, f.shape, v.shape, b.shape)
    delta = f - b
    stats.append((k, (delta.norm() / (b.norm() + 1e-8)).item(), ((v.float() - b).norm() / (b.norm() + 1e-8)).item()))
    merged[k] = (b + sf_scale * (v.float() - b) + alpha * delta).to(v.dtype)
for k in fun:
    if k not in sf: dropped.append(k)
print("dropped fun-only keys:", len(dropped), dropped[:6])
import statistics
print("rel |fun-base| median %.4f  max %.4f ; rel |sf-base| median %.4f" % (
    statistics.median(s[1] for s in stats), max(s[1] for s in stats), statistics.median(s[2] for s in stats)))
os.makedirs(f"{OUT}/transformer", exist_ok=True)
save_file(merged, f"{OUT}/transformer/diffusion_pytorch_model.safetensors")
cfg = json.load(open(f"{SF}/transformer/config.json")); cfg["in_channels"] = 48
json.dump(cfg, open(f"{OUT}/transformer/config.json", "w"), indent=2)
for sub in ["scheduler", "text_encoder", "tokenizer", "vae"]:
    if os.path.lexists(f"{OUT}/{sub}"): os.remove(f"{OUT}/{sub}")
    os.symlink(f"{FUN}/{sub}", f"{OUT}/{sub}")
os.remove(f"{OUT}/scheduler"); os.symlink(f"{SF}/scheduler", f"{OUT}/scheduler")
shutil.copy(f"{SF}/model_index.json", f"{OUT}/model_index.json")
print("wrote", OUT)
