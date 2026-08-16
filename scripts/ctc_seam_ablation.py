"""The padding ablation, repeated on real CTC frames at native resolution.

The first ablation ran on 512x512 OpenImages crops: 16 tiles of 128px. A 1080p
frame padded to 1920x1152 is 15x9 = 135 tiles. The per-pixel share of border is
the same, but the content is not — video frames carry large smooth regions and
long straight edges that a 512 crop of OpenImages does not, and a padding scheme
that extrapolates could behave differently on either. Assuming the crop result
transfers is exactly the kind of assumption that produced the replicate error
earlier, so it gets measured instead.

Reference is always the stock full-frame decode of the SAME latent, so what is
isolated is the seam and nothing else.
"""
import argparse, json, sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.backbone.padding import wrap_tile_padding

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/warmstart/ckpt_warmstart.pth.tar")
ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
ap.add_argument("--frames", type=int, default=2)
ap.add_argument("--patch", type=int, default=8)      # 8 latent -> 128px RGB
ap.add_argument("--split", type=int, default=2)
ap.add_argument("--device", default="cuda:6")
a = ap.parse_args()

seqs, missing = C.discover([])
print(f"{len(seqs)} sekans olculuyor, {len(missing)} yok\n")
ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(split_depth=a.split, latent_patch=a.patch, tile_pad_mode="zeros",
                   seam_repair="none")
net = FlexUFIntra(cfg).to(a.device).eval()
load_flexuf_state(net, ck)

frames = []
for s in seqs:
    x, planes = C.read_frames(s["path"], s["w"], s["h"], a.frames,
                              max(1, s["frames"] // max(a.frames, 1)))
    if x is None:
        continue
    for i in range(x.shape[0]):
        frames.append((s, x[i:i+1], planes[i]))
print(f"{len(frames)} kare\n")

print(f"SAF DIKIS CEZASI, dB — j={a.split}, {a.patch*16}px tile, CTC native res\n")
print(f"  {'mod':<11}" + ''.join(f"{'qp'+str(q):>9}" for q in a.qps))
res = {}
for mode in ("zeros", "replicate", "linear", "arls"):
    undo = wrap_tile_padding(net.dec.groups, cfg.split_depth, mode)
    row = []
    with torch.no_grad():
        for qp_v in a.qps:
            gap = 0.0
            for s, x, pl in frames:
                x = x.to(a.device)
                _, _, H, W = x.shape
                import torch.nn.functional as F
                ph, pw = (-H) % cfg.rgb_patch, (-W) % cfg.rgb_patch
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=a.device)
                y_hat, q_dec, _ = net._encode_to_latent(xp, qp)
                nt = ((H + ph) // cfg.rgb_patch) * ((W + pw) // cfg.rgb_patch)
                em = torch.full((nt,), cfg.num_exits - 1, device=a.device)
                d = C.psnr_611_420(net.dec.forward_full(y_hat, q_dec)[:, :, :H, :W], pl)
                r = C.psnr_611_420(net.dec(y_hat, q_dec, exit_map=em)[:, :, :H, :W], pl)
                gap += d - r
            row.append(gap / len(frames))
    undo(); res[mode] = row
    extra = "" if mode == "zeros" else "   " + ' '.join(
        f"{res['zeros'][i]-row[i]:+7.3f}" for i in range(len(a.qps)))
    print(f"  {mode:<11}" + ''.join(f"{v:>9.4f}" for v in row) + extra, flush=True)

Path("results").mkdir(exist_ok=True)
json.dump({"qps": a.qps, "patch": a.patch*16, "split": a.split,
           "sequences": [s["name"] for s in seqs], "res": res},
          open(f"results/ctc_seam_p{a.patch*16}.json", "w"), indent=2)
