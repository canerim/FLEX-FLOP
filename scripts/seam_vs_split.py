"""Does the contamination law predict the seam?

Section 4 claims the corrupted fraction of a tile is 1 - ((F-2b)/F)^2 with b the
number of per-tile blocks and F the tile side in feature pixels. That is a
counting argument about how far a 3x3 can reach, and it has never been tested
against the thing it is supposed to explain -- the measured seam penalty.

The split depth j sets b directly: b = (K - j) * blocks_per_exit. Sweeping j
from 0 (everything per tile) to K (nothing per tile) sweeps b from 12 to 0 and
should sweep the seam from its maximum to exactly zero. j = K is the control
that must give 0.000 dB; if it does not, the measurement is wrong rather than
the law.

Untrained warm start, every tile at full depth, so nothing but geometry is in
the number.
"""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402
from ckpt import pinned as _pin  # noqa: E402
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=_pin("runs/RECIPE512/ckpt_eval.pth.tar"))
ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
ap.add_argument("--max_seqs", type=int, default=12)
ap.add_argument("--device", default=_gpu("cuda:7"))
ap.add_argument("--out", default="results/seam_vs_split.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg0 = FlexUFConfig(**ck["config"])
warm = torch.load(reference_for(cfg0, None), map_location="cpu",
                  weights_only=False)
K, P = cfg0.num_exits, cfg0.rgb_patch
b_per = cfg0.blocks_per_exit
Fp = cfg0.feature_patch

seqs, _ = C.discover([])
seqs = seqs[:a.max_seqs]
frames = []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
    if x is not None:
        frames.append((x[0:1], pl[0]))
print(f"  {len(frames)} frames, tile {P}px = {Fp} feature px, "
      f"{b_per} blocks per exit\n", flush=True)

def build(j):
    c = FlexUFConfig(**{**cfg0.__dict__, "split_depth": j,
                        "tile_pad_mode": "replicate", "seam_repair": "none"})
    m = FlexUFIntra(c).to(dev).eval(); load_flexuf_state(m, warm)
    return c, m

rows = []
print(f"  {'j':>3}{'b':>4}{'predicted frac':>16}" +
      "".join(f"{'q'+str(q):>10}" for q in a.qps))
with torch.no_grad():
    for j in range(0, K + 1):
        cfg, net = build(min(j, K))
        b = (K - j) * b_per
        frac = 1 - max(0.0, (Fp - 2 * b) / Fp) ** 2 if b > 0 else 0.0
        vals = []
        for qp_v in a.qps:
            tot = 0.0
            for x, pl in frames:
                x = x.to(dev); _, _, H, W = x.shape
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(xp, qp)
                nt = ((H + ph) // P) * ((W + pw) // P)
                em = torch.full((nt,), K - 1, device=dev)
                d0 = C.psnr_611_420(net.dec.forward_full(y, q)[:, :, :H, :W], pl)
                d1 = C.psnr_611_420(net.dec(y, q, exit_map=em)[:, :, :H, :W], pl)
                tot += d0 - d1
            vals.append(tot / len(frames))
        rows.append({"j": j, "b": b, "predicted_fraction": frac,
                     "seam_db": dict(zip(map(str, a.qps), vals))})
        print(f"  {j:>3}{b:>4}{frac:>16.3f}" +
              "".join(f"{v:>10.4f}" for v in vals), flush=True)
        del net; torch.cuda.empty_cache()

# Provenance, said properly. This file recorded "ckpt": a.ckpt, and a.ckpt is
# only where the CONFIG is read from -- the weights are the warm start, loaded
# on line 62, because the point is to measure geometry with nothing trained
# into it. A reader of the old field would conclude the sweep is on the pinned
# checkpoint, and I did, for about ten minutes.
json.dump({"weights": str(reference_for(cfg0, None)),
           "config_from": a.ckpt,
           "ckpt": str(reference_for(cfg0, None)),
           "seam_repair": "none", "tile_pad_mode": "replicate",
           "tile_px": P, "feature_px": Fp,
           "blocks_per_exit": b_per, "n_frames": len(frames),
           "sequences": [s["name"] for s in seqs], "rows": rows},
          open(a.out, "w"), indent=2)
print(f"\n  -> {a.out}")
