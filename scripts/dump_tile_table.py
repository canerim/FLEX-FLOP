"""Dump the per-tile, per-exit distortion table for one frame.

The router figures need the real thing rather than a plausible-looking synthetic
array: the whole point of the Lagrangian picture is the SHAPE of D(t,k) across
tiles, and a synthetic table with the wrong dynamic range makes the lambda sweep
degenerate to a single column, which is exactly what a reader would otherwise
conclude about the method.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.eval import reference_frame_mse, tiled_exit_mses, true_frame_mse
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_eval.pth.tar")
ap.add_argument("--seq", default="Bosphorus")
ap.add_argument("--qp", type=int, default=32)
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--out", default="results/tile_table.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"])
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(reference_for(cfg, None), map_location="cpu",
                                  weights_only=False))
cost = exit_costs(cfg, "head").to(dev)
K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch

seqs, _ = C.discover([])
s = next(x for x in seqs if a.seq.lower() in x["name"].lower())
x, _pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
x = x[0:1].to(dev); _, _, H, W = x.shape
ph, pw = (-H) % P, (-W) % P
xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
nh, nw = (H + ph) // P, (W + pw) // P

with torch.no_grad():
    qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
    y, q, aux = net._encode_to_latent(xp, qp)
    M = tiled_exit_mses(net.dec, y, q, xp, cfg)          # [T, K]
    R = reference_frame_mse(ref.dec, y, q, xp)
    # bits per tile, the free decoder-side difficulty signal
    bits = net.get_y_bits(net.add_noise(aux["y_res"]), aux["scales_hat"]).sum(1,
                                                                keepdim=True)
    L = P // 16
    tb = (bits.view(1, 1, nh, L, nw, L).permute(0, 1, 2, 4, 3, 5)
              .reshape(nh * nw, L * L).sum(1))
    # a lambda sweep, and the frame dB each allocation actually delivers
    sweep = []
    # The interesting range is narrow and entirely at the low end: above ~5e-5
    # every tile is already at the cheapest rung and the sweep is a flat line.
    # A geomspace over six decades spends 80% of its points there.
    for lam in np.geomspace(2e-7, 2e-4, 60):
        k = (M + float(lam) * cost[None, :]).argmin(1).clamp(min=j)
        db = (10 * torch.log10(true_frame_mse(net.dec, y, q, xp, k) / R)).item()
        sweep.append({"lam": float(lam), "db": db,
                      "saving": 100 * (1 - cost[k].mean()).item(),
                      "map": k.tolist()})

json.dump({"ckpt": a.ckpt, "seq": s["name"], "qp": a.qp, "nh": nh, "nw": nw,
           "K": K, "j": j, "tile_px": P, "cost": cost.tolist(),
           "ref_mse": R.item(), "D": M.cpu().tolist(),
           "bits_per_tile": tb.cpu().tolist(), "sweep": sweep},
          open(a.out, "w"))
print(f"  {s['name']} qp{a.qp}: {nh}x{nw} tiles, K={K}, j={j}")
print(f"  -> {a.out}")
