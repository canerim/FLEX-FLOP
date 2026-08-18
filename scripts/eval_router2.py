"""How good is the router, in the units that decide the result?

Plain agreement counts every tile once. That is the wrong weighting and it cuts
both ways:

  - On tiles where two exits are within a hair of each other, disagreeing costs
    nothing. Counting those as failures understates the router.
  - On the few tiles where the wrong exit is most of the frame's error,
    disagreeing costs everything. Counting those as one vote among many
    overstates it.

So three numbers, not one:

  agreement            fraction of tiles matching the oracle's exit
  cost-weighted agr.   the same, weighted by what the decision is worth
  realised regret      the excess Lagrangian cost actually paid, which is the
                       only one that appears in the frontier

Measured on CTC, on a checkpoint the router never trained against.
"""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.eval import tiled_exit_mses
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.router.head2 import StemRouterHeadV2

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--router", required=True)
ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
ap.add_argument("--frames", type=int, default=2)
ap.add_argument("--device", default="cuda:4")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
rk = torch.load(a.router, map_location="cpu", weights_only=False)
head = StemRouterHeadV2(384, 256, cfg.num_exits, min_exit=cfg.split_depth).to(dev).eval()
head.load_state_dict(rk["router_head_v2"])
lam = rk["lam"]
cost = exit_costs(cfg, "head").to(dev)

seqs, _ = C.discover([])
frames = []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
    if x is not None:
        for i in range(x.shape[0]):
            frames.append(x[i:i+1])
print(f"  lam={lam:g}, {len(frames)} CTC karesi, {cfg.rgb_patch}px tile\n")
# Entropy of the ORACLE's own exit distribution, in bits. This is the guard
# against a result that looks perfect and means nothing: as lambda rises the
# oracle itself collapses onto one exit, and a constant router agrees with a
# constant oracle 100% of the time while doing no routing at all. v1 hit exactly
# that (agreement 1.000, distribution [0,0,100,0,0,0]) and it would have been
# reported as success. Entropy near zero says "there was nothing to decide".
print(f"  {'qp':>4}{'uyum':>8}{'mal-agir':>10}{'regret':>10}"
      f"{'tasarruf':>10}{'oracle':>9}{'oracle entropi':>16}{'':>3}")
with torch.no_grad():
    for qp_v in a.qps:
        A = W = Rg = SV = SVo = ENT = n = 0.0
        for x in frames:
            x = x.to(dev); _, _, H, Wd = x.shape; P = cfg.rgb_patch
            ph, pw = (-H) % P, (-Wd) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, aux = net._encode_to_latent(xp, qp)
            nh, nw = (H + ph) // P, (Wd + pw) // P
            # DEPLOYED path -- see flexuf/eval.py. The agreement this script
            # reports is agreement with the oracle for the decoder that ships,
            # not for a full-frame one.
            M = tiled_exit_mses(net.dec, y, q, xp, cfg)
            stem = net.dec.upsample(y)
            for g in range(cfg.split_depth):
                stem = net.dec.groups[g](stem)
            k = head(stem, y, aux["scales_hat"], qp,
                     cfg.feature_patch, cfg.latent_patch).argmax(1)
            lag = M + lam * cost[None, :]
            best, ks = lag.min(1)
            stake = lag.max(1).values - best
            hit = (k == ks).float()
            A += hit.mean().item()
            W += (hit * stake).sum().item() / stake.sum().clamp_min(1e-12).item()
            Rg += ((lag.gather(1, k[:, None]).squeeze(1) - best) / best).mean().item()
            SV += (1 - cost[k].mean() / cost[-1]).item()
            SVo += (1 - cost[ks].mean() / cost[-1]).item()
            pk = torch.bincount(ks, minlength=cfg.num_exits).float()
            pk = pk / pk.sum()
            ENT += -(pk * (pk.clamp_min(1e-12)).log2()).sum().item()
            n += 1
        e = ENT / n
        flag = "  <- ORACLE SABIT, uyum anlamsiz" if e < 0.15 else ""
        print(f"  {qp_v:>4}{A/n:>8.3f}{W/n:>10.3f}{Rg/n:>10.5f}"
              f"{100*SV/n:>9.1f}%{100*SVo/n:>8.1f}%{e:>16.3f}{flag}")
print("\n  oracle entropi: oracle'in kendi cikis dagiliminin entropisi (bit).")
print("  0'a yakinsa oracle da sabittir ve %100 uyum yonlendirme DEGIL demektir.")
