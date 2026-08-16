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
print(f"  {'qp':>4}{'uyum':>8}{'maliyet-agirlikli':>19}{'gerceklesen regret':>20}"
      f"{'tasarruf':>10}{'oracle tasarruf':>17}")
with torch.no_grad():
    for qp_v in a.qps:
        A = W = Rg = SV = SVo = n = 0.0
        for x in frames:
            x = x.to(dev); _, _, H, Wd = x.shape; P = cfg.rgb_patch
            ph, pw = (-H) % P, (-Wd) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, aux = net._encode_to_latent(xp, qp)
            nh, nw = (H + ph) // P, (Wd + pw) // P
            M = torch.stack([
                (((o - xp) ** 2).mean(1).view(1, nh, P, nw, P)
                 .permute(0, 1, 3, 2, 4).reshape(nh * nw, P * P).mean(1))
                for o in net.dec.forward_all_exits(y, q)], 1)
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
            n += 1
        print(f"  {qp_v:>4}{A/n:>8.3f}{W/n:>19.3f}{Rg/n:>20.5f}"
              f"{100*SV/n:>9.1f}%{100*SVo/n:>16.1f}%")
print("\n  maliyet-agirlikli uyum: kil payi tile'lardaki uyusmazlik ucuz sayilir,")
print("  onemli tile'lardaki pahali. Ikisi ayrilmadan %86 tek basina yorumlanamaz.")
