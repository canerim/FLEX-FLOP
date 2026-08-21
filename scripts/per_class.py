"""Saving broken down by CTC class, at one global operating point.

The headline number is a mean over 53 sequences spanning four resolutions, and a
mean over four resolutions hides the thing that turns out to matter most. A
256 px tile is 40 tiles on a 1080p frame, 18 on 720p, 8 at 832x480 and 2 at
416x240, and with two tiles there is barely an allocation to make.

The operating point is global, not per class: lambda is bisected once so the
whole test set lands on the budget, exactly as it would be in deployment, and
each class is then reported at that lambda. Bisecting per class would answer a
different and easier question.

Measured on the deployed tiled decode path (flexuf/eval.py) with the delivered
distortion, like every other number in this project.
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.eval import reference_frame_mse, tiled_exit_mses, true_frame_mse
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.3, 0.5])
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--device", default=_gpu("cuda:2"))
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"])
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(reference_for(cfg, None), map_location="cpu",
                                  weights_only=False))
sa, sb = net.enc.state_dict(), ref.enc.state_dict()
assert max((sa[k] - sb[k]).abs().max().item() for k in sa) == 0.0
cost = exit_costs(cfg, "head").to(dev)
K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch

seqs, _ = C.discover([])
frames = []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
    if x is not None:
        for i in range(x.shape[0]):
            frames.append((s["cls"], f"{s['w']}x{s['h']}", x[i:i + 1]))
cls_n = defaultdict(int)
for c, _r, _x in frames:
    cls_n[c] += 1
print(f"  {len(frames)} frames, classes " +
      ", ".join(f"{k}:{v}" for k, v in sorted(cls_n.items())) + "\n", flush=True)

rows = []
with torch.no_grad():
    for qp_v in a.qps:
        cache = []
        for cls, res, x in frames:
            x = x.to(dev); _, _, H, W = x.shape
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net._encode_to_latent(xp, qp)
            ntile = ((H + ph) // P) * ((W + pw) // P)
            cache.append((cls, res, ntile,
                          tiled_exit_mses(net.dec, y, q, xp, cfg),
                          reference_frame_mse(ref.dec, y, q, xp), y, q, xp))

        def maps_at(lam):
            return [(M + lam * cost[None, :]).argmin(1).clamp(min=j)
                    for _c, _r, _n, M, *_ in cache]

        def db_at(maps):
            tot = 0.0
            for m, (_c, _r, _n, _M, R, y_, q_, xp_) in zip(maps, cache):
                tot += (10 * torch.log10(
                    true_frame_mse(net.dec, y_, q_, xp_, m) / R)).item()
            return tot / len(cache)

        for budget in a.budgets:
            lo, hi = 0.0, 1.0
            if db_at(maps_at(hi)) >= budget:
                for _ in range(22):
                    mid = 0.5 * (lo + hi)
                    if db_at(maps_at(mid)) <= budget:
                        lo = mid
                    else:
                        hi = mid
            else:
                lo = hi
            maps = maps_at(lo)
            per = defaultdict(lambda: {"sv": [], "db": [], "tiles": 0, "res": "",
                                       "hist": [0] * cfg.num_exits})
            for m, (cls, res, nt, _M, R, y_, q_, xp_) in zip(maps, cache):
                d = per[cls]
                d["sv"].append(100 * (1 - cost[m].mean()).item())
                d["db"].append((10 * torch.log10(
                    true_frame_mse(net.dec, y_, q_, xp_, m) / R)).item())
                d["tiles"], d["res"] = nt, res
                # Which rung each tile actually took. This is the thing the
                # saving is an average of, and the average hides whether the
                # ladder is being used or whether one exit is doing all the
                # work -- which is exactly what a collapsed router looks like.
                b = torch.bincount(m, minlength=cfg.num_exits)
                for k_ in range(cfg.num_exits):
                    d["hist"][k_] += int(b[k_])
            out = {c: {"saving": sum(v["sv"]) / len(v["sv"]),
                       "db": sum(v["db"]) / len(v["db"]),
                       "tiles": v["tiles"], "res": v["res"], "n": len(v["sv"]),
                       "hist": v["hist"],
                       "exit_share": [100 * h / max(1, sum(v["hist"]))
                                      for h in v["hist"]]}
                   for c, v in per.items()}
            rows.append({"qp": qp_v, "budget_db": budget, "lam": lo,
                         "overall_db": db_at(maps),
                         "overall_saving": 100 * sum(
                             (1 - cost[m].mean()).item() for m in maps) / len(maps),
                         "per_class": out})
            print(f"  qp {qp_v:>2}  budget {budget:.2f}  "
                  f"(delivered {db_at(maps):.4f} dB, overall "
                  f"{rows[-1]['overall_saving']:.2f}%)")
            for c in sorted(out, key=lambda c: -out[c]["tiles"]):
                v = out[c]
                sh = "  ".join(f"e{k_}:{p_:>4.1f}%"
                               for k_, p_ in enumerate(v["exit_share"])
                               if k_ >= j)
                print(f"      {c:<9} {v['res']:>9}  {v['tiles']:>3} tiles  "
                      f"n={v['n']:>2}   {v['saving']:>6.2f}%   {v['db']:>7.4f} dB"
                      f"   {sh}")
            print(flush=True)

json.dump({"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
           "n_frames": len(frames), "tile_px": P, "rows": rows},
          open(a.out, "w"), indent=2)
print(f"  -> {a.out}")
