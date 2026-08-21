"""What does canvas coupling actually remove, measured on a trained model?

The paper's largest claimed remaining gain is Section 4.4: give the per-tile 3x3
its real neighbour from a shared canvas, for +0.032% of the decode, and the seam
stops existing at uniform depth. That claim has two halves and only one of them
has ever been checked.

  exactness   at uniform depth a coupled tiled decode should be bit-identical to
              a full-frame decode. Checked here as max|difference|.
  worth       what it removes from the FLOOR, and what that is worth in saving.

The caveat is the same one the seam-repair ablation carries: this model was
trained with replicate padding, so switching to coupling at inference is a
distribution shift and the honest reading is a BOUND, not a deployment number. If
coupling is exact at uniform depth the floor should fall to the weight drift and
no further, and the routed case is where the approximation bites -- neighbours at
different depths are real but shallower.
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
from flexuf.cost import exit_costs
from flexuf.eval import reference_frame_mse, tiled_exit_mses, true_frame_mse
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=_pin("runs/RECIPE512/ckpt_eval.pth.tar"))
ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
ap.add_argument("--budget", type=float, default=0.1)
ap.add_argument("--max_seqs", type=int, default=16)
ap.add_argument("--device", default=_gpu("cuda:7"))
ap.add_argument("--out", default="results/coupling_ablation.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg0 = FlexUFConfig(**ck["config"])
K, j, P = cfg0.num_exits, cfg0.split_depth, cfg0.rgb_patch


def build(coupled):
    c = FlexUFConfig(**{**cfg0.__dict__, "tile_coupling": coupled})
    m = FlexUFIntra(c).to(dev).eval(); load_flexuf_state(m, ck)
    return c, m


cfg_pad, net_pad = build(False)
cfg_cpl, net_cpl = build(True)
ref = FlexUFIntra(cfg0).to(dev).eval()
load_flexuf_state(ref, torch.load(reference_for(cfg0, None), map_location="cpu",
                                  weights_only=False))
cost = exit_costs(cfg0, "head").to(dev)

seqs, _ = C.discover([])
seqs = seqs[:a.max_seqs]
frames = []
for s in seqs:
    x, _pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
    if x is not None:
        frames.append(x[0:1])
print(f"  {len(frames)} frames, budget {a.budget} dB\n", flush=True)

rows = []
with torch.no_grad():
    for qp_v in a.qps:
        cache = []
        exact = 0.0
        for x in frames:
            x = x.to(dev); _, _, H, W = x.shape
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net_pad._encode_to_latent(xp, qp)
            nt = ((H + ph) // P) * ((W + pw) // P)
            deep = torch.full((nt,), K - 1, dtype=torch.long, device=dev)
            # exactness of the coupled decode at uniform depth
            full = net_pad.dec.forward_full(y, q)
            cpl = net_cpl.dec(y, q, exit_map=deep)
            exact = max(exact, (full - cpl).abs().max().item())
            cache.append((y, q, xp, reference_frame_mse(ref.dec, y, q, xp), nt))

        out = {}
        for name, net in (("padded", net_pad), ("coupled", net_cpl)):
            # floor: every tile at full depth
            fl = 0.0
            for y, q, xp, R, nt in cache:
                em = torch.full((nt,), K - 1, dtype=torch.long, device=dev)
                fl += (10 * torch.log10(
                    true_frame_mse(net.dec, y, q, xp, em) / R)).item()
            fl /= len(cache)
            # routed at the budget, with the table built on that same path
            tabs = [tiled_exit_mses(net.dec, y, q, xp, cfg0)
                    for y, q, xp, _R, _n in cache]

            def db_sv(lam):
                d = s_ = 0.0
                for (y, q, xp, R, _n), M in zip(cache, tabs):
                    k = (M + lam * cost[None, :]).argmin(1).clamp(min=j)
                    d += (10 * torch.log10(
                        true_frame_mse(net.dec, y, q, xp, k) / R)).item()
                    s_ += 100 * (1 - cost[k].mean()).item()
                return d / len(cache), s_ / len(cache)

            lo, hi = 0.0, 1.0
            if db_sv(hi)[0] >= a.budget:
                for _ in range(20):
                    mid = 0.5 * (lo + hi)
                    if db_sv(mid)[0] <= a.budget:
                        lo = mid
                    else:
                        hi = mid
            else:
                lo = hi
            db, sv = db_sv(lo)
            out[name] = {"floor_db": fl, "routed_db": db, "saving": sv}
            del tabs

        rows.append({"qp": qp_v, "uniform_depth_max_diff": exact, **out})
        p_, c_ = out["padded"], out["coupled"]
        print(f"  qp {qp_v:>2}   uniform-depth max|diff| = {exact:.3e}")
        print(f"     floor   padded {p_['floor_db']:.4f}   "
              f"coupled {c_['floor_db']:.4f}   ({c_['floor_db']-p_['floor_db']:+.4f})")
        print(f"     routed  padded {p_['saving']:.2f}%   "
              f"coupled {c_['saving']:.2f}%   "
              f"({c_['saving']-p_['saving']:+.2f} pts)\n", flush=True)

json.dump({"ckpt": a.ckpt, "budget_db": a.budget, "n_frames": len(frames),
           "note": "trained with replicate padding; coupling switched on at "
                   "inference only, so these are bounds not deployment numbers",
           "rows": rows}, open(a.out, "w"), indent=2)
print(f"  -> {a.out}")
