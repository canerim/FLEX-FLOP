"""Does the exit map have to be recomputed for every frame and every rate?

Configuration A costs the encoder about one extra decode per frame to search for
the map. If the map is stable -- across adjacent frames of a sequence, or across
quality indices of the same frame -- then that cost amortises and the side
information can be sent less often. Nobody has checked, and the answer bears on
whether A is practical rather than merely optimal.

Two transfers, both measured as: take the oracle map computed HERE, apply it
THERE, and report what the substitution costs in delivered dB and in saving
against recomputing.

  time   the map from frame 0 applied to frames 1, 2, 4, 8 of the same sequence
  rate   the map from qp 0 applied at qp 32 and 63, and the reverse

The control in both cases is the oracle map computed in place.
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
ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
ap.add_argument("--offsets", type=int, nargs="+", default=[0, 1, 2, 4, 8])
ap.add_argument("--budget", type=float, default=0.1)
ap.add_argument("--max_seqs", type=int, default=8)
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--out", default="results/map_transfer.json")
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
seqs = [s for s in seqs if s["w"] >= 1280][:a.max_seqs]
print(f"  {len(seqs)} sequences, offsets {a.offsets}, qps {a.qps}\n", flush=True)


def prep(x, qp_v):
    x = x.to(dev); _, _, H, W = x.shape
    ph, pw = (-H) % P, (-W) % P
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
    qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
    y, q, _ = net._encode_to_latent(xp, qp)
    return y, q, xp


def solve(M, R_list, cells, budget):
    """Bisect lambda over a set of frames; return maps, dB, saving."""
    def at(lam):
        ms = [(m + lam * cost[None, :]).argmin(1).clamp(min=j) for m in M]
        d = np.mean([(10 * torch.log10(
            true_frame_mse(net.dec, y, q, xp, k) / R)).item()
            for k, (y, q, xp), R in zip(ms, cells, R_list)])
        return ms, d
    lo, hi = 0.0, 1.0
    if at(hi)[1] >= budget:
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            if at(mid)[1] <= budget:
                lo = mid
            else:
                hi = mid
    else:
        lo = hi
    ms, d = at(lo)
    sv = float(np.mean([100 * (1 - cost[k].mean()).item() for k in ms]))
    return ms, d, sv


def apply(maps, cells, R_list):
    d = np.mean([(10 * torch.log10(
        true_frame_mse(net.dec, y, q, xp, k) / R)).item()
        for k, (y, q, xp), R in zip(maps, cells, R_list)])
    sv = float(np.mean([100 * (1 - cost[k].mean()).item() for k in maps]))
    return d, sv


rows = []
with torch.no_grad():
    # ---------------- transfer across time ---------------------------------
    for qp_v in a.qps:
        cells, Rs, tabs = {}, {}, {}
        for off in a.offsets:
            cc, rr, tt = [], [], []
            for s in seqs:
                x, _ = C.read_frames(s["path"], s["w"], s["h"], off + 1, 1)
                if x is None or x.shape[0] <= off:
                    continue
                y, q, xp = prep(x[off:off + 1], qp_v)
                cc.append((y, q, xp))
                rr.append(reference_frame_mse(ref.dec, y, q, xp))
                tt.append(tiled_exit_mses(net.dec, y, q, xp, cfg))
            cells[off], Rs[off], tabs[off] = cc, rr, tt
        m0, d0, s0 = solve(tabs[0], Rs[0], cells[0], a.budget)
        for off in a.offsets:
            if off == 0:
                rows.append({"kind": "time", "qp": qp_v, "offset": 0,
                             "in_place_db": d0, "in_place_saving": s0,
                             "transfer_db": d0, "transfer_saving": s0})
                continue
            n = min(len(m0), len(cells[off]))
            td, ts = apply(m0[:n], cells[off][:n], Rs[off][:n])
            _, id_, is_ = solve(tabs[off][:n], Rs[off][:n], cells[off][:n],
                                a.budget)
            rows.append({"kind": "time", "qp": qp_v, "offset": off,
                         "in_place_db": id_, "in_place_saving": is_,
                         "transfer_db": td, "transfer_saving": ts})
            print(f"  time  qp{qp_v:>2} +{off:<2} frames   in place "
                  f"{id_:.4f} dB / {is_:.2f}%   transferred {td:.4f} dB / "
                  f"{ts:.2f}%", flush=True)
        del cells, Rs, tabs

    # ---------------- transfer across rate ---------------------------------
    cells, Rs, tabs = {}, {}, {}
    for qp_v in a.qps:
        cc, rr, tt = [], [], []
        for s in seqs:
            x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
            if x is None:
                continue
            y, q, xp = prep(x[0:1], qp_v)
            cc.append((y, q, xp))
            rr.append(reference_frame_mse(ref.dec, y, q, xp))
            tt.append(tiled_exit_mses(net.dec, y, q, xp, cfg))
        cells[qp_v], Rs[qp_v], tabs[qp_v] = cc, rr, tt
    solved = {q_: solve(tabs[q_], Rs[q_], cells[q_], a.budget) for q_ in a.qps}
    for src in a.qps:
        for dst in a.qps:
            td, ts = apply(solved[src][0], cells[dst], Rs[dst])
            rows.append({"kind": "rate", "from": src, "to": dst,
                         "in_place_db": solved[dst][1],
                         "in_place_saving": solved[dst][2],
                         "transfer_db": td, "transfer_saving": ts})
            if src != dst:
                print(f"  rate  q{src} -> q{dst}   in place "
                      f"{solved[dst][1]:.4f} dB / {solved[dst][2]:.2f}%   "
                      f"transferred {td:.4f} dB / {ts:.2f}%", flush=True)

json.dump({"ckpt": a.ckpt, "budget_db": a.budget,
           "sequences": [s["name"] for s in seqs], "rows": rows},
          open(a.out, "w"), indent=2)
print(f"\n  -> {a.out}")
