"""Is per-tile adaptivity necessary, or would a uniformly shallower decoder do?

This is the first question a reviewer asks about any adaptive-inference paper and
it deserves a direct answer rather than an appeal to the idea. Three allocations
at matched compute, on the deployed decode path:

  UNIFORM     every tile at the same exit k. This is what a static, shallower
              decoder would give -- no signalling, no router, no per-tile
              anything. The ladder still provides the exits, so this is the
              strongest possible version of "just make it shallower".

  RANDOM      a per-tile map drawn to match the oracle's exit HISTOGRAM at the
              budget, but assigned to tiles at random. Same average cost, same
              mix of depths, none of the content dependence. The gap between
              this and the oracle is what knowing WHICH tile is easy is worth,
              separated from the gain of merely running some tiles shallower.

  RATE-RANK   the same histogram again, but assigned by a free decoder-side
              signal instead of at random: the number of bits the entropy model
              spent on each tile. A tile that cost more bits carries more
              detail, so it is the one that should run deep. This is the
              cheapest possible router -- zero parameters, zero training, and
              the decoder has the number before the trunk starts -- and it is
              the natural analogue of the confidence rule that early-exit
              classifiers use, which is likewise read off the network's own
              output rather than learned.

  ORACLE      argmin_k [ D(t,k) + lambda c_k ], lambda bisected to the budget.

Because RANDOM, RATE-RANK and ORACLE share a histogram they share an average
cost exactly, so the three differ only in WHICH tile got which depth. That makes
the comparison a measurement of ranking quality and nothing else.

Reported against the released decoder's full-frame decode of the same latent, so
the tiling penalty is charged to every row.
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
from flexuf.eval import reference_frame_mse, tiled_exit_mses, true_frame_mse
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
ap.add_argument("--budget", type=float, default=0.1)
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--device", default="cuda:0")
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
            frames.append(x[i:i + 1])
print(f"  {len(frames)} CTC karesi / {len(seqs)} sekans, budget {a.budget} dB\n")

g = torch.Generator(device="cpu").manual_seed(a.seed)
rows = []
with torch.no_grad():
    for qp_v in a.qps:
        cache = []
        for x in frames:
            x = x.to(dev); _, _, H, W = x.shape
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net._encode_to_latent(xp, qp)
            # Bits per tile, from the entropy model, as a free difficulty
            # proxy. bits_y is [1,C,h,w] over the latent grid; a tile of P RGB
            # pixels is P/16 latent positions.
            _, _, aux = net._encode_to_latent(xp, qp)
            bits = net.get_y_bits(net.add_noise(aux["y_res"]),
                                  aux["scales_hat"]).sum(1, keepdim=True)
            L = P // 16
            nh_, nw_ = bits.shape[-2] // L, bits.shape[-1] // L
            tb = (bits.view(1, 1, nh_, L, nw_, L).permute(0, 1, 2, 4, 3, 5)
                      .reshape(nh_ * nw_, L * L).sum(1))
            cache.append((tiled_exit_mses(net.dec, y, q, xp, cfg),
                          reference_frame_mse(ref.dec, y, q, xp), y, q, xp, tb))

        def db_of(maps):
            return sum((10 * torch.log10(
                true_frame_mse(net.dec, y_, q_, xp_, m) / R)).item()
                for m, (_, R, y_, q_, xp_, _tb) in zip(maps, cache)) / len(cache)

        def sv_of(maps):
            return 100 * sum((1 - cost[m].mean()).item() for m in maps) / len(maps)

        # ---- UNIFORM: the static, shallower decoder --------------------
        uni = []
        for k in range(j, K):
            maps = [torch.full((M.shape[0],), k, dtype=torch.long, device=dev)
                    for M, *_ in cache]
            uni.append({"exit": k, "db": db_of(maps), "saving": sv_of(maps)})

        # ---- ORACLE at the budget --------------------------------------
        def at_lam(lam):
            return [(M + lam * cost[None, :]).argmin(1).clamp(min=j)
                    for M, *_ in cache]
        lo, hi = 0.0, 1.0
        if db_of(at_lam(hi)) >= a.budget:
            for _ in range(24):
                mid = 0.5 * (lo + hi)
                if db_of(at_lam(mid)) <= a.budget:
                    lo = mid
                else:
                    hi = mid
        else:
            lo = hi
        omaps = at_lam(lo)
        oracle = {"db": db_of(omaps), "saving": sv_of(omaps), "lam": lo}

        # ---- RANDOM at the oracle's own histogram ----------------------
        rmaps = []
        for m in omaps:
            perm = torch.randperm(m.numel(), generator=g).to(dev)
            rmaps.append(m[perm])
        rnd = {"db": db_of(rmaps), "saving": sv_of(rmaps)}

        # ---- RATE-RANK: same histogram, ordered by bits per tile -------
        # Sorting the oracle's own multiset of depths onto the tiles by
        # descending bit cost. Deepest to the most expensive tile.
        hmaps = []
        for m, (_M, _R, _y, _q, _xp, tb) in zip(omaps, cache):
            depths = torch.sort(m, descending=True).values      # the multiset
            rank = torch.argsort(tb, descending=True)           # hardest first
            out = torch.empty_like(m)
            out[rank] = depths
            hmaps.append(out)
        heur = {"db": db_of(hmaps), "saving": sv_of(hmaps),
                "agreement": sum((h == o).float().mean().item()
                                 for h, o in zip(hmaps, omaps)) / len(omaps)}

        # The static row a reviewer cares about: the shallowest UNIFORM depth
        # whose distortion still fits the budget. If none does, adaptivity is
        # not an improvement on the static option -- it is the only option.
        feasible = [u for u in uni if u["db"] <= a.budget]
        best_static = max(feasible, key=lambda u: u["saving"]) if feasible else None

        rows.append({"qp": qp_v, "uniform": uni, "oracle": oracle,
                     "random": rnd, "rate_rank": heur,
                     "best_static": best_static})
        print(f"  qp {qp_v}")
        for u in uni:
            mark = "  <- fits the budget" if u["db"] <= a.budget else ""
            print(f"    uniform exit {u['exit']}   {u['db']:8.4f} dB"
                  f"   {u['saving']:6.2f}%{mark}")
        print(f"    RANDOM     (same histogram, shuffled)"
              f" {rnd['db']:8.4f} dB   {rnd['saving']:6.2f}%")
        print(f"    RATE-RANK  (same histogram, by bits) "
              f" {heur['db']:8.4f} dB   {heur['saving']:6.2f}%"
              f"   agree {heur['agreement']:.3f}")
        print(f"    ORACLE     at the budget            "
              f" {oracle['db']:8.4f} dB   {oracle['saving']:6.2f}%")
        if best_static:
            print(f"    -> best static saving {best_static['saving']:.2f}%, "
                  f"adaptive {oracle['saving']:.2f}%")
        else:
            print(f"    -> NO uniform depth fits {a.budget} dB; "
                  f"adaptive gives {oracle['saving']:.2f}%")
        print(flush=True)

json.dump({"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "budget_db": a.budget,
           "n_frames": len(frames), "n_sequences": len(seqs),
           "seed": a.seed, "rows": rows}, open(a.out, "w"), indent=2)
print(f"  -> {a.out}")
