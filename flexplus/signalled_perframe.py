"""The signalled curve, with the multiplier bisected per frame instead of per set.

scripts/signalled_curve.py bisects one lambda per rate so the MEAN per-frame
decibel lands on the budget. Appendix A shows what that leaves: on the dumped
tables, 138 of 265 frame-rate pairs are served worse than the budget and the
worst is 0.240 dB. It also shows the fix costs nothing -- the encoder already
holds the per-tile table, and bisecting per frame RAISES the mean saving,
because one shared multiplier solves a pooled-MSE Lagrangian rather than a
constraint on a mean of logarithms.

Those numbers are on tables taken with forward_all_exits, which carry no
tiling penalty. This measures the same thing on the DEPLOYED tiled path, one
real decode at a time, so the headline can be stated as a guarantee with a
number that a decoder actually delivers.

Structure follows signalled_curve.py deliberately: the same per-tile table, the
same reference, the same two-stage bisection (cheap table first, then corrected
against a real decode), the same measured saving counted with hooks on the
forward pass that ran. The only change is that all of it happens inside a
frame rather than across the set.

Nothing under ~/FLEX-UF is written; this file lives in FLEX-PLUS.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
sys.path.insert(0, str(UF)); sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(UF / "scripts"))
import ctc_intra as C                                              # noqa: E402
from flexuf.config import FlexUFConfig                             # noqa: E402
from flexuf.cost import exit_costs                                 # noqa: E402
from flexuf.eval import (reference_frame_mse, tiled_exit_mses,     # noqa: E402
                         true_frame_mse)
from flexuf.measure import measured_saving_pct                     # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state            # noqa: E402
from flexuf.reference import reference_for                         # noqa: E402

RES = Path(__file__).resolve().parent / "results"


def map_bits(k, K):
    """Same order-0 entropy plus per-frame histogram as the shipped script.

    Plus sixteen bits for the multiplier itself, which is the whole added cost
    of the guarantee and would be dishonest to omit while quoting the map's.
    """
    p = torch.bincount(k, minlength=K).float()
    p = p / p.sum()
    H = -(p * p.clamp_min(1e-12).log2()).sum().item()
    return H * k.numel() + K * 8 + 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PIN_e9.pth.tar"))
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--global_lambda", action="store_true",
                    help="the shipped protocol instead: one multiplier per "
                         "rate, bisected on the set mean, reported per frame. "
                         "This is what the appendix compares against, measured "
                         "the same way and on the same checkpoint so the two "
                         "differ only in where the multiplier is chosen.")
    ap.add_argument("--out", default=str(RES / "signalled_perframe.json"))
    a = ap.parse_args()
    dev = torch.device(a.device)

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    sa, sb = net.enc.state_dict(), ref.enc.state_dict()
    assert max((sa[k] - sb[k]).abs().max().item() for k in sa) == 0.0
    cost = exit_costs(cfg, "head").to(dev)
    K, j = cfg.num_exits, cfg.split_depth

    seqs, _ = C.discover([])
    if a.max_seqs:
        seqs = seqs[: a.max_seqs]
    frames, names = [], []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append(x[0:1]); names.append(s["name"])
    print(f"  {len(frames)} CTC intra karesi, epoch {ck.get('epoch')}, "
          f"{cfg.rgb_patch}px karo, butce {a.budget} dB\n", flush=True)

    rows, t0 = [], time.time()
    with torch.no_grad():
        for qp_v in a.qps:
            per = []
            for x, nm in zip(frames, names):
                x = x.to(dev); _, _, H, W = x.shape; P = cfg.rgb_patch
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, aux = net._encode_to_latent(xp, qp)
                M = tiled_exit_mses(net.dec, y, q, xp, cfg)
                R = reference_frame_mse(ref.dec, y, q, xp)

                def table_db(lam):
                    k = (M + lam * cost[None, :]).argmin(1)
                    return float(10 * torch.log10(
                        M.gather(1, k[:, None]).squeeze(1).mean() / R))

                def true_db(lam):
                    k = (M + lam * cost[None, :]).argmin(1).clamp(min=j)
                    return float(10 * torch.log10(
                        true_frame_mse(net.dec, y, q, xp, k) / R))

                if a.global_lambda:
                    # Collected now, solved after the loop: one multiplier has
                    # to see every frame before it can be chosen.
                    per.append({"seq": nm, "_M": M, "_R": R, "_y": y,
                                "_q": q, "_xp": xp, "floor_db": true_db(0.0)})
                    continue
                # The ceiling as well as the floor: what this frame
                # delivers when every tile takes the shallowest exit it may.
                # A frame whose ceiling is already under the budget cannot
                # spend the rest of it, so the tolerance above that point is
                # wasted on it -- which is what picking an operating point is
                # about, and what a budget quoted without it hides.
                ceil = true_db(1e9)
                floor = true_db(0.0)
                if floor > a.budget:
                    # The ladder cannot reach the budget on this frame at all:
                    # cutting it into tiles already costs more than the budget
                    # allows, before a single tile has exited early. Recorded
                    # as infeasible rather than quietly served over budget.
                    lam, td, feas = 0.0, floor, False
                else:
                    # Bisect on the TRUE delivered decibel, not on the table.
                    #
                    # The first version bisected the cheap table and corrected
                    # against a decode six times. That converges for most
                    # frames and leaves the rest above the budget -- 17 of 53
                    # at qp0, 25 at qp32 -- because nothing in it ever enforces
                    # the bound it exists to deliver. A guarantee measured by a
                    # solver that does not guarantee is not a guarantee.
                    #
                    # So: seed a bracket from the table, which costs no decode,
                    # then bisect on real decodes while keeping only
                    # multipliers whose delivered decibel is at or under the
                    # budget. The lambda returned is by construction one of
                    # those, so the bound holds for every feasible frame by
                    # construction rather than by convergence.
                    seed = 1.0
                    if table_db(seed) >= a.budget:
                        lo, hi = 0.0, seed
                        for _ in range(40):
                            mid = 0.5 * (lo + hi)
                            if table_db(mid) <= a.budget:
                                lo = mid
                            else:
                                hi = mid
                        seed = lo
                    lam, td, feas = 0.0, floor, True
                    hi = min(1.0, max(seed * 4.0, 1e-6))
                    for _ in range(16):
                        mid = 0.5 * (lam + hi)
                        t = true_db(mid)
                        if t <= a.budget:
                            lam, td = mid, t
                        else:
                            hi = mid
                k = (M + lam * cost[None, :]).argmin(1).clamp(min=j)
                per.append({
                    "seq": nm, "db": td, "floor_db": floor,
                    "ceiling_db": ceil, "saturated": bool(ceil <= a.budget),
                    "lam": lam, "feasible": feas,
                    "saving_pct_measured": measured_saving_pct(
                        net.dec, ref.dec, y, q, k),
                    "saving_pct_vs_release": float(100 * (1 - cost[k].mean())),
                    "map_bits": map_bits(k, K),
                    "bpp_added": map_bits(k, K) / (H * W),
                    # Which exits the frame actually used. Left out of the
                    # first version, which meant the configuration being
                    # quoted had no exit distribution of its own and the
                    # nearest available one came from a file whose argmin runs
                    # over all K before the clamp -- so its "e0" column is
                    # really e2, and reading it as written would put 87% of
                    # tiles at an exit the decoder cannot reach.
                    "hist": torch.bincount(k, minlength=K).tolist()})
            if a.global_lambda:
                def setmean(lam):
                    t = 0.0
                    for z in per:
                        k = (z["_M"] + lam * cost[None, :]).argmin(1).clamp(min=j)
                        t += float(10 * torch.log10(true_frame_mse(
                            net.dec, z["_y"], z["_q"], z["_xp"], k) / z["_R"]))
                    return t / len(per)
                # Seed from the table before touching a decode. The
                # multiplier this budget needs is around 1e-4, and bisecting
                # [0, 1] twelve times cannot resolve that -- the smallest value
                # it tries is already too large, so lo stays at zero and the
                # answer comes back as "decode everything deepest": 3.40% saved
                # at 0.023 dB, which is not an operating point anyone asked
                # for. The table costs no decode to bisect finely, and the real
                # decodes then only have to correct it.
                def setmean_table(lam):
                    t = 0.0
                    for z in per:
                        M_ = z["_M"]
                        k = (M_ + lam * cost[None, :]).argmin(1)
                        t += float(10 * torch.log10(
                            M_.gather(1, k[:, None]).squeeze(1).mean() / z["_R"]))
                    return t / len(per)
                lo, hi = 0.0, 1.0
                if setmean_table(hi) <= a.budget:
                    seed = hi
                else:
                    for _ in range(45):
                        mid = 0.5 * (lo + hi)
                        if setmean_table(mid) <= a.budget:
                            lo = mid
                        else:
                            hi = mid
                    seed = lo
                lam_g, hi = 0.0, min(1.0, max(seed * 4.0, 1e-6))
                for _ in range(12):
                    mid = 0.5 * (lam_g + hi)
                    if setmean(mid) <= a.budget:
                        lam_g = mid
                    else:
                        hi = mid
                for z in per:
                    k = (z["_M"] + lam_g * cost[None, :]).argmin(1).clamp(min=j)
                    z["lam"] = lam_g
                    z["db"] = float(10 * torch.log10(true_frame_mse(
                        net.dec, z["_y"], z["_q"], z["_xp"], k) / z["_R"]))
                    z["feasible"] = z["floor_db"] <= a.budget
                    z["saving_pct_measured"] = measured_saving_pct(
                        net.dec, ref.dec, z["_y"], z["_q"], k)
                    z["saving_pct_vs_release"] = float(100 * (1 - cost[k].mean()))
                    z["map_bits"] = map_bits(k, K)
                    z["bpp_added"] = 0.0
                    for kk in ("_M", "_R", "_y", "_q", "_xp"):
                        z.pop(kk)
            db = np.array([p["db"] for p in per])
            sv = np.array([p["saving_pct_measured"] for p in per])
            rows.append({"qp": qp_v, "n": len(per),
                         "mean_db": float(db.mean()),
                         "p95_db": float(np.percentile(db, 95)),
                         "max_db": float(db.max()),
                         "over_budget": int((db > a.budget + 1e-9).sum()),
                         "infeasible": int(sum(not z["feasible"] for z in per)),
                         "saturated": int(sum(z.get("saturated", False)
                                              for z in per)),
                         "mean_ceiling_db": float(np.mean(
                             [z["ceiling_db"] for z in per])),
                         "min_ceiling_db": float(np.min(
                             [z["ceiling_db"] for z in per])),
                         "saving_pct_measured": float(sv.mean()),
                         "map_bits": float(np.mean([p["map_bits"] for p in per])),
                         "hist_pct": (lambda h: (100 * h / h.sum()).tolist())(
                             np.array([p["hist"] for p in per], float).sum(0)),
                         "per_frame": per})
            print(f"   q{qp_v:>3}  tasarruf {sv.mean():6.2f}%   ort dB "
                  f"{db.mean():.4f}  p95 {np.percentile(db,95):.4f}  "
                  f"max {db.max():.4f}  asan "
                  f"{int((db>a.budget+1e-9).sum())}/{len(per)} "
                  f"(ulasilamaz {int(sum(not z['feasible'] for z in per))}, "
                  f"doymus {int(sum(z.get('saturated', False) for z in per))})  "
                  f"harita {np.mean([p['map_bits'] for p in per]):.0f} bit",
                  flush=True)

    allv = np.concatenate([[p["db"] for p in r["per_frame"]] for r in rows])
    alls = np.concatenate([[p["saving_pct_measured"] for p in r["per_frame"]]
                           for r in rows])
    print(f"\n  HEPSI  tasarruf {alls.mean():.2f}%   max {allv.max():.4f} dB   "
          f"asan {int((allv > a.budget + 1e-9).sum())}/{allv.size}")
    Path(a.out).write_text(json.dumps(
        {"script": "flexplus/signalled_perframe.py", "budget_db": a.budget,
         "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
         "decode": "deployed tiled path, one real decode per correction step",
         "lambda": "bisected per frame on the true delivered decibel",
         "map_bits_note": "order-0 entropy + per-frame histogram + 16 bits of lambda",
         "n_frames": len(frames), "rows": rows}, indent=2))
    print(f"  {time.time()-t0:.0f}s, yazildi {a.out}")


if __name__ == "__main__":
    main()
