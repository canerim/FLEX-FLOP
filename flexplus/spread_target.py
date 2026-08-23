"""What would have to be true for 60 per cent at 0.1 dB.

The ceiling says what the ladder could save if the router were free; 39.1 per
cent as shipped, 68.9 at split depth 0, 76.4 at twelve exits. The paper
realises 70.7 per cent of its ceiling, and the gap is not the router being
bad at its job. It is that a tile which drops one exit pays real distortion:
the training log's spread_dB is 0.64 to 1.07 across the six exits on the
reported run, against a budget of 0.1. Most tiles simply cannot afford to
move, which is why the allocation sits at mean exit 3.37 of 5 with only 22
per cent of tiles at the shallowest available exit -- the allocation is
distortion-bound, not floor-bound, and nothing that lowers the floor alone
will fix it.

So invert the question. Take the measured per-tile MSE table, compress the
gap between each exit and the deepest by a factor s, re-run the paper's own
Lagrangian bisection at 0.1 dB, and read the saving. The s that reaches 60
per cent is the answer to "how flat would the ladder have to be", in the one
unit that matters -- and it can then be compared against what deep
supervision and self-distillation actually deliver in the literature.

Nothing under ~/FLEX-UF is modified; its cost model and bisection are
imported and reused so the two numbers stay the same number.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(UF)); sys.path.insert(0, str(UF / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C                                        # noqa: E402
from flexuf.config import FlexUFConfig                       # noqa: E402
from flexuf.cost import exit_costs                           # noqa: E402
from flexuf.eval import per_tile_mse, tiled_exit_mses        # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state      # noqa: E402
from flexuf.reference import reference_for                   # noqa: E402


def bisect(M, R, cost, groups, target, steps=60):
    """The paper's operating point: largest saving whose dB <= target.

    Bisected on the PER-FRAME decibel, which is what ~/DCVC/test_video.py
    averages and therefore what a published DCVC-UF number means. Pooling
    every tile into one MSE is the natural form for the Lagrangian but it is
    the flattering one, by 0.023 to 0.033 dB on an identical allocation --
    a third of the budget. Bisected pooled, this file put the qp32 oracle at
    32.50 per cent against the 26.70 in results/raterank_RECIPE512_b01.json;
    bisected per frame it reproduces that number to four decimals, which is
    the check that says the extrapolations below are on the paper's axis.
    """
    def at(lam):
        k = (M + lam * cost[None, :]).argmin(1)
        per_frame = torch.stack([
            10 * torch.log10(M[g].gather(1, k[g][:, None]).squeeze(1).mean()
                             / R[g].mean()) for g in groups]).mean().item()
        return (per_frame,
                100 * (1 - cost[k].mean() / cost[-1]).item(),
                100 * (1 - cost[k].mean()).item(), k)
    if at(0.0)[0] > target:
        return None
    lo, hi = 0.0, 1.0
    if at(hi)[0] < target:
        return at(hi)
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if at(mid)[0] <= target:
            lo = mid
        else:
            hi = mid
    return at(lo)


def costs_at_split(cfg, j):
    """Exit costs with the shared trunk moved to depth j.

    Split depth is the tiled decoder's only defence against the seam: the
    first j exits' worth of blocks run on the whole frame, so the cut happens
    where the features are already settled. Per-position depth removes the
    seam outright, so j becomes a free parameter -- this is what that is worth
    before any of it is trained.
    """
    d = dict(cfg.__dict__)
    d["split_depth"] = j
    return exit_costs(FlexUFConfig(**d), "head")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PAPER.pth.tar"))
    ap.add_argument("--ref", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--target", type=float, default=0.10)
    ap.add_argument("--out", default=str(HERE / "results/spread_target.json"))
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    print(f"  {a.ckpt}\n  K={cfg.num_exits} j={cfg.split_depth} "
          f"tile={cfg.rgb_patch}px", flush=True)

    seqs, _ = C.discover([])
    frames, measured = [], []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            frames.append((x[0:1], len(measured))); measured.append(s["name"])
    print(f"  {len(frames)} CTC karesi", flush=True)

    cost = exit_costs(cfg, "head").to(dev)
    out = {"ckpt": a.ckpt, "frames": len(frames), "target_db": a.target,
           "split_depth": cfg.split_depth, "rows": []}
    with torch.no_grad():
        for qp_v in a.qps:
            per_exit, ref_mse, tile_seq = [], [], []
            for x, seq_i in frames:
                x = x.to(dev); _, _, H, W = x.shape; P = cfg.rgb_patch
                ph, pw = (-H) % P, (-W) % P
                xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(xp, qp)
                nh, nw = (H + ph) // P, (W + pw) // P
                per_exit.append(tiled_exit_mses(net.dec, y, q, xp, cfg))
                ref_mse.append(per_tile_mse(ref.dec.forward_full(y, q), xp,
                                            nh, nw, P))
                tile_seq.append(torch.full((nh * nw,), seq_i,
                                           dtype=torch.long, device=dev))
            M = torch.cat(per_exit); R = torch.cat(ref_mse)
            SEQ = torch.cat(tile_seq)

            # Before anything is extrapolated, the pipeline has to reproduce
            # the stored oracle. The anchor drift is the tell: at the deepest
            # exit everywhere the decode is stock UF up to -0.0114 dB at q32,
            # so a lambda-0 point far from that means the reference is not the
            # one the paper measured against.
            groups = [g for g in ((SEQ == i) for i in range(len(measured)))
                      if bool(g.any())]
            uni = [10 * torch.log10(M[:, k].mean() / R.mean()).item()
                   for k in range(cost.numel())]
            print("    tekduze cikis dB:",
                  " ".join(f"{k}:{v:+.4f}" for k, v in enumerate(uni)), flush=True)
            k0 = (M + 0.0 * cost[None, :]).argmin(1)
            print(f"    lambda=0 dB {10 * torch.log10(M.gather(1, k0[:, None]).mean() / R.mean()).item():+.4f}"
                  f"   hist {torch.bincount(k0, minlength=cost.numel()).tolist()}",
                  flush=True)
            # Which budget reproduces the stored oracle tells us which decibel
            # the stored number was bisected on. results/raterank_*_b01.json
            # says 26.70 per cent vs release at qp32 "at 0.1 dB"; pooling every
            # tile into one MSE and averaging a per-frame decibel differ by
            # 0.023-0.033 dB on an identical allocation, and at ~1 saving point
            # per 0.01 dB that is the whole discrepancy or none of it.
            probe = []
            for t in (0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15):
                r = bisect(M, R, cost, groups, t)
                probe.append((t, r[2]) if r else (t, None))
            print("    hedef->tasarruf(release):",
                  " ".join(f"{t:.2f}:{'-' if v is None else f'{v:.2f}'}"
                           for t, v in probe), flush=True)
            # A budget met ON AVERAGE is not a budget met on every frame. One
            # lambda for the whole set lets a frame that gives up little
            # subsidise one that gives up a lot; a per-frame lambda forbids it.
            # The three readings of "0.1 dB" are worth 3 to 4 saving points, so
            # which one the stored oracle used decides whether an extrapolation
            # from it is on the paper's axis at all.
            per_seq = []
            for g in groups:
                r = bisect(M[g], R[g], cost, [torch.ones_like(g[g])],
                           a.target)
                if r is not None:
                    per_seq.append(r[2])
            if per_seq:
                print(f"    kare basina lambda, {a.target} dB: "
                      f"{sum(per_seq) / len(per_seq):.2f}% "
                      f"({len(per_seq)}/{len(groups)} kare butceye giriyor)",
                      flush=True)
            base = bisect(M, R, cost, groups, a.target)
            # The exit spread this qp actually has, in dB, tile-averaged: what
            # a tile gives up by taking the shallowest available exit instead
            # of the deepest. Exits below the split are unreachable -- forward()
            # clamps to j -- so the shallowest AVAILABLE one is exit j.
            deep = M[:, -1]
            spread = (10 * torch.log10(M[:, cfg.split_depth] / deep)).mean().item()
            row = {"qp": qp_v, "spread_db": spread,
                   "saving_pct": base[1], "saving_pct_vs_release": base[2],
                   "db": base[0], "sweep": []}
            print(f"\n  qp {qp_v}: yayilim {spread:.3f} dB, "
                  f"{a.target} dB'de {base[1]:.2f}% "
                  f"({base[2]:.2f}% release'e karsi)", flush=True)

            # Two exits below the split do exist in the ladder, but
            # forward() clamps to j, so M[:, 0] and M[:, 1] carry exit 2's
            # error, not their own. Billing them at a 2- and 4-block cost while
            # crediting them with 6-block quality is a free lunch, and the
            # first version of this sweep took it: it read "60 per cent at
            # j=0" off exits that have never been measured. So the shallow
            # exits are not assumed -- they are PARAMETERISED. Give the
            # 2-block exit an error sigma dB above the deepest, put the
            # 4-block one on the same depth-vs-quality curve, and ask what
            # sigma buys 60 per cent. That is a target a training run can be
            # held to, and the measured anchors say what is plausible: exit 2
            # (6 blocks) sits 0.163 dB above the deepest here, exit 3 0.059,
            # exit 4 0.024.
            anchors = {2: uni[2] - uni[5], 3: uni[3] - uni[5],
                       4: uni[4] - uni[5]}
            row["anchor_db_above_deepest"] = anchors
            print(f"    olculen: 6blok {anchors[2]:+.3f} dB, "
                  f"8blok {anchors[3]:+.3f}, 10blok {anchors[4]:+.3f} "
                  f"(en derinin ustunde)", flush=True)

            cj = costs_at_split(cfg, 0).to(dev)
            deep = M[:, -1]
            # A shallow exit does not cost every tile the same. Flat sky
            # gives up almost nothing by leaving early and dense texture gives
            # up a great deal, and that spread IS the thing early exiting
            # exploits. The first version of this sweep set every tile's
            # 2-block error to the same sigma above the deepest, which erases
            # exactly that heterogeneity and understates the saving; it read
            # 40 per cent where the heterogeneity-preserving form reads more.
            # So the hypothetical exit inherits each tile's OWN measured
            # penalty ratio at 6 blocks, raised to a power: a tile that loses
            # nothing at 6 blocks still loses little at 2.
            ratio = (M[:, cfg.split_depth] / deep).clamp_min(1.0)
            for r0 in (1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0):
                r1 = 1.0 + 0.5 * (r0 - 1.0)     # 4 blocks, half the way there
                Ms = M.clone()
                Ms[:, 0] = deep * ratio ** r0
                Ms[:, 1] = deep * ratio ** r1
                sig0 = (10 * torch.log10(Ms[:, 0].mean() / deep.mean())).item()
                sig1 = (10 * torch.log10(Ms[:, 1].mean() / deep.mean())).item()
                r = bisect(Ms, R, cj, groups, a.target)
                if r is None:
                    continue
                row["sweep"].append({"split_depth": 0, "sigma0_db": sig0,
                                     "sigma1_db": sig1, "power": r0,
                                     "saving_pct": r[1],
                                     "saving_pct_vs_release": r[2],
                                     "db": r[0],
                                     "hist": torch.bincount(
                                         r[3], minlength=cost.numel()).tolist()})
            print("    2-blok cikis kalitesi -> tasarruf(release):", flush=True)
            for w in row["sweep"]:
                print(f"      us={w['power']:.2f} (2blok {w['sigma0_db']:.3f} dB) -> "
                      f"{w['saving_pct_vs_release']:5.2f}%   hist {w['hist']}",
                      flush=True)
            got = [w for w in row["sweep"]
                   if w["saving_pct_vs_release"] >= 60.0]
            if got:
                b = max(got, key=lambda w: w["power"])
                print(f"    60% icin 2-blok cikisi en derinin "
                      f"{b['sigma0_db']:.3f} dB ustunde olmali", flush=True)
            else:
                best = max(row["sweep"], key=lambda w: w["saving_pct_vs_release"])
                print(f"    60% bu tarama icinde yok; en iyi "
                      f"{best['saving_pct_vs_release']:.2f}% "
                      f"(2blok {best['sigma0_db']:.3f} dB)", flush=True)
            out["rows"].append(row)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
