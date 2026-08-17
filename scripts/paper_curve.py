"""The number that belongs in a paper: saving against RELEASED DCVC-UF.

Three defects in what came before, all of which flattered or distorted the
result, and all of which this fixes at once.

1. **The reference was our own deepest exit, not DCVC-UF.** Every previous table
   reported "x% saved at y dB" where y was measured against the trained model's
   own deepest exit. That exit has drifted from the released decoder (0.025 /
   0.051 / 0.074 dB at qp0/32/63), so the quoted y understated the real loss by
   exactly that much. A reviewer compares against the published model; so does
   this. Frozen encoder makes it exact -- both decoders read the identical latent
   from the identical bitstream, so the gap is the synthesis and nothing else.

2. **The allocation was a tau threshold, not the Pareto bound.** Taking, per tile,
   the cheapest exit within tau of full decode is a rule, not an optimum: it
   spends the same tolerance on a tile that would cost 0.001 dB to improve as on
   one that would cost 0.5. The Pareto-optimal allocation at a given cost is
   argmin_k (mse_k + lambda * C_k), swept over lambda. That is what an ideal
   router would do, and a tau sweep UNDERSTATES it -- so previous tables were
   conservative rather than wrong, but they were not the bound they claimed.

3. **Two datasets were quoted interchangeably.** The same checkpoint gave 22.8%
   on CTC and 17.2% on OpenImages crops at qp32. Both were reported as "the
   oracle". This measures CTC only, because that is the benchmark the field uses
   and the one the drift was measured on.

Still an ORACLE: it assumes a router that always picks the argmin. The gap
between this and a trained router is the open question and is stated as such
wherever these numbers appear.
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

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ref", default="runs/warmstart/ckpt_warmstart.pth.tar",
                help="the released decoder, warm-started into ladder form; its "
                     "deepest exit is bit-exact DCVC-UF (asserted in tests)")
ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--device", default="cuda:4")
ap.add_argument("--out", default="results/paper_curve.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(a.ref, map_location="cpu", weights_only=False))

# The comparison is only the one claimed if the latents are identical.
sa, sb = net.enc.state_dict(), ref.enc.state_dict()
worst = max((sa[k] - sb[k]).abs().max().item() for k in sa)
assert worst == 0.0, f"encoders differ by {worst}; the latent is not shared"
print(f"  encoder ayni       max|diff| = {worst}")

cost = exit_costs(cfg, "head").to(dev)
seqs, missing = C.discover([])
frames, measured = [], []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
    if x is not None:
        frames.append((x[0:1], pl[0]))
        measured.append(s["name"])
# The test set has to travel with the number. It grew from 10 sequences to 40
# when MCL-JCV finished downloading, and two JSONs written either side of that
# are indistinguishable by content while their numbers are not comparable.
print(f"  {len(frames)} CTC karesi from {len(measured)} sequences "
      f"({len(missing)} not on disk), {cfg.rgb_patch}px tile\n")

LAMBDAS = [0.0] + [10 ** e for e in torch.linspace(-6, -1.5, 22).tolist()]
rows, op_rows = [], []
print(f"  {'qp':>4}{'tasarruf':>10}{'gercek UF alti dB':>20}")
with torch.no_grad():
    for qp_v in a.qps:
        # Per-tile MSE at every exit, and the reference's own per-tile MSE, all
        # from the same latent.
        per_exit, ref_mse, npx = [], [], 0
        for x, pl in frames:
            x = x.to(dev); _, _, H, W = x.shape
            P = cfg.rgb_patch
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net._encode_to_latent(xp, qp)
            nh, nw = (H + ph) // P, (W + pw) // P
            def tiles(img):
                e = ((img - xp) ** 2).mean(1)
                return (e.view(1, nh, P, nw, P).permute(0, 1, 3, 2, 4)
                         .reshape(nh * nw, P * P).mean(1))
            outs = net.dec.forward_all_exits(y, q)
            per_exit.append(torch.stack([tiles(o) for o in outs], 1))
            ref_mse.append(tiles(ref.dec.forward_full(y, q)))
        M = torch.cat(per_exit)          # [tiles, K] our exits
        R = torch.cat(ref_mse)           # [tiles]    released DCVC-UF
        best = None
        for lam in LAMBDAS:
            k = (M + lam * cost[None, :]).argmin(1)
            mse = M.gather(1, k[:, None]).squeeze(1).mean()
            # dB below the RELEASED decoder, on the same tiles.
            db = 10 * torch.log10(mse / R.mean())
            sv = 100 * (1 - cost[k].mean() / cost[-1])
            rows.append({"qp": qp_v, "lam": lam, "db_vs_uf": db.item(),
                         "saving_pct": sv.item(),
                         "hist": torch.bincount(k, minlength=cfg.num_exits).tolist()})
            if db.item() <= 0.1 and (best is None or sv.item() > best["saving_pct"]):
                best = rows[-1]

        # Exact operating points, by bisection on lambda.
        #
        # Quoting "saving at 0.1 dB" off a 23-point log sweep meant interpolating
        # between whatever samples happened to bracket the budget, and the answer
        # then depended on the sweep rather than the decoder: comparing two
        # frontiers that way made qp16 look 9.7 points better when the true gap
        # was 4.6, purely because one curve had a sample sitting on 0.1 dB.
        #
        # Both dB and saving increase with lambda -- a larger lambda prices
        # compute higher, so tiles move to shallower exits -- so bisection is
        # valid. It costs nothing: M and R are already computed, and each step is
        # one argmin over a [tiles, K] tensor.
        #
        # The allocation is discrete, so a target is not exactly attainable. The
        # invariant kept is the one that matters for a claim: dB never EXCEEDS
        # the budget, and the saving reported is the largest achievable under it.
        def at_lam(lam):
            k = (M + lam * cost[None, :]).argmin(1)
            mse = M.gather(1, k[:, None]).squeeze(1).mean()
            return (10 * torch.log10(mse / R.mean()).item(),
                    100 * (1 - cost[k].mean() / cost[-1]).item(), k)

        ops = []
        for target in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
            lo, hi = 0.0, 1.0
            if at_lam(hi)[0] < target:       # budget above the ceiling
                ops.append({"target_db": target, "db_vs_uf": at_lam(hi)[0],
                            "saving_pct": at_lam(hi)[1], "saturated": True})
                continue
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if at_lam(mid)[0] <= target:
                    lo = mid
                else:
                    hi = mid
            db_o, sv_o, k_o = at_lam(lo)
            ops.append({"target_db": target, "lam": lo, "db_vs_uf": db_o,
                        "saving_pct": sv_o, "saturated": False,
                        "hist": torch.bincount(k_o, minlength=cfg.num_exits).tolist()})
        op_rows.extend({"qp": qp_v, **o} for o in ops)
        if best:
            print(f"  {qp_v:>4}{best['saving_pct']:>9.1f}%{best['db_vs_uf']:>19.4f}")
        else:
            b = min((r for r in rows if r['qp'] == qp_v), key=lambda r: r['db_vs_uf'])
            print(f"  {qp_v:>4}{'—':>10}{b['db_vs_uf']:>19.4f}  (0.1 dB'ye hic ulasilamiyor)")

Path(a.out).parent.mkdir(exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"ckpt": a.ckpt, "ref": a.ref, "frames_per_seq": a.frames,
     "n_sequences": len(measured), "measured": measured,
     "not_measured": [m["name"] for m in missing],
     "rows": rows, "op_points": op_rows}, indent=2))
print(f"\n  wrote {a.out}")
