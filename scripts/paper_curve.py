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
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ref", default=None,
                help="the released decoder, warm-started into ladder form; its "
                     "deepest exit is bit-exact DCVC-UF (asserted in tests). "
                     "Default: the file matching this checkpoint's K, since the "
                     "key layout depends on how the 12 blocks are grouped")
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
load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                 map_location="cpu", weights_only=False))

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
        frames.append((x[0:1], pl[0], len(measured)))
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
        per_exit, ref_mse, tile_seq, npx = [], [], [], 0
        for x, pl, seq_i in frames:
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
            tile_seq.append(torch.full((nh * nw,), seq_i, dtype=torch.long,
                                       device=dev))
        M = torch.cat(per_exit)          # [tiles, K] our exits
        R = torch.cat(ref_mse)           # [tiles]    released DCVC-UF
        SEQ = torch.cat(tile_seq)        # [tiles]    which sequence each came from
        # One frame per sequence is read, so SEQ groups frames as well.
        groups = [g for g in ((SEQ == i) for i in range(len(measured)))
                  if bool(g.any())]

        best = None
        for lam in LAMBDAS:
            k = (M + lam * cost[None, :]).argmin(1)
            mse = M.gather(1, k[:, None]).squeeze(1).mean()
            # Two decibels on every sweep point, not just the operating points.
            #
            # `db_vs_uf` pools every tile of every frame into one MSE, the
            # natural form for the Lagrangian the theory is about.
            # `db_vs_uf_per_frame` averages a per-frame decibel, which is what
            # ~/DCVC/test_video.py does and so what published DCVC-UF numbers
            # mean. They differ by 0.023-0.033 dB on an identical allocation --
            # a quarter to a third of a 0.1 dB budget, with pooling always the
            # flattering one. BD-saving integrates these rows, so it could not
            # be quoted in the codec convention until both were here.
            db = 10 * torch.log10(mse / R.mean())
            dbf = torch.stack([
                10 * torch.log10(M[g].gather(1, k[g][:, None]).squeeze(1).mean()
                                 / R[g].mean()) for g in groups]).mean()
            sv = 100 * (1 - cost[k].mean() / cost[-1])
            # cost[-1] is OUR ladder at full depth (1.0095 stock decodes: the
            # deepest exit pays seam repair, the release does not), so `sv`
            # measures early exiting against our own full-depth path. Every
            # claim built on it says "against the released DCVC-UF decoder",
            # which needs denominator 1 -- costs are already in units of one
            # stock decode. Kept side by side rather than swapped, because
            # every stored curve and every BD comparison uses saving_pct.
            svr = 100 * (1 - cost[k].mean())
            rows.append({"qp": qp_v, "lam": lam, "db_vs_uf": db.item(),
                         "db_vs_uf_per_frame": dbf.item(),
                         "saving_pct": sv.item(),
                         "saving_pct_vs_release": svr.item(),
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
            """(pooled dB, per-frame dB, saving, assignment) at one lambda.

            Two decibels, because the two are not the same number and both are
            quoted somewhere. Pooling every tile into one MSE is the natural
            form for the Lagrangian the theory is about; averaging a per-frame
            decibel is what ~/DCVC/test_video.py does, so it is what published
            DCVC-UF numbers mean. Measured on an identical allocation the two
            differ by 0.023-0.033 dB -- a quarter to a third of the 0.1 dB
            budget -- with pooling always the flattering one. Reporting only
            one of them silently picks a side.
            """
            k = (M + lam * cost[None, :]).argmin(1)
            mse = M.gather(1, k[:, None]).squeeze(1).mean()
            pooled = 10 * torch.log10(mse / R.mean()).item()
            per_frame = torch.stack([
                10 * torch.log10(M[g].gather(1, k[g][:, None]).squeeze(1).mean()
                                 / R[g].mean()) for g in groups]).mean().item()
            return (pooled, per_frame,
                    100 * (1 - cost[k].mean() / cost[-1]).item(), k)

        ops = []
        for target in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.50):
            lo, hi = 0.0, 1.0
            # Budget below the FLOOR: at lambda = 0 every tile takes its
            # lowest-MSE exit, so that is the least distortion available. If it
            # exceeds the budget, no allocation meets the budget -- and without
            # this the bisection reports the lambda-0 point as if it did.
            if at_lam(0.0)[0] > target:
                p0, f0, s0, _ = at_lam(0.0)
                ops.append({"target_db": target, "db_vs_uf": p0,
                            "db_vs_uf_per_frame": f0, "saving_pct": None,
                            "floor_db": p0, "budget_reachable": False})
                continue
            if at_lam(hi)[0] < target:       # budget above the ceiling
                p_, f_, s_, _ = at_lam(hi)
                ops.append({"target_db": target, "db_vs_uf": p_,
                            "db_vs_uf_per_frame": f_,
                            "saving_pct": s_, "saturated": True})
                continue
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if at_lam(mid)[0] <= target:
                    lo = mid
                else:
                    hi = mid
            db_o, dbf_o, sv_o, k_o = at_lam(lo)
            # Per-sequence breakdown at the SAME lambda.
            #
            # The headline is an average over 40 sequences, and an average hides
            # the question a deployment actually asks: is this saving uniform,
            # or does it come from a few easy clips? The allocation is global --
            # one lambda prices compute for every tile -- so each sequence's own
            # saving and dB at that lambda are read out directly, and their
            # spread is the answer.
            per_seq = []
            for i, nm in enumerate(measured):
                sel = SEQ == i
                if not bool(sel.any()):
                    continue
                ks = k_o[sel]
                mse_s = M[sel].gather(1, ks[:, None]).squeeze(1).mean()
                per_seq.append({
                    "seq": nm,
                    "db_vs_uf": (10 * torch.log10(mse_s / R[sel].mean())).item(),
                    "saving_pct": (100 * (1 - cost[ks].mean() / cost[-1])).item(),
                    "saving_pct_vs_release": (100 * (1 - cost[ks].mean())).item()})
            ops.append({"target_db": target, "lam": lo, "db_vs_uf": db_o,
                        "db_vs_uf_per_frame": dbf_o,
                        "saving_pct": sv_o, "saturated": False,
                        "hist": torch.bincount(k_o, minlength=cfg.num_exits).tolist(),
                        "per_sequence": per_seq})
        op_rows.extend({"qp": qp_v, **o} for o in ops)
        if best:
            print(f"  {qp_v:>4}{best['saving_pct']:>9.1f}%{best['db_vs_uf']:>19.4f}")
        else:
            b = min((r for r in rows if r['qp'] == qp_v), key=lambda r: r['db_vs_uf'])
            print(f"  {qp_v:>4}{'—':>10}{b['db_vs_uf']:>19.4f}  (0.1 dB'ye hic ulasilamiyor)")

Path(a.out).parent.mkdir(exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"ckpt": a.ckpt, "ref": reference_for(cfg, a.ref),
     # WHICH snapshot, not just which path. ckpt_step.pth.tar is overwritten
     # every --ckpt_every steps, so the path is stable while the weights are
     # not: two files naming it can be hours of training apart, and nothing in
     # them said so. That let a comparison of curve_BEST128 against
     # curve_FINE12 silently weigh different amounts of training.
     "ckpt_epoch": ck.get("epoch"), "ckpt_step": ck.get("step"),
     "frames_per_seq": a.frames,
     "n_sequences": len(measured), "measured": measured,
     "not_measured": [m["name"] for m in missing],
     "rows": rows, "op_points": op_rows}, indent=2))
print(f"\n  wrote {a.out}")
