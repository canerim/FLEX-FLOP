"""Signal the exit map instead of predicting it — the deliverable, honestly billed.

Why this replaces the router
----------------------------
The router could never reach the oracle, and 12000 steps of training proved the
limit is not optimisation: CE fell from 0.5 to 0.0935 while qp0 agreement moved
0.743 -> 0.741 and its regret did not change to five decimals. The model fit the
training data four times better and its test behaviour was identical. That is
the signature of missing information, not missing capacity.

The information is missing for a structural reason. The oracle picks
argmin(mse_k + lambda*C_k), and mse_k is the error against the SOURCE. The
decoder never sees the source. So the router was being asked to infer something
that is not in its input, and no architecture fixes that.

But the ENCODER sees the source. It can compute the oracle exactly. And in video
coding, mode decisions are signalled rather than inferred -- HEVC and VVC
transmit block partitioning, prediction mode and transform tree; the decoder does
not guess them. Predicting them at the decoder was the unusual choice here, not
the conventional one.

The bill, and it IS billed
--------------------------
Four selectable exits is at most 2 bits per tile, and the map is entropy coded
because its distribution is far from uniform. At 1080p with 256px tiles that is
40 tiles, 80 raw bits, against ~400k bits for the frame at qp0 -- 0.02%.

Every number below adds the map's cost to the bitstream before computing bpp.
Leaving it out would repeat exactly the accounting errors caught earlier today:
a benefit measured in one place and its bill paid in another.
"""
import argparse, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.eval import (per_tile_mse, reference_frame_mse, tiled_exit_mses,
                         true_frame_mse)
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ref", default=None,
                help="reference checkpoint. Default: the warm start matching "
                     "this checkpoint's K")
ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
ap.add_argument("--frames", type=int, default=2)
ap.add_argument("--max_seqs", type=int, default=0,
                help="cap the test set (0 = all present). The mid-epoch health "
                     "check runs often -- every ~1.2 h per run once --ckpt_every "
                     "is on -- and the full set is 40 sequences since MCL-JCV "
                     "landed, which would keep the evaluation card permanently "
                     "busy competing with a training run on the same GPU. The "
                     "subset is deterministic (first N by discovery order) and "
                     "the JSON records exactly which sequences it was, so a "
                     "health check can never be mistaken for the headline.")
ap.add_argument("--latent_patch", type=int, default=None,
                help="override the tile size the checkpoint was trained at, to "
                     "separate 'larger tiles' from 'a different training run'. "
                     "Most of the ladder permits this -- adapters are 1x1 and the "
                     "split is a reshape -- but GridSeamRepair does NOT: its gate "
                     "is P x P, indexed by position within a tile, so it is tied "
                     "to the size it was trained at. Pass --seam_repair none on "
                     "both sides of such a comparison.")
ap.add_argument("--seam_repair", default=None,
                help="override the seam-repair module, e.g. 'none' when changing "
                     "tile size (see --latent_patch).")
ap.add_argument("--device", default="cuda:4")
ap.add_argument("--budgets", type=float, nargs="+", default=None,
                help="evaluate several quality budgets in ONE pass. Everything "
                     "expensive -- encoding, decoding all K exits, the reference "
                     "-- depends only on the latent, so it is cached per QP and "
                     "shared; only the bisection repeats, and each of its steps "
                     "is one argmin on a [tiles, K] table. Three budgets this "
                     "way cost what one used to. Rows carry `budget_db`, so a "
                     "reader must filter on it.")
ap.add_argument("--budget", type=float, default=0.1,
                help="quality budget in dB below the release. 0.1 is the "
                     "project target; larger budgets buy more compute but "
                     "saturate at the ladder's ceiling, first at low rate "
                     "where the frontier is steepest.")
ap.add_argument("--out", default="results/signalled_curve.json")
a = ap.parse_args()
TARGETS = sorted(a.budgets) if a.budgets else [a.budget]
TARGET = TARGETS[0]
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
ov = {}
if a.latent_patch: ov["latent_patch"] = a.latent_patch
if a.seam_repair: ov["seam_repair"] = a.seam_repair
if ov: cfg = FlexUFConfig(**{**cfg.__dict__, **ov})
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                 map_location="cpu", weights_only=False))
sa, sb = net.enc.state_dict(), ref.enc.state_dict()
assert max((sa[k] - sb[k]).abs().max().item() for k in sa) == 0.0
cost = exit_costs(cfg, "head").to(dev)

seqs, missing = C.discover([])
if a.max_seqs:
    missing = missing + [{"name": s["name"]} for s in seqs[a.max_seqs:]]
    seqs = seqs[:a.max_seqs]
frames, measured = [], []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
    if x is not None:
        for i in range(x.shape[0]):
            frames.append((x[i:i+1], pl[i]))
        measured.append(s["name"])
# See paper_curve.py: the test set must be recorded with the result, because it
# changed size mid-experiment and the JSON is otherwise silent about it.
print(f"  encoder ayni, {len(frames)} CTC karesi from {len(measured)} sequences "
      f"({len(missing)} not on disk), {cfg.rgb_patch}px tile\n")

def map_bits(k, K):
    """Entropy of the exit map in bits -- what an ideal entropy coder would spend.

    Not 2 bits per tile: the distribution is far from uniform, and an arithmetic
    coder with a per-frame histogram spends its entropy. Reported this way rather
    than as the 2-bit worst case because the worst case is not what a codec
    would ship, and overstating our own cost is as wrong as understating it.

    One overstatement remains, deliberately. Under j=2 the exits 0, 1 and 2 all
    cost the same -- the first j groups run full-frame for every tile -- so a
    real codec would signal four symbols, not six, and argmin scatters the
    cheapest allocation across three indices that this entropy counts
    separately. Measured, that inflates the map by at most 7 bits per frame at
    qp0 and by 0 at qp63, against 218-266 bits total. Left as it is: the number
    is then an upper bound on a cost already four orders of magnitude below the
    frame, and a bound in the safe direction is worth more than 3% of a
    negligible quantity.
    """
    p = torch.bincount(k, minlength=K).float()
    p = p / p.sum()
    H = -(p * p.clamp_min(1e-12).log2()).sum().item()
    return H * k.numel() + K * 8   # payload + a small per-frame histogram

rows = []
print(f"  {'qp':>4}{'butce':>7}{'tasarruf':>10}{'dB (gercek UF)':>16}"
      f"{'bpp artisi':>12}{'harita biti':>13}")
print("  tasarruf = saving_pct_vs_release: payda YAYINLANMIS decoder'in 1.0'i, "
      "bizim en derin exitimizin 1.0095'i degil\n")
with torch.no_grad():
    for qp_v in a.qps:
        # Everything a frame contributes that does NOT depend on lambda, cached
        # once per QP.
        #
        # This loop used to sit INSIDE the lambda sweep, so all 25 lambdas
        # re-encoded and re-decoded all 80 frames: 25x the work for identical
        # numbers, because the per-tile MSEs, the reference and the bitrate are
        # all functions of the latent alone. Only the argmin, the exit map and
        # the aggregates move with lambda, and those are tensor ops on a
        # [tiles, K] table. Measured at 40 sequences the old form took over half
        # an hour per checkpoint, which also meant every training run queued
        # behind it.
        cache = []
        for x, pl in frames:
            x = x.to(dev); _, _, H, W = x.shape; P = cfg.rgb_patch
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, aux = net._encode_to_latent(xp, qp)
            # The per-exit table is built on the DEPLOYED path -- one tiled
            # decode per exit -- not with dec.forward_all_exits, which runs full
            # frame and cancels the tiling penalty out of the reported dB. See
            # flexuf/eval.py for the measurement that forced this change.
            M = tiled_exit_mses(net.dec, y, q, xp, cfg)
            R = reference_frame_mse(ref.dec, y, q, xp)
            cache.append((M, R, H * W, y, q, xp))

        # dB is averaged PER FRAME, then over frames -- the convention
        # test_video.py uses and the one every published DCVC-UF number follows.
        # paper_curve pools all tiles into one MSE instead, which is the natural
        # form for the Lagrangian the theory is about. The two differ; the
        # difference is measured in scripts/db_convention.py rather than left
        # for a reader to discover by comparing two tables.
        def at_lam(lam):
            SV = SVR = DB = EXTRA = MB = n = 0.0
            for M, R, npx, _y, _q, _xp in cache:
                # The ENCODER's decision: it has the source, so this is exact.
                k = (M + lam * cost[None, :]).argmin(1)
                mb = map_bits(k, cfg.num_exits)
                SV += (1 - cost[k].mean() / cost[-1]).item()
                # cost[-1] is OUR ladder at full depth, 1.0095 stock decodes:
                # the deepest exit pays seam repair and the released decoder
                # does not. Dividing by it answers "what does early exiting
                # save against our own full-depth path", while every claim
                # around this number says "against the released DCVC-UF
                # decoder" -- and drops the seam-repair tax out of the figure
                # it is supposed to be net of. Costs are already in units of
                # one stock decode, so the release denominator is exactly 1.
                SVR += (1 - cost[k].mean()).item()
                DB += (10 * torch.log10(M.gather(1, k[:, None]).squeeze(1).mean() / R)).item()
                EXTRA += mb / npx                     # bpp added by the map
                MB += mb; n += 1
            return 100 * SV / n, DB / n, EXTRA / n, MB / n, 100 * SVR / n

        def true_db(lam):
            """The dB an actual decode delivers at this lambda.

            The table above measures each tile with every OTHER tile at the same
            exit. A routed frame is mixed, so a tile's border sees whatever depth
            its neighbour chose, and only a real decode of the real map settles
            it. One decode per frame, so this is affordable outside the
            bisection but not inside it.
            """
            tot = 0.0
            for M, R, _npx, y_, q_, xp_ in cache:
                k = (M + lam * cost[None, :]).argmin(1).clamp(min=cfg.split_depth)
                mse = true_frame_mse(net.dec, y_, q_, xp_, k)
                tot += (10 * torch.log10(mse / R)).item()
            return tot / len(cache)

        # Bisection on lambda. Reading the best sample of a 25-point grid under
        # the budget left this number short of what the system reaches: the
        # grid's closest point at qp0 sat at 0.0975 dB, and the saving between
        # there and 0.1 is not small. Both dB and saving increase with lambda,
        # so bisection is valid.
        #
        # TARGETS is set once from --budget/--budgets at module scope. The
        # target used to be re-assigned inside this loop: with --budget wired up
        # but that line left in place, every run reported 0.1 dB results in a
        # file named for 0.3 or 0.5, and nothing downstream could tell.
        #
        # The FLOOR first, on the deployed path. At lambda = 0 nothing is
        # charged for compute, so every tile takes its lowest-MSE exit -- the
        # least distortion this ladder can produce. If that already exceeds the
        # budget, no allocation meets it and there is nothing to report.
        #
        # Measured with a real decode, not with the table: the floor is exactly
        # where the tiling penalty is largest, because every tile is running its
        # full per-tile depth.
        floor_db = true_db(0.0)

        def bisect_to(t):
            """Largest lambda whose TABLE dB stays under t. 60 halvings."""
            if at_lam(1.0)[1] < t:
                return 1.0
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if at_lam(mid)[1] <= t:
                    lo = mid
                else:
                    hi = mid
            return lo

        for target in TARGETS:
            if floor_db > target:
                print(f"  {qp_v:>4}{target:>7.2f}{'—':>10}"
                      f"{'floor ' + format(floor_db, '.4f') + ' dB':>18}"
                      f"{'exceeds the budget':>22}")
                rows.append({"qp": qp_v, "saving_pct": None, "db_vs_uf": None,
                             "floor_db": floor_db, "budget_db": target,
                             "budget_reachable": False})
                continue

            # Bisect on the cheap table, then correct with a real decode. The
            # table measures each tile with its neighbours at the SAME exit; a
            # routed frame is mixed, so the two differ by a small, nearly
            # constant residual. Shifting the proxy target by that residual and
            # re-bisecting converges in two or three passes, and each pass costs
            # one decode per frame rather than one per bisection step.
            inner, lam, td = target, None, None
            for _ in range(6):
                lam = bisect_to(inner)
                td = true_db(lam)
                if abs(td - target) < 5e-4:
                    break
                inner = max(1e-4, min(1.0, inner + (target - td)))
            sv, _db_table, extra, mb, svr = at_lam(lam)
            rows.append({"qp": qp_v, "saving_pct": sv,
                         # db_vs_uf is now what a decoder DELIVERS, from one
                         # real decode of the chosen map. db_vs_uf_table is the
                         # uniform-exit table's estimate, kept so the residual
                         # between them stays visible instead of being an
                         # unexamined modelling assumption.
                         "db_vs_uf": td, "db_vs_uf_table": _db_table,
                         "saving_pct_vs_release": svr,
                         "lam": lam, "bpp_added": extra, "map_bits": mb,
                         "floor_db": floor_db, "budget_db": target,
                         "budget_reachable": True})
            print(f"  {qp_v:>4}{target:>7.2f}{svr:>9.2f}%{td:>16.4f}"
                  f"{extra:>12.6f}{mb:>13.0f}")

Path(a.out).parent.mkdir(exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"ckpt": a.ckpt,
     # Which checkpoint this is, not which step the run has since reached.
     # compare_runs.py was reading the run's CURRENT step and labelling a
     # step-4000 measurement as 6,800.
     "ckpt_epoch": ck.get("epoch"), "ckpt_step": ck.get("step"),
     "budgets": TARGETS, "frames_per_seq": a.frames,
     "n_sequences": len(measured), "measured": measured,
     "not_measured": [m["name"] for m in missing], "rows": rows}, indent=2))
print(f"\n  wrote {a.out}")
