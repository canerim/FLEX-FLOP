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
from flexuf.model import FlexUFIntra, load_flexuf_state

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--ref", default="runs/warmstart/ckpt_warmstart.pth.tar")
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
ap.add_argument("--out", default="results/signalled_curve.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
ov = {}
if a.latent_patch: ov["latent_patch"] = a.latent_patch
if a.seam_repair: ov["seam_repair"] = a.seam_repair
if ov: cfg = FlexUFConfig(**{**cfg.__dict__, **ov})
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(a.ref, map_location="cpu", weights_only=False))
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
    """
    p = torch.bincount(k, minlength=K).float()
    p = p / p.sum()
    H = -(p * p.clamp_min(1e-12).log2()).sum().item()
    return H * k.numel() + K * 8   # payload + a small per-frame histogram

rows = []
print(f"  {'qp':>4}{'tasarruf':>10}{'dB (gercek UF)':>16}{'bpp artisi':>12}{'harita biti':>13}")
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
            nh, nw = (H + ph) // P, (W + pw) // P
            def tl(img):
                e = ((img - xp) ** 2).mean(1)
                return (e.view(1, nh, P, nw, P).permute(0, 1, 3, 2, 4)
                         .reshape(nh * nw, P * P).mean(1))
            M = torch.stack([tl(o) for o in net.dec.forward_all_exits(y, q)], 1)
            R = tl(ref.dec.forward_full(y, q)).mean()
            cache.append((M, R, H * W))

        # dB is averaged PER FRAME, then over frames -- the convention
        # test_video.py uses and the one every published DCVC-UF number follows.
        # paper_curve pools all tiles into one MSE instead, which is the natural
        # form for the Lagrangian the theory is about. The two differ; the
        # difference is measured in scripts/db_convention.py rather than left
        # for a reader to discover by comparing two tables.
        def at_lam(lam):
            SV = DB = EXTRA = MB = n = 0.0
            for M, R, npx in cache:
                # The ENCODER's decision: it has the source, so this is exact.
                k = (M + lam * cost[None, :]).argmin(1)
                mb = map_bits(k, cfg.num_exits)
                SV += (1 - cost[k].mean() / cost[-1]).item()
                DB += (10 * torch.log10(M.gather(1, k[:, None]).squeeze(1).mean() / R)).item()
                EXTRA += mb / npx                     # bpp added by the map
                MB += mb; n += 1
            return 100 * SV / n, DB / n, EXTRA / n, MB / n

        # Bisection on lambda, as in paper_curve. Reading the best sample of a
        # 25-point grid under the budget left this number short of what the
        # system reaches: the grid's closest point at qp0 sat at 0.0975 dB, and
        # the saving between there and 0.1 is not small. Both dB and saving
        # increase with lambda, so bisection is valid, and with the per-frame
        # work cached each step costs one argmin.
        TARGET = 0.1
        best = None
        if at_lam(1.0)[1] >= TARGET:
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if at_lam(mid)[1] <= TARGET:
                    lo = mid
                else:
                    hi = mid
            best = at_lam(lo)
        else:                       # budget above what the ladder can spend
            best = at_lam(1.0)
        if best:
            sv, db, extra, mb = best
            rows.append({"qp": qp_v, "saving_pct": sv, "db_vs_uf": db,
                         "bpp_added": extra, "map_bits": mb})
            print(f"  {qp_v:>4}{sv:>9.1f}%{db:>16.4f}{extra:>12.6f}{mb:>13.0f}")
        else:
            print(f"  {qp_v:>4}{'—':>10}{'0.1 dB ulasilamiyor':>16}")

Path(a.out).parent.mkdir(exist_ok=True)
Path(a.out).write_text(json.dumps(
    {"ckpt": a.ckpt, "frames_per_seq": a.frames,
     "n_sequences": len(measured), "measured": measured,
     "not_measured": [m["name"] for m in missing], "rows": rows}, indent=2))
print(f"\n  wrote {a.out}")
