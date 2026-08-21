"""Saving at a chosen dB budget, on the CTC sequences, per QP.

The tau sweep in oracle_diagnostic reports whatever dB each threshold happens to
land on, which is fine for a shape but wrong for a headline: "22.2% at 0.320 dB"
is not an answer to "how much at 0.3 dB". This bisects tau so the ACHIEVED loss
lands on the budget asked for, and reports the saving there.

Oracle, i.e. a perfect router. Stated in every caption because it is an upper
bound and the gap to a real router is the open question, not a rounding error.
"""
import argparse, json, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402
import torch.nn.functional as F
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.model import FlexUFIntra, load_flexuf_state

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--budgets", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--device", default=_gpu("cuda:4"))
ap.add_argument("--out", default="results/budget_table.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
cost = exit_costs(cfg, "head").to(dev)

seqs, _ = C.discover([])
frames = []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
    if x is not None:
        frames.append((s["cls"], x[0:1], pl[0]))
print(f"{len(frames)} CTC karesi, {cfg.rgb_patch}px tile, j={cfg.split_depth}\n")

rows = []
with torch.no_grad():
    for qp_v in a.qps:
        # Per-tile MSE at every exit, plus the frame's dense PSNR, gathered once.
        mses, dense_db = [], []
        for _, x, pl in frames:
            x = x.to(dev); _, _, H, W = x.shape
            ph, pw = (-H) % cfg.rgb_patch, (-W) % cfg.rgb_patch
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net._encode_to_latent(xp, qp)
            outs = net.dec.forward_all_exits(y, q)
            P = cfg.rgb_patch
            nh, nw = (H + ph) // P, (W + pw) // P
            per = []
            for xh in outs:
                e = ((xh - xp) ** 2).mean(1)
                per.append(e.view(1, nh, P, nw, P).permute(0, 1, 3, 2, 4)
                            .reshape(nh * nw, P * P).mean(1))
            mses.append(torch.stack(per, 1))
        M = torch.cat(mses)                     # [tiles, K]
        deep = M[:, -1].mean()

        def at(tau):
            # Per-tile, against the tile's OWN deepest-exit error. Comparing to
            # the frame mean instead made the threshold meaningless: easy tiles
            # sat below it at every exit and hard tiles above it at all of them,
            # so the table came out as "0% until a cliff, then 25%" -- a property
            # of the wrong denominator, not of the content.
            ok = M <= M[:, -1:] * (10 ** (tau / 10))
            ok[:, -1] = True
            k = ok.float().argmax(1)
            db = 10 * torch.log10(M.gather(1, k[:, None]).mean() / deep)
            return db.item(), 100 * (1 - cost[k].mean().item() / cost[-1].item()), k

        # Dense grid, then the best FEASIBLE point. Bisection was wrong here and
        # the table said so: with four usable exits over a few hundred tiles the
        # achieved dB moves in jumps, so a bisection converges onto a
        # discontinuity and reports the value on the wrong side of it -- the
        # "<= 0.1 dB" column came out at 0.22 dB. Scanning and then filtering on
        # achieved <= budget cannot do that, because the constraint is checked on
        # the number actually reported.
        # Linear and fine over the range that matters. A log grid put only six
        # points between 0 and 1 dB -- and its low end was NEGATIVE tau, which
        # silently means "every tile takes the deepest exit" and filled the table
        # with zeros that looked like a finding.
        grid = [i / 400.0 for i in range(0, 1601)]     # 0 .. 4 dB, 0.0025 steps
        pts = []
        for t in grid:
            d, sv, k = at(t)
            pts.append((d, sv, k))
        for B in a.budgets:
            feas = [pt for pt in pts if pt[0] <= B + 1e-9]
            d, sv, k = max(feas, key=lambda pt: pt[1]) if feas else (0.0, 0.0, pts[0][2])
            hist = torch.bincount(k, minlength=cfg.num_exits).tolist()
            rows.append({"qp": qp_v, "budget": B, "achieved_db": d,
                         "saving_pct": sv, "exit_hist": hist})

print(f"  {'qp':>4}" + "".join(f"{'≤'+str(b)+' dB':>14}" for b in a.budgets))
for qp_v in a.qps:
    r = [x for x in rows if x["qp"] == qp_v]
    print(f"  {qp_v:>4}" + "".join(f"{x['saving_pct']:>9.1f}% {x['achieved_db']:>4.2f}" for x in r))
Path(a.out).parent.mkdir(exist_ok=True)
Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "rows": rows}, indent=2))
print(f"\n  -> {a.out}   (ORACLE = kusursuz router; ust sinir)")
