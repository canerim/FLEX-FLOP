"""What does configuration A's search actually cost the encoder?

The paper says "about 1.21 decodes per frame", which came from the cost model of
`dec.forward_all_exits`: one shared trunk pass and K adapter-and-head taps. That
is the FULL-FRAME table -- the one Section 5 shows is not the table the decoder
delivers against. Building the table on the deployed path means K tiled decodes,
which is a different number entirely, and the paper has been quoting the cheap
one.

Measured here, all three, interleaved on one GPU:

  decode           one tiled decode at full depth, the unit
  full-frame table dec.forward_all_exits -- the cheap approximation
  deployed table   K tiled decodes -- what the exact search costs

and the quality the approximation gives up, so the trade is visible rather than
assumed.
"""
import argparse, json, statistics, sys
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
ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_eval.pth.tar")
ap.add_argument("--qp", type=int, default=32)
ap.add_argument("--iters", type=int, default=15)
ap.add_argument("--seq", default=None,
                help="pick a sequence by name. The default first-1920-wide is "
                     "Beauty, which is easy enough that the ladder saturates at "
                     "0.1 dB and both searches trivially agree -- an "
                     "uninformative comparison.")
ap.add_argument("--device", default="cuda:7")
ap.add_argument("--out", default="results/encoder_cost.json")
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
s = (next(x for x in seqs if a.seq.lower() in x["name"].lower()) if a.seq
     else next(x for x in seqs if x["w"] == 1920))
x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
x = x[0:1].to(dev); _, _, H, W = x.shape
ph, pw = (-H) % P, (-W) % P
xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
nh, nw = (H + ph) // P, (W + pw) // P
nt = nh * nw

with torch.no_grad():
    qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
    y, q, _ = net._encode_to_latent(xp, qp)
    deep = torch.full((nt,), K - 1, dtype=torch.long, device=dev)

    def timeit(fn):
        for _ in range(4):
            fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.iters):
            e0, e1 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            torch.cuda.synchronize(); e0.record(); fn(); e1.record()
            torch.cuda.synchronize(); ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)

    t_dec = timeit(lambda: net.dec(y, q, exit_map=deep))
    t_ff = timeit(lambda: net.dec.forward_all_exits(y, q))
    t_dep = timeit(lambda: tiled_exit_mses(net.dec, y, q, xp, cfg))

print(f"  {s['name'][:26]} q{a.qp}, {nt} tiles, median of {a.iters}\n")
print(f"  one tiled decode (the unit)      {t_dec:8.1f} ms   1.00x")
print(f"  full-frame table (approximate)   {t_ff:8.1f} ms   {t_ff/t_dec:.2f}x")
print(f"  deployed table (exact)           {t_dep:8.1f} ms   {t_dep/t_dec:.2f}x")

# and what the approximation gives up
with torch.no_grad():
    Rm = reference_frame_mse(ref.dec, y, q, xp)
    Mdep = tiled_exit_mses(net.dec, y, q, xp, cfg)
    outs = net.dec.forward_all_exits(y, q)

    def tl(img):
        e = ((img - xp) ** 2).mean(1)
        return (e.view(1, nh, P, nw, P).permute(0, 1, 3, 2, 4)
                 .reshape(nt, P * P).mean(1))
    Mff = torch.stack([tl(o) for o in outs], 1)

    def solve(M):
        lo, hi = 0.0, 1.0
        for _ in range(28):
            mid = 0.5 * (lo + hi)
            k = (M + mid * cost[None, :]).argmin(1).clamp(min=j)
            d = (10 * torch.log10(true_frame_mse(net.dec, y, q, xp, k) / Rm)).item()
            if d <= 0.1:
                lo = mid
            else:
                hi = mid
        k = (M + lo * cost[None, :]).argmin(1).clamp(min=j)
        return (10 * torch.log10(true_frame_mse(net.dec, y, q, xp, k) / Rm)).item(), \
               100 * (1 - cost[k].mean()).item(), k
    d_dep, s_dep, k_dep = solve(Mdep)
    d_ff, s_ff, k_ff = solve(Mff)
    agree = (k_dep == k_ff).float().mean().item()

print(f"\n  exact search:       {s_dep:6.2f}% at {d_dep:.4f} dB")
print(f"  approximate search: {s_ff:6.2f}% at {d_ff:.4f} dB   "
      f"(maps agree on {agree:.0%} of tiles)")

json.dump({"seq": s["name"], "qp": a.qp, "n_tiles": nt,
           "ms_one_decode": t_dec, "ms_full_frame_table": t_ff,
           "ms_deployed_table": t_dep,
           "x_full_frame": t_ff / t_dec, "x_deployed": t_dep / t_dec,
           "exact": {"saving": s_dep, "db": d_dep},
           "approx": {"saving": s_ff, "db": d_ff, "agreement": agree}},
          open(a.out, "w"), indent=2)
print(f"\n  -> {a.out}")
