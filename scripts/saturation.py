"""The exact budget at which the ladder saturates, per rate.

Beyond some quality budget a looser budget buys nothing: every tile has already
taken the cheapest rung the split permits, saving is pinned at 100*(1-c_j), and
spending more dB is simply throwing quality away. That threshold is not something
to search for -- it is one forward pass. Set every tile to exit j and measure the
distortion; any budget at or above that number reaches the ceiling and any budget
below it does not, because both saving and distortion increase monotonically with
lambda.

Reported against the RELEASED decoder's full-frame decode of the same latent, so
the tiling penalty is inside the number, and in units where the release is 1.0.
"""
import argparse, json, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--qps", type=int, nargs="+",
                default=[0, 8, 16, 24, 32, 40, 48, 56, 63])
ap.add_argument("--frames", type=int, default=1)
ap.add_argument("--device", default=_gpu("cuda:0"))
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
CEIL = 100.0 * (1.0 - cost[j].item())

seqs, missing = C.discover([])
frames = []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
    if x is not None:
        for i in range(x.shape[0]):
            frames.append((x[i:i + 1], pl[i]))
print(f"  {len(frames)} CTC karesi / {len(seqs)} sekans, ceiling = {CEIL:.2f}% "
      f"(every tile at exit {j}, cost {cost[j].item():.4f})\n", flush=True)

print(f"  {'qp':>4}{'saturation dB':>16}{'saving there':>15}"
      f"{'floor dB':>11}{'usable band':>14}")
rows = []
with torch.no_grad():
    for qp_v in a.qps:
        db_sat = db_floor = 0.0
        for x, pl in frames:
            x = x.to(dev); _, _, H, W = x.shape
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net._encode_to_latent(xp, qp)
            nt = ((H + ph) // P) * ((W + pw) // P)
            R = ((ref.dec.forward_full(y, q) - xp) ** 2).mean()
            for e, which in ((j, "sat"), (K - 1, "floor")):
                em = torch.full((nt,), e, device=dev)
                M = ((net.dec(y, q, exit_map=em) - xp) ** 2).mean()
                d = (10 * torch.log10(M / R)).item()
                if which == "sat":
                    db_sat += d
                else:
                    db_floor += d
        n = len(frames)
        r = {"qp": qp_v, "saturation_db": db_sat / n, "floor_db": db_floor / n,
             "saving_pct_vs_release": CEIL}
        rows.append(r)
        print(f"  {qp_v:>4}{r['saturation_db']:>16.4f}{CEIL:>14.2f}%"
              f"{r['floor_db']:>11.4f}{r['saturation_db']-r['floor_db']:>14.4f}",
              flush=True)

json.dump({"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "ceiling_pct": CEIL,
           "exit_j": j, "cost_j": cost[j].item(), "n_frames": len(frames),
           "note": "saturation_db = every tile at exit j; any budget at or above "
                   "it reaches the ceiling, any budget below it does not",
           "rows": rows}, open(a.out, "w"), indent=2)
print(f"\n  -> {a.out}")
