"""Why does saving collapse as QP rises? Measure the mechanism, do not assume it.

The frontier is built from one quantity: how much distortion each exit adds
relative to the full decode. If that penalty is small, tiles can leave early
cheaply and the saving is large. So the QP dependence has to live there, and
this prints it directly -- per-exit distortion ratio D_k / D_K, per QP, on CTC.

Reported as a RATIO rather than an absolute MSE on purpose. Absolute error falls
steeply with QP (a qp63 reconstruction is simply better), so absolute numbers
would show the trivial fact that high-rate coding has less error, not the thing
that decides the frontier -- how much WORSE a shallow exit is than the deep one
at the same rate.
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
ap.add_argument("--ckpt", default="runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar")
ap.add_argument("--ref", default="runs/warmstart/ckpt_warmstart.pth.tar")
ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
ap.add_argument("--frames", type=int, default=2)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--out", default="results/why_qp.json")
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(a.ref, map_location="cpu", weights_only=False))
cost = exit_costs(cfg, "head").to(dev)

seqs, _ = C.discover([])
frames = []
for s in seqs:
    x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
    if x is not None:
        for i in range(x.shape[0]):
            frames.append(x[i:i+1])

K = cfg.num_exits
print(f"  {len(frames)} CTC karesi, {cfg.rgb_patch}px tile\n")
print(f"  cikis basina, YAYINLANMIS DCVC-UF'e gore dB kaybi (tum tile'lar o cikista)\n")
print(f"  {'qp':>4}" + "".join(f"{'cikis '+str(k):>11}" for k in range(cfg.split_depth, K))
      + f"{'bpp':>9}")
rows = []
with torch.no_grad():
    for qp_v in a.qps:
        acc = torch.zeros(K, device=dev); R = 0.0; bp = 0.0; n = 0
        for x in frames:
            x = x.to(dev); _, _, H, W = x.shape; P = cfg.rgb_patch
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, aux = net._encode_to_latent(xp, qp)
            b, _, _ = net._rate(aux, qp, H * W)
            for k, o in enumerate(net.dec.forward_all_exits(y, q)):
                acc[k] += ((o - xp) ** 2).mean()
            R += ((ref.dec.forward_full(y, q) - xp) ** 2).mean().item()
            bp += b.item(); n += 1
        db = (10 * torch.log10(acc / n / (R / n))).tolist()
        rows.append({"qp": qp_v, "db_per_exit": db, "bpp": bp / n})
        print(f"  {qp_v:>4}" + "".join(f"{db[k]:>11.4f}" for k in range(cfg.split_depth, K))
              + f"{bp/n:>9.4f}")

Path(a.out).write_text(json.dumps({"cost": cost.tolist(), "rows": rows}, indent=2))
print(f"\n  cost/cikis: " + "  ".join(f"C{k}={cost[k]:.4f}" for k in range(cfg.split_depth, K)))
print(f"  wrote {a.out}")
