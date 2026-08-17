"""Measure the quantities the theory is about, so no claim in it is decorative.

Three things, each a direct consequence of the derivation:

  1. Delta(lambda) = J_blind(lambda) - J_oracle(lambda) >= 0, with equality iff
     every tile agrees on the argmin. This is min-of-average >= average-of-min,
     and it is exactly the value of per-tile routing in Lagrangian units.
  2. Delta grows with rate, because the tiles' distortion vectors become more
     heterogeneous as the latent carries more detail.
  3. On a single hull segment, dB(S) = 10 log10(a + bS) is CONCAVE in S, so the
     marginal dB per extra point of saving falls within a segment and jumps at
     vertices.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C_
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.model import FlexUFIntra, load_flexuf_state

dev = "cuda:0"
CKPT = "runs/wdec_j2_p128_grid/ckpt_epo0.pth.tar"
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load("runs/warmstart/ckpt_warmstart.pth.tar",
                                  map_location="cpu", weights_only=False))
C = exit_costs(cfg, "head").to(dev)
J = cfg.split_depth

seqs, _ = C_.discover([])
frames, measured = [], []
for s in seqs:
    x, pl = C_.read_frames(s["path"], s["w"], s["h"], 2, 1)
    if x is not None:
        for i in range(x.shape[0]):
            frames.append(x[i:i+1])
        measured.append(s["name"])

out = {}
print(f"  {len(frames)} CTC karesi from {len(measured)} sequences\n")
print("  1) Delta(lambda) = J_blind - J_oracle  (Lagrange biriminde, MSE olcegi)")
print(f"  {'qp':>4}" + "".join(f"{('l='+f'{l:.0e}'):>13}" for l in (1e-5, 3e-5, 1e-4, 3e-4)))
with torch.no_grad():
    for qp_v in (0, 16, 32, 48, 63):
        Ms, R = [], 0.0
        for x in frames:
            x = x.to(dev); _, _, H, W = x.shape; P = cfg.rgb_patch
            ph, pw = (-H) % P, (-W) % P
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
            y, q, _ = net._encode_to_latent(xp, qp)
            nh, nw = (H + ph) // P, (W + pw) // P
            def tl(img):
                e = ((img - xp) ** 2).mean(1)
                return (e.view(1, nh, P, nw, P).permute(0, 1, 3, 2, 4)
                         .reshape(nh * nw, P * P).mean(1))
            Ms.append(torch.stack([tl(o) for o in net.dec.forward_all_exits(y, q)], 1))
            R += tl(ref.dec.forward_full(y, q)).mean().item()
        M = torch.cat(Ms); R /= len(frames)
        Dbar = M.mean(0)                                   # blind: average first
        row, rec = [], {}
        for lam in (1e-5, 3e-5, 1e-4, 3e-4):
            Jb = (Dbar[J:] + lam * C[J:]).min().item()     # min of average
            Jo = (M[:, J:] + lam * C[None, J:]).min(1).values.mean().item()  # average of min
            row.append(Jb - Jo); rec[f"{lam:.0e}"] = {"J_blind": Jb, "J_oracle": Jo,
                                                      "delta": Jb - Jo}
        out[qp_v] = {"delta": rec, "Dbar": Dbar.tolist(), "D_ref": R,
                     "cost": C.tolist()}
        print(f"  {qp_v:>4}" + "".join(f"{v:>13.3e}" for v in row))

print("\n  2) Delta / J_oracle  (olcekten bagimsiz, oranin QP ile buyumesi)")
print(f"  {'qp':>4}" + "".join(f"{('l='+f'{l:.0e}'):>13}" for l in (1e-5, 3e-5, 1e-4, 3e-4)))
for qp_v, d in out.items():
    print(f"  {qp_v:>4}" + "".join(
        f"{d['delta'][f'{l:.0e}']['delta']/d['delta'][f'{l:.0e}']['J_oracle']:>12.2%}"
        for l in (1e-5, 3e-5, 1e-4, 3e-4)))

print("\n  3) Icbukeylik: ayni segmentte marjinal dB / ek %5 tasarruf DUSMELI")
rows = json.load(open("results/paper_curve_grid128.json"))["rows"]
def db_at(qp, s):
    c = [r for r in rows if r["qp"] == qp and r["saving_pct"] >= s]
    return min(c, key=lambda r: r["db_vs_uf"])["db_vs_uf"] if c else float("nan")
print(f"  {'qp':>4}{'10->15':>10}{'15->20':>10}{'20->25':>10}{'25->30':>10}")
for qp_v in (0, 16, 32, 48, 63):
    d = [db_at(qp_v, s) for s in (10, 15, 20, 25, 30)]
    print(f"  {qp_v:>4}" + "".join(f"{d[i+1]-d[i]:>10.4f}" for i in range(4)))

# See why_qp.py: the table this feeds is meaningless without knowing which
# model and which sequences it was measured on.
out["_provenance"] = {"ckpt": CKPT, "n_sequences": len(measured),
                      "measured": measured}
Path("results/theory_check.json").write_text(json.dumps(out, indent=2))
print("\n  wrote results/theory_check.json")
