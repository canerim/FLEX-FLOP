"""What the saving looks like. Released against ours at the working budget.

A compression paper's most persuasive figure is the one that shows the reader
there is nothing to see. Two crops from the same bitstream -- one decoded by the
released decoder, one by ours at the 0.1 dB operating point with roughly a third
of the multiply-accumulates -- and the amplified difference between them.

The crop is chosen automatically as the tile that gave up the most quality, so
the figure shows the method's worst case rather than a flattering one.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R)); sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402
from ckpt import pinned as _pin  # noqa: E402
import naturestyle as ns
ns.apply()
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.eval import reference_frame_mse, tiled_exit_mses, true_frame_mse
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.reference import reference_for

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=_pin("runs/RECIPE512/ckpt_eval.pth.tar"))
ap.add_argument("--seq", default="Bosphorus")
ap.add_argument("--qp", type=int, default=32)
ap.add_argument("--budget", type=float, default=0.1)
ap.add_argument("--crop", type=int, default=320)
ap.add_argument("--device", default=_gpu("cuda:2"))
ap.add_argument("--out", default="docs/figures/qualitative.png")
# Provenance beside the picture. Every earlier version of this figure wrote a
# PNG and nothing else, so which checkpoint decoded it could not be recovered
# once the per-epoch file it defaulted to had been overwritten -- and a
# qualitative figure whose checkpoint cannot be stated is not evidence.
ap.add_argument("--sidecar", default=None,
                help="JSON file recording the checkpoint, sequence, rate, "
                     "budget and the numbers printed on the panels")
a = ap.parse_args()
dev = a.device

ck = torch.load(R / a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"])
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load(reference_for(cfg, None), map_location="cpu",
                                  weights_only=False))
cost = exit_costs(cfg, "head").to(dev)
K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch

seqs, _ = C.discover([])
s = next(x for x in seqs if a.seq.lower() in x["name"].lower())
x, pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
x = x[0:1].to(dev); _, _, H, W = x.shape
ph, pw = (-H) % P, (-W) % P
xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
nh, nw = (H + ph) // P, (W + pw) // P

with torch.no_grad():
    qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
    y, q, aux = net._encode_to_latent(xp, qp)
    # Rate is charged on the TRUE pixel count. It is identical for both panels:
    # early exit changes the synthesis, never the bitstream, which is the whole
    # reason the two decodes can be compared on one latent.
    bpp = float(net._rate(aux, qp, H * W)[0].item())
    M = tiled_exit_mses(net.dec, y, q, xp, cfg)
    Rm = reference_frame_mse(ref.dec, y, q, xp)
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        k = (M + mid * cost[None, :]).argmin(1).clamp(min=j)
        if (10 * torch.log10(true_frame_mse(net.dec, y, q, xp, k) / Rm)).item() \
                <= a.budget:
            lo = mid
        else:
            hi = mid
    k = (M + lo * cost[None, :]).argmin(1).clamp(min=j)
    saving = 100 * (1 - cost[k].mean()).item()
    db = (10 * torch.log10(true_frame_mse(net.dec, y, q, xp, k) / Rm)).item()
    rel = ref.dec.forward_full(y, q)[:, :, :H, :W]
    our = net.dec(y, q, exit_map=k)[:, :, :H, :W]
    # The metric DCVC-UF's own paper reports: (6Y+U+V)/8 in 4:2:0 against the
    # uint8 source planes, not against a chroma upsample of them.
    psnr_rel = C.psnr_611_420(rel, pl[0])
    psnr_our = C.psnr_611_420(our, pl[0])

# the worst tile, so the crop is not a flattering choice
per = ((our - rel) ** 2).mean(1)[0]
tt = (per.view(nh, P, nw, P) if False else
      F.pad(per, (0, pw, 0, ph)).view(nh, P, nw, P).permute(0, 2, 1, 3)
      .reshape(nh * nw, P * P).mean(1))
ti = int(tt.argmax())
cy, cx = (ti // nw) * P + P // 2, (ti % nw) * P + P // 2
h2 = a.crop // 2
cy = int(np.clip(cy, h2, H - h2)); cx = int(np.clip(cx, h2, W - h2))
sl = (slice(cy - h2, cy + h2), slice(cx - h2, cx + h2))

# The decoder's tensors are YCbCr 4:4:4 shifted by -0.5, not RGB. Displaying
# them directly renders the picture pink; the conversion is the same one
# test_video.py applies before writing pixels.
from src.utils.transforms import ycbcr2rgb


def to_img(t):
    rgb = ycbcr2rgb(t + 0.5)
    return np.clip(rgb[0].permute(1, 2, 0).cpu().numpy(), 0, 1)
A_, B_ = to_img(rel), to_img(our)
err = np.abs(A_ - B_).mean(2)

# Three panels in a row at column width, not four across the page.
#
# A full-width figure has to force a page break -- reportlab has no float
# mechanism -- and wherever the break lands, the rest of that page's columns are
# lost. This one left page 15's right column ending at 31% of the page. The
# fourth panel was the whole frame with a box on it, which is context the caption
# can carry in words; the three that matter are the two crops and the difference,
# and they are square, so they tile a column exactly.
fig = plt.figure(figsize=(ns.W1, 1.42))
gs = fig.add_gridspec(1, 3, wspace=0.045)
ax = [fig.add_subplot(gs[i]) for i in range(3)]
ax[0].imshow(A_[sl])
ax[0].set_title(f"released, {psnr_rel:.2f} dB", fontsize=6, color=ns.INK2,
                loc="left", pad=2)
ax[1].imshow(B_[sl])
ax[1].set_title(f"ours, {saving:.0f}% fewer, {psnr_our:.2f} dB",
                fontsize=6, color=ns.INK2, loc="left", pad=2)
im = ax[2].imshow(err[sl] * 20, cmap="magma", vmin=0, vmax=1)
ax[2].set_title("|difference| ×20", fontsize=6, color=ns.INK2,
                loc="right", pad=2)
for b in ax:
    b.set_xticks([]); b.set_yticks([]); b.grid(False)
    # Square cells. The full frame is 16:9 and the crops are square, so without
    # this the first panel is half again as wide as the others and the grid
    # reads as a mistake. Letterboxing the frame costs nothing; it is context.
    for sp in b.spines.values():
        sp.set_linewidth(0.4); sp.set_color("#c8ccd0")
out = R / a.out
out.parent.mkdir(parents=True, exist_ok=True)
for _d in (out, R / "paper/figures" / out.name):
    _d.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(_d, dpi=500, bbox_inches="tight", pad_inches=0.02,
                facecolor="white")
print(f"  {s['name'][:26]} q{a.qp}: {saving:.2f}% saved at {db:.4f} dB")
print(f"  {bpp:.4f} bpp; PSNR released {psnr_rel:.3f}, ours {psnr_our:.3f}")
print(f"  worst tile {ti} of {nh*nw}, exit {int(k[ti])}")
print(f"  -> {out}")

if a.sidecar:
    import json
    Path(R / a.sidecar).write_text(json.dumps(
        {"figure": str(a.out), "what": "released decode and ours from one "
                                       "latent, at a stated budget",
         "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"), "ckpt_step": ck.get("step"),
         "seq": s["name"], "cls": s["cls"], "resolution": [s["w"], s["h"]],
         "qp": a.qp, "budget_db": a.budget, "delivered_db": db,
         "saving_pct_vs_release": saving, "bpp": bpp,
         "psnr_released": psnr_rel, "psnr_routed": psnr_our,
         "psnr_convention": "(6Y+U+V)/8 in 4:2:0 on 0..255",
         "crop_px": a.crop, "crop_centre_yx": [cy, cx],
         "worst_tile": ti, "worst_tile_exit": int(k[ti]),
         "n_tiles": nh * nw, "tile_px": P,
         "exit_hist": torch.bincount(k, minlength=K).tolist(),
         "amplification": 20}, indent=2))
    print(f"  -> {a.sidecar}")
