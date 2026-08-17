"""Figures the slides need that no earlier script produced.

Two of them, both showing rather than telling:

  exits.png     one CTC crop decoded at each exit, with its error map against
                the released decoder underneath. The point of the ladder is that
                shallow exits are cheap and worse; a reader should see how much
                worse before being shown a percentage.
  architecture.png  where the split sits and what each part costs, drawn to the
                measured MAC shares rather than sketched.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C
from src.utils.transforms import ycbcr2rgb
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.model import FlexUFIntra, load_flexuf_state

SUR, INK, INK2, INK3 = "#ffffff", "#0b0b0b", "#52514e", "#8a8985"
TUM = "#0065BD"
dev = "cuda:0"   # index within CUDA_VISIBLE_DEVICES, not the physical card

ck = torch.load("runs/CONTROL/ckpt_epo0.pth.tar", map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
ref = FlexUFIntra(cfg).to(dev).eval()
load_flexuf_state(ref, torch.load("runs/warmstart/ckpt_warmstart.pth.tar",
                                  map_location="cpu", weights_only=False))
cost = exit_costs(cfg, "head").to(dev)

seqs, _ = C.discover([])
s = [q for q in seqs if "Bosphorus" in q["name"]] or seqs
s = s[0]
x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
x = x[0:1].to(dev)
_, _, H, W = x.shape; P = cfg.rgb_patch
ph, pw = (-H) % P, (-W) % P
xp = F.pad(x, (0, pw, 0, ph), mode="replicate")
qp = torch.full((1,), 32, dtype=torch.int32, device=dev)

with torch.no_grad():
    y, q, _ = net._encode_to_latent(xp, qp)
    outs = net.dec.forward_all_exits(y, q)
    full = ref.dec.forward_full(y, q)

def rgb(t, r, c, n=384):
    """A crop, converted to RGB for display.

    The model works in shifted YCbCr, so plotting its three channels directly as
    red/green/blue produces a picture that is wrong in a way that looks almost
    plausible -- pink sky, green water. The conversion is DCVC's own.
    """
    v = ycbcr2rgb((t[:, :, r:r+n, c:c+n] + 0.5).clamp(0, 1))
    return v[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

R0, C0 = 380, 900
ks = [2, 3, 4, 5]
fig, ax = plt.subplots(2, len(ks) + 1, figsize=(15, 6.4), facecolor=SUR)
ref_c = rgb(full, R0, C0)
ax[0][0].imshow(ref_c); ax[0][0].set_title("yayinlanmis DCVC-UF", color=TUM, fontsize=11)
ax[1][0].axis("off")
ax[1][0].text(0.5, 0.5, "hata haritalari\n(x25 guclendirilmis)", ha="center", va="center",
              color=INK2, fontsize=10, transform=ax[1][0].transAxes)
for j, k in enumerate(ks, start=1):
    c = rgb(outs[k], R0, C0)
    e = np.abs(c - ref_c).mean(-1)
    ax[0][j].imshow(c)
    sv = 100 * (1 - cost[k] / cost[-1]).item()
    d = 10 * torch.log10(((outs[k] - xp) ** 2).mean() / ((full - xp) ** 2).mean())
    ax[0][j].set_title(f"cikis {k}\n%{sv:.0f} tasarruf, {d.item():+.3f} dB",
                       color=INK, fontsize=10)
    ax[1][j].imshow(np.clip(e * 25, 0, 1), cmap="inferno", vmin=0, vmax=1)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
for a in ax[1][1:]:
    a.set_xticks([]); a.set_yticks([])
fig.suptitle("Bosphorus 1080p, qp32 — ayni bitstream, farkli cikislar",
             color=INK, fontsize=12.5, x=0.02, ha="left")
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig("results/fig_exits.png", dpi=140, facecolor=SUR)
print("wrote results/fig_exits.png")

# ---- architecture, drawn to measured MAC shares -------------------------
fig, a = plt.subplots(figsize=(13.5, 3.5), facecolor=SUR)
a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis("off")
UP, TR, HD = 0.0816, 0.8944, 0.0240
x0 = 0.04; w_up = 0.10
a.add_patch(Rectangle((x0, .42), w_up, .28, fc="#cfe3f5", ec=TUM, lw=1.4))
a.text(x0 + w_up/2, .56, "upsample", ha="center", va="center", fontsize=9)
a.text(x0 + w_up/2, .36, f"{100*UP:.1f}%", ha="center", fontsize=8.5, color=INK3)
bw = 0.052; gap = .004; xb = x0 + w_up + .03
for b in range(12):
    shared = b < 4
    a.add_patch(Rectangle((xb + b*(bw+gap), .42), bw, .28,
                          fc="#cfe3f5" if shared else "#ffe6cc",
                          ec=TUM if shared else "#d97b28", lw=1.3))
    a.text(xb + b*(bw+gap) + bw/2, .56, str(b), ha="center", va="center", fontsize=7.5)
    if b in (5, 7, 9, 11):
        a.annotate("", xy=(xb + b*(bw+gap) + bw/2, .30), xytext=(xb + b*(bw+gap) + bw/2, .42),
                   arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.6))
        k = {5: 2, 7: 3, 9: 4, 11: 5}[b]
        a.text(xb + b*(bw+gap) + bw/2, .245, f"cikis {k}", ha="center", fontsize=8, color="#c0392b")
        a.text(xb + b*(bw+gap) + bw/2, .17,
               f"%{100*(1-cost[k]/cost[-1]).item():.0f}", ha="center", fontsize=8, color=INK3)
xh = xb + 12*(bw+gap) + .02
a.add_patch(Rectangle((xh, .42), .07, .28, fc="#cfe3f5", ec=TUM, lw=1.4))
a.text(xh + .035, .56, "head", ha="center", va="center", fontsize=9)
a.text(xh + .035, .36, f"{100*HD:.1f}%", ha="center", fontsize=8.5, color=INK3)
a.text(x0, .84, "tam kare (dikis yok)", color=TUM, fontsize=9.5)
a.text(xb + 4*(bw+gap), .84, "tile basina — atlanabilen kisim, decode'un %89.4'u",
       color="#d97b28", fontsize=9.5)
a.plot([xb + 4*(bw+gap) - gap/2]*2, [.38, .78], color=INK3, ls="--", lw=1.2)
a.text(xb + 4*(bw+gap) - gap/2, .05, "j = 2 bolunme noktasi", ha="center",
       fontsize=8.5, color=INK3)
fig.tight_layout()
fig.savefig("results/fig_arch.png", dpi=140, facecolor=SUR)
print("wrote results/fig_arch.png")
