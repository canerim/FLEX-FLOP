"""Everything the first trained checkpoint says, on one page.

Includes the pictures, not only the curves: the exit map and the error map are
the only place you can SEE whether routing is picking up structure or scattering,
and whether the seam is present. A saving number cannot show either.

Every panel says ORACLE where it means a perfect router, because the gap between
that and a real one is the project's open question rather than a detail.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
import ctc_intra as C
from flexuf.config import FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.model import FlexUFIntra, load_flexuf_state

CK = sys.argv[1] if len(sys.argv) > 1 else "runs/heads_only_j2_p256/ckpt_epo0.pth.tar"
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:4"
S1, S2, S3, S4, S5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
SURF, INK, INK2, INK3, GRIDC = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8985", "#e6e5e2"

def style(a, title, sub=None):
    a.set_facecolor(SURF)
    a.set_title(title, color=INK, fontsize=11, loc="left", pad=15 if sub else 7)
    if sub:
        a.text(0, 1.012, sub, transform=a.transAxes, color=INK3, fontsize=8, va="bottom")
    a.grid(True, color=GRIDC, lw=0.7, zorder=0); a.set_axisbelow(True)
    a.tick_params(colors=INK2, labelsize=8.5, length=0)
    for k, sp in a.spines.items():
        sp.set_visible(k == "bottom"); sp.set_color(GRIDC)

bt = json.loads(Path("results/budget_table.json").read_text())["rows"]
ck = torch.load(CK, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(DEV).eval(); load_flexuf_state(net, ck)
cost = exit_costs(cfg, "head").to(DEV)

seqs, _ = C.discover([])
s = seqs[0]
x, pl = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
x = x[:1].to(DEV); _, _, H, W = x.shape
P = cfg.rgb_patch
ph, pw = (-H) % P, (-W) % P
xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
QP = 32
with torch.no_grad():
    qp = torch.full((1,), QP, dtype=torch.int32, device=DEV)
    y, q, _ = net._encode_to_latent(xp, qp)
    outs = net.dec.forward_all_exits(y, q)
    nh, nw = (H + ph) // P, (W + pw) // P
    per = []
    for xh in outs:
        e = ((xh - xp) ** 2).mean(1)
        per.append(e.view(1, nh, P, nw, P).permute(0, 1, 3, 2, 4)
                    .reshape(nh * nw, P * P).mean(1))
    M = torch.stack(per, 1)
    tau = 10 ** (0.3 / 10)                      # the 0.3 dB budget, per tile
    ok = M <= M[:, -1:] * tau; ok[:, -1] = True
    k = ok.float().argmax(1)
    dense = net.dec.forward_full(y, q)
    routed = net.dec(y, q, exit_map=k)
    sv = 100 * (1 - cost[k].mean().item() / cost[-1].item())
    d_db = C.psnr_611_420(dense[:, :, :H, :W], pl[0])
    r_db = C.psnr_611_420(routed[:, :, :H, :W], pl[0])

fig = plt.figure(figsize=(16.5, 9.4), facecolor=SURF)
gs = fig.add_gridspec(2, 3, hspace=0.34, wspace=0.22,
                      left=0.05, right=0.985, top=0.87, bottom=0.06)
fig.suptitle("FLEX-UF — ilk egitilmis checkpoint", fontsize=16, color=INK,
             x=0.05, ha="left", y=0.965)
fig.text(0.05, 0.925, f"{Path(CK).parent.name} / {Path(CK).name} · govde donuk, "
         f"sadece exit head'leri egitildi · {cfg.rgb_patch}px tile, j={cfg.split_depth}",
         color=INK2, fontsize=9.5)

# 1 saving vs budget
a = fig.add_subplot(gs[0, 0])
for qpv, c in zip((0, 16, 32, 48, 63), (S1, S3, S4, S2, S5)):
    r = sorted([z for z in bt if z["qp"] == qpv], key=lambda z: z["budget"])
    a.plot([z["achieved_db"] for z in r], [z["saving_pct"] for z in r],
           color=c, lw=2, marker="o", ms=4.5, markeredgecolor=SURF,
           markeredgewidth=1.2, label=f"qp {qpv}", zorder=3)
for b, lab in ((0.1, "0.1 dB"), (0.3, "0.3 dB")):
    a.axvline(b, color=INK3, lw=1.3, ls=(0, (4, 3)), zorder=2)
    a.text(b + 0.012, 1.5, lab, color=INK3, fontsize=8, rotation=90, va="bottom")
a.set_xlabel("kaybedilen PSNR (dB)", color=INK2, fontsize=9)
a.set_ylabel("tasarruf (%)", color=INK2, fontsize=9)
a.set_xlim(0, 1.05); a.set_ylim(0, 46)
a.legend(fontsize=8, frameon=False, loc="lower right")
style(a, "1 · Butce vs tasarruf", "ORACLE — kusursuz router. Ust sinir.")

# 2 the 0.3 dB row
a = fig.add_subplot(gs[0, 1])
r3 = sorted([z for z in bt if abs(z["budget"] - 0.3) < 1e-9], key=lambda z: z["qp"])
xs = range(len(r3))
a.bar(list(xs), [z["saving_pct"] for z in r3], 0.55, color=S3, zorder=3)
for i, z in enumerate(r3):
    a.text(i, z["saving_pct"] + 0.8, f"{z['saving_pct']:.1f}%", ha="center",
           color=INK2, fontsize=8.5)
r1 = sorted([z for z in bt if abs(z["budget"] - 0.1) < 1e-9], key=lambda z: z["qp"])
a.bar(list(xs), [z["saving_pct"] for z in r1], 0.26, color=S1, zorder=4)
a.set_xticks(list(xs)); a.set_xticklabels([f"qp{z['qp']}" for z in r3])
a.set_ylabel("tasarruf (%)", color=INK2, fontsize=9); a.set_ylim(0, 46)
a.legend(handles=[plt.Rectangle((0,0),1,1,color=S3), plt.Rectangle((0,0),1,1,color=S1)],
         labels=["≤0.3 dB", "≤0.1 dB"], fontsize=8.5, frameon=False)
style(a, "2 · Iki butcede tasarruf", "0.1 -> 0.3 dB gecisi tasarrufu ~1.7x artiriyor.")

# 3 exit histogram
a = fig.add_subplot(gs[0, 2])
h = torch.bincount(k, minlength=cfg.num_exits).cpu().numpy()
cols = [INK3] * cfg.split_depth + [S1, S3, S4, S2][: cfg.num_exits - cfg.split_depth]
a.bar(range(cfg.num_exits), h, 0.6, color=cols, zorder=3)
for i, v in enumerate(h):
    if v: a.text(i, v + 0.4, str(v), ha="center", color=INK2, fontsize=8.5)
a.set_xticks(range(cfg.num_exits))
a.set_xticklabels([f"c{i}" + ("\n(yasak)" if i < cfg.split_depth else "") for i in range(cfg.num_exits)])
a.set_ylabel("tile sayisi", color=INK2, fontsize=9)
style(a, f"3 · 0.3 dB'de cikis dagilimi", f"{s['name'][:28]}, qp{QP} — {int(h.sum())} tile")

# 4-6 pictures
def show(ax, img, title, sub=None, cmap=None, vmax=None):
    ax.imshow(img, cmap=cmap, vmin=0 if cmap else None, vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(GRIDC)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=15 if sub else 7)
    if sub: ax.text(0, 1.012, sub, transform=ax.transAxes, color=INK3, fontsize=8, va="bottom")

def rgb(t):
    """YCbCr -> RGB before display.

    The codec works in shifted YCbCr; showing those three planes as if they were
    R, G, B produced a magenta-and-green picture that looked like a bug in the
    decoder rather than in the plotting. Uses DCVC's own conversion so the
    picture matches what its test harness would write out.
    """
    from src.utils.transforms import ycbcr2rgb
    v = ycbcr2rgb((t[:, :, :H, :W] + 0.5).clamp(0, 1))
    return v[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

a = fig.add_subplot(gs[1, 0])
show(a, rgb(dense), "4 · Stok DCVC-UF (tam decode)", f"6:1:1 PSNR {d_db:.2f} dB")
a = fig.add_subplot(gs[1, 1])
show(a, rgb(routed), "5 · Yonlendirilmis decode",
     f"PSNR {r_db:.2f} dB  ({r_db-d_db:+.3f})   tasarruf %{sv:.1f}")

a = fig.add_subplot(gs[1, 2])
emap = k.view(nh, nw).cpu().numpy()
cmap = ListedColormap([S1, S3, S4, S2])
im = a.imshow(emap, cmap=cmap, vmin=cfg.split_depth - 0.5, vmax=cfg.num_exits - 0.5,
              interpolation="nearest")
a.set_xticks([]); a.set_yticks([])
for sp in a.spines.values(): sp.set_color(GRIDC)
a.set_title("6 · Hangi tile hangi cikisa gitti", color=INK, fontsize=11, loc="left", pad=15)
a.text(0, 1.012, "Yapisiz olsaydi rastgele gurultu gorurduk.", transform=a.transAxes,
       color=INK3, fontsize=8, va="bottom")
cb = fig.colorbar(im, ax=a, fraction=0.046, ticks=range(cfg.split_depth, cfg.num_exits))
cb.ax.set_yticklabels([f"c{i}" for i in range(cfg.split_depth, cfg.num_exits)])
cb.ax.tick_params(colors=INK2, labelsize=8, length=0)
cb.outline.set_edgecolor(GRIDC)

fig.savefig("results/checkpoint_report.png", dpi=130, facecolor=SURF)
print("wrote results/checkpoint_report.png")
print(f"  {s['name']}  qp{QP}: dense {d_db:.2f} -> routed {r_db:.2f} dB "
      f"({r_db-d_db:+.3f}), saving {sv:.1f}%")
