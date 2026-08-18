"""The seam artefact at pixel scale, and what each fix does to it.

Every seam number in this project is a dB average over 40 sequences. That is
the right unit for a decision and the wrong one for understanding: it says the
penalty is 0.548 dB with zeros and 0.107 with replicate, and it does not say
what the reader would actually SEE.

This crops a window straddling a tile boundary and decodes it four ways, with
the exit map held at full depth throughout -- so the ONLY difference between
the panels is how a 3x3 depthwise convolution treats the tile border. Nothing
here is about early exit; it isolates the artefact the tiling itself creates.

Error maps are against the full-frame decode of the SAME latent, so what they
show is the seam and nothing else: identical weights, identical bitstream,
identical arithmetic away from the border.

    python scripts/seam_patches.py --device cuda:0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

R = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(R))
sys.path.insert(0, str(R / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import naturestyle as ns  # noqa: E402

ns.apply()


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/BEST/ckpt_eval.pth.tar")
    ap.add_argument("--seq", default="Bosphorus")
    ap.add_argument("--qp", type=int, default=63)
    ap.add_argument("--row", type=int, default=384)
    ap.add_argument("--col", type=int, default=880)
    ap.add_argument("--win", type=int, default=192)
    ap.add_argument("--amp", type=float, default=40.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="docs/figures/seam_patches.png")
    a = ap.parse_args(argv)

    import ctc_intra as C
    from src.utils.transforms import ycbcr2rgb
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.reference import reference_for

    dev = a.device
    ck = torch.load(R / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    K, j, P = cfg.num_exits, cfg.split_depth, cfg.rgb_patch

    seqs, _ = C.discover([])
    s = ([q for q in seqs if a.seq in q["name"]] or seqs)[0]
    x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
    x = x[0:1].to(dev)
    _, _, H, W = x.shape
    ph, pw = (-H) % P, (-W) % P
    xp = F.pad(x, (0, pw, 0, ph), mode="replicate")
    nt = ((H + ph) // P) * ((W + pw) // P)
    qp = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
    deep = torch.full((nt,), K - 1, dtype=torch.long, device=dev)

    # Snap the window so a tile boundary runs through its middle -- otherwise
    # the figure is a picture of a tile interior, which has no seam in it.
    col = (a.col // P) * P - a.win // 2
    row = a.row
    saved_mode = cfg.tile_pad_mode
    variants, out = [], {}
    with torch.no_grad():
        y, q, _ = net._encode_to_latent(xp, qp)
        full = ref.dec.forward_full(y, q)
        keep_repair = net.dec.seam_repair
        for label, mode, repair in (
                ("zeros padding", "zeros", False),
                ("replicate padding", "replicate", False),
                ("replicate + grid repair", "replicate", True)):
            cfg.tile_pad_mode = mode
            net.dec.seam_repair = keep_repair if repair else None
            out[label] = net.dec(y, q, exit_map=deep)
            variants.append(label)
        net.dec.seam_repair = keep_repair
        cfg.tile_pad_mode = saved_mode

    def crop_rgb(t):
        v = ycbcr2rgb((t[:, :, row:row + a.win, col:col + a.win] + 0.5).clamp(0, 1))
        return v[0].clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    ref_c = crop_rgb(full)
    fig, ax = plt.subplots(2, len(variants) + 1, figsize=(ns.W2, 3.0))
    ax[0][0].imshow(ref_c)
    ax[0][0].set_title("Full-frame decode\n(no tiling at all)", fontsize=6,
                       color=ns.INK)
    ax[1][0].axis("off")
    ax[1][0].text(0.5, 0.5, f"|error| vs full-frame\namplified x{a.amp:.0f}",
                  ha="center", va="center", fontsize=6, color=ns.INK2,
                  transform=ax[1][0].transAxes)
    for i, label in enumerate(variants, start=1):
        c = crop_rgb(out[label])
        e = np.abs(c - ref_c).mean(-1)
        # dB over the WHOLE frame, so the caption is the population number and
        # not whatever this particular crop happens to show.
        db = 10 * torch.log10(((out[label] - xp) ** 2).mean()
                              / ((full - xp) ** 2).mean())
        ax[0][i].imshow(c)
        ax[0][i].set_title(f"{label}\n{db.item():+.3f} dB, whole frame",
                           fontsize=6, color=ns.INK)
        ax[1][i].imshow(np.clip(e * a.amp, 0, 1), cmap="inferno", vmin=0, vmax=1)
        # mark where the tile boundary falls inside the crop
        for A in (ax[0][i], ax[1][i]):
            bx = (col // P + 1) * P - col
            if 0 < bx < a.win:
                A.axvline(bx, color="#00e5ff", lw=0.6, ls=(0, (4, 3)))
    for A in ax.ravel():
        A.set_xticks([]); A.set_yticks([])
    fig.suptitle(f"{s['name'].split('_')[0]}, qp {a.qp} — every tile at FULL "
                 f"depth, so the only difference is the tile border "
                 f"(dashed line)", fontsize=6.5, color=ns.INK2, x=0.01,
                 ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    o = R / a.out
    o.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(o, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  window {a.win}px at row {row}, col {col}; tile pitch {P}")
    for label in variants:
        db = 10 * torch.log10(((out[label] - xp) ** 2).mean()
                              / ((full - xp) ** 2).mean())
        print(f"    {label:<26} {db.item():+.4f} dB")
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
