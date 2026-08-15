"""Decode one image at every exit and write a side-by-side comparison.

Numbers say how much quality an exit costs; they do not say what that cost looks
like. A 0.8 dB drop can be an invisible softening or a visible seam grid, and
the two call for different fixes — so this renders the reconstructions and the
error maps rather than only tabulating them.

Each panel is the SAME latent decoded to a different depth through the deployed
patched path, so what the panels differ by is exactly what routing chooses
between. The error map is amplified because at these dB levels the residual is
invisible at native scale.

    python scripts/sample_exits.py --ckpt runs/warmstart/ckpt_warmstart.pth.tar \
        --qp 63 --n 2 --device 7
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

DCVC_ROOT = Path.home() / "DCVC"
sys.path.insert(0, str(DCVC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.image_dataset import ImageFolder  # noqa: E402
from src.utils.common import get_training_lambdas  # noqa: E402
from src.utils.transforms import ycbcr2rgb  # noqa: E402

from flexuf.config import QP_LEVELS, FlexUFConfig  # noqa: E402
from flexuf.cost import saving  # noqa: E402
from flexuf.model import FlexUFIntra  # noqa: E402


def to_rgb8(x: torch.Tensor) -> np.ndarray:
    """Codec-space YCbCr (shifted by -0.5) back to displayable RGB."""
    rgb = ycbcr2rgb(x + 0.5, clamp=True)
    return (rgb[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/data10/shareddata/openimages/dcvc_train")
    ap.add_argument("--qp", type=int, default=63)
    ap.add_argument("--n", type=int, default=2, help="how many images")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--device", default="0")
    ap.add_argument("--outdir", default="results/samples")
    a = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.device)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval()
    net.load_state_dict(ck.get("state_dict", ck.get("net")), strict=False)

    ds = ImageFolder(a.dataset, a.crop, a.crop, QP_LEVELS,
                     get_training_lambdas([10.0, 2048.0], QP_LEVELS))
    v = Path(a.dataset) / "description_val.json"
    if v.exists():
        ds.dataset = json.loads(v.read_text())
        ds.dataset_length = len(ds.dataset)

    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    exits = list(range(cfg.split_depth, cfg.num_exits))
    nt = (a.crop // cfg.rgb_patch) ** 2

    for idx in range(a.n):
        x = ds[idx * 7][0][None].to(dev)
        qp = torch.tensor([a.qp], dtype=torch.int32, device=dev)

        y_hat, q_dec, _ = net._encode_to_latent(x, qp)
        full = net.dec.forward_full(y_hat, q_dec)
        ref_mse = net.get_mse(x, full).mean().item()

        panels = [("orijinal", to_rgb8(x), None, None),
                  ("tam decode (UF)", to_rgb8(full), 0.0, 0.0)]
        for k in exits:
            em = torch.full((nt,), k, device=dev)
            r = net.forward_routed(x, qp, em)
            db = 10 * np.log10(max(r["mse"].mean().item(), 1e-12) / max(ref_mse, 1e-12))
            panels.append((f"çıkış {k}", to_rgb8(r["x_hat"]), db,
                           100 * saving(em, cfg, "head")))

        cols = len(panels)
        fig, ax = plt.subplots(2, cols, figsize=(3.1 * cols, 6.6))
        ref_img = panels[1][1].astype(np.int16)
        for c, (name, img, db, sv) in enumerate(panels):
            ax[0, c].imshow(img); ax[0, c].axis("off")
            t = name if db is None else f"{name}\n{db:+.3f} dB · {sv:.0f}% tasarruf"
            ax[0, c].set_title(t, fontsize=9)
            if c == 0:
                ax[1, c].axis("off")
                ax[1, c].text(0.5, 0.5, "hata haritası\n(tam decode'a göre, 12x)",
                              ha="center", va="center", fontsize=9)
            else:
                err = np.abs(img.astype(np.int16) - ref_img).mean(2)
                ax[1, c].imshow(np.clip(err * 12, 0, 255), cmap="inferno", vmin=0, vmax=255)
                ax[1, c].axis("off")
                ax[1, c].set_title(f"ort |hata| {err.mean():.2f}", fontsize=8)
        fig.suptitle(
            f"DCVC-UF çıkışları — qp {a.qp}, {cfg.rgb_patch}px tile, j={cfg.split_depth}"
            f"   ({Path(a.ckpt).name})", fontsize=11)
        fig.tight_layout()
        p = out / f"exits_qp{a.qp}_{idx}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {p}")

        for name, _, db, sv in panels[1:]:
            print(f"    {name:<18} {db:+.3f} dB   {sv:.1f}% tasarruf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
