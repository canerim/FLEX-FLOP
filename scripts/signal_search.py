"""Which cheap per-tile quantity actually predicts how much decode a tile needs?

The oracle diagnostic established that the headroom is real (routing beats any
uniform depth by up to +13.6pp at equal quality) but that the router's four
hand-made signals correlate with the oracle at |r| < 0.1. A router cannot exploit
what its inputs cannot see, so before tuning the router, find better inputs.

The candidates, and why each might work
---------------------------------------
The four current ones are statistics of the latent tile: a Gaussian rate
surrogate, channel sparsity, gradient energy, spatial variance. They are guesses
about what makes content hard.

The new ones come from the codec's own machinery rather than from guessing:

  scales_hat  — the entropy model's predicted Gaussian scale per latent element.
                This is the codec's OWN estimate of how unpredictable each
                element is, it is already computed during entropy decoding, and
                it costs nothing to read. If anything in the pipeline knows which
                tiles are hard, it is this.
  bits        — the implied code length, log2(scale) summed over the tile. The
                rate a tile actually consumed is a direct measure of how much
                information it carries.
  y_energy    — magnitude of the latent itself.

Correlation is computed against the ORACLE's exit choice, which is the quantity
the router must reproduce. A signal that cannot rank tiles the way the oracle
does is useless however principled it looks.

    python scripts/signal_search.py --ckpt runs/e1_j2_p128/status_latest.pth.tar --device 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

DCVC_ROOT = Path.home() / "DCVC"
sys.path.insert(0, str(DCVC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.image_dataset import ImageFolder  # noqa: E402
from src.utils.common import get_training_lambdas  # noqa: E402

from flexuf.config import QP_LEVELS, FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.router.router import latent_tiles_with_halo, tile_signals  # noqa: E402


def tile_reduce(x: torch.Tensor, p: int, how: str = "mean") -> torch.Tensor:
    """[B,C,H,W] -> [B*nh*nw] by reducing over channels and each pxp tile."""
    B, C, H, W = x.shape
    nh, nw = H // p, W // p
    t = (x.mean(1)
         .view(B, nh, p, nw, p)
         .permute(0, 1, 3, 2, 4)
         .reshape(B * nh * nw, p * p))
    return t.mean(1) if how == "mean" else t.amax(1)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/data10/shareddata/openimages/dcvc_train")
    ap.add_argument("--qp", type=int, default=63)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.device)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(device).eval()
    load_flexuf_state(net, ck)

    ds = ImageFolder(a.dataset, a.crop, a.crop, QP_LEVELS,
                     get_training_lambdas([10.0, 2048.0], QP_LEVELS))
    val = Path(a.dataset) / "description_val.json"
    if val.exists():
        ds.dataset = json.loads(val.read_text())
        ds.dataset_length = len(ds.dataset)
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=4,
                        sampler=SequentialSampler(ds))

    feats: dict[str, list[torch.Tensor]] = {}
    mses = []
    p = cfg.latent_patch

    for i, batch in enumerate(loader):
        if i >= a.batches:
            break
        x = batch[0].to(device)
        B = x.shape[0]
        qp = torch.full((B,), a.qp, dtype=torch.int32, device=device)

        # per-tile MSE at each exit
        out = net.forward_all_exits(x, qp)
        rgb_p = cfg.rgb_patch
        H, W = x.shape[-2:]
        nh, nw = H // rgb_p, W // rgb_p
        per_exit = []
        for x_hat in out["x_hats"]:
            err = (x_hat - x) ** 2
            per_exit.append(err.mean(1)
                            .view(B, nh, rgb_p, nw, rgb_p)
                            .permute(0, 1, 3, 2, 4)
                            .reshape(B * nh * nw, rgb_p * rgb_p).mean(1))
        mses.append(torch.stack(per_exit, dim=1).cpu())

        # the codec's own internals
        y_hat, _, aux = net._encode_to_latent(x, qp)
        scales = aux["scales_hat"]
        # scales_hat is [B, 2*C, h, w] in some configs; take the first C if so
        if scales.shape[1] != y_hat.shape[1]:
            scales = scales[:, : y_hat.shape[1]]
        bits = torch.log2(scales.clamp_min(1e-6) * (2 * torch.pi * torch.e) ** 0.5)

        # Features after the SHARED STEM.
        #
        # At j=2 the opening upsample and groups 0..1 run full-frame for every
        # tile regardless of where it exits, so their output is already computed
        # and reading it costs nothing. It is also much closer to what the oracle
        # measures than the raw latent is: the oracle is about how the DECODER
        # behaves on this tile, and this is the decoder's own intermediate state.
        stem = net.dec.upsample(y_hat)
        for g in range(cfg.split_depth):
            stem = net.dec.groups[g](stem)
        # stem is at 2x the latent grid, so tiles are 2p on a side there
        sp = p * 2

        cand = {
            "stem_energy": tile_reduce(stem.abs(), sp),
            "stem_std": tile_reduce((stem - stem.mean(1, keepdim=True)).abs(), sp),
            "stem_max": tile_reduce(stem.abs(), sp, "max"),
            "scales_mean": tile_reduce(scales, p),
            "scales_max": tile_reduce(scales, p, "max"),
            "bits": tile_reduce(bits, p),
            "y_energy": tile_reduce(y_hat.abs(), p),
        }
        # the four current hand-made signals
        sig = tile_signals(latent_tiles_with_halo(y_hat, cfg))
        for j, nm in enumerate(["s1_rate_surrogate", "s2_sparsity",
                                "s3_gradient", "s4_spatial_var"]):
            cand[nm] = sig[:, j]
        for k, v in cand.items():
            feats.setdefault(k, []).append(v.float().cpu())

    mses = torch.cat(mses)
    feats = {k: torch.cat(v) for k, v in feats.items()}

    db = 10 * torch.log10(mses / mses[:, -1:].clamp_min(1e-12))
    ok = db <= a.tau
    ok[:, -1] = True
    choice = ok.float().argmax(dim=1).float()
    costs = exit_costs(cfg, "head")

    print(f"\ncheckpoint {a.ckpt}")
    print(f"tiles {mses.shape[0]}   qp {a.qp}   tau {a.tau} dB   "
          f"oracle saving {100*(1-costs[choice.long()].mean().item()):.1f}%")
    print(f"oracle choice: mean {choice.mean():.2f}  std {choice.std():.3f}\n")

    if choice.std() < 1e-9:
        print("oracle is constant here — nothing to correlate against")
        return 1

    print(f"  {'signal':<20} {'Pearson r':>10} {'|r|':>7}   {'Spearman':>9}")
    rows = []
    for k, v in feats.items():
        if v.std() < 1e-9:
            rows.append((0.0, k, float("nan"), float("nan")))
            continue
        r = torch.corrcoef(torch.stack([v, choice]))[0, 1].item()
        # rank correlation: robust to the monotone-but-nonlinear case
        rv = v.argsort().argsort().float()
        rc = choice.argsort().argsort().float()
        rho = torch.corrcoef(torch.stack([rv, rc]))[0, 1].item()
        rows.append((abs(r), k, r, rho))
    for _, k, r, rho in sorted(rows, reverse=True):
        print(f"  {k:<20} {r:>+10.3f} {abs(r):>7.3f}   {rho:>+9.3f}")

    best = max(rows)[1]
    print(f"\nbest single signal: {best}")
    print("A signal the router cannot rank tiles by is useless however")
    print("principled it looks — this is the number that decides which inputs")
    print("the Class-Module gets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
