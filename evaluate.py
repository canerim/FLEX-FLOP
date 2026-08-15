"""Measure the frontier: decoder compute saved versus PSNR lost.

This produces the only two numbers the project is ultimately judged on, so the
protocol is spelled out rather than left implicit.

Protocol
--------
1. **Matched anchor.** Every routed decode is compared against the *full* decode
   of the same crop at the same QP, not against a published number or a
   different frame set. FLEX-FLOP's note applies: numbers from different frame
   lists are never compared.

2. **Bit-exact control first.** Before any frontier point is reported, the
   deepest exit is checked against stock UF. If `max|Δ| != 0` the run is not a
   re-expression of UF and everything downstream is meaningless.

3. **Reference curves before routing.** "Every tile at exit k" is measured for
   each k. This is the curve a router has to beat — if routing does not sit
   above the uniform-depth curve, the router is adding nothing and the honest
   conclusion is that a plain shallower decoder would do.

4. **Saving is charged honestly.** `flexuf.cost.saving` amortises the shared
   stem, charges each early exit for its adapter, and charges the halo where it
   is actually carried. A saving number that ignores any of those is not real.

5. **Rate is reported but is not a variable.** Entropy decoding happens once,
   full-frame, so bpp is identical across exits by construction. It is logged to
   prove that, not because it moves.

Usage
-----
    python evaluate.py --ckpt runs/e1_j2_p128/ckpt.pth.tar \
        --dataset /data10/.../dcvc_train --router runs/.../router.pth.tar \
        --n_frames 192 --qps 30 42 48 54 63
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.image_dataset import ImageFolder  # noqa: E402
from src.models.image_model import IntraDecoder  # noqa: E402
from src.utils.common import get_training_lambdas  # noqa: E402

from flexuf.config import QP_LEVELS, FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs, saving  # noqa: E402
from flexuf.losses import psnr_from_mse  # noqa: E402
from flexuf.backbone.warmstart import remap_ladder_to_stock  # noqa: E402
from flexuf.model import FlexUFIntra  # noqa: E402
from flexuf.router.router import ExitRouter, latent_tiles_with_halo, tile_signals  # noqa: E402


@torch.no_grad()
def control_bit_exact(net, cfg, device) -> float:
    """The deepest exit must equal a stock UF decoder holding the same weights.

    Returns max|Δ|, which must be exactly 0.0.

    The ladder and stock `IntraDecoder` use disjoint key names, so the ladder's
    tensors are mapped back through `remap_ladder_to_stock` before loading. Doing
    it with a bare `strict=False` would match nothing, leave the stock model at
    random init, and turn this control into theatre.
    """
    stock = IntraDecoder().to(device).eval()
    ladder_state = {
        k[len("dec.") :]: v for k, v in net.state_dict().items() if k.startswith("dec.")
    }
    mapped = remap_ladder_to_stock(ladder_state, cfg.blocks_per_exit)
    missing, unexpected = stock.load_state_dict(mapped, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"control remap incomplete: {len(missing)} missing, "
            f"{len(unexpected)} unexpected — e.g. {list(missing)[:3]} {list(unexpected)[:3]}"
        )
    y = torch.randn(1, 256, 32, 32, device=device)
    q = torch.rand(1, 384, 1, 1, device=device) + 0.5
    return (stock(y, q) - net.dec.forward_full(y, q)).abs().max().item()


@torch.no_grad()
def evaluate(net, router, loader, cfg, device, qp_list):
    results = {}
    K = cfg.num_exits
    costs = exit_costs(cfg, "head")

    for qp_val in qp_list:
        acc_uniform = torch.zeros(K, dtype=torch.float64)
        acc_routed_mse = 0.0
        acc_routed_saving = 0.0
        acc_bpp = 0.0
        n = 0
        share = torch.zeros(K, dtype=torch.long)

        for batch in loader:
            x = batch[0].to(device)
            B = x.shape[0]
            qp = torch.full((B,), qp_val, dtype=torch.int32, device=device)

            out = net.forward_all_exits(x, qp)
            for k in range(K):
                acc_uniform[k] += out["mses"][k].double().sum().cpu()
            acc_bpp += out["bpp"].double().sum().cpu().item()

            if router is not None:
                y_hat, _ = net.latent_of(x, qp)
                tiles = latent_tiles_with_halo(y_hat, cfg)
                nt = tiles.shape[0] // B
                qp_t = qp.long().repeat_interleave(nt)
                em = router.assign(tile_signals(tiles), qp_t)
                share += torch.bincount(em, minlength=K).cpu()

                # decode each frame with its own tile map
                for b in range(B):
                    r = net.forward_routed(
                        x[b : b + 1], qp[b : b + 1], em[b * nt : (b + 1) * nt]
                    )
                    acc_routed_mse += r["mse"].double().sum().cpu().item()
                    acc_routed_saving += saving(em[b * nt : (b + 1) * nt], cfg, "head")
            n += B

        rec = {
            "qp": qp_val,
            "n_frames": n,
            "bpp": acc_bpp / n,
            "uniform": [
                {
                    "exit": k,
                    "psnr": psnr_from_mse(acc_uniform[k] / n).item(),
                    "saving_pct": round(100 * (1 - costs[k].item()), 2),
                }
                for k in range(K)
            ],
        }
        if router is not None:
            p_routed = psnr_from_mse(torch.tensor(acc_routed_mse / n)).item()
            p_full = psnr_from_mse(acc_uniform[K - 1] / n).item()
            rec["routed"] = {
                "psnr": p_routed,
                "psnr_loss_dB": round(p_full - p_routed, 4),
                "saving_pct": round(100 * acc_routed_saving / n, 2),
                "exit_share": share.tolist(),
            }
        results[qp_val] = rec
        print(json.dumps(rec, indent=2), flush=True)
    return results


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--router", default=None)
    ap.add_argument("--n_frames", type=int, default=192)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--qps", type=int, nargs="+", default=[30, 42, 48, 54, 63])
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.device)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(device).eval()
    net.load_state_dict(ck.get("state_dict", ck.get("net")))

    err = control_bit_exact(net, cfg, device)
    print(f"CONTROL deepest exit vs stock UF: max|diff| = {err}", flush=True)

    router = None
    if a.router:
        rk = torch.load(a.router, map_location="cpu", weights_only=False)
        router = ExitRouter(cfg.num_exits).to(device).eval()
        router.load_state_dict(rk["router"])

    ds = ImageFolder(a.dataset, a.crop, a.crop, QP_LEVELS,
                     get_training_lambdas([10.0, 2048.0], QP_LEVELS))
    ds.dataset = ds.dataset[: a.n_frames]
    ds.dataset_length = len(ds.dataset)
    loader = DataLoader(ds, batch_size=a.batch_size, num_workers=4,
                        sampler=SequentialSampler(ds), drop_last=False)

    res = evaluate(net, router, loader, cfg, device, a.qps)
    out = a.out or str(Path(a.ckpt).parent / "eval.json")
    Path(out).write_text(json.dumps(
        {"config": cfg.__dict__, "control_max_diff": err, "results": res}, indent=2))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
