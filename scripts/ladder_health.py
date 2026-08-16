"""Is the exit ladder monotonic THROUGH THE DEPLOYED PATH?

The single property everything else rests on. If a deeper exit is not reliably
better than a shallower one, no router can help: routing is the act of spending
more compute where it buys more quality, and that trade does not exist if the
ordering is broken.

This is measured through `forward_routed` — the real patched decode — not through
`forward_all_exits`. The distinction matters and has already misled this project
once: the full-frame path has no seams, so it can look perfectly ordered while
the deployed path is not. Measured on e1 at epoch 2:

    exit 2   43.79% saved   0.1787 dB
    exit 3   28.88% saved   0.2137 dB   <- WORSE than exit 2, while costing more
    exit 4   13.98% saved   0.1006 dB
    exit 5    0.00% saved   0.0432 dB

Exit 3 costs 15 points more compute than exit 2 and returns worse quality. A
router faced with that has nothing to exploit, and the honest reading is not
"the router is bad" but "the ladder is not ready".

    python scripts/ladder_health.py --ckpt runs/e1_j2_p128/status_latest.pth.tar --device 4
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.image_dataset import ImageFolder  # noqa: E402
from src.utils.common import get_training_lambdas  # noqa: E402

from flexuf.config import QP_LEVELS, FlexUFConfig  # noqa: E402
from flexuf.cost import saving  # noqa: E402
from flexuf.losses import psnr_from_mse  # noqa: E402
from flexuf.model import DeterministicCrop, FlexUFIntra, load_flexuf_state  # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/data10/shareddata/openimages/dcvc_train")
    ap.add_argument("--qp", type=int, default=63)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", a.device)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)

    ds = ImageFolder(a.dataset, a.crop, a.crop, QP_LEVELS,
                     get_training_lambdas([10.0, 2048.0], QP_LEVELS))
    v = Path(a.dataset) / "description_val.json"
    if v.exists():
        ds.dataset = json.loads(v.read_text())[: a.frames]
        ds.dataset_length = len(ds.dataset)
    # Deterministic: measurement must not depend on which crop was drawn.
    ds = DeterministicCrop(ds)
    ld = DataLoader(ds, batch_size=2, num_workers=4, sampler=SequentialSampler(ds))
    nt = (a.crop // cfg.rgb_patch) ** 2

    res = {}
    for k in list(range(cfg.split_depth, cfg.num_exits)) + ["full"]:
        tot, n, sv = 0.0, 0, 0.0
        for b in ld:
            x = b[0].to(dev); B = x.shape[0]
            qp = torch.full((B,), a.qp, dtype=torch.int32, device=dev)
            if k == "full":
                tot += net.forward_all_exits(x, qp)["mses"][-1].sum().item(); n += B
            else:
                for i in range(B):
                    em = torch.full((nt,), k, device=dev)
                    tot += net.forward_routed(x[i:i+1], qp[i:i+1], em)["mse"].sum().item()
                    sv += saving(em, cfg, "head"); n += 1
        res[k] = (tot / n, sv / max(n, 1) if k != "full" else 0.0)

    pf = psnr_from_mse(torch.tensor(res["full"][0])).item()
    print(f"\n{a.ckpt}   qp {a.qp}   {a.frames} held-out frames\n")
    print(f"  {'exit':>5} {'saving':>9} {'PSNR':>9} {'loss':>11}")
    losses = []
    for k in range(cfg.split_depth, cfg.num_exits):
        m, sv = res[k]
        p = psnr_from_mse(torch.tensor(m)).item()
        losses.append((k, pf - p, 100 * sv))
        print(f"  {k:>5} {100*sv:>8.2f}% {p:>8.2f} {pf-p:>+10.4f} dB")
    print(f"  {'full':>5} {0.0:>8.2f}% {pf:>8.2f} {0.0:>+10.4f} dB")

    # deeper must not be worse
    bad = [(k1, k2) for (k1, l1, _), (k2, l2, _) in zip(losses, losses[1:]) if l2 > l1]
    print()
    if bad:
        print(f"  NOT MONOTONIC: {bad} — a deeper exit costs more and returns less.")
        print("  Routing cannot help until this is fixed; a uniform depth is the")
        print("  honest choice while it holds.")
        return 1
    print("  monotonic: deeper is better at every step — routing has a trade to exploit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
