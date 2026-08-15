"""Saving and dB cost at every exit, for each QP separately.

Rate changes what early exit is worth. A low-QP latent carries little detail, so
truncating the decoder costs little; a high-QP latent is dense and the same
truncation costs more. A single qp63 number hides that entirely, and the
operating point a deployment actually cares about may be at the other end.

Everything is measured through forward_routed — the real patched decode — so the
seam is included, and against that same model's own full decode.

    python scripts/per_qp_saving.py --ckpt <ckpt> --device 6
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader, SequentialSampler

sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.datasets.image_dataset import ImageFolder            # noqa: E402
from src.utils.common import get_training_lambdas             # noqa: E402
from flexuf.config import QP_LEVELS, FlexUFConfig             # noqa: E402
from flexuf.cost import saving                                # noqa: E402
from flexuf.losses import psnr_from_mse                       # noqa: E402
from flexuf.model import FlexUFIntra                          # noqa: E402


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/data10/shareddata/openimages/dcvc_train")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 8, 16, 24, 32, 40, 48, 56, 63])
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", default=None)
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
        ds.dataset = json.loads(v.read_text())[: a.frames]
        ds.dataset_length = len(ds.dataset)
    ld = DataLoader(ds, batch_size=2, num_workers=3, sampler=SequentialSampler(ds))
    nt = (a.crop // cfg.rgb_patch) ** 2
    exits = list(range(cfg.split_depth, cfg.num_exits))

    print(f"\n{Path(a.ckpt).name}   tile {cfg.rgb_patch}px   j={cfg.split_depth}   "
          f"{a.frames} held-out frames\n")
    hdr = "  qp  |   bpp  |  PSNR  |" + "".join(
        f"  exit{k} ({100*saving(torch.full((nt,),k),cfg,'head'):.0f}%) |" for k in exits)
    print(hdr); print("  " + "-" * (len(hdr) - 2))

    rows = []
    for qp_val in a.qps:
        tot = {k: 0.0 for k in exits}
        full, bpp, n = 0.0, 0.0, 0
        for b in ld:
            x = b[0].to(dev); B = x.shape[0]
            qp = torch.full((B,), qp_val, dtype=torch.int32, device=dev)
            out = net.forward_all_exits(x, qp)
            full += out["mses"][-1].sum().item()
            bpp += out["bpp"].sum().item()
            for i in range(B):
                for k in exits:
                    em = torch.full((nt,), k, device=dev)
                    tot[k] += net.forward_routed(x[i:i+1], qp[i:i+1], em)["mse"].sum().item()
            n += B
        pf = psnr_from_mse(torch.tensor(full / n)).item()
        cells, rec = [], {"qp": qp_val, "bpp": bpp / n, "psnr_full": pf, "exits": {}}
        for k in exits:
            db = pf - psnr_from_mse(torch.tensor(tot[k] / n)).item()
            sv = 100 * saving(torch.full((nt,), k), cfg, "head")
            cells.append(f" {db:+7.3f} dB |")
            rec["exits"][k] = {"saving_pct": sv, "db_cost": db}
        rows.append(rec)
        print(f"  {qp_val:>3} | {bpp/n:6.3f} | {pf:6.2f} |" + "".join(cells))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"ckpt": a.ckpt, "config": cfg.__dict__,
                                           "rows": rows}, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
