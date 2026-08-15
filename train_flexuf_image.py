"""Train the multi-exit DCVC-UF intra codec from scratch, on Microsoft's recipe.

Fidelity to the recipe
----------------------
Everything that defines Microsoft's `train_image.py` is reproduced exactly, and
where a line differs it is because the model has K exits instead of one:

| ingredient        | Microsoft (`~/DCVC/train_image.py`)         | here            |
|-------------------|---------------------------------------------|-----------------|
| schedule          | `get_training_strategy()` lines 22-32       | copied verbatim |
| epochs            | 105 entries (45+25+20+5+4+4+2+1)            | same            |
| lr per epoch      | 2e-4 -> 5e-5 -> 1e-5 -> 2e-4 -> ... -> 1e-6 | same            |
| patch size        | 256 for epochs 0-89, 512 from epoch 90      | same            |
| optimiser         | `AdamW(lr=1e-4)`, overridden per epoch      | same            |
| batch size        | 16                                          | same            |
| grad clip         | `clip_grad_norm_(max_norm=0.1)`, skip NaN   | same            |
| dataset           | `ImageFolder` + `description.json`          | same            |
| lambdas           | log-spaced 10 -> 2048 over 64 QPs           | same            |
| QP sampling       | uniform over 64, per sample                 | same            |
| loss              | `lambda*mse + bpp`                          | Eq.(6)-(7) form |

The loss is the single deliberate change, and it is weight-normalised precisely
so the rate-distortion operating point the lambdas encode is unchanged — see
`flexuf/losses.py`.

Why one GPU per run rather than DDP across three
------------------------------------------------
Three *independent* experiments answer more than one three-times-faster run of a
single configuration: the axes we need to separate (split depth j, tile size,
adapter capacity) are exactly the ones nobody has measured on UF. Each run is
therefore rank -1 single-GPU, pinned with CUDA_VISIBLE_DEVICES.

Usage
-----
    python train_flexuf_image.py \
        --train_dataset /path/to/openimages_dcvc \
        --save_dir runs/e1_j2_p128 \
        --lambdas 10 2048 \
        --split_depth 2 --latent_patch 8 --latent_halo 2 \
        --device 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

DCVC_ROOT = Path.home() / "DCVC"
sys.path.insert(0, str(DCVC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.image_dataset import ImageFolder  # noqa: E402
from src.utils.common import create_folder, get_training_lambdas  # noqa: E402

from flexuf.config import QP_LEVELS, FlexUFConfig  # noqa: E402
from flexuf.losses import (  # noqa: E402
    exit_weights,
    multi_exit_rd_loss,
    per_exit_rd,
    psnr_from_mse,
)
from flexuf.model import FlexUFIntra  # noqa: E402


def get_training_strategy():
    """Verbatim from `~/DCVC/train_image.py:19-33`. Do not "improve" this.

    Each entry is [reference_epoch, lr, patch_w, patch_h]; the list is indexed by
    epoch, so its length defines the recipe's 105 epochs.
    """
    return (
        [[0,   2e-4, 256, 256]] * 45 +
        [[49,  5e-5, 256, 256]] * 25 +
        [[69,  1e-5, 256, 256]] * 20 +
        [[90,  2e-4, 512, 512]] * 5 +
        [[95,  5e-5, 512, 512]] * 4 +
        [[99,  1e-5, 512, 512]] * 4 +
        [[103, 1e-6, 512, 512]] * 2 +
        [[105, 1e-6, 512, 512]]
    )


def parse_args(argv):
    p = argparse.ArgumentParser()
    # -- Microsoft's arguments, same names and defaults --------------------
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("-e", "--epochs", type=int, default=105)
    p.add_argument("--lambdas", type=float, nargs="+", default=[10.0, 2048.0])
    p.add_argument("-n", "--num_workers", type=int, default=8)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--train_dataset", type=str, required=True)
    # -- FLEX-UF's arguments ------------------------------------------------
    p.add_argument("--num_exits", type=int, default=6, help="K")
    p.add_argument("--split_depth", type=int, default=2, help="j")
    p.add_argument("--latent_patch", type=int, default=8, help="p; 8 -> 128x128 RGB")
    p.add_argument("--latent_halo", type=int, default=2, help="h, in latent px")
    p.add_argument("--adapter_kind", choices=["conv1x1", "ffn"], default="conv1x1")
    p.add_argument("--aux_weight", type=float, default=1.0, help="alpha_i of Eq.(6)")
    p.add_argument("--aux_schedule", choices=["constant", "warmup"], default="constant")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--tag", type=str, default="", help="experiment label for logs")
    p.add_argument("--pretrain", type=str, default=None,
                   help="warm-start checkpoint to initialise from (e.g. the ladder "
                        "built from Microsoft's release)")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="train ONLY the exit adapters, leaving every inherited "
                        "tensor untouched")
    return p.parse_args(argv)


def build_cfg(args) -> FlexUFConfig:
    return FlexUFConfig(
        num_exits=args.num_exits,
        split_depth=args.split_depth,
        latent_patch=args.latent_patch,
        latent_halo=args.latent_halo,
        adapter_kind=args.adapter_kind,
        aux_weight=args.aux_weight,
        aux_schedule=args.aux_schedule,
    )


def train_one_epoch(net, loader, optimizer, epoch, cfg, args, device, logf):
    strategy = get_training_strategy()
    net.train()

    idx = min(len(strategy) - 1, epoch)
    _, lr, patch_w, patch_h = strategy[idx]
    for g in optimizer.param_groups:
        g["lr"] = lr
    loader.dataset.set_patch_size(patch_w, patch_h)

    w = exit_weights(
        cfg.num_exits,
        cfg.aux_weight,
        cfg.aux_schedule,
        epoch,
        cfg.aux_warmup_epochs,
        device=device,
    )

    t0 = time.time()
    n_skipped = 0
    for i, batch in enumerate(loader):
        batch = [t.to(device, non_blocking=True) for t in batch]
        x, qp, lambdas = batch[0], batch[-2], batch[-1]

        out = net.forward_all_exits(x, qp)
        ld = multi_exit_rd_loss(out["mses"], out["bpp"], lambdas, w)

        optimizer.zero_grad(set_to_none=True)
        ld["loss"].backward()

        # Microsoft's exact guard: clip to 0.1, and drop the batch on a
        # non-finite norm rather than letting it poison the weights.
        clip_params = [p_ for p_ in net.parameters() if p_.requires_grad]
        total_norm = clip_grad_norm_(
            clip_params, max_norm=0.1, error_if_nonfinite=False
        ).item()
        if math.isnan(total_norm) or math.isinf(total_norm):
            n_skipped += 1
            continue
        optimizer.step()

        if i % args.log_every == 0:
            with torch.no_grad():
                psnrs = [psnr_from_mse(m.mean()).item() for m in out["mses"]]
                rd = per_exit_rd(out["mses"], out["bpp"], lambdas).tolist()
            t1 = time.time()
            seen = i * args.batch_size
            rec = {
                "epoch": epoch,
                "step": i,
                "seen": seen,
                "total": len(loader.dataset),
                "lr": lr,
                "patch": patch_w,
                "loss": ld["loss"].item(),
                "bpp": ld["bpp"].item(),
                # The ladder's shape. A healthy run has these monotonically
                # increasing; all-equal means the exits have collapsed, which is
                # the failure FLEX hit when training from scratch.
                "psnr_per_exit": [round(v, 3) for v in psnrs],
                "spread_dB": round(psnrs[-1] - psnrs[0], 3),
                "rd_per_exit": [round(v, 5) for v in rd],
                "grad_norm": round(total_norm, 5),
                "skipped": n_skipped,
                "sec": round(t1 - t0, 1),
            }
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            t0 = t1


def main(argv):
    args = parse_args(argv)
    cfg = build_cfg(args)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    create_folder(args.save_dir)

    dataset = ImageFolder(
        args.train_dataset, 256, 256, QP_LEVELS,
        get_training_lambdas(args.lambdas, QP_LEVELS),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=RandomSampler(dataset),
        prefetch_factor=2,
        persistent_workers=args.num_workers > 0,
    )

    net = FlexUFIntra(cfg).to(device)

    if args.pretrain:
        pre = torch.load(args.pretrain, map_location="cpu", weights_only=False)
        net.load_state_dict(pre.get("state_dict", pre.get("net")), strict=False)
        print(f"warm-started from {args.pretrain}", flush=True)

    # Stage A of FLEX's recipe: freeze everything inherited, train only the
    # adapters.
    #
    # Those inherited tensors are 99.66% of decoder MACs and every 1x1 in it, and
    # they are what makes the deepest exit bit-exact the reference codec. Training
    # them would throw away the warm start the whole approach depends on — and the
    # warm start is precisely what fixes the non-monotonic ladder that the
    # from-scratch runs produced. So the trainable set is ~739k parameters out of
    # 42.9M, and one pass costs hours rather than the ~37 days a full-recipe run
    # of this size takes.
    if args.freeze_backbone:
        for name, prm in net.named_parameters():
            prm.requires_grad = ".adapters." in name
        trainable = [p_ for p_ in net.parameters() if p_.requires_grad]
        n_tr = sum(p_.numel() for p_ in trainable)
        n_all = sum(p_.numel() for p_ in net.parameters())
        print(f"frozen backbone: training {n_tr:,} / {n_all:,} params "
              f"({100*n_tr/n_all:.2f}%)", flush=True)
        optimizer = torch.optim.AdamW(trainable, lr=1e-4)
    else:
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)

    begin_epoch = 0
    ckpt_path = Path(args.save_dir) / "status_latest.pth.tar"
    if ckpt_path.exists():
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        net.load_state_dict(st["net"])
        optimizer.load_state_dict(st["opt"])
        begin_epoch = st["epoch"] + 1
        print(f"resumed from {ckpt_path} at epoch {begin_epoch}", flush=True)

    meta = {
        "tag": args.tag,
        "config": cfg.__dict__,
        "recipe": "microsoft train_image.py, verbatim schedule",
        "params_total": sum(p.numel() for p in net.parameters()),
        "params_adapters": sum(p.numel() for p in net.dec.adapters.parameters()),
        "dataset": args.train_dataset,
        "dataset_size": len(dataset),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "device": args.device,
        "rgb_patch": cfg.rgb_patch,
        "tiles_per_256crop": cfg.tiles_for_crop(256),
        "tiles_per_512crop": cfg.tiles_for_crop(512),
    }
    (Path(args.save_dir) / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2), flush=True)

    logf = open(Path(args.save_dir) / "train_log.jsonl", "a")
    try:
        for epoch in range(begin_epoch, args.epochs):
            train_one_epoch(net, loader, optimizer, epoch, cfg, args, device, logf)
            torch.save(
                {"net": net.state_dict(), "opt": optimizer.state_dict(), "epoch": epoch},
                ckpt_path,
            )
            if epoch % 5 == 0:
                torch.save(
                    {"state_dict": net.state_dict(), "epoch": epoch, "config": cfg.__dict__},
                    Path(args.save_dir) / f"ckpt_epo{epoch}.pth.tar",
                )
    finally:
        logf.close()

    torch.save(
        {"state_dict": net.state_dict(), "config": cfg.__dict__},
        Path(args.save_dir) / "ckpt.pth.tar",
    )
    print("training complete", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
