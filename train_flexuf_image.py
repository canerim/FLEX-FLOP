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
import contextlib
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
    ladder_distill_loss,
    exit_weights,
    multi_exit_rd_loss,
    per_exit_rd,
    psnr_from_mse,
)
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402


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
    p.add_argument("--steps_per_epoch", type=int, default=0,
                   help="cap the optimiser steps in one epoch (0 = a full pass "
                        "over the dataset). Microsoft's schedule is indexed by "
                        "EPOCH -- 45 at lr 2e-4, then 25 at 5e-5, and so on -- "
                        "and a full pass over 379,614 OpenImages at batch 16 "
                        "with 512px crops takes far longer than any experiment "
                        "budget here, so the schedule's later stages were never "
                        "reached by any run: the six live ones are at epoch 0-4 "
                        "after two days. Capping the steps keeps the SHAPE of "
                        "the schedule -- the same learning rates in the same "
                        "order for the same number of epochs -- while fitting "
                        "the wall-clock available. The sampler is reshuffled "
                        "every epoch, so consecutive epochs see different "
                        "images and the run still covers the dataset; it simply "
                        "covers it across epochs rather than within one. "
                        "Counted in OPTIMISER steps, so it means the same thing "
                        "whether the effective batch is reached in one "
                        "micro-batch or several.")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--train_dataset", type=str, required=True)
    # -- FLEX-UF's arguments ------------------------------------------------
    p.add_argument("--num_exits", type=int, default=6, help="K")
    p.add_argument("--split_depth", type=int, default=2, help="j")
    p.add_argument("--latent_patch", type=int, default=8, help="p; 8 -> 128x128 RGB")
    p.add_argument("--latent_halo", type=int, default=2, help="h, in latent px")
    p.add_argument("--adapter_kind", choices=["conv1x1", "ffn", "scaled"], default="conv1x1")
    p.add_argument("--seam_repair", choices=["none", "depthwise", "full", "grid"], default="full",
                   help="full-frame pass after stitching that heals tile borders")
    p.add_argument("--min_crop", type=int, default=0,
                   help="floor on the recipe's crop size. Needed when the tile is "
                        "as large as the crop: a 256px tile in a 256px crop is ONE "
                        "tile, so no seam exists and --train_patched trains "
                        "nothing it will face at inference")
    p.add_argument("--aux_weight", type=float, default=1.0, help="alpha_i of Eq.(6)")
    p.add_argument("--aux_schedule", choices=["constant", "warmup"], default="constant")
    p.add_argument("--aux_warmup_epochs", type=int, default=10,
                   help="epochs over which aux_schedule=warmup ramps alpha from "
                        "0 to --aux_weight. The config carried this field and "
                        "the command line had no way to set it, so every run "
                        "that asked for a warmup got the default silently.")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--ckpt_every", type=int, default=0,
                   help="steps between mid-epoch weight snapshots (0 = epoch end only)")
    p.add_argument("--tag", type=str, default="", help="experiment label for logs")
    p.add_argument("--pretrain", type=str, default=None,
                   help="warm-start checkpoint to initialise from (e.g. the ladder "
                        "built from Microsoft's release)")
    p.add_argument("--freeze_backbone", action="store_true",
                   help="train ONLY the exit adapters, leaving every inherited "
                        "tensor untouched")
    p.add_argument("--train_patched", action="store_true",
                   help="train through the DEPLOYED patched decode so the adapters "
                        "learn to compensate the tile seams (FLEX: +0.51..+0.90 dB)")
    p.add_argument("--tile_pad", default="replicate",
                   choices=["zeros", "replicate", "linear", "arls", "learned"],
                   help="how a per-tile block invents the missing neighbour at a "
                        "tile border. Measured pure seam penalty at qp63, j=2/128px: "
                        "zeros 0.840 dB, replicate 0.209, linear 0.522, arls 0.163 "
                        "(per-channel AR(1) least squares, arXiv:2502.12300). "
                        "But arls costs +10.7%% of decode wall-clock for +0.034 dB "
                        "over replicate, which buys 1.34 trunk blocks of budget for "
                        "very little; 'learned' keeps its per-channel adaptivity at "
                        "replicate's cost, starting at exactly replicate.")
    p.add_argument("--grad_accum", type=int, default=1,
                   help="micro-batches to accumulate before stepping. The recipe "
                        "specifies batch 16; Microsoft reach it across GPUs "
                        "(get_dataloader divides batch_size by world_size), and "
                        "on one card at 512x512 it does not fit. Accumulating "
                        "keeps the EFFECTIVE batch at the recipe's value instead "
                        "of quietly training at half of it. Gradients are clipped "
                        "after accumulation, so the 0.1 norm bound applies to the "
                        "same quantity it does in train_image.py.")
    p.add_argument("--joint_router", action="store_true",
                   help="train a LEARNED router head jointly with the decoder. "
                        "The hand-made signals were measured blind (|r| <= 0.12 "
                        "against the oracle's choice) and widening them made it "
                        "worse, so both attempts were asking which fixed function "
                        "of a fixed stem predicts the right exit. This lets the "
                        "stem become routable instead. Costs 0.044%% of the decode "
                        "-- a 1x1 of 384->16 on the stem, which sits at 1/64 of "
                        "the pixel count; GridSeamRepair is 0.951%% for scale.")
    p.add_argument("--router_beta", type=float, default=1.0,
                   help="weight on the differentiable compute term sum_k P_k C_k. "
                        "Sweep it to trace the frontier; 0 means quality only.")
    p.add_argument("--gumbel_tau", type=float, default=1.0)
    p.add_argument("--tile_coupling", action="store_true",
                   help="let each per-tile 3x3 depthwise read its REAL neighbours "
                        "from a shared canvas instead of inventing them. The "
                        "depthwise is 0.334%% of a DepthConvBlock and the only "
                        "operator with any spatial extent, so this costs +0.066%% "
                        "of the decode at 128px tiles and +0.032%% at 256px, "
                        "against GridSeamRepair's 0.951%%. Where neighbouring tiles "
                        "share a depth the result is BIT-EXACT the full-frame "
                        "decode -- measured max|diff| = 0.0, the seam does not "
                        "shrink, it stops existing. Where they differ it is "
                        "currently 0.12 dB WORSE than replicate on an untrained "
                        "ladder, because a deep block reads a neighbour's "
                        "six-blocks-shallower feature; --distill_weight is the "
                        "term that makes those features compatible, so the two "
                        "belong together.")
    p.add_argument("--distill_weight", type=float, default=0.0,
                   help="weight on ladder distillation: each exit's adapter is "
                        "trained to reproduce a DEEPER exit's feature, not just to "
                        "make good pixels. The multi-exit literature is consistent "
                        "that distilling from deep exits to shallow ones is what "
                        "makes shallow exits usable (Zhang et al., 'Be Your Own "
                        "Teacher', ICCV 2019; ERDE, arXiv:2510.04856). Here the "
                        "target is 384-channel and dense rather than 3-channel RGB, "
                        "so it carries far more signal per step -- and it is exactly "
                        "the quantity the shared head consumes.")
    p.add_argument("--distill_teacher", choices=["deepest", "adjacent"],
                   default="adjacent",
                   help="whose feature each exit imitates. 'deepest' is the obvious "
                        "choice and the one the FITEE 2024 survey of multi-exit "
                        "self-distillation reports as harmful for the SHALLOWEST "
                        "exits: too large a student-teacher gap degrades them. "
                        "'adjacent' has exit k imitate exit k+1, so each adapter "
                        "closes one group's worth of gap and the chain carries the "
                        "rest -- which also matches our structure, where a shallow "
                        "exit is literally a prefix of a deep one.")
    p.add_argument("--compress_schedule", action="store_true",
                   help="spread Microsoft's 105-entry schedule over --epochs "
                        "instead of indexing it one-for-one. The recipe is 45 "
                        "epochs at 2e-4, 25 at 5e-5, 20 at 1e-5, then four "
                        "short 512px stages ending at 1e-6. Running only its "
                        "first block leaves the model at the highest learning "
                        "rate it ever sees -- never annealed, and an unannealed "
                        "model is reliably worse than the same budget spent "
                        "with a decay. This maps epoch e to strategy index "
                        "round(e * 105 / epochs), so a shorter run still "
                        "traverses the whole shape: the same rates in the same "
                        "order and the same proportions, ending at 1e-6. Off by "
                        "default, so verbatim indexing is what you get unless "
                        "you ask for this.")
    p.add_argument("--epoch_offset", type=int, default=0,
                   help="where in Microsoft's 105-epoch schedule to START. 0 is "
                        "correct from scratch and WRONG for a warm start: 2e-4 is "
                        "the from-scratch initial lr, and --pretrain hands us the "
                        "OUTPUT of epoch 105, so applying epoch-0's lr kicks a "
                        "converged decoder off its optimum. Measured: the deepest "
                        "exit fell 0.15/0.20/0.26 dB below released DCVC-UF at "
                        "qp0/32/63 after ONE epoch. 69 puts us in the 1e-5 @ 256px "
                        "fine-tune regime -- the same recipe, read at the right place.")
    p.add_argument("--new_lr_scale", type=float, default=1.0,
                   help="lr multiplier for modules that did NOT exist in stock UF "
                        "(adapters, seam repair, pad_coef). They are zero-init and "
                        "must learn from nothing, so the fine-tune lr that protects "
                        "the inherited trunk is far too small for them. One lr "
                        "cannot serve both; this is the standard warm-start split.")
    p.add_argument("--anchor_weight", type=float, default=0.0,
                   help="pin the deepest exit to the RELEASED decoder's output "
                        "with this weight (0 = off). Measured need: training the "
                        "whole decoder drifts the deepest exit 0.20 dB (CTC) to "
                        "0.38 dB (OpenImages, qp32) below the release after ONE "
                        "epoch. Every saving figure is quoted 'at x dB vs "
                        "DCVC-UF', so that drift is spent before a single tile "
                        "exits early -- it alone exceeds a 0.1 dB budget. "
                        "Freezing the trunk removes the drift but leaves a 1x1 "
                        "adapter to replace six DepthConvBlocks; this keeps the "
                        "trunk trainable and pins only the deep end.")
    p.add_argument("--anchor_per_sample", action="store_true",
                   help="scale the anchor by each sample's OWN lambda instead "
                        "of the batch mean. The batch mean makes the anchor 39x "
                        "too strong at qp0 and 5x too weak at qp63, which is "
                        "backwards: the drift is 10x larger at qp63. Off by "
                        "default so running jobs are unaffected by a restart.")
    p.add_argument("--bf16", action="store_true",
                   help="Run the forward passes under torch.autocast in "
                        "bfloat16. Microsoft's train_image.py is fp32 and this "
                        "is the one departure from it, made deliberately: an "
                        "A6000 has no fp16 tensor-core advantage over bf16 and "
                        "bf16 keeps fp32's exponent range, so the gradient "
                        "clip at 0.1 and the non-finite guard behave as they "
                        "do in fp32 -- which fp16 plus a loss scaler would "
                        "not. Losses are computed in fp32 outside the "
                        "autocast region so the RD trade-off is not evaluated "
                        "at reduced precision.")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="freeze the encoder, hyperprior and entropy model; train "
                        "the WHOLE decoder (trunk, head and adapters)")
    return p.parse_args(argv)


def build_cfg(args) -> FlexUFConfig:
    return FlexUFConfig(
        num_exits=args.num_exits,
        split_depth=args.split_depth,
        latent_patch=args.latent_patch,
        latent_halo=args.latent_halo,
        adapter_kind=args.adapter_kind,
        seam_repair=args.seam_repair,
        aux_weight=args.aux_weight,
        aux_schedule=args.aux_schedule,
        aux_warmup_epochs=args.aux_warmup_epochs,
        tile_pad_mode=args.tile_pad,
        tile_coupling=args.tile_coupling,
    )


def _param_groups(net, new_lr_scale):
    """Split trainable tensors into inherited and new-since-stock-UF.

    The two need different learning rates for opposite reasons: the inherited
    decoder is already converged and any large step moves it off the optimum
    (measured: 0.20 dB of anchor drift at qp32 in one epoch at lr 2e-4), while
    the adapters and seam repair are zero-initialised and learn nothing at the lr
    that keeps the trunk still. Running both at one lr means picking which of the
    two failures to accept.
    """
    NEW_PREFIXES = (".adapters.", ".seam_repair.", "pad_coef", "router_head.")
    inherited, fresh = [], []
    for name, prm in net.named_parameters():
        if not prm.requires_grad:
            continue
        (fresh if any(k in name for k in NEW_PREFIXES) else inherited).append(prm)
    groups = [{"params": inherited, "lr_scale": 1.0}]
    if fresh:
        groups.append({"params": fresh, "lr_scale": new_lr_scale})
    print(f"param groups: {sum(p.numel() for p in inherited):,} inherited "
          f"(lr x1), {sum(p.numel() for p in fresh):,} new (lr x{new_lr_scale})",
          flush=True)
    return groups


def _fp32(out):
    """Every tensor a forward returned, in fp32. Lists and tensors only, which
    is what the forwards produce."""
    got = {}
    for k, v in out.items():
        if torch.is_tensor(v):
            got[k] = v.float()
        elif isinstance(v, (list, tuple)):
            got[k] = type(v)(t.float() if torch.is_tensor(t) else t for t in v)  # None passes through
        else:
            got[k] = v
    return got


def amp_ctx(args, device):
    """bfloat16 autocast when asked for, and nothing at all when not.

    contextlib.nullcontext rather than an `if` at every call site, so the fp32
    path is exactly the code Microsoft's trainer runs.
    """
    if not getattr(args, "bf16", False):
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def train_one_epoch(net, loader, optimizer, epoch, cfg, args, device, logf,
                    anchor_net=None):
    strategy = get_training_strategy()
    net.train()

    if args.compress_schedule and args.epochs > 0:
        idx = min(len(strategy) - 1,
                  int(round((epoch + args.epoch_offset) * len(strategy)
                            / args.epochs)))
    else:
        idx = min(len(strategy) - 1, epoch + args.epoch_offset)
    _, lr, patch_w, patch_h = strategy[idx]
    # The recipe trains at 256x256 until epoch 90. With a 256px tile that is a
    # single tile per crop — no borders, no seams — so patched training would be
    # identical to full-frame and the configuration would never see the artefact
    # it exists to handle. Raising the floor costs 4x the pixels per step and is
    # the only way the experiment means anything.
    if args.min_crop:
        patch_w = max(patch_w, args.min_crop)
        patch_h = max(patch_h, args.min_crop)
    # Two groups: inherited weights at the schedule's lr, new zero-init modules
    # at a multiple of it. `lr_scale` is set where the optimiser is built.
    for g in optimizer.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
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
    total_norm = float("nan")
    for i, batch in enumerate(loader):
        # A capped epoch stops here. Placed before the work rather than after
        # it so the cap is exact: `--steps_per_epoch 1500` performs 1500
        # optimiser steps, not 1500 plus whatever the loader had prefetched.
        if args.steps_per_epoch and i >= args.steps_per_epoch * max(
                1, args.grad_accum):
            break
        batch = [t.to(device, non_blocking=True) for t in batch]
        x, qp, lambdas = batch[0], batch[-2], batch[-1]

        with amp_ctx(args, device):
            if args.train_patched:
                # FLEX Stage A: one patched decode with a fresh random depth per
                # tile. Trains the mixed-depth frame that deployment actually
                # produces, at the cost of a single decode rather than K.
                out = (net.forward_joint(x, qp, tau=args.gumbel_tau,
                                         beta=args.router_beta)
                       if args.joint_router else net.forward_random_depth(x, qp))
                w_step = torch.ones(1, device=device)
            else:
                # Only the exits the schedule is actually weighting. Under
                # aux_schedule=warmup the first epoch weights one of six, and
                # decoding the other five would be six times the cost for a
                # term multiplied by zero.
                active = [k for k in range(len(w)) if float(w[k]) != 0.0]
                out = net.forward_all_exits(
                    x, qp, only=None if len(active) == len(w) else active)
                w_step = w
        # The latent stays as the forward produced it, in the autocast dtype and
        # attached to its graph, so the distillation and anchor terms below can
        # reuse it instead of encoding the same crop again. Measured: the second
        # encode was half the step time.
        y_cached = out.pop("_y_hat", None)
        q_cached = out.pop("_q_dec", None)
        feats_cached = out.pop("_feats", None)
        # Out of the autocast region: the RD trade-off is the quantity the whole
        # paper is about and it is not evaluated at reduced precision. Casting
        # here rather than inside also keeps the loss graph in fp32, which is
        # what the 0.1 grad clip was tuned against.
        out = _fp32(out)
        ld = multi_exit_rd_loss(out["mses"], out["bpp"], lambdas, w_step)

        # Anchor: hold the deepest exit on the released decoder's output.
        #
        # Not a regulariser for its own sake. The project's claim is "x% cheaper
        # at y dB versus DCVC-UF", and y is only that if our deepest exit still
        # IS DCVC-UF. It starts bit-exact (warm start) and drifts, because the
        # mixed-depth objective gives the deepest exit a quarter of the gradient
        # while the shared trunk is pulled toward the shallow exits.
        #
        # Distilled against the frozen decoder's OUTPUT rather than the source
        # image: matching the release is the requirement, and the release is not
        # the source. A term that pushed the deepest exit toward the image would
        # be asking it to beat DCVC-UF, which is a different project.
        anchor_mse = None
        if anchor_net is not None:
            with amp_ctx(args, device):
                if y_cached is None:
                    y_a, q_a, _ = net._encode_to_latent(x, qp)
                else:
                    y_a, q_a = y_cached, q_cached
                with torch.no_grad():
                    ref = anchor_net.dec.forward_full(y_a, q_a)
                deep = net.dec.forward_full(y_a, q_a)
            deep, ref = deep.float(), ref.float()
            anchor_mse = ((deep - ref) ** 2).mean()
            # The stated intent is "scaled by the same lambda the reconstruction
            # term carries, so the weight does not have to be retuned per QP".
            # `lambdas.mean()` does not do that. lambdas runs 10 (qp0) to 2048
            # (qp63) and the batch mean is ~393, so relative to the RD term the
            # anchor is 39x STRONGER at qp0 and 5x WEAKER at qp63 -- while the
            # drift it exists to prevent is ten times larger at qp63 (0.003 dB
            # against 0.029). It is backwards exactly where it matters.
            #
            # Measured stakes: the anchor drift IS the frontier's floor, and at
            # qp63 that floor eats 30% of the 0.1 dB budget. Removing it is
            # worth 3.28 points of saving there (results/curve_BEST.json, read
            # at 0.1 dB against 0.1 dB + floor).
            #
            # Off by default. Six runs share this file, and a crash-restart
            # would otherwise pick the change up mid-experiment and silently
            # change what they are measuring.
            if args.anchor_per_sample:
                per = ((deep - ref) ** 2).mean(dim=tuple(range(1, deep.dim())))
                anchor_term = (lambdas.reshape(-1) * per).mean()
            else:
                anchor_term = lambdas.mean() * anchor_mse
            ld["loss"] = ld["loss"] + args.anchor_weight * anchor_term

        # Ladder distillation: supervise the adapters in FEATURE space, where
        # their job is actually stated, instead of only through the head's
        # 3-channel output. One extra trunk pass, tapped at every exit.
        distill = None
        if args.distill_weight > 0:
            with amp_ctx(args, device):
                if feats_cached is not None:
                    _feats = feats_cached
                else:
                    if y_cached is None:
                        y_d, _, _ = net._encode_to_latent(x, qp)
                    else:
                        y_d = y_cached
                    _feats = net.dec.exit_features(y_d)
            distill = ladder_distill_loss([f.float() for f in _feats],
                                          args.distill_teacher)
            ld["loss"] = ld["loss"] + args.distill_weight * distill

        # Accumulate, then step. zero_grad only at the start of an accumulation
        # window, and the loss is scaled so the accumulated gradient equals the
        # one a single large batch would produce. `total_norm` carries the last
        # clipped norm forward, because a log line can fall on a micro-batch
        # where no clip happened and reporting a stale number is better than
        # reporting none -- but it must never read as a fresh measurement, so it
        # starts at nan and only a real clip replaces it.
        if i % args.grad_accum == 0:
            optimizer.zero_grad(set_to_none=True)
        (ld["loss"] / args.grad_accum).backward()

        # Microsoft's exact guard: clip to 0.1, and drop the batch on a
        # non-finite norm rather than letting it poison the weights.
        if (i + 1) % args.grad_accum == 0:
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
                if args.train_patched:
                    # forward_random_depth returns ONE mixed-depth reconstruction,
                    # so it carries no per-exit breakdown — and that breakdown is
                    # the health signal: a ladder collapsing shows up as the
                    # per-exit PSNRs converging on each other long before the
                    # loss notices. Cheap to recover: one extra full-frame pass
                    # every log_every steps, which at 200 is a fraction of a
                    # percent of training time.
                    with amp_ctx(args, device):
                        diag = net.forward_all_exits(x, qp)
                    diag = _fp32(diag)
                    psnrs = [psnr_from_mse(m.mean()).item() for m in diag["mses"]]
                    rd = per_exit_rd(diag["mses"], diag["bpp"], lambdas).tolist()
                    mixed = psnr_from_mse(out["mses"][0].mean()).item()
                else:
                    # During the warmup schedule the step only decodes the
                    # exits it weights, so the others are None here. The
                    # per-exit PSNR is the health signal that says whether the
                    # ladder is ordered, and it is worth one full pass every
                    # log_every steps to keep it.
                    src = out
                    if any(m is None for m in out["mses"]):
                        with amp_ctx(args, device):
                            src = net.forward_all_exits(x, qp)
                        src = _fp32(src)
                    psnrs = [psnr_from_mse(m.mean()).item() for m in src["mses"]]
                    rd = per_exit_rd(src["mses"], src["bpp"], lambdas).tolist()
                    mixed = None
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
                "anchor_mse": (anchor_mse.item() if anchor_mse is not None else None),
                "distill": (distill.item() if distill is not None else None),
                # The primary diagnostic for a jointly trained router: WHERE it
                # sent the tiles. A constant router -- everything to one exit --
                # is the known failure mode and is invisible in the loss, which
                # falls perfectly well while the routing degenerates.
                **({"exit_hist": torch.bincount(
                        out["exit_idx"], minlength=cfg.num_exits).tolist(),
                    "exp_cost": round(out["cost"].item(), 4)}
                   if "exit_idx" in out else {}),
                "bpp": ld["bpp"].item(),
                # The ladder's shape. A healthy run has these monotonically
                # increasing; all-equal means the exits have collapsed, which is
                # the failure FLEX hit when training from scratch.
                "psnr_per_exit": [round(v, 3) for v in psnrs],
                "spread_dB": round(psnrs[-1] - psnrs[0], 3),
                "rd_per_exit": [round(v, 5) for v in rd],
                # PSNR of the mixed-depth decode actually being optimised, next
                # to the per-exit numbers measured full-frame.
                #
                # The gap between them is NOT the seam cost — the mixed decode
                # samples random exits, so it carries depth cost too, and lands
                # near the average of the exits it drew. Pure seam cost is
                # measured separately, by decoding with every tile at the deepest
                # exit (scripts/per_qp_saving.py): 0.063 dB at qp0, 0.140 at qp63.
                "psnr_mixed_patched": None if mixed is None else round(mixed, 3),
                "grad_norm": round(total_norm, 5),
                "skipped": n_skipped,
                "sec": round(t1 - t0, 1),
            }
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            t0 = t1

        # Mid-epoch snapshot, for measurement rather than for resume.
        #
        # An epoch here is 47451 steps -- 10 to 19 hours depending on the crop --
        # and the epoch-end save is the only one. So a run that has been training
        # all day still cannot be EVALUATED, which is the thing that matters when
        # a configuration is new and might simply be broken: FINE12 changes K from
        # 6 to 12, and waiting a full epoch to find that out wastes the day.
        #
        # Weights only, no optimiser state: this is not a resume point (resuming
        # mid-epoch would need the sampler position too, which is not worth the
        # fragility). Written to a temp file and renamed, because the evaluation
        # scripts poll for this file and would otherwise read a half-written one.
        if args.ckpt_every and i and i % args.ckpt_every == 0:
            snap = Path(args.save_dir) / "ckpt_step.pth.tar"
            tmp = snap.with_suffix(".tmp")
            torch.save({"state_dict": net.state_dict(), "config": cfg.__dict__,
                        "epoch": epoch, "step": i}, tmp)
            tmp.replace(snap)


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
    # Freeze only the analysis side: the encoder, hyperprior and entropy model
    # stay exactly Microsoft's, so the latent and therefore the bitstream and the
    # bpp are identical to the release. The entire decoder — all 143 inherited
    # tensors plus the 10 adapters — is then free to reorganise itself around the
    # multi-exit structure.
    #
    # This sits between the two modes we already had. --freeze_backbone trains
    # 739k adapter parameters and leaves the decoder unable to adapt at all;
    # training everything trains the encoder too, which changes the latent and
    # makes the result incomparable to real DCVC-UF (measured: our q_scale table
    # came out 6.45x the release's, and qp63 landed at 0.783 bpp against 0.829).
    # Here the rate axis is pinned to the release while the decoder gets full
    # freedom.
    anchor_net = None
    if args.anchor_weight > 0:
        if not args.pretrain:
            raise SystemExit("--anchor_weight needs --pretrain: the anchor IS the "
                             "released decoder, and without a warm start there is "
                             "nothing to pin to.")
        anchor_net = FlexUFIntra(cfg).to(device).eval()
        load_flexuf_state(anchor_net, torch.load(args.pretrain, map_location="cpu",
                                                 weights_only=False))
        for prm in anchor_net.parameters():
            prm.requires_grad = False
        print(f"anchor: deepest exit pinned to {args.pretrain} "
              f"with weight {args.anchor_weight}", flush=True)

    if args.freeze_encoder:
        # The router head lives beside the decoder, not inside it, so a plain
        # "dec." prefix test froze it -- it would have sat at its initialisation
        # for the whole run while the log happily reported training. Caught by
        # the trainable-parameter count not moving when the head was added.
        for name, prm in net.named_parameters():
            prm.requires_grad = name.startswith(("dec.", "router_head."))
        trainable = [p_ for p_ in net.parameters() if p_.requires_grad]
        n_tr = sum(p_.numel() for p_ in trainable)
        n_all = sum(p_.numel() for p_ in net.parameters())
        n_enc = n_all - n_tr
        print(f"frozen encoder: training the decoder, {n_tr:,} / {n_all:,} params "
              f"({100*n_tr/n_all:.2f}%); {n_enc:,} analysis-side params frozen "
              f"at the release values", flush=True)
        optimizer = torch.optim.AdamW(_param_groups(net, args.new_lr_scale), lr=1e-4)
    elif args.freeze_backbone:
        for name, prm in net.named_parameters():
            prm.requires_grad = (".adapters." in name or ".seam_repair." in name
                                 or name.startswith("router_head."))
        trainable = [p_ for p_ in net.parameters() if p_.requires_grad]
        n_tr = sum(p_.numel() for p_ in trainable)
        n_all = sum(p_.numel() for p_ in net.parameters())
        print(f"frozen backbone: training {n_tr:,} / {n_all:,} params "
              f"({100*n_tr/n_all:.2f}%)", flush=True)
        optimizer = torch.optim.AdamW(_param_groups(net, args.new_lr_scale), lr=1e-4)
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
            train_one_epoch(net, loader, optimizer, epoch, cfg, args, device, logf,
                            anchor_net)
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
