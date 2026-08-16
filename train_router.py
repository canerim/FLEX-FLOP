"""Stage B — train the router against a frozen multi-exit decoder.

ClassSR's own recipe (Sec. 3.7) is three stages: pre-train the branches, then
train the Class-Module against them, then optionally fine-tune jointly. We follow
it, with the branches being the exit ladder trained by `train_flexuf_image.py`.

Why the router is trained *after* and *separately*
--------------------------------------------------
FLEX-FLOP found a specific failure worth avoiding: a router trained against a
decoder that is still changing goes stale. It learns that some exit is bad,
stops sending tiles there, that exit then receives no gradient and stays bad, and
the ladder collapses to two live exits. Freezing the decoder first removes the
feedback loop entirely — the oracle table is fixed, so the router is solving a
stationary problem.

The objective (ClassSR Eq. 2, plus one term)
--------------------------------------------
    L = w1*L_image + w2*L_c + w3*L_a + beta*L_comp

  L_image  expected distortion of the routing decision, sum_i P_i * mse_i
  L_c      Eq.(3) Class-Loss — drives P toward one-hot, so the differentiable
           soft blend used in training matches the argmax used at inference
  L_a      Eq.(4) Average-Loss — keeps every exit in use; without it the router
           sends everything to the deepest exit, which is optimal for distortion
           and saves nothing
  L_comp   ours: beta * sum_i P_i * C_i, with C_i the measured relative cost.
           ClassSR had three fixed branches and did not need this; a ladder has a
           continuum of costs, and sweeping beta is what traces the frontier.

Defaults w1=2000, w2=1, w3=6 are ClassSR's. Batch is large for the same reason
ClassSR raised theirs to 96: L_a is a batch statistic and needs enough tiles per
step to mean anything.

Usage
-----
    python train_router.py --ckpt runs/e1_j2_p128/ckpt.pth.tar \
        --train_dataset /data10/.../dcvc_train --beta 1.0 --save_dir runs/e1_j2_p128/router_b1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, RandomSampler

DCVC_ROOT = Path.home() / "DCVC"
sys.path.insert(0, str(DCVC_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.image_dataset import ImageFolder  # noqa: E402
from src.utils.common import create_folder, get_training_lambdas  # noqa: E402

from flexuf.config import QP_LEVELS, FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs, saving  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.router.losses import regret_objective, router_objective  # noqa: E402
from flexuf.router.router import (  # noqa: E402
    N_STEM_SIGNALS,
    ExitRouter,
    stem_signals,
)


def parse_args(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True, help="trained decoder checkpoint")
    p.add_argument("--train_dataset", type=str, required=True)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch_size", type=int, default=16, help="frames per step")
    p.add_argument("--crop", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("-n", "--num_workers", type=int, default=8)
    # ClassSR Eq.(2) weights, rebalanced for the normalised image loss.
    #
    # ClassSR uses w1:w2:w3 = 2000:1:6 against an L1 loss of order 0.02-0.05, so
    # its image term sits ~50-100x above the two regularisers. That ratio is the
    # intent: reconstruction is primary, Class-Loss and Average-Loss only shape
    # the distribution. Our L_image is normalised to a ratio near 1.0, so
    # reproducing the same 50-100x separation means w_image ~= 50, not 2000.
    # Passing --no_normalize_image restores ClassSR's literal numbers.
    p.add_argument("--w_image", type=float, default=50.0)
    p.add_argument("--w_class", type=float, default=1.0)
    p.add_argument("--w_avg", type=float, default=6.0)
    p.add_argument("--no_normalize_image", action="store_true",
                   help="use raw MSE in L_image instead of the ratio to the deepest exit")
    # beta is THE frontier knob. At beta=0 the router only cares about quality
    # and parks everything on the deepest exit; as beta grows it buys compute
    # savings at the cost of dB. One router is trained per beta and each becomes
    # one point on the reported frontier.
    p.add_argument("--beta", type=float, default=25.0, help="complexity weight; SWEEP THIS")
    p.add_argument("--objective", choices=["classsr", "regret"], default="classsr",
                   help="'classsr' is Eq.(2) plus our complexity term: an expected "
                        "distortion, a Class-Loss to repair the soft/argmax "
                        "mismatch, and an Average-Loss to stop collapse -- three "
                        "surrogates, none of them the quantity we measure against. "
                        "'regret' minimises the gap to the oracle directly, which "
                        "is zero exactly when the router picks argmin(mse + lam*C), "
                        "and whose VALUE is the excess cost being paid.")
    p.add_argument("--lam", type=float, default=None,
                   help="Lagrange multiplier for --objective regret; sweep it to "
                        "trace the frontier the way beta does. Defaults to beta so "
                        "one sweep script drives both.")
    p.add_argument("--gumbel_tau", type=float, default=1.0,
                   help="Gumbel-Softmax temperature. hard=True straight-through, so "
                        "training decides exactly as inference does -- one exit, "
                        "chosen -- while gradients flow through the soft path.")
    p.add_argument("--soft", action="store_true",
                   help="regret with a plain softmax instead of the hard "
                        "straight-through sample, to measure what the hard "
                        "decision is worth rather than assume it.")
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args(argv)


@torch.no_grad()
def per_tile_mse(net, x, qp, cfg):
    """MSE of every tile at every exit, for the frozen decoder.

    This is the oracle table ClassSR's Class-Module is trained against. It is
    computed with one trunk pass (`forward_all_exits`) and then reduced per tile,
    so building it costs one full decode per frame rather than K.

    Returns:
        mses  [P, K]  — per-tile MSE at each exit
        tiles [P, C, h, w] — the latent tiles the router will read
        qp_t  [P] — the QP of the frame each tile came from
    """
    out = net.forward_all_exits(x, qp)
    B = x.shape[0]
    rgb_p = cfg.rgb_patch
    H, W = x.shape[-2:]
    nh, nw = H // rgb_p, W // rgb_p

    per_exit = []
    for x_hat in out["x_hats"]:
        err = (x_hat - x) ** 2                       # [B, 3, H, W]
        # mean over channels and within each tile
        t = (
            err.mean(1)
            .view(B, nh, rgb_p, nw, rgb_p)
            .permute(0, 1, 3, 2, 4)
            .reshape(B * nh * nw, rgb_p * rgb_p)
            .mean(1)
        )
        per_exit.append(t)
    mses = torch.stack(per_exit, dim=1)              # [P, K]

    # Signals come from the SHARED STEM, not the raw latent: measured about
    # twice as predictive of the oracle (stem_max r=+0.440 vs the best latent
    # statistic at +0.206), and free, because under a j-split the stem runs
    # full-frame for every tile anyway. See scripts/signal_search.py.
    y_hat, _, aux = net._encode_to_latent(x, qp)
    scales = aux["scales_hat"]
    if scales.shape[1] != y_hat.shape[1]:
        scales = scales[:, : y_hat.shape[1]]
    stem = net.dec.upsample(y_hat)
    for g in range(cfg.split_depth):
        stem = net.dec.groups[g](stem)
    sig = stem_signals(stem, y_hat, scales, cfg)
    qp_t = qp.long().repeat_interleave(nh * nw)
    return mses, sig, qp_t


def main(argv):
    args = parse_args(argv)
    import os

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.device)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    create_folder(args.save_dir)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(device)
    load_flexuf_state(net, ck)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)

    router = ExitRouter(cfg.num_exits, n_signals=N_STEM_SIGNALS, min_exit=cfg.split_depth).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=args.lr)
    costs = exit_costs(cfg, halo_scope="head").to(device)

    dataset = ImageFolder(
        args.train_dataset, args.crop, args.crop, QP_LEVELS,
        get_training_lambdas([10.0, 2048.0], QP_LEVELS),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=False, pin_memory=True, drop_last=True,
        sampler=RandomSampler(dataset, replacement=True,
                              num_samples=args.steps * args.batch_size),
    )

    print(json.dumps({
        "ckpt": args.ckpt, "config": cfg.__dict__, "beta": args.beta,
        "exit_costs": [round(c, 4) for c in costs.tolist()],
        "router_params": sum(p.numel() for p in router.parameters()),
        "tiles_per_frame": cfg.tiles_for_crop(args.crop),
    }, indent=2), flush=True)

    logf = open(Path(args.save_dir) / "router_log.jsonl", "a")
    t0 = time.time()
    for step, batch in enumerate(loader):
        x, qp = batch[0].to(device), batch[-2].to(device)
        mses, sig, qp_t = per_tile_mse(net, x, qp, cfg)
        if args.objective == "regret":
            logits = router(sig, qp_t)
            obj = regret_objective(
                mses, logits, costs,
                lam=(args.lam if args.lam is not None else args.beta),
                tau=args.gumbel_tau, hard=not args.soft, w_avg=0.0)
            probs = obj["probs"]
        else:
            probs = router.probabilities(sig, qp_t)
        if args.objective == "classsr":
          obj = router_objective(
            mses, probs, costs,
            w_image=args.w_image, w_class=args.w_class,
            w_avg=args.w_avg, beta=args.beta,
            normalize_image=not args.no_normalize_image,
        )

        opt.zero_grad(set_to_none=True)
        obj["loss"].backward()
        opt.step()

        if step % args.log_every == 0:
            with torch.no_grad():
                hard = probs.argmax(1)
                sv = saving(hard, cfg, "head")
                # dB lost versus every tile taking the deepest exit
                chosen = mses.gather(1, hard[:, None]).squeeze(1).mean()
                deepest = mses[:, -1].mean()
                d_db = 10 * torch.log10(chosen.clamp_min(1e-10) / deepest.clamp_min(1e-10))
            rec = {
                "step": step,
                "loss": round(obj["loss"].item(), 4),
                # Only the terms this objective actually has. The regret run has
                # one term, and logging zeros for the ClassSR terms would make the
                # two look like the same experiment in the same table.
                **{k: round(obj[k].item(), 8) for k in
                   ("l_image", "l_class", "l_avg", "l_comp", "l_regret",
                    "oracle_agree") if k in obj},
                "exit_share_soft": [round(v, 3) for v in obj["exit_share"].tolist()],
                "exit_share_hard": torch.bincount(hard, minlength=cfg.num_exits).tolist(),
                # The two numbers the whole project is about:
                "saving_pct": round(100 * sv, 2),
                "psnr_loss_dB": round(d_db.item(), 4),
                "sec": round(time.time() - t0, 1),
            }
            print(json.dumps(rec), flush=True)
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if step + 1 >= args.steps:
            break

    logf.close()
    torch.save({"router": router.state_dict(), "config": cfg.__dict__, "beta": args.beta},
               Path(args.save_dir) / "router.pth.tar")
    print("router training complete", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
