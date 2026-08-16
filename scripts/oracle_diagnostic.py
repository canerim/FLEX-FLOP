"""Is there any content adaptivity to exploit, and can the signals see it?

The frontier sweep produced routers that send *every* tile to the same exit,
changing only which exit as beta moves. That is a constant function, not
routing, and it invalidates the premise the project rests on — ClassSR works
because sub-images differ in difficulty. So before tuning the router, establish
whether the thing it is meant to find exists at all.

Three questions, in the order that makes the answer diagnostic:

  1. **Does the ORACLE vary?** For each tile, the cheapest exit whose distortion
     stays within `tau` of the full decode. If the oracle itself is constant,
     there is no adaptivity in the content at this checkpoint and no router
     could help — the honest conclusion would be that a uniformly shallower
     decoder is the right answer. If the oracle varies a lot, the headroom is
     real and any failure is the router's.

  2. **How much is that headroom worth?** Oracle saving at a given dB budget,
     against the best uniform-depth choice at the same budget. This is the
     ceiling on what routing can buy, and if it is small the whole mechanism is
     not worth its complexity.

  3. **Can the signals see it?** Correlation between each router input and the
     oracle's choice. A signal uncorrelated with the oracle cannot drive it, and
     four uncorrelated signals explain a constant router precisely.

    python scripts/oracle_diagnostic.py --ckpt runs/e3_j2_p64/ckpt_epo0.pth.tar --device 7
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


@torch.no_grad()
def collect(net, loader, cfg, device, qp_val, max_batches):
    """Per-tile MSE at every exit, plus the router's view of each tile."""
    mses, sigs = [], []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch[0].to(device)
        B = x.shape[0]
        qp = torch.full((B,), qp_val, dtype=torch.int32, device=device)

        out = net.forward_all_exits(x, qp)
        rgb_p = cfg.rgb_patch
        H, W = x.shape[-2:]
        nh, nw = H // rgb_p, W // rgb_p
        per_exit = []
        for x_hat in out["x_hats"]:
            err = (x_hat - x) ** 2
            t = (err.mean(1)
                 .view(B, nh, rgb_p, nw, rgb_p)
                 .permute(0, 1, 3, 2, 4)
                 .reshape(B * nh * nw, rgb_p * rgb_p)
                 .mean(1))
            per_exit.append(t)
        mses.append(torch.stack(per_exit, dim=1).cpu())

        y_hat, _ = net.latent_of(x, qp)
        sigs.append(tile_signals(latent_tiles_with_halo(y_hat, cfg)).cpu())
    return torch.cat(mses), torch.cat(sigs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", default="/data10/shareddata/openimages/dcvc_train")
    ap.add_argument("--qp", type=int, default=63)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--batches", type=int, default=12)
    ap.add_argument("--batch_size", type=int, default=4)
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

    mses, sigs = collect(net, loader, cfg, device, a.qp, a.batches)
    K = mses.shape[1]
    costs = exit_costs(cfg, "head")
    print(f"\ncheckpoint : {a.ckpt}")
    print(f"tiles      : {mses.shape[0]}   exits: {K}   qp: {a.qp}   tile {cfg.rgb_patch}px")

    # ---- 1. does the oracle vary? -------------------------------------
    # dB penalty of each exit, per tile, against that tile's own full decode.
    db = 10 * torch.log10(mses / mses[:, -1:].clamp_min(1e-12))
    print(f"\nper-tile dB penalty by exit (mean +- std across tiles):")
    for k in range(K):
        print(f"  exit {k}: {db[:, k].mean():+7.3f} +- {db[:, k].std():5.3f}"
              f"   (saving {100 * (1 - costs[k].item()):5.1f}%)")

    print(f"\nORACLE: cheapest exit within tau dB of the full decode")
    print(f"  {'tau':>6} {'distinct':>9} {'share over exits':>34} {'saving':>8}")
    for tau in (0.1, 0.3, 0.5, 1.0, 2.0):
        ok = db <= tau                       # [T, K]
        ok[:, -1] = True                     # the deepest always qualifies
        choice = ok.float().argmax(dim=1)    # first (cheapest) qualifying exit
        share = torch.bincount(choice, minlength=K)
        frac = (share.float() / share.sum()).tolist()
        distinct = int((share > 0).sum())
        sv = 100 * (1 - costs[choice].mean().item())
        print(f"  {tau:>6.1f} {distinct:>9} "
              f"{'[' + ' '.join(f'{v:.2f}' for v in frac) + ']':>34} {sv:>7.1f}%")

    # ---- 2. what is the headroom worth? -------------------------------
    # Compare at MATCHED ACHIEVED QUALITY, not at a matched constraint.
    #
    # The first version of this compared the oracle under a per-tile constraint
    # ("every tile within tau") against uniform under a mean constraint ("the
    # average is within tau"), and unsurprisingly made routing look worse than
    # uniform at loose budgets. That is impossible on the merits — the oracle can
    # always imitate a uniform choice — and it was purely an artefact of the two
    # sides solving different problems.
    #
    # So: sweep tau to trace the oracle's (achieved mean dB, saving) curve, take
    # the uniform curve as the K discrete points (mean_db[k], 1-cost[k]), and
    # read the uniform curve at the oracle's achieved dB by linear interpolation.
    # The gap between them at equal quality is what routing actually buys, and it
    # is the number that decides whether the mechanism earns its complexity.
    mean_db = db.mean(dim=0)
    uni = sorted((mean_db[k].item(), 100 * (1 - costs[k].item())) for k in range(K))

    def uniform_saving_at(target_db: float) -> float:
        """Saving of the best uniform depth achieving `target_db` mean penalty."""
        if target_db <= uni[0][0]:
            return uni[0][1]
        if target_db >= uni[-1][0]:
            return uni[-1][1]
        for (d0, s0), (d1, s1) in zip(uni, uni[1:]):
            if d0 <= target_db <= d1:
                t = 0.0 if d1 == d0 else (target_db - d0) / (d1 - d0)
                return s0 + t * (s1 - s0)
        return uni[-1][1]

    # The TRUE upper bound is the Lagrangian-optimal per-tile allocation.
    #
    # The tau-thresholded oracle above ("cheapest exit whose penalty is within
    # tau") is a heuristic, not the Pareto frontier of (mean dB, saving). It is
    # a *constraint* satisfier: it refuses any tile above tau even when letting
    # one tile go slightly further would buy a lot of compute elsewhere. A
    # router minimising a Lagrangian is not bound by that and can legitimately
    # beat it — which is exactly what happened, the measured router landing at
    # 56.36% / 0.278 dB against the tau-oracle's 55.9% / 0.351 dB. Reporting the
    # tau-oracle as "the upper bound" would have been wrong.
    #
    # The real bound: for each tile independently choose k minimising
    # mse_k + lam * cost_k, and sweep lam. Because the tiles are independent and
    # the cost is additive, that sweep traces the exact Pareto frontier.
    print(f"\nTRUE UPPER BOUND: Lagrangian-optimal per-tile allocation")
    print(f"  {'lambda':>10} {'mean dB':>9} {'saving':>9}")
    c = costs.to(mses.device)
    for lam in (0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1):
        k = (mses + lam * c[None, :]).argmin(dim=1)
        got = db.gather(1, k[:, None]).mean().item()
        sv = 100 * (1 - c[k].mean().item())
        print(f"  {lam:>10.0e} {got:>9.3f} {sv:>8.1f}%")

    print(f"\nHEADROOM: routing vs a uniformly shallower decoder, at EQUAL quality")
    print(f"  {'tau':>6} {'achieved dB':>12} {'oracle':>9} {'uniform':>9} {'gain':>9}")
    for tau in (0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0):
        ok = db <= tau
        ok[:, -1] = True
        choice = ok.float().argmax(dim=1)
        got_db = db.gather(1, choice[:, None]).mean().item()
        sv = 100 * (1 - costs[choice].mean().item())
        u = uniform_saving_at(got_db)
        print(f"  {tau:>6.2f} {got_db:>11.3f}  {sv:>8.1f}% {u:>8.1f}% {sv - u:>+8.1f}pp")
    print("  (gain > 0 means per-tile routing beats any single depth at that quality)")

    # ---- 3. can the signals see it? -----------------------------------
    print(f"\nSIGNALS vs oracle choice (tau=0.5): Pearson r")
    ok = db <= 0.5
    ok[:, -1] = True
    choice = ok.float().argmax(dim=1).float()
    names = ["s1 rate-surrogate", "s2 sparsity", "s3 gradient", "s4 spatial-var"]
    for i, nm in enumerate(names):
        s = sigs[:, i]
        if s.std() < 1e-9 or choice.std() < 1e-9:
            print(f"  {nm:<20} r = n/a (constant)   [std {s.std():.3e}]")
            continue
        r = torch.corrcoef(torch.stack([s, choice]))[0, 1].item()
        print(f"  {nm:<20} r = {r:+.3f}   [range {s.min():.3g} .. {s.max():.3g}]")

    # Exit code is the point of this script when it runs unattended: it gates
    # the frontier sweep, which costs about an hour of GPU time on a card that
    # is also training. Measuring a frontier on a checkpoint with no headroom
    # produces a table of zeros and teaches nothing.
    if choice.std() < 1e-9:
        print("\n  => VERDICT: no headroom. The ORACLE is constant, so no router")
        print("     could help here and a uniformly shallower decoder would be the")
        print("     honest answer. Expected before the auxiliary losses switch on:")
        print("     with alpha=0 the shallow exits are never trained, so their")
        print("     adapters sit at zero-init and their features were never shaped")
        print("     for the head. Re-check once alpha has ramped.")
        return 1
    print(f"\n  => VERDICT: headroom exists (oracle std {choice.std():.3f} exits).")
    print("     A constant router would now be the router's failure, not the")
    print("     content's, and the frontier is worth measuring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
