"""Train router v2 to AGREE with the oracle, not merely to regret less.

Objective, in the order the literature suggests and measurement confirmed:

  cost-sensitive CE to the oracle's choice   the direct signal for agreement,
                                             weighted by what each decision costs
  + alpha * expected regret                  keeps it honest: agreement on ties
                                             is worth nothing, regret says so
  + loss-free balancing bias                 collapse control WITHOUT an
                                             auxiliary loss, so no interference
                                             gradient enters the objective
                                             (arXiv:2408.15664)

Held-out agreement is reported every log line, on tiles the router never trained
on. In-sample agreement on 144k parameters and a few thousand tiles would climb
to anything at all and mean nothing -- the first version of the signal probe did
exactly that, reporting 1.000 on data it had fitted.

--inputs runs the input ablation
-------------------------------
The parameter-free rule beats this head, so the reviewer's question is which of
the head's inputs carries any signal at all. `--inputs` names the groups that
stay live; the rest are zeroed after their projection, which leaves the
parameter count, the architecture and the optimiser identical and changes only
what the head is allowed to know. Naming it at all switches the bits pathway on,
so every variant of the ablation is one architecture; omitting it reproduces the
paper's head exactly. See flexuf/router/head2.py and scripts/router_ablation.sh.

The last log line's agreement is 12 held-out tiles per step, which is far too
few to separate five variants. So the run ends with `--eval_steps` batches drawn
FRESH from the same shuffled epoch -- images no step of this run has touched,
379,614 of them against the 18,000 training consumes -- and reports agreement
with a standard error taken across frames, plus what the best single constant
exit would have scored on the same tiles. That last number is the floor: a
variant that does not beat it has learned nothing about individual tiles.
"""
import argparse, json, math, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path.home() / "DCVC"))
from torch.utils.data import DataLoader
from src.datasets.image_dataset import ImageFolder
from src.utils.common import get_training_lambdas
from flexuf.config import QP_LEVELS, FlexUFConfig
from flexuf.cost import exit_costs
from flexuf.eval import tiled_exit_mses
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.router.head2 import StemRouterHeadV2, oracle_ce_loss
from flexuf.router.losses import regret_objective

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--lam", type=float, required=True)
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--batch_size", type=int, default=6)
ap.add_argument("--crop", type=int, default=512)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--alpha", type=float, default=1.0)
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--inputs", default=None,
                help="ablation: comma-separated live input groups "
                     "(stem,latent,scales,bits,qp) or 'all'. Omit for the "
                     "paper's head, which has no bits pathway at all.")
ap.add_argument("--label", default=None,
                help="name this variant carries in the results JSON")
ap.add_argument("--eval_steps", type=int, default=100,
                help="batches of unseen images to measure agreement on at the "
                     "end; 0 skips it")
ap.add_argument("--seed", type=int, default=0,
                help="fixes the head's init, the image order and the qp draws, "
                     "so two variants differ only in their inputs")
ap.add_argument("--results", default=None,
                help="results/*.json to write, with the checkpoint and its "
                     "epoch, so provenance survives the run")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = a.device
torch.manual_seed(a.seed)

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
for p in net.parameters():
    p.requires_grad = False
head = StemRouterHeadV2(384, 256, cfg.num_exits, min_exit=cfg.split_depth,
                        inputs=a.inputs).to(dev)
n_param = sum(p.numel() for p in head.parameters())
opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
cost = exit_costs(cfg, "head").to(dev)
print(f"  v2 basligi {n_param:,} param, "
      f"decoder donuk, lam={a.lam:g}, {a.steps} adim", flush=True)
print(f"  girdiler: {head.inputs}   (bits yolu {'var' if head.with_bits else 'yok'})"
      f"   ckpt {a.ckpt} epoch {ck.get('epoch')}", flush=True)

D = "/data10/shareddata/openimages/dcvc_train"
ds = ImageFolder(D, a.crop, a.crop, QP_LEVELS, get_training_lambdas([10., 2048.], QP_LEVELS))
ld = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=5, drop_last=True)

t0, step, n_ho = time.time(), 0, 0
it = iter(ld)


def next_batch():
    """The next batch, reopening the epoch only if it actually runs out.

    An explicit iterator rather than a `for` over the loader, because the final
    measurement keeps drawing from the SAME shuffled epoch: 379,614 images
    against the 18,000 training consumes, so what it draws is images this run
    has never seen. That is what makes the closing number a held-out one rather
    than a re-read of the training set.
    """
    global it
    try:
        return next(it)
    except StopIteration:
        it = iter(ld)
        return next(it)


def decode_batch(b):
    """Everything the FROZEN decoder contributes to one step."""
    x = b[0].to(dev); B = x.shape[0]
    qp = torch.randint(0, 64, (B,), dtype=torch.int32, device=dev)
    with torch.no_grad():
        y, q, aux = net._encode_to_latent(x, qp)
        # The oracle the router imitates must be the oracle for the DEPLOYED
        # decode. dec.forward_all_exits runs full frame, so a router trained
        # against it learns to predict the best exit for a decoder that is
        # not the one running at inference -- and the error is not uniform
        # across exits, it grows with how many blocks ran per tile.
        M = tiled_exit_mses(net.dec, y, q, x, cfg)
        stem = net.dec.upsample(y)
        for g in range(cfg.split_depth):
            stem = net.dec.groups[g](stem)
        sc = aux["scales_hat"]
        bits = None
        if head.with_bits:
            # Built exactly as scripts/raterank_curve.py builds it, so the
            # learned head and the parameter-free rule are reading one signal
            # and not two that merely have the same name.
            bits = net.get_y_bits(net.add_noise(aux["y_res"]), aux["scales_hat"])
    return x, qp, y, sc, M, stem, bits


def oracle_labels(M):
    """The oracle's own choice per tile, at this lambda.

    Columns below the split depth are duplicates of exit j -- same decode, same
    cost -- so argmin can land on one arbitrarily. The head can no longer emit
    them at all, so an unclamped label would score every such tile as a
    disagreement.
    """
    return (M + a.lam * cost[None, :]).argmin(1).clamp(min=cfg.split_depth)


hist = []
while step < a.steps:
    x, qp, y, sc, M, stem, bits = decode_batch(next_batch())
    # Half the tiles train, half are held out. Agreement is only ever
    # reported on the half the optimiser never saw.
    n = M.shape[0]; tr = torch.arange(n, device=dev) % 2 == 0
    n_ho = int((~tr).sum())
    head.train()
    logits = head(stem, y, sc, qp, cfg.feature_patch, cfg.latent_patch, bits=bits)
    ce, k_star, _ = oracle_ce_loss(logits[tr], M[tr], cost, a.lam,
                                   min_exit=cfg.split_depth)
    reg = regret_objective(M[tr], logits[tr], cost, lam=a.lam, hard=True)
    loss = ce + a.alpha * reg["loss"]
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
    with torch.no_grad():
        kk = oracle_labels(M)
        head.rebalance(logits.argmax(1)[tr], torch.bincount(kk, minlength=cfg.num_exits).float() / n)
        ho = (logits.argmax(1)[~tr] == kk[~tr]).float().mean().item()
    hist.append(ho)
    if step % 250 == 0:
        print(f"    s{step:<5} CE {ce.item():.4f}  regret {reg['loss'].item():.5f}  "
              f"AYRILMIS agree {ho:.3f}  dagilim "
              f"{torch.bincount(logits.argmax(1), minlength=cfg.num_exits).tolist()}  "
              f"{time.time()-t0:.0f}s", flush=True)
    step += 1

final = sum(hist[-50:]) / len(hist[-50:])
final250 = sum(hist[-250:]) / len(hist[-250:])
print(f"  AYRILMIS agreement (son 50 adim ortalamasi): {final:.3f}")

ev = None
if a.eval_steps > 0:
    # The measurement the ablation is actually read off: fresh images, every
    # tile counted, and a spread quoted across FRAMES rather than tiles because
    # tiles of one image are not independent draws.
    head.eval()
    per_frame, pred_h, orc_h, n_tiles = [], torch.zeros(cfg.num_exits), torch.zeros(cfg.num_exits), 0
    with torch.no_grad():
        for _ in range(a.eval_steps):
            x, qp, y, sc, M, stem, bits = decode_batch(next_batch())
            logits = head(stem, y, sc, qp, cfg.feature_patch, cfg.latent_patch, bits=bits)
            pred, kk = logits.argmax(1), oracle_labels(M)
            per_frame.append((pred == kk).float().view(x.shape[0], -1).mean(1))
            pred_h += torch.bincount(pred, minlength=cfg.num_exits).float().cpu()
            orc_h += torch.bincount(kk, minlength=cfg.num_exits).float().cpu()
            n_tiles += pred.numel()
    fr = torch.cat(per_frame)
    se = float(fr.std(unbiased=True) / math.sqrt(fr.numel()))
    ev = {"agree": float(fr.mean()), "stderr_across_frames": se,
          "n_frames": int(fr.numel()), "n_tiles": int(n_tiles),
          "batches": a.eval_steps,
          # What the best single constant exit would have scored on these same
          # tiles. A variant that does not clear this has learned nothing about
          # individual tiles, whatever its agreement looks like on its own.
          "constant_best_agree": float(orc_h.max() / orc_h.sum().clamp_min(1)),
          "pred_hist": pred_h.tolist(), "oracle_hist": orc_h.tolist(),
          "note": "fresh draws from the same shuffled epoch, never trained on"}
    print(f"  GORULMEMIS agreement {ev['agree']:.4f} +- {se:.4f} over "
          f"{ev['n_frames']} frames / {ev['n_tiles']} tiles   "
          f"(sabit en iyi cikis {ev['constant_best_agree']:.4f})")
    print(f"  dagilim  router {[int(v) for v in ev['pred_hist']]}  "
          f"oracle {[int(v) for v in ev['oracle_hist']]}")

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"router_head_v2": head.state_dict(), "lam": a.lam, "ckpt": a.ckpt,
            "ckpt_epoch": ck.get("epoch"), "inputs": head.inputs,
            "with_bits": head.with_bits, "heldout_agree": final,
            "eval": ev}, a.out)
print(f"  wrote {a.out}")

if a.results:
    # Provenance in the result file itself: which checkpoint, and which EPOCH of
    # it, because a checkpoint path alone can name different weights on
    # different days.
    res = {"script": "scripts/train_router2.py",
           "label": a.label or (head.inputs if a.inputs else "paper"),
           "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
           "router_ckpt": a.out, "inputs": head.inputs,
           "with_bits": head.with_bits, "router_params": n_param,
           "lam": a.lam, "steps": a.steps, "batch_size": a.batch_size,
           "crop": a.crop, "lr": a.lr, "alpha": a.alpha, "seed": a.seed,
           "device": a.device,
           # The device string alone is a lie under CUDA_VISIBLE_DEVICES, which
           # is how these are launched: cuda:0 inside the process is whichever
           # physical GPU the mask exposed.
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
           "gpu_name": (torch.cuda.get_device_name(dev)
                        if str(dev).startswith("cuda") else None),
           "tile_px": cfg.rgb_patch,
           "split_depth": cfg.split_depth, "num_exits": cfg.num_exits,
           "heldout_agree_last50": final, "heldout_agree_last250": final250,
           "heldout_tiles_per_step": n_ho, "eval": ev,
           "wall_s": time.time() - t0,
           "finished": time.strftime("%Y-%m-%dT%H:%M:%S")}
    Path(a.results).parent.mkdir(parents=True, exist_ok=True)
    Path(a.results).write_text(json.dumps(res, indent=2))
    print(f"  wrote {a.results}")
