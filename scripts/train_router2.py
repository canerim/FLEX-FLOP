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
"""
import argparse, json, sys, time
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
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
for p in net.parameters():
    p.requires_grad = False
head = StemRouterHeadV2(384, 256, cfg.num_exits, min_exit=cfg.split_depth).to(dev)
opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
cost = exit_costs(cfg, "head").to(dev)
print(f"  v2 basligi {sum(p.numel() for p in head.parameters()):,} param, "
      f"decoder donuk, lam={a.lam:g}, {a.steps} adim", flush=True)

D = "/data10/shareddata/openimages/dcvc_train"
ds = ImageFolder(D, a.crop, a.crop, QP_LEVELS, get_training_lambdas([10., 2048.], QP_LEVELS))
ld = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=5, drop_last=True)

P, t0, step = cfg.rgb_patch, time.time(), 0
hist = []
while step < a.steps:
    for b in ld:
        if step >= a.steps:
            break
        x = b[0].to(dev); B = x.shape[0]
        qp = torch.randint(0, 64, (B,), dtype=torch.int32, device=dev)
        with torch.no_grad():
            y, q, aux = net._encode_to_latent(x, qp)
            nh, nw = x.shape[-2] // P, x.shape[-1] // P
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
        # Half the tiles train, half are held out. Agreement is only ever
        # reported on the half the optimiser never saw.
        n = M.shape[0]; tr = torch.arange(n, device=dev) % 2 == 0
        head.train()
        logits = head(stem, y, sc, qp, cfg.feature_patch, cfg.latent_patch)
        ce, k_star, _ = oracle_ce_loss(logits[tr], M[tr], cost, a.lam,
                                       min_exit=cfg.split_depth)
        reg = regret_objective(M[tr], logits[tr], cost, lam=a.lam, hard=True)
        loss = ce + a.alpha * reg["loss"]
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0); opt.step(); sched.step()
        with torch.no_grad():
            lag = M + a.lam * cost[None, :]
            kk = lag.argmin(1)
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
print(f"  AYRILMIS agreement (son 50 adim ortalamasi): {final:.3f}")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"router_head_v2": head.state_dict(), "lam": a.lam, "ckpt": a.ckpt,
            "heldout_agree": final}, a.out)
print(f"  wrote {a.out}")
