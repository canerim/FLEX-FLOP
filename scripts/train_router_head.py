"""Train the REAL router: one decision per tile, taken before the deep blocks run.

Every number reported until now assumed an oracle -- a router that reads each
tile's true error at every exit and takes the argmin. No decoder can do that:
knowing the error at exit k means having decoded to exit k, which is the cost
being avoided. This trains the deployable thing.

    stem (already computed, groups 0..j-1)
      -> 1x1 conv 384->16          learned view, 0.044% of the decode
      -> per-tile (mean, std)      one 32-vector per tile
      -> + QP -> MLP -> K logits   8,726 parameters
      -> argmax                    the decision, before any deep block runs

The decoder is frozen throughout. That is not a shortcut, it is the fix for a
failure FLEX-FLOP hit: a router chasing a moving decoder goes stale, stops
feeding an exit, that exit stops improving, and the ladder collapses. With the
decoder fixed the router solves a stationary problem.

Objective is expected regret against the Lagrangian optimum (see
flexuf/router/losses.py): zero exactly when the router picks argmin(mse + lam*C),
and its value IS the excess cost being paid, in the frontier's own units. lam
sets the operating point; sweeping it traces the frontier.
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
from flexuf.model import FlexUFIntra, load_flexuf_state
from flexuf.router.losses import regret_objective

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--lam", type=float, required=True)
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--batch_size", type=int, default=6)
ap.add_argument("--crop", type=int, default=512)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--tau", type=float, default=1.0)
ap.add_argument("--device", default="cuda:2")
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = a.device

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
net = FlexUFIntra(cfg).to(dev); load_flexuf_state(net, ck)
net.eval()
for p in net.parameters():
    p.requires_grad = False
head = net.router_head
for p in head.parameters():
    p.requires_grad = True
head.train()
opt = torch.optim.AdamW(head.parameters(), lr=a.lr)
cost = exit_costs(cfg, "head").to(dev)
print(f"  router basligi {sum(p.numel() for p in head.parameters()):,} parametre, "
      f"decoder donuk, lam={a.lam:g}", flush=True)

D = "/data10/shareddata/openimages/dcvc_train"
ds = ImageFolder(D, a.crop, a.crop, QP_LEVELS, get_training_lambdas([10., 2048.], QP_LEVELS))
ld = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=5, drop_last=True)

P, t0, step = cfg.rgb_patch, time.time(), 0
while step < a.steps:
    for b in ld:
        if step >= a.steps:
            break
        x = b[0].to(dev)
        B = x.shape[0]
        qp = torch.randint(0, 64, (B,), dtype=torch.int32, device=dev)
        with torch.no_grad():
            y, q, _ = net._encode_to_latent(x, qp)
            nh, nw = x.shape[-2] // P, x.shape[-1] // P
            outs = net.dec.forward_all_exits(y, q)
            M = torch.stack([
                (((o - x) ** 2).mean(1).view(B, nh, P, nw, P)
                 .permute(0, 1, 3, 2, 4).reshape(B * nh * nw, P * P).mean(1))
                for o in outs], 1)
            stem = net.dec.upsample(y)
            for g in range(cfg.split_depth):
                stem = net.dec.groups[g](stem)
        obj = regret_objective(M, head(stem, qp, cfg.feature_patch), cost,
                               lam=a.lam, tau=a.tau, hard=True)
        opt.zero_grad(set_to_none=True)
        obj["loss"].backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"    s{step:<5} regret {obj['loss'].item():.5f}  "
                  f"oracle_agree {obj['oracle_agree'].item():.3f}  "
                  f"dagilim {obj['exit_share'].mul(100).round().int().tolist()}  "
                  f"{time.time()-t0:.0f}s", flush=True)
        step += 1

Path(a.out).parent.mkdir(parents=True, exist_ok=True)
torch.save({"router_head": head.state_dict(), "lam": a.lam, "ckpt": a.ckpt}, a.out)
print(f"  wrote {a.out}")
