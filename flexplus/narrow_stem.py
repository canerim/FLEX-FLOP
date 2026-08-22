"""A slimmable stand-in for the always-on stem, trained to match it.

The ceiling is 39.12% because a frame with every tile on its shallowest rung
still pays 0.6088 of a released decode, and 0.2981 of that -- 49% -- is the
four trunk blocks below the split. They run full-frame, before tiling, so no
exit can skip them. flexplus/PLAN.md has the arithmetic; halving their cost
puts the ceiling at 54.0%, quartering it at 61.5%.

flexplus/stem_tolerance.py asked whether the stem can simply be cut, and the
answer was no: untrained, dropping a quarter of its channels costs 0.50 dB at
the lowest rate and 2.21 dB at the highest, against a 0.2 dB budget. But an
untrained slice is the wrong measurement of a slimmable design. Slimmable
Networks (Yu et al., ICLR 2019) trains one network at several widths at once,
sharing weights, and reports each width close to a network trained at that
width alone; Dynamic Slimmable Network (CVPR 2021) makes the width
input-dependent. The claim under test is that the gap closes with training.

So: build a narrow stem of width w, plug it in place of the four blocks, and
train it to reproduce the full stem's output feature. Nothing else moves --
the encoder, the entropy model, the rest of the trunk, the exits and the head
are all frozen -- so the target is a pure regression and any quality change is
attributable to the stem alone.

    python flexplus/narrow_stem.py --width 0.5 --steps 4000 --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(ROOT / "scripts"))

from flexuf.config import FlexUFConfig                      # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state     # noqa: E402
from flexuf.reference import reference_for                  # noqa: E402


class NarrowStem(nn.Module):
    """The four stem blocks at width w, with 1x1 projections either side.

    Same operator sequence as the block it replaces, at int(w*C) channels, so
    its cost is w^2 of the original plus two pointwise projections. The
    projections are what make it drop-in: everything downstream still sees C
    channels and needs no change at all.

    Residual around the whole thing and the output projection zero-initialised,
    so at step zero the module is the identity and the decode is exactly the
    one that skips the stem -- a known, measurable starting point rather than
    noise.
    """

    def __init__(self, channels: int, width: float, blocks: int = 4):
        super().__init__()
        c = max(8, int(round(channels * width / 8) * 8))
        self.c_in, self.c = channels, c
        self.down = nn.Conv2d(channels, c, 1)
        body = []
        for _ in range(blocks):
            body += [nn.Conv2d(c, c, 3, padding=1, groups=c),
                     nn.SiLU(),
                     nn.Conv2d(c, c, 1),
                     nn.SiLU(),
                     nn.Conv2d(c, c, 1)]
        self.body = nn.Sequential(*body)
        self.up = nn.Conv2d(c, channels, 1)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(self.body(self.down(x)))

    def macs_per_px(self):
        """MAC per feature pixel, for the cost the ceiling is computed from."""
        c, C = self.c, self.c_in
        per_block = c * 9 + c * c + c * c          # depthwise 3x3 + two 1x1
        return C * c + 4 * per_block + c * C


def stem_full(dec, y_hat, j):
    feat = dec.upsample(y_hat)
    for g in range(j):
        feat = dec.groups[g](feat)
    return feat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--width", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--train_dataset",
                    default="/data10/shareddata/openimages/dcvc_train")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None)
    # Matching the stem's output feature is a proxy, and measurement says it
    # is a poor one: a 5% relative error in the feature came out as 5 to 10
    # times the dB at the exit, because the three trunk blocks below the split
    # amplify stem error rather than absorbing it. "decode" trains on the
    # thing that is actually reported -- the reconstruction at an exit --
    # which is the same quantity the budget is measured in.
    ap.add_argument("--objective", choices=("feature", "decode"),
                    default="feature")
    a = ap.parse_args()
    dev = a.device
    tag = a.tag or (f"w{a.width:g}" if a.objective == "feature"
                    else f"w{a.width:g}_decode")

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    for p in net.parameters():
        p.requires_grad_(False)
    j = cfg.split_depth

    # Channel count of the stem's output, read off the module rather than
    # assumed.
    with torch.no_grad():
        probe = torch.zeros(1, 3, 256, 256, device=dev)
        qp = torch.zeros(1, dtype=torch.int32, device=dev)
        y, q, _ = net._encode_to_latent(probe, qp)
        C = stem_full(net.dec, y, j).shape[1]

    K = cfg.num_exits
    narrow = NarrowStem(C, a.width).to(dev)
    n_par = sum(p.numel() for p in narrow.parameters())
    full_par = sum(p.numel() for k in range(j)
                   for p in net.dec.groups[k].parameters())
    print(f"  stem width {a.width:g}: {narrow.c}/{C} channels, "
          f"{n_par/1e6:.2f} M parameters against the full stem's "
          f"{full_par/1e6:.2f} M")

    # The same loader the main training uses, so the crops, the colour
    # conversion and the qp sampling are identical to what produced the
    # checkpoint this is fitted against.
    from src.datasets.image_dataset import ImageFolder      # noqa: E402
    ds = ImageFolder(a.train_dataset, a.crop, a.crop, 64, [1.0] * 64)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True,
        pin_memory=True)
    opt = torch.optim.Adam(narrow.parameters(), lr=a.lr)
    log = HERE / f"logs/narrow_{tag}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)

    step, t0 = 0, time.time()
    while step < a.steps:
        for x in dl:
            if step >= a.steps:
                break
            img, _qp, _lam = x
            img = img.to(dev, non_blocking=True)
            # The loader samples a qp per image; the stem is shared across
            # rates, so sweeping the three the paper reports keeps the fit
            # from specialising on one of them.
            qp_v = a.qps[step % len(a.qps)]
            qp = torch.full((img.shape[0],), qp_v, dtype=torch.int32,
                            device=dev)
            x = img
            with torch.no_grad():
                y, q, _ = net._encode_to_latent(x, qp)
                base = net.dec.upsample(y)
                target = (stem_full(net.dec, y, j)
                          if a.objective == "feature" else None)
            if a.objective == "feature":
                loss = F.mse_loss(narrow(base), target)
                rel_den = target.pow(2).mean()
            else:
                # Every exit each step, not one sampled from them. Sampling
                # gave each exit a quarter of the updates the feature
                # objective gives all of them at once, so a run matched on
                # steps was still not matched on updates per exit -- and the
                # decode objective lost the comparison by about that much.
                feat = narrow(base)
                loss = 0.0
                for g in range(j, K):
                    feat = net.dec.groups[g](feat)
                    rec = net.dec._apply_head(net.dec._at_exit(feat, g), q)
                    loss = loss + F.mse_loss(rec, x)
                loss = loss / (K - j)
                rel_den = x.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % 100 == 0 or step == 1:
                rel = (loss / rel_den).item()
                rec = {"step": step, "loss": loss.item(),
                       "relative_mse": rel, "qp": qp_v,
                       "sec_per_step": (time.time() - t0) / step}
                with open(log, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"  {step:>6}  mse {loss.item():.6f}  "
                      f"relative {rel:.5f}  qp {qp_v}", flush=True)

    out_p = HERE / f"results/narrow_stem_{tag}.pth"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": narrow.state_dict(), "width": a.width,
                "channels": narrow.c, "channels_full": C,
                "macs_per_px": narrow.macs_per_px(),
                "ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
                "steps": a.steps, "objective": a.objective}, out_p)
    print(f"  wrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
