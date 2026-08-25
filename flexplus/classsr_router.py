"""A ClassSR-style Class-Module for the exit ladder, trained without labels.

ClassSR (Kong et al., CVPR 2021) splits an image into sub-images, sends each to
one of M super-resolution branches of different width, and picks the branch
with a small classification network -- five convolutions, an average pool and a
fully-connected layer -- trained with three losses: an Image-Loss for quality,
a Class-Loss that sharpens the probability vector, and an Average-Loss that
stops every sub-image collapsing onto the most expensive branch.

The structure maps onto this decoder almost exactly. Sub-image -> cell; SR
branch -> exit; branch width -> trunk depth. What differs is where the
supervision comes from, and that is the point of running it:

  our routers are told the answer. They regress the measured per-cell error
  m(x,k), tabulated from K full-frame decodes, and the allocation follows from
  a Lagrangian over that table.

  the Class-Module is not. It sees only the reconstruction cost of the branch
  it chose, so the classes are DISCOVERED. ClassSR needs its two extra losses
  precisely because nothing else stops the discovery degenerating.

Two more differences, both deliberate. The input is the cell's content -- the
stem feature the trunk is about to consume -- rather than four hand-made
scalars, which is what ClassSR's Class-Module reads and what
flexuf/router/head.py had to invent a 1x1 projection to approximate. And the
architecture is ClassSR's, not ours.

The trade-off is swept the way ClassSR would sweep it: by retraining at
several lambda in the RD term, since a Class-Module commits to a hard class
and has no test-time knob. Each run contributes one point to its own frontier,
which is then read at the paper's 0.1 dB budget on the TRUE error.

Nothing under ~/FLEX-UF is modified.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(UF)); sys.path.insert(0, str(UF / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C                                          # noqa: E402
from flexuf.config import FlexUFConfig                         # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state        # noqa: E402
from flexuf.reference import reference_for                     # noqa: E402
sys.path.insert(0, str(HERE))
from adaptive_depth import dilate_depth                        # noqa: E402


class ClassModule(nn.Module):
    """ClassSR's Class-Module: five convolutions, average pool, one linear.

    Kept at ClassSR's shape rather than tuned, so the comparison is about the
    method and not about who searched harder. Width 32 puts it at 116k
    parameters against the shipped head's 148k, which keeps the router's own
    cost in the same place in the accounting.
    """

    def __init__(self, in_ch: int, num_exits: int, width: int = 32,
                 min_exit: int = 0):
        super().__init__()
        w = width
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(w, w, 3, 1, 1), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w, num_exits)
        self.min_exit = min_exit
        self.num_exits = num_exits

    def forward(self, x):
        h = self.pool(self.body(x)).flatten(1)
        logits = self.fc(h)
        if self.min_exit > 0:
            # Exits below the split decode identically to the split, so leaving
            # their logits live would let the Average-Loss spend mass on
            # distinctions the decoder cannot make. The shipped ExitRouter
            # masks them for the same reason.
            logits = logits.clone()
            logits[:, :self.min_exit] = -1e4
        return logits


def class_loss(p):
    """ClassSR Eq. (2): reward a peaked probability vector.

    Lc = - sum_{i<j} |P_i - P_j|, so minimising it maximises the spread
    between the largest probability and the rest. Without it the module is
    free to output a near-uniform vector, which at test time -- where only the
    argmax branch runs -- is a coin flip wearing a distribution.
    """
    d = (p[:, :, None] - p[:, None, :]).abs()
    m = torch.triu(torch.ones_like(d[0]), diagonal=1)
    return -(d * m[None]).sum(dim=(1, 2)).mean()


def average_loss(p, active):
    """ClassSR Eq. (3): keep the batch's class usage even.

    La = sum_i | sum_batch P_i - B/M |. Without it, Image-Loss alone drives
    every cell to the most expensive branch, which is the degenerate solution
    this decoder would also find: the deepest exit is always the best
    reconstruction.
    """
    b = p.shape[0]
    target = b / max(1, len(active))
    return (p[:, active].sum(0) - target).abs().sum() / b


def cells_of(feat, cell_feat):
    """[1,C,H,W] stem -> [N, C, cell, cell] in the row-major cell order."""
    _, ch, H, W = feat.shape
    nh, nw = H // cell_feat, W // cell_feat
    x = feat[:, :, :nh * cell_feat, :nw * cell_feat]
    x = x.reshape(1, ch, nh, cell_feat, nw, cell_feat)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(nh * nw, ch, cell_feat, cell_feat)
    return x, nh, nw


def cell_mse(img, target, cell):
    e = ((img - target) ** 2).mean(1, keepdim=True)
    return F.avg_pool2d(e, cell, cell)[0, 0].reshape(-1)


def frames_openimages(n, crop, dev):
    import numpy as _np
    from PIL import Image
    from src.utils.transforms import rgb2ycbcr_np
    root = Path("/data10/shareddata/openimages/dcvc_train")
    out = []
    for sub in sorted(root.glob("train_*")):
        for p in sorted(sub.iterdir()):
            if len(out) >= n:
                return out
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            if im.width < crop or im.height < crop:
                continue
            a = _np.array(im).astype(_np.float32) / 255.0
            i = torch.randint(0, a.shape[0] - crop + 1, (1,)).item()
            j = torch.randint(0, a.shape[1] - crop + 1, (1,)).item()
            a = a[i:i + crop, j:j + crop]
            out.append(torch.as_tensor(rgb2ycbcr_np(a) - 0.5,
                                       dtype=torch.float32).permute(2, 0, 1)[None])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PAPER.pth.tar"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--images", type=int, default=400)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lam", type=float, nargs="+",
                    default=[3e-4, 1e-3, 3e-3, 1e-2, 3e-2])
    ap.add_argument("--w_class", type=float, default=0.05)
    ap.add_argument("--w_avg", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--split", type=int, default=None)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--out", default=str(HERE / "results/classsr_router.json"))
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    for p in net.parameters():
        p.requires_grad_(False)
    K = cfg.num_exits
    j = cfg.split_depth if a.split is None else a.split
    stride = cfg.rgb_patch // cfg.feature_patch
    cf = a.cell // stride
    cost_k = torch.tensor([(k + 1) * cfg.blocks_per_exit for k in range(K)],
                          dtype=torch.float32, device=dev)
    cc = json.loads((HERE / "results/cost_constants.json").read_text())
    UPS, TRUNK, HEAD = cc["SHARE_UPSAMPLE"], cc["SHARE_TRUNK"], cc["SHARE_HEAD"]
    NBLK = cc["N_TRUNK_BLOCKS"]
    ADAPT = torch.tensor(cc["adapter"], dtype=torch.float32)
    print(f"  K={K} j={j} hucre={a.cell}px -> {cf}x{cf} stem yamasi", flush=True)

    torch.manual_seed(0)
    train = frames_openimages(a.images, a.crop, dev)
    print(f"  {len(train)} egitim goruntusu", flush=True)
    seqs, _ = C.discover([])
    test = []
    for s in seqs:
        x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
        if x is not None:
            test.append(x[0:1])
    print(f"  {len(test)} CTC karesi", flush=True)

    @torch.no_grad()
    def tables(x, qp_v):
        """Per-cell stem patches, per-exit MSE, and the reference MSE."""
        x = x.to(dev)
        ph, pw = (-x.shape[-2]) % a.cell, (-x.shape[-1]) % a.cell
        xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
        qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
        y, q, _ = net._encode_to_latent(xp, qp)
        stem = net.dec.upsample(y)
        recs = net.dec.forward_all_exits(y, q)
        rr = ref.dec.forward_full(y, q)
        cells, nh, nw = cells_of(stem, cf)
        M = torch.stack([cell_mse(r, xp, a.cell) for r in recs], 1)
        R = cell_mse(rr, xp, a.cell)
        n = min(cells.shape[0], M.shape[0])
        return cells[:n], M[:n], R[:n], nh, nw

    results = {"ckpt": a.ckpt, "cell_px": a.cell, "split_depth": j,
               "w_class": a.w_class, "w_avg": a.w_avg, "runs": []}
    active = list(range(j, K))

    for lam in a.lam:
        torch.manual_seed(0)
        cm = ClassModule(stem_ch(net, dev), K, min_exit=j).to(dev)
        npar = sum(p.numel() for p in cm.parameters())
        opt = torch.optim.Adam(cm.parameters(), lr=a.lr)
        t0 = time.time()
        for ep in range(a.epochs):
            for x in train:
                qv = int(a.qps[torch.randint(0, len(a.qps), (1,)).item()])
                cells, M, R, _, _ = tables(x, qv)
                logits = cm(cells)
                p = logits.softmax(1)
                # Image-Loss analogue: the expected Lagrangian under P. ClassSR
                # backpropagates the L1 of the chosen branch; here the cost side
                # is explicit, which is what the ladder is actually trading.
                rd = (p * (M / M[:, -1:].clamp_min(1e-12)
                           + lam * cost_k[None, :])).sum(1).mean()
                loss = rd + a.w_class * class_loss(p) + a.w_avg * average_loss(p, active)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
        cm.eval()
        row = {"lam": lam, "params": npar, "train_s": time.time() - t0,
               "per_qp": {}}
        with torch.no_grad():
            for qv in a.qps:
                tot_db = 0.0
                hist = torch.zeros(K, dtype=torch.long)
                maps = []
                agree = tot = 0
                for x in test:
                    cells, M, R, nh, nw = tables(x, qv)
                    k = cm(cells).argmax(1).clamp(min=j)
                    tot_db += (10 * torch.log10(
                        M.gather(1, k[:, None]).squeeze(1).mean()
                        / R.mean())).item()
                    hist += torch.bincount(k.cpu(), minlength=K)
                    star = (M[:, j:] + lam * cost_k[None, j:]).argmin(1) + j
                    agree += int((k == star).sum()); tot += k.numel()
                    # The saving this map actually costs, with the dilation
                    # band charged rather than modelled -- the same accounting
                    # every other router in this project is scored by.
                    e = k.reshape(nh, nw).float()
                    e = e.repeat_interleave(cf, 0).repeat_interleave(cf, 1)
                    D = dilate_depth((e + 1.0) * cfg.blocks_per_exit)
                    ran = sum(int((D >= g * cfg.blocks_per_exit + 1 - 1e-6).sum())
                              * cfg.blocks_per_exit for g in range(K))
                    maps.append((ran / D.numel(), ADAPT[k.cpu()].mean().item()))
                frac = sum(m[0] for m in maps) / len(maps)
                adm = sum(m[1] for m in maps) / len(maps)
                cost = (UPS + TRUNK * frac / NBLK + adm + HEAD)
                sv = 100 * (1 - cost)
                row["per_qp"][qv] = {
                    "db": tot_db / len(test), "saving_pct": sv,
                    "hist": hist.tolist(),
                    "agreement": agree / max(1, tot)}
                print(f"   lam {lam:.0e} qp{qv:>3}: dB {tot_db/len(test):+.4f}  "
                      f"tasarruf {sv:5.2f}%  uyum {agree/max(1,tot):.3f}  "
                      f"mix {hist.tolist()}", flush=True)
        results["runs"].append(row)
        Path(a.out).write_text(json.dumps(results, indent=2))
    print(f"\n  yazildi {a.out}")


def stem_ch(net, dev):
    with torch.no_grad():
        x = torch.zeros(1, 3, 256, 256, device=dev)
        qp = torch.zeros(1, dtype=torch.int32, device=dev)
        y, _, _ = net._encode_to_latent(x, qp)
        return net.dec.upsample(y).shape[1]


if __name__ == "__main__":
    main()
