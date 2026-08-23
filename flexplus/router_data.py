"""Per-cell features and the per-exit error they have to predict.

The paper's router classifies a tile's exit from a small CNN over the stem,
and its input ablation is already done: stem alone scores 0.751 held-out
agreement against 0.750 for every input together, with bits at 0.569 and the
QP at 0.408. That axis is spent.

Agreement is the wrong target anyway. The allocation is a Lagrangian --
argmin_k (m_k + lam c_k) over cells -- so what a router needs is not the
argmin but the CURVE m_k it is taken over. A cell whose exits are nearly tied
can be misclassified at no cost; a cell on a cliff cannot. Adaptive Patch
Exiting (ECCV 2022) makes the same move for super-resolution, regressing each
layer's incremental capacity rather than predicting a class, and it is the
natural fit here because the cost side of the Lagrangian is already exact.

So this dumps, for every cell: cheap decoder-side features in named groups,
and the per-exit MSE the decoder actually achieved. Nothing here is signalled
-- every feature is computable at the decoder from the bitstream it already
has, which is what keeps the comparison against the parameter-free rate-rank
surrogate honest.

Train on OpenImages, test on the CTC frames: the paper's protocol, so the
router never sees a test sequence.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import torch
import torch.nn.functional as F

UF = Path.home() / "FLEX-UF"
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(UF)); sys.path.insert(0, str(UF / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))

import ctc_intra as C                                          # noqa: E402
from flexuf.config import FlexUFConfig                         # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state        # noqa: E402
from flexuf.reference import reference_for                     # noqa: E402

GROUPS = {
    "bits":   ["bits_sum", "bits_mean", "bits_max"],
    "latent": ["lat_absmean", "lat_std", "lat_max", "lat_fraczero",
               "lat_gradx", "lat_grady", "lat_s1", "lat_s2", "lat_s3",
               "lat_s4"],
    "scales": ["sc_mean", "sc_std", "sc_max"],
    "stem":   ["stem_absmean", "stem_std", "stem_grad", "stem_localvar"],
    "qp":     ["qp", "quant_step"],
    "nbr":    ["nbr_bits", "nbr_stemgrad", "nbr_latabs", "nbr_range"],
}
FEATURES = [f for g in GROUPS.values() for f in g]


def _pool(x, k):
    """Mean over k x k cells, keeping [H/k, W/k]."""
    return F.avg_pool2d(x, k, k)


def cell_features(net, y_hat, q, aux, cell_rgb, rgb_hw):
    """Decoder-side features for every cell, in the order of FEATURES."""
    H, W = rgb_hw
    _, _, hl, wl = y_hat.shape
    cl = max(1, cell_rgb // (H // hl))                 # cell on the latent grid
    stem = net.dec.upsample(y_hat)
    hs, ws = stem.shape[-2:]
    cs = max(1, cell_rgb // (H // hs))                 # cell on the stem grid

    y = y_hat.float()
    bits = net.get_y_bits(net.add_noise(aux["y_res"]), aux["scales_hat"])
    b1 = bits.sum(1, keepdim=True)
    f = {}
    f["bits_sum"] = _pool(b1, cl)[0, 0] * (cl * cl)
    f["bits_mean"] = _pool(b1, cl)[0, 0]
    f["bits_max"] = F.max_pool2d(b1, cl, cl)[0, 0]

    ab = y.abs().mean(1, keepdim=True)
    f["lat_absmean"] = _pool(ab, cl)[0, 0]
    f["lat_std"] = (_pool(ab ** 2, cl) - _pool(ab, cl) ** 2).clamp_min(0)[0, 0].sqrt()
    f["lat_max"] = F.max_pool2d(ab, cl, cl)[0, 0]
    f["lat_fraczero"] = _pool((y.abs() < 1e-2).float().mean(1, keepdim=True), cl)[0, 0]
    gx = (y[:, :, :, 1:] - y[:, :, :, :-1]).abs().mean(1, keepdim=True)
    gy = (y[:, :, 1:, :] - y[:, :, :-1, :]).abs().mean(1, keepdim=True)
    f["lat_gradx"] = _pool(F.pad(gx, (0, 1)), cl)[0, 0]
    f["lat_grady"] = _pool(F.pad(gy, (0, 0, 0, 1)), cl)[0, 0]

    # The four closed-form signals the shipped router uses, per cell.
    mu = _pool(y.mean(1, keepdim=True), cl)
    v = (_pool(y.pow(2).mean(1, keepdim=True), cl) - mu ** 2).clamp_min(1e-6)
    f["lat_s1"] = (0.5 * (1.0 + torch.log(2 * torch.pi * v)) / 0.6931)[0, 0]
    f["lat_s2"] = f["lat_fraczero"]
    f["lat_s3"] = f["lat_gradx"] + f["lat_grady"]
    pn = y.pow(2).sum(1, keepdim=True).sqrt()
    f["lat_s4"] = (_pool(pn ** 2, cl) - _pool(pn, cl) ** 2).clamp_min(0)[0, 0]

    sc = aux["scales_hat"].float().abs().mean(1, keepdim=True)
    f["sc_mean"] = _pool(sc, cl)[0, 0]
    f["sc_std"] = (_pool(sc ** 2, cl) - _pool(sc, cl) ** 2).clamp_min(0)[0, 0].sqrt()
    f["sc_max"] = F.max_pool2d(sc, cl, cl)[0, 0]

    sa = stem.float().abs().mean(1, keepdim=True)
    f["stem_absmean"] = _pool(sa, cs)[0, 0]
    f["stem_std"] = (_pool(sa ** 2, cs) - _pool(sa, cs) ** 2).clamp_min(0)[0, 0].sqrt()
    sgx = (stem[:, :, :, 1:] - stem[:, :, :, :-1]).abs().mean(1, keepdim=True)
    sgy = (stem[:, :, 1:, :] - stem[:, :, :-1, :]).abs().mean(1, keepdim=True)
    f["stem_grad"] = (_pool(F.pad(sgx, (0, 1)), cs)
                      + _pool(F.pad(sgy, (0, 0, 0, 1)), cs))[0, 0]
    lv = F.avg_pool2d(sa ** 2, 3, 1, 1) - F.avg_pool2d(sa, 3, 1, 1) ** 2
    f["stem_localvar"] = _pool(lv.clamp_min(0), cs)[0, 0]

    nh, nw = f["bits_sum"].shape
    for k in list(f):
        f[k] = f[k][:nh, :nw]

    def nbr(t):
        return F.avg_pool2d(t[None, None], 3, 1, 1)[0, 0]
    f["nbr_bits"] = nbr(f["bits_sum"])
    f["nbr_stemgrad"] = nbr(f["stem_grad"])
    f["nbr_latabs"] = nbr(f["lat_absmean"])
    f["nbr_range"] = (F.max_pool2d(f["bits_sum"][None, None], 3, 1, 1)[0, 0]
                      + F.max_pool2d(-f["bits_sum"][None, None], 3, 1, 1)[0, 0])
    return f, (nh, nw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(UF / "runs/RECIPE512/ckpt_PAPER.pth.tar"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--source", choices=["openimages", "ctc"], required=True)
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--images", type=int, default=300)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, None),
                                      map_location="cpu", weights_only=False))
    K = cfg.num_exits

    imgs = []
    if a.source == "ctc":
        seqs, _ = C.discover([])
        for s in seqs:
            x, _ = C.read_frames(s["path"], s["w"], s["h"], 1, 1)
            if x is not None:
                imgs.append((x[0:1], s["name"]))
    else:
        # The exact chain the decoder was trained on, from
        # ~/DCVC/src/datasets/image_dataset.py: /255, RGB to YCbCr, minus 0.5.
        # Feeding [0,1] RGB instead put the deepest exit 0.3 to 0.9 dB off the
        # released decoder on these images -- structurally impossible, since
        # that exit takes the raw feature -- which is how the wrong colour
        # space was caught. CTC frames come out of ctc_intra.read_frames
        # already in this domain, so only this branch was ever wrong.
        import numpy as _np
        from PIL import Image
        from src.utils.transforms import rgb2ycbcr_np
        root = Path("/data10/shareddata/openimages/dcvc_train")
        files = []
        for sub in sorted(root.glob("train_*")):
            for p in sorted(sub.iterdir()):
                files.append(p)
                if len(files) >= a.images * 3:
                    break
            if len(files) >= a.images * 3:
                break
        torch.manual_seed(0)
        for p in files:
            if len(imgs) >= a.images:
                break
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                continue
            if im.width < a.crop or im.height < a.crop:
                continue
            # Crop before converting: rgb2ycbcr_np asserts even sides, and
            # the crop is even by construction while the source may not be.
            arr = _np.array(im).astype(_np.float32) / 255.0
            i = torch.randint(0, arr.shape[0] - a.crop + 1, (1,)).item()
            jj = torch.randint(0, arr.shape[1] - a.crop + 1, (1,)).item()
            arr = arr[i:i + a.crop, jj:jj + a.crop]
            t = torch.as_tensor(rgb2ycbcr_np(arr) - 0.5,
                                dtype=torch.float32).permute(2, 0, 1)[None]
            imgs.append((t, p.name))
    print(f"  {len(imgs)} goruntu, hucre {a.cell}px", flush=True)

    X, M, R, QP, IMG, GRID = [], [], [], [], [], []
    with torch.no_grad():
        for gi, (x, name) in enumerate(imgs):
            x = x.to(dev)
            ph, pw = (-x.shape[-2]) % a.cell, (-x.shape[-1]) % a.cell
            xp = F.pad(x, (0, pw, 0, ph), mode="replicate") if (ph or pw) else x
            for qp_v in a.qps:
                qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                y, q, aux = net._encode_to_latent(xp, qp)
                recs = net.dec.forward_all_exits(y, q)
                rr = ref.dec.forward_full(y, q)
                f, (nh, nw) = cell_features(net, y, q, aux, a.cell,
                                            xp.shape[-2:])
                vals = []
                for k in FEATURES:
                    if k == "qp":
                        vals.append(torch.full((nh * nw,), float(qp_v),
                                               device=dev))
                    elif k == "quant_step":
                        vals.append(torch.full((nh * nw,), float(q.mean()),
                                               device=dev))
                    else:
                        vals.append(f[k].reshape(-1).float())
                X.append(torch.stack(vals, 1).cpu())

                def cm(img):
                    e = ((img - xp) ** 2).mean(1, keepdim=True)
                    return F.avg_pool2d(e, a.cell, a.cell)[0, 0][:nh, :nw].reshape(-1)
                M.append(torch.stack([cm(r) for r in recs], 1).cpu())
                R.append(cm(rr).cpu())
                QP.append(torch.full((nh * nw,), qp_v, dtype=torch.int32))
                IMG.append(torch.full((nh * nw,), gi, dtype=torch.int32))
                GRID.append((gi, qp_v, nh, nw, name))
            if (gi + 1) % 25 == 0:
                print(f"    {gi + 1}/{len(imgs)}", flush=True)

    import numpy as np
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, X=torch.cat(X).numpy(), M=torch.cat(M).numpy(),
        R=torch.cat(R).numpy(), qp=torch.cat(QP).numpy(),
        img=torch.cat(IMG).numpy(),
        features=np.array(FEATURES),
        groups=np.array(json.dumps(GROUPS)),
        grid=np.array([f"{g[0]},{g[1]},{g[2]},{g[3]},{g[4]}" for g in GRID]),
        K=np.array([K]), cell=np.array([a.cell]),
        split_depth=np.array([cfg.split_depth]))
    print(f"  yazildi {out}  ({torch.cat(X).shape[0]} hucre, "
          f"{len(FEATURES)} oznitelik)")


if __name__ == "__main__":
    main()
