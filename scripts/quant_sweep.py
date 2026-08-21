"""What does quantising the decoder cost, in the same currency as everything else?

Early exit removes MACs. Quantisation makes the MACs that remain cheaper. They
are orthogonal levers on the same budget, and this project has exactly one
currency -- dB below the released decoder -- so they can be priced against each
other directly.

What must NOT be quantised
--------------------------
The encoder, hyperprior and entropy model. They produce `y_hat` and the symbol
probabilities; perturbing either changes the bitstream, and an entropy model that
disagrees between encoder and decoder does not degrade gracefully, it destroys
the stream. Only `dec.*` (synthesis) and `router_head.*` are touched, and the
encoder state is asserted bit-identical afterwards -- the same check every other
measurement in this project makes.

The unit
--------
MAC counts do not move under quantisation, so "% of MACs saved" cannot express
it. The standard unit that does is **BOPs** -- bit-operations, MACs weighted by
the product of operand widths (Baskin et al., "UNIQ"; van Baalen et al.). A MAC
with b_w-bit weights and b_a-bit activations costs b_w * b_a, so int8/int8 is
1024/64 = 16x cheaper than fp32. That is an idealised hardware model -- real
speedup depends on kernel support -- so BOPs are reported alongside, never
instead of, the MAC-based saving.

    saving_bops = 1 - (cost_MAC_fraction * b_w * b_a) / (1.0 * 32 * 32)

What is measured
----------------
Weight-only, symmetric, per-output-channel post-training quantisation, no
calibration set and no fine-tuning -- deliberately the weakest form, so the
numbers are a floor on what quantisation can do rather than a best case.

  * the FLOOR: how far the deepest exit drifts from the release. This is the
    quantity that already eats 30% of the 0.1 dB budget at qp 63, so if
    quantisation moves it, it moves it where the project is most sensitive.
  * per-exit dB, because the interesting question is not "does quantisation cost
    quality" but "does it cost the SHALLOW exits more". If it does, the two
    levers compete rather than compose, and the exit ladder's advantage shrinks
    under aggressive quantisation.

    python scripts/quant_sweep.py --ckpt runs/BEST/ckpt_eval.pth.tar \
        --bits 8 6 4 --out results/quant_BEST.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402

from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402


@torch.no_grad()
def fake_quantise_(module: nn.Module, bits: int) -> int:
    """Symmetric per-output-channel weight quantisation, in place.

    Per-channel rather than per-tensor because a single scale over a 384x384
    kernel is dominated by whichever channel has the largest weight, and the
    rest lose most of their range. Per-channel is the standard minimum for
    weight PTQ and needs no calibration data.
    """
    n = 0
    qmax = 2 ** (bits - 1) - 1
    for m in module.modules():
        if not isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            continue
        w = m.weight
        flat = w.reshape(w.shape[0], -1)
        scale = flat.abs().amax(dim=1).clamp_min(1e-12) / qmax
        shape = (-1,) + (1,) * (w.dim() - 1)
        m.weight.copy_((flat / scale[:, None]).round().clamp(-qmax - 1, qmax)
                       .mul(scale[:, None]).reshape(w.shape))
        n += 1
    return n


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--bits", type=int, nargs="+", default=[8, 6, 4])
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--images", type=int, default=32)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])

    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))

    # Held-out images, same loader and same colour space as why_qp_val.py --
    # the training pipeline converts RGB to YCbCr, and feeding raw RGB produced
    # meaningless per-exit numbers once already.
    sys.path.insert(0, str(ROOT / "scripts"))
    from why_qp_val import load_val
    imgs = load_val(a.images, a.crop, dev)
    if not imgs:
        raise SystemExit("no usable validation images")

    K, P = cfg.num_exits, cfg.rgb_patch
    rows = []
    print(f"  {len(imgs)} held-out images at {a.crop}px, {cfg.rgb_patch}px tile\n")
    print(f"  {'bits':>5}{'BOPs vs fp32':>14}{'qp':>5}"
          + "".join(f"{'exit ' + str(k):>9}" for k in range(K)))

    for bits in [32] + list(a.bits):
        net = FlexUFIntra(cfg).to(dev).eval()
        load_flexuf_state(net, ck)
        if bits != 32:
            # dec.* and router_head.* only. Quantising net.enc would change
            # y_hat and therefore the bitstream.
            nq = fake_quantise_(net.dec, bits)
            nq += fake_quantise_(net.router_head, bits)
        else:
            nq = 0
        sa, sb = net.enc.state_dict(), ref.enc.state_dict()
        worst = max((sa[k] - sb[k]).abs().max().item() for k in sa)
        assert worst == 0.0, (
            f"the encoder moved (max|diff| = {worst}) at {bits} bits; the "
            f"bitstream is no longer the release's")

        bops = (bits * bits) / (32 * 32)
        with torch.no_grad():
            for qp_v in a.qps:
                se = torch.zeros(K, device=dev)
                sr = torch.zeros((), device=dev)
                for x in imgs:
                    qp = torch.full((1,), qp_v, dtype=torch.int32, device=dev)
                    y, q, _ = net._encode_to_latent(x, qp)
                    # DEPLOYED path: one tiled decode per exit. The full-frame
                    # forward_all_exits cancels the tiling penalty against a
                    # full-frame reference (flexuf/eval.py), and quantisation
                    # interacts with tile borders -- a border already fed
                    # invented values is where a coarser weight grid shows up
                    # first -- so measuring it full frame would understate
                    # exactly the effect being looked for.
                    nt = (x.shape[-2] // P) * (x.shape[-1] // P)
                    for i in range(K):
                        em = torch.full((nt,), max(i, cfg.split_depth),
                                        dtype=torch.long, device=dev)
                        se[i] += ((net.dec(y, q, exit_map=em) - x) ** 2).mean()
                    sr += ((ref.dec.forward_full(y, q) - x) ** 2).mean()
                db = (10 * torch.log10(se / sr)).tolist()
                rows.append({"bits": bits, "qp": qp_v, "db_per_exit": db,
                             "bops_vs_fp32": bops, "layers_quantised": nq})
                print(f"  {bits:>5}{bops:>13.3f}x{qp_v:>5}"
                      + "".join(f"{v:>9.4f}" for v in db))
        print()

    (ROOT / a.out).write_text(json.dumps(
        {"ckpt": a.ckpt, "n_images": len(imgs), "crop": a.crop,
         "scope": "dec.* and router_head.* only; encoder asserted unchanged",
         "method": "weight-only, symmetric, per-output-channel, no calibration",
         "rows": rows}, indent=2))
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
