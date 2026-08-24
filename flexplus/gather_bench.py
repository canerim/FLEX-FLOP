"""Does the time follow the MAC count? A gather is enough to make it.

The saving is counted in multiply-accumulates, and a reviewer is right to ask
whether a mask that skips 40 per cent of positions skips 40 per cent of the
time. For a spatial convolution the answer is usually no -- AdaDSR's own
authors say pixel-wise sparse convolution is not hardware friendly.

But 99.708 per cent of a DepthConvBlock's arithmetic is pointwise: 8C^2 per
position against 9C for the depthwise 3x3, at C=384. A 1x1 convolution is a
per-position matmul over channels, so the needed positions can be gathered
into a dense [N_kept, C] matrix, multiplied, and scattered back -- no spatial
structure required, and the cost is linear in how many positions were kept.

This times exactly that, against the dense path, at the shapes the decoder
actually runs. If the gathered time tracks the kept fraction, the MAC count
is the honest unit after all.

One card, seconds of work; run it on a card that is free.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent


def timeit(fn, iters=30, warmup=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--channels", type=int, default=384)
    ap.add_argument("--hw", type=int, nargs=2, default=[272, 480])
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
    ap.add_argument("--out", default=str(HERE / "results/gather_bench.json"))
    a = ap.parse_args()

    dev = torch.device(a.device)
    C, (H, W) = a.channels, a.hw
    x = torch.randn(1, C, H, W, device=dev, dtype=torch.float16)
    up = nn.Conv2d(C, 4 * C, 1).to(dev).half()
    dn = nn.Conv2d(4 * C, C, 1).to(dev).half()
    dw = nn.Conv2d(C, C, 3, padding=1, groups=C).to(dev).half()
    Wu = up.weight.reshape(4 * C, C).t().contiguous()
    Wd = dn.weight.reshape(C, 4 * C).t().contiguous()
    bu, bd = up.bias, dn.bias

    with torch.no_grad():
        dense = timeit(lambda: dn(F.silu(up(dw(x)))))
        print(f"  yogun yol: {dense:.3f} ms  ({C}ch, {H}x{W}, fp16)", flush=True)
        flat = x[0].reshape(C, H * W).t().contiguous()
        rows = []
        for f in a.fracs:
            n = int(H * W * f)
            idx = torch.randperm(H * W, device=dev)[:n]

            def run(idx=idx):
                g = flat.index_select(0, idx)
                h = F.silu(torch.addmm(bu, g, Wu))
                o = torch.addmm(bd, h, Wd)
                out = torch.zeros_like(flat)
                out.index_copy_(0, idx, o)
                return out

            t = timeit(run)
            rows.append({"frac": f, "ms": t, "ms_dense": dense,
                         "speedup": dense / t})
            print(f"    tutulan {f:.2f}: {t:.3f} ms   hizlanma "
                  f"{dense / t:.2f}x   (ideal {1 / f:.2f}x)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"channels": C, "hw": [H, W], "dense_ms": dense, "rows": rows},
        indent=2))
    print(f"\n  yazildi {a.out}")


if __name__ == "__main__":
    main()
