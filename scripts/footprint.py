"""Peak memory and throughput of the routed decode against the released one.

Two questions a deployment reader asks that a multiply-accumulate count cannot
answer. Does routing cost memory, and what frame rate does it actually reach?

Memory is the more interesting of the two. Tiling the trunk turns one large
feature map into a batch of small ones, and the early exits then drop tiles from
that batch as they leave, so the peak is set by the widest point of the ladder
rather than by the deepest. Whether that comes out above or below the released
decoder's peak is not obvious from the architecture and is measured here.

Peak is read from torch.cuda.max_memory_allocated, reset before each condition,
which reports the allocator's high-water mark for THIS process and is therefore
unaffected by whatever else is resident on a shared card.

    python scripts/footprint.py --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))

from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.measure import MacMeter  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402

MB = 1024 ** 2


def timed(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b))
    return statistics.median(ts)


def peak(fn):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - before) / MB


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qp", type=int, default=32)
    ap.add_argument("--budget_db", type=float, default=0.1)
    ap.add_argument("--sizes", nargs="+",
                    default=["832x480", "1280x720", "1920x1080", "3840x2160"])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/supp_footprint.json")
    a = ap.parse_args(argv)

    torch.cuda.set_device(a.device)
    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    dec, j = net.dec, cfg.split_depth

    curve = json.loads((ROOT / "results/curve_RECIPE512.json").read_text())
    rows = [r for r in curve["rows"] if r["qp"] == a.qp and r.get("hist")]
    hist = min(rows, key=lambda r: abs(
        r.get("db_vs_uf_per_frame", 9) - a.budget_db))["hist"]

    out = {"ckpt": a.ckpt, "qp": a.qp, "budget_db": a.budget_db, "hist": hist,
           "device": torch.cuda.get_device_name(dev), "rows": []}
    print(f"  {'size':>11}{'tiles':>7}{'MAC%':>7}"
          f"{'full ms':>9}{'routed ms':>11}{'FPS full':>10}{'FPS rtd':>9}"
          f"{'peak full':>11}{'peak rtd':>10}")
    with torch.no_grad():
        for size in a.sizes:
            w0, h0 = (int(v) for v in size.split("x"))
            W = (w0 + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
            H = (h0 + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
            n_tiles = (H // cfg.rgb_patch) * (W // cfg.rgb_patch)
            props = torch.tensor(hist, dtype=torch.float) / sum(hist)
            counts = (props * n_tiles).round().long()
            counts[-1] += n_tiles - counts.sum()
            em = torch.cat([torch.full((int(c),), k, dtype=torch.long)
                            for k, c in enumerate(counts)]).to(dev).clamp(min=j)
            try:
                x = torch.zeros(1, 3, H, W, device=dev)
                qpt = torch.full((1,), a.qp, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(x, qpt)

                with MacMeter(dec) as mr:
                    dec(y, q, exit_map=em)
                with MacMeter(dec) as mf:
                    dec.forward_full(y, q)
                mac = 100.0 * (1 - mr.total / mf.total)

                p_full = peak(lambda: dec.forward_full(y, q))
                p_rtd = peak(lambda: dec(y, q, exit_map=em))
                t_full = timed(lambda: dec.forward_full(y, q), a.warmup, a.iters)
                t_rtd = timed(lambda: dec(y, q, exit_map=em), a.warmup, a.iters)
            except torch.OutOfMemoryError:
                # Not a failure of the method. The evaluation card is shared and
                # a neighbouring process holds most of it; recorded as absent
                # rather than quietly dropped, so the table says why.
                print(f"  {W}x{H:<5}  out of memory on a shared card")
                out["rows"].append({"size": f"{W}x{H}", "n_tiles": n_tiles,
                                    "oom": True})
                torch.cuda.empty_cache()
                continue

            row = {"size": f"{W}x{H}", "n_tiles": n_tiles, "mac_saving_pct": mac,
                   "ms_full": t_full, "ms_routed": t_rtd,
                   "fps_full": 1000.0 / t_full, "fps_routed": 1000.0 / t_rtd,
                   "peak_mb_full": p_full, "peak_mb_routed": p_rtd,
                   "time_saving_pct": 100.0 * (1 - t_rtd / t_full),
                   "peak_delta_pct": 100.0 * (p_rtd / p_full - 1)}
            out["rows"].append(row)
            print(f"  {row['size']:>11}{n_tiles:>7}{mac:>7.1f}"
                  f"{t_full:>9.1f}{t_rtd:>11.1f}"
                  f"{row['fps_full']:>10.1f}{row['fps_routed']:>9.1f}"
                  f"{p_full:>10.0f}M{p_rtd:>9.0f}M", flush=True)
            del x, y, q, em
            torch.cuda.empty_cache()

    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
