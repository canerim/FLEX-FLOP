"""Wall clock for the routed decode on a second device class, and at batch sizes
the deployed decoder does not use.

Two disclosures the compute axis is missing, measured with one timer so the
numbers are comparable to each other.

**A second device class.** Every latency figure in this project was taken on an
RTX A6000, and all eight cards in this machine are A6000s, so the only second
class reachable without another machine is the CPU. It is the unflattering end:
a tiled decode with a shrinking active set launches many small kernels on a GPU
and the launch overhead is what a MAC count cannot see, while on a CPU the work
is closer to arithmetic-bound and the MAC model should look better. Reporting
the direction the trend runs in is the point; a device sweep that only shows the
favourable end is not evidence.

**Batch size.** ALGM batches 32 duplicates of one image and says so, because
naive batched throughput would misrepresent an adaptive method. Our argument is
the opposite one and it needs the same statement: batch 1 is what a decoder
does, one frame arrives at a time, and every other timing here is at batch 1.
Measuring larger batches shows what that choice costs. A batch is genuinely
different work here, not a repetition: `exit_map` is indexed over the whole
batch's tiles, so B frames of 40 tiles are 40B tiles in one shrinking active
set, and the groups run wider.

Three variants per measurement, all on the same latent so nothing but the
synthesis differs:

  stock   the released decoder's full-frame forward, the number to beat
  deep    our tiled forward with every tile at the deepest exit: the overhead
          floor, same arithmetic as stock plus the tiling machinery
  routed  our tiled forward with the exit map the system picks at the budget

The timer is `time.perf_counter` around a synchronised call, not CUDA events,
because events do not exist on the CPU and the two device classes have to be
timed the same way to be compared. Warm-up first, then interleaved rounds with
the median reported: this card is shared, and timing all of A then all of B
puts any drift in the other job's load straight into the ratio.

    python scripts/latency_device.py --ckpt runs/RECIPE512/ckpt_PAPER.pth.tar \
        --device cpu --qps 32 --iters 3 --out results/supp_latency_cpu.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.cost import exit_costs, frame_relative_cost  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402
from flexuf.reference import reference_for  # noqa: E402


def _sync(dev):
    if str(dev).startswith("cuda"):
        torch.cuda.synchronize(dev)


def timeit_interleaved(fns, warmup, iters, dev):
    """Median seconds per call, measured one round of each at a time."""
    for _ in range(warmup):
        for fn in fns.values():
            fn()
    _sync(dev)
    ts = {k: [] for k in fns}
    for _ in range(iters):
        for k, fn in fns.items():
            _sync(dev)
            t0 = time.perf_counter()
            fn()
            _sync(dev)
            ts[k].append(time.perf_counter() - t0)
    return {k: statistics.median(v) for k, v in ts.items()}


def content(b, H, W, dev):
    """A deterministic non-uniform batch, one distinct image per element.

    Constant images would give every tile the same latent, so the entropy model
    would have nothing to do and any data-dependent kernel choice would be
    unrepresentative. Each batch element gets its own texture seed so a batch is
    B different frames rather than B copies of one.
    """
    xs = []
    for i in range(b):
        g = torch.Generator(device="cpu").manual_seed(i)
        yy = torch.linspace(0, 1, H).view(1, 1, H, 1)
        xx = torch.linspace(0, 1, W).view(1, 1, 1, W)
        base = (yy * xx).expand(1, 3, H, W).clone()
        tex = torch.rand(1, 3, H // 16, W // 16, generator=g)
        tex = torch.nn.functional.interpolate(tex, size=(H, W), mode="bilinear",
                                              align_corners=False)
        xs.append(0.6 * base + 0.4 * tex - 0.5)
    return torch.cat(xs).to(dev)


def hist_at(curve, qp, budget, K):
    """The measured exit histogram at this rate and budget.

    Read from an operating point of a frontier file rather than from the nearest
    sample of its lambda sweep: the sweep's closest point can sit a long way
    from the budget, and the exit mix is what is being timed.
    """
    ops = [o for o in curve.get("op_points", [])
           if o["qp"] == qp and o.get("hist")]
    if not ops:
        raise SystemExit(f"no operating point with a histogram at qp {qp}")
    o = min(ops, key=lambda r: abs(r["target_db"] - budget))
    if abs(o["target_db"] - budget) > 1e-9:
        print(f"    qp{qp}: nearest operating point is {o['target_db']:g} dB, "
              f"not {budget:g}")
    h = list(o["hist"])
    return h + [0] * (K - len(h)), o["target_db"]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--curve", default="results/supp_paper_curve_PAPER.json",
                    help="frontier file the exit histogram is taken from; must "
                         "be the same checkpoint being timed")
    ap.add_argument("--qps", type=int, nargs="+", default=[32])
    ap.add_argument("--batches", type=int, nargs="+", default=[1])
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--budget_db", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0,
                    help="CPU threads; 0 leaves torch's default. Recorded "
                         "either way, because a CPU time without a thread "
                         "count is not a measurement")
    ap.add_argument("--mem_fraction", type=float, default=0.0,
                    help="cap this process's share of the CUDA card. This is a "
                         "shared machine and a batch sweep is the one job here "
                         "that can grow without bound")
    ap.add_argument("--note", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    dev = a.device
    on_cuda = str(dev).startswith("cuda")
    if on_cuda:
        torch.cuda.set_device(dev)
        if a.mem_fraction > 0:
            torch.cuda.set_per_process_memory_fraction(a.mem_fraction, dev)
    if a.threads:
        torch.set_num_threads(a.threads)

    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"]) if "config" in ck else FlexUFConfig()
    net = FlexUFIntra(cfg).to(dev).eval(); load_flexuf_state(net, ck)
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    cost = exit_costs(cfg, "head")
    curve = json.loads((ROOT / a.curve).read_text())
    if curve.get("ckpt") != a.ckpt:
        print(f"    exit map comes from {curve.get('ckpt')}, timing "
              f"{a.ckpt}: the mix is another checkpoint's")

    P = cfg.rgb_patch
    W = (a.width + P - 1) // P * P
    H = (a.height + P - 1) // P * P
    n_tiles = (H // P) * (W // P)
    dev_name = (torch.cuda.get_device_name(dev) if on_cuda
                else f"CPU, {torch.get_num_threads()} threads")
    print(f"  {a.width}x{a.height} padded to {W}x{H}, {P}px tiles, "
          f"{n_tiles} tiles/frame")
    print(f"  {dev_name}, {a.iters} interleaved iterations after {a.warmup} "
          f"warm-up, median\n")
    print(f"  {'qp':>4}{'batch':>7}{'stock':>11}{'deep':>11}{'routed':>11}"
          f"{'overhead':>10}{'realised':>10}{'predicted':>11}")

    rows, oom = [], []
    with torch.no_grad():
        for qp_v in a.qps:
            h, got_db = hist_at(curve, qp_v, a.budget_db, cfg.num_exits)
            props = torch.tensor(h, dtype=torch.float) / max(sum(h), 1)
            counts = (props * n_tiles).round().long()
            counts[-1] += n_tiles - counts.sum()
            em1 = torch.cat([torch.full((int(c),), k, dtype=torch.long)
                             for k, c in enumerate(counts)])
            predicted = 100 * (1 - frame_relative_cost(em1, cfg, "head"))
            for b in a.batches:
                try:
                    x = content(b, H, W, dev)
                    qp = torch.full((b,), qp_v, dtype=torch.int32, device=dev)
                    y, q, _ = net._encode_to_latent(x, qp)
                    # exit_map is indexed over the WHOLE batch's tiles: B frames
                    # of n_tiles are B*n_tiles entries, not n_tiles.
                    em = em1.repeat(b).to(dev)
                    deep = torch.full((b * n_tiles,), cfg.num_exits - 1,
                                      dtype=torch.long, device=dev)
                    t = timeit_interleaved(
                        {"stock": lambda: ref.dec.forward_full(y, q),
                         "deep": lambda: net.dec(y, q, exit_map=deep),
                         "routed": lambda: net.dec(y, q, exit_map=em)},
                        a.warmup, a.iters, dev)
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    oom.append({"qp": qp_v, "batch": b})
                    print(f"  {qp_v:>4}{b:>7}   out of memory; larger batches "
                          f"not attempted at this rate")
                    if on_cuda:
                        torch.cuda.empty_cache()
                    break
                ms = {k: 1e3 * v for k, v in t.items()}
                row = {"qp": qp_v, "batch": b,
                       "ms_stock": ms["stock"], "ms_deep": ms["deep"],
                       "ms_routed": ms["routed"],
                       "ms_stock_per_frame": ms["stock"] / b,
                       "ms_deep_per_frame": ms["deep"] / b,
                       "ms_routed_per_frame": ms["routed"] / b,
                       "overhead_pct": 100 * (ms["deep"] / ms["stock"] - 1),
                       "realised_saving_pct": 100 * (1 - ms["routed"] / ms["stock"]),
                       "predicted_saving_pct": predicted,
                       "hist_per_frame": h, "hist_from_db": got_db}
                rows.append(row)
                print(f"  {qp_v:>4}{b:>7}{ms['stock']:>10.1f}ms"
                      f"{ms['deep']:>10.1f}ms{ms['routed']:>10.1f}ms"
                      f"{row['overhead_pct']:>9.1f}%"
                      f"{row['realised_saving_pct']:>9.1f}%"
                      f"{predicted:>10.1f}%")
                del x, y, q, em, deep
                if on_cuda:
                    torch.cuda.empty_cache()

    print("\n  overhead = what the tiling machinery costs before routing has")
    print("  saved anything; realised = wall clock; predicted = the MAC model.")
    out = {"what": "wall clock by device class and batch size",
           "ckpt": a.ckpt, "device": dev_name, "device_arg": dev,
           "torch_threads": torch.get_num_threads(),
           "resolution": [a.width, a.height], "padded": [W, H],
           "tile_px": P, "tiles_per_frame": n_tiles,
           "budget_db": a.budget_db, "exit_map_from": a.curve,
           "timer": "time.perf_counter around a synchronised call",
           "warmup": a.warmup, "iters": a.iters, "interleaved": True,
           "entropy_decode_timed": False,
           "note": a.note, "oom": oom, "rows": rows}
    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
