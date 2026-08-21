"""What the saving is worth in seconds and in joules.

The paper's saving is a multiply-accumulate count. That is the right unit to
optimise, because it is the only one that is independent of the machine, the
driver and whatever else happens to be resident on the card. It is not the unit
anyone deploying a decoder cares about. This measures the two units they do care
about, on the same exit maps the paper reports, and puts the three side by side.

They do not agree, and the disagreement is the result. A tiled decode runs its
groups on a shrinking set of tiles, so the last groups launch kernels over a
handful of 32x32 feature maps and the card is mostly idle inside them. Arithmetic
saved is not time saved at the same rate, and time saved is not energy saved at
the same rate either, because a partly idle GPU still draws most of its static
power.

Power is sampled from the driver at 100 ms, which is roughly the rate the sensor
itself updates. Each condition therefore runs for a fixed WALL TIME rather than a
fixed iteration count: a 30 ms decode measured over ten iterations gives the
sensor three samples and a meaningless mean. Idle is measured immediately before
and after every condition, on the same card, so a neighbouring process waking up
mid-measurement shows up as a drift between the two rather than as a saving.

    python scripts/power_profile.py --device cuda:0 --gpu_index 2
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402

from flexuf.config import FlexUFConfig  # noqa: E402
from flexuf.measure import MacMeter  # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state  # noqa: E402


class PowerSampler:
    """Board power in watts, sampled in the background at 100 ms.

    nvidia-smi is started once in streaming mode rather than called per sample:
    a fresh process per sample costs about 30 ms of CPU and perturbs the thing
    being measured.
    """

    def __init__(self, index: int, period_ms: int = 100):
        self.index, self.period_ms = index, period_ms
        self.samples: list[float] = []
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def __enter__(self):
        self._proc = subprocess.Popen(
            ["nvidia-smi", f"--id={self.index}", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits", f"-lms={self.period_ms}"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)

        def pump():
            for line in self._proc.stdout:
                if self._stop.is_set():
                    break
                try:
                    self.samples.append(float(line.strip()))
                except ValueError:
                    pass
        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)
        return False

    def mark(self):
        return len(self.samples)

    def since(self, start: int) -> list[float]:
        return self.samples[start:]


def run_for(fn, seconds: float, sampler, warmup: int = 10):
    """Run fn in a loop for `seconds`, return (median ms per call, watts)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    mark = sampler.mark()
    t_end = time.perf_counter() + seconds
    n, per_call = 0, []
    while time.perf_counter() < t_end:
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        per_call.append(a.elapsed_time(b))
        n += 1
    torch.cuda.synchronize()
    watts = sampler.since(mark)
    return {
        "ms_median": statistics.median(per_call),
        "ms_mean": statistics.fmean(per_call),
        "iters": n,
        "watts_mean": statistics.fmean(watts) if watts else None,
        "watts_max": max(watts) if watts else None,
        "power_samples": len(watts),
    }


def idle(sampler, seconds: float = 4.0):
    torch.cuda.synchronize()
    mark = sampler.mark()
    time.sleep(seconds)
    w = sampler.since(mark)
    return statistics.fmean(w) if w else None


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 32, 63])
    ap.add_argument("--budget_db", type=float, default=0.1)
    ap.add_argument("--sizes", nargs="+", default=["1280x720", "1920x1080"])
    ap.add_argument("--seconds", type=float, default=20.0,
                    help="wall time per condition; the power sensor needs it")
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--gpu_index", type=int, default=2,
                    help="index nvidia-smi uses, which is NOT the index torch "
                         "uses when CUDA_VISIBLE_DEVICES is set")
    ap.add_argument("--out", default="results/supp_power.json")
    a = ap.parse_args(argv)

    torch.cuda.set_device(a.device)
    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    dec, K, j = net.dec, cfg.num_exits, cfg.split_depth

    curve = json.loads((ROOT / "results/curve_RECIPE512.json").read_text())

    def hist_for(qp):
        rows = [r for r in curve["rows"] if r["qp"] == qp and r.get("hist")]
        if not rows:
            return None
        return min(rows, key=lambda r: abs(
            r.get("db_vs_uf_per_frame", 9) - a.budget_db))["hist"]

    out = {"ckpt": a.ckpt, "budget_db": a.budget_db,
           "device": torch.cuda.get_device_name(dev),
           "power_limit_w": None, "rows": []}
    try:
        out["power_limit_w"] = float(subprocess.run(
            ["nvidia-smi", f"--id={a.gpu_index}", "--query-gpu=power.limit",
             "--format=csv,noheader,nounits"], capture_output=True,
            text=True).stdout.strip())
    except Exception:
        pass

    with PowerSampler(a.gpu_index) as sampler, torch.no_grad():
        time.sleep(1.0)
        for size in a.sizes:
            w0, h0 = (int(v) for v in size.split("x"))
            W = (w0 + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
            H = (h0 + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
            n_tiles = (H // cfg.rgb_patch) * (W // cfg.rgb_patch)

            for qp in a.qps:
                hist = hist_for(qp)
                if hist is None:
                    continue
                props = torch.tensor(hist, dtype=torch.float) / sum(hist)
                counts = (props * n_tiles).round().long()
                counts[-1] += n_tiles - counts.sum()
                em = torch.cat([torch.full((int(c),), k, dtype=torch.long)
                                for k, c in enumerate(counts)]
                               ).to(dev).clamp(min=j)

                x = torch.zeros(1, 3, H, W, device=dev)
                qpt = torch.full((1,), qp, dtype=torch.int32, device=dev)
                y, q, _ = net._encode_to_latent(x, qpt)

                # The arithmetic, from hooks off these exact calls, so the MAC
                # saving quoted next to the seconds is the saving of the decode
                # that was actually timed and not of a nominally similar one.
                with MacMeter(dec) as m_routed:
                    dec(y, q, exit_map=em)
                with MacMeter(dec) as m_full:
                    dec.forward_full(y, q)
                mac_saving = 100.0 * (1.0 - m_routed.total / m_full.total)

                idle_before = idle(sampler)
                full = run_for(lambda: dec.forward_full(y, q),
                               a.seconds, sampler)
                idle_mid = idle(sampler)
                routed = run_for(lambda: dec(y, q, exit_map=em),
                                 a.seconds, sampler)
                idle_after = idle(sampler)

                def energy(r, base):
                    if r["watts_mean"] is None:
                        return None, None
                    j_total = r["watts_mean"] * r["ms_median"] / 1000.0
                    j_above = ((r["watts_mean"] - base) * r["ms_median"] / 1000.0
                               if base is not None else None)
                    return j_total, j_above

                base = statistics.fmean(
                    [v for v in (idle_before, idle_mid, idle_after)
                     if v is not None]) if idle_before is not None else None
                jf, jf_ab = energy(full, base)
                jr, jr_ab = energy(routed, base)

                row = {
                    "size": f"{W}x{H}", "qp": qp, "n_tiles": n_tiles,
                    "hist": hist,
                    "mac_saving_pct": mac_saving,
                    "idle_w": {"before": idle_before, "mid": idle_mid,
                               "after": idle_after, "mean": base},
                    "full": full, "routed": routed,
                    "time_saving_pct": 100.0 * (
                        1 - routed["ms_median"] / full["ms_median"]),
                    "joules_per_frame_full": jf,
                    "joules_per_frame_routed": jr,
                    "energy_saving_pct": (100.0 * (1 - jr / jf)
                                          if jf and jr else None),
                    "joules_above_idle_full": jf_ab,
                    "joules_above_idle_routed": jr_ab,
                    "energy_saving_above_idle_pct": (
                        100.0 * (1 - jr_ab / jf_ab)
                        if jf_ab and jr_ab else None),
                }
                out["rows"].append(row)
                print(f"  {row['size']:>10}  qp{qp:<3} "
                      f"MAC {mac_saving:5.1f}%   "
                      f"time {full['ms_median']:6.1f} -> "
                      f"{routed['ms_median']:6.1f} ms "
                      f"({row['time_saving_pct']:5.1f}%)   "
                      f"power {full['watts_mean'] or 0:5.1f} -> "
                      f"{routed['watts_mean'] or 0:5.1f} W   "
                      f"energy {jf or 0:6.3f} -> {jr or 0:6.3f} J/frame "
                      f"({row['energy_saving_pct'] or 0:5.1f}%)", flush=True)
                del x, y, q, em
                torch.cuda.empty_cache()

    (ROOT / a.out).write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
