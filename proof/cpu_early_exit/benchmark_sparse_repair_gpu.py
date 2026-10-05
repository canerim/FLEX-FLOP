"""Matched CUDA-event repair-stage benchmark, only on an idle selected GPU.

The script refuses to create a CUDA context if nvidia-smi reports a compute
process on the selected device. It compares the trained dense repair, dense
thresholded repair and the packed sparse pointwise path on the same tensor.
It does not time the complete decoder or codec.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = Path("/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/shared_metric_audit/source")
CHECKPOINT = ROOT / "runs/RECIPE512/ckpt_PIN_e15.pth.tar"
sys.path[:0] = [str(SNAPSHOT), str(ROOT)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_idle(gpu_index: int) -> str:
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
    found = {}
    for line in devices.splitlines():
        index, uuid = [part.strip() for part in line.split(",", 1)]
        found[int(index)] = uuid
    if gpu_index not in found:
        raise RuntimeError(f"GPU {gpu_index} is unavailable")
    target = found[gpu_index]
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True)
    busy = [line.strip() for line in apps.splitlines() if line.startswith(target)]
    if busy:
        raise RuntimeError(f"GPU {gpu_index} has compute processes: {busy}")
    return target


def other_compute_pids(uuid: str) -> list[int]:
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True)
    return [int(line.split(",", 1)[1].strip()) for line in apps.splitlines()
            if line.startswith(uuid) and int(line.split(",", 1)[1].strip()) != os.getpid()]


def ci(values: list[float]) -> list[float]:
    rng = random.Random(20261005)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(5000))
    return [means[124], means[4874]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--feature-height", type=int, default=160)
    parser.add_argument("--feature-width", type=int, default=256)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.feature_height > 0 and args.feature_width > 0 and args.trials >= 10
    uuid = require_idle(args.gpu)

    import torch
    from flexuf.backbone.decoder import GridSeamRepair
    from flexuf.kernels.depthwise3x3 import depthwise3x3
    from flexuf.kernels.fused_seam import FusedGridSeamRepair, gated_projection
    from proof.cpu_early_exit.sparse_grid_repair import SparseGridRepairWrapper

    torch.manual_seed(20261005)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    prefix = "dec.seam_repair."
    module = GridSeamRepair(channels=384, patch=32).eval()
    module.load_state_dict({k[len(prefix):]: v for k, v in checkpoint["state_dict"].items()
                            if k.startswith(prefix)}, strict=True)
    device = torch.device(f"cuda:{args.gpu}")
    module = module.to(device)
    x = torch.randn(1, 384, args.feature_height, args.feature_width,
                    device=device, dtype=torch.float32)
    sparse = SparseGridRepairWrapper(module, threshold=.25).eval()

    def dense_thresholded():
        correction = module.pw(module.act(module.dw(x)))
        H, W = x.shape[-2:]
        gate = module.gate.repeat(1, 1, -(-H // 32), -(-W // 32))[:, :, :H, :W]
        return x + gate * (gate >= .25) * correction

    fused = FusedGridSeamRepair(module).eval()

    def fused_thresholded():
        activated = depthwise3x3(x, module.dw, activate=True)
        return gated_projection(activated, module.pw, module.gate, x,
                                module.patch, threshold=.25)

    arms = {"trained_dense": lambda: module(x),
            "thresholded_dense": dense_thresholded,
            "trained_fused": lambda: fused(x),
            "thresholded_fused": fused_thresholded,
            "thresholded_packed": lambda: sparse(x)}
    with torch.inference_mode():
        for _ in range(10):
            for fn in arms.values():
                fn()
        torch.cuda.synchronize(device)
        reference = dense_thresholded()
        candidate = sparse(x)
        max_abs = float((reference - candidate).abs().max().item())
        if max_abs > 2e-5:
            raise RuntimeError(f"Packed and dense thresholded paths differ: {max_abs}")
        fused_candidate = fused_thresholded()
        fused_max_abs = float((reference - fused_candidate).abs().max().item())
        if fused_max_abs > 2e-5:
            raise RuntimeError(f"Fused and dense thresholded paths differ: {fused_max_abs}")
        trained_fused_max_abs = float((module(x) - fused(x)).abs().max().item())
        if trained_fused_max_abs > 2e-5:
            raise RuntimeError(f"Fused and trained dense paths differ: {trained_fused_max_abs}")
        measurements = {name: [] for name in arms}
        rng = random.Random(20261005)
        for _ in range(args.trials):
            names = list(arms)
            rng.shuffle(names)
            for name in names:
                start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                arms[name]()
                stop.record()
                stop.synchronize()
                measurements[name].append(start.elapsed_time(stop))
        ratios_raw = [a / b for a, b in zip(measurements["thresholded_dense"],
                                             measurements["thresholded_packed"])]
        ratios_fused = [a / b for a, b in zip(measurements["thresholded_fused"],
                                               measurements["thresholded_packed"])]
    contention = other_compute_pids(uuid)
    if contention:
        raise RuntimeError(f"GPU acquired other compute processes during benchmark: {contention}")
    result = {
        "scope": "Repair stage only, FP32 CUDA events, same frozen checkpoint and random feature tensor; includes a matched thresholded fused dense baseline. No full decoder or codec latency.",
        "gpu_index": args.gpu, "gpu_uuid": uuid,
        "device_name": torch.cuda.get_device_name(device),
        "feature_shape": list(x.shape), "threshold": .25,
        "active_pointwise_fraction": next(iter(sparse._plans.values())).active_fraction,
        "max_abs_packed_vs_dense_thresholded": max_abs,
        "max_abs_fused_vs_dense_thresholded": fused_max_abs,
        "max_abs_trained_fused_vs_dense": trained_fused_max_abs,
        "trials": args.trials,
        "timings_ms": measurements,
        "medians_ms": {name: statistics.median(values) for name, values in measurements.items()},
        "paired_dense_thresholded_over_packed_mean_ratio": statistics.mean(ratios_raw),
        "paired_fused_thresholded_over_packed_mean_ratio": statistics.mean(ratios_fused),
        "paired_raw_ratio_mean_ci95": ci(ratios_raw),
        "paired_fused_ratio_mean_ci95": ci(ratios_fused),
        "checkpoint_sha256": sha(CHECKPOINT), "script_sha256": sha(Path(__file__)),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("medians_ms", "paired_dense_thresholded_over_packed_mean_ratio",
                                              "paired_fused_thresholded_over_packed_mean_ratio",
                                              "paired_fused_ratio_mean_ci95",
                                              "max_abs_packed_vs_dense_thresholded",
                                              "max_abs_fused_vs_dense_thresholded")}, indent=2))


if __name__ == "__main__":
    main()
