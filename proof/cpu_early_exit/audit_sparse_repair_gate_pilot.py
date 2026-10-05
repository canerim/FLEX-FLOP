"""CPU-only pilot: truncate the frozen grid-repair gate on three fixed frames.

The pointwise convolution is still executed densely here to measure quality.
Sparse pointwise MAC values are projections for a hypothetical exact sparse
kernel, not observed execution counts or latency.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

ROOT = Path(__file__).resolve().parents[2]
BASE = Path("/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research")
SNAPSHOT = BASE / "shared_metric_audit/source"
UPSTREAM = Path("/home/can_karsal/DCVC")
SOURCE = ROOT / "cvpr2027/data/research20260927/component_interventions/analysis.json"
CHECKPOINT = ROOT / "runs/RECIPE512/ckpt_PIN_e15.pth.tar"
OUT = ROOT / "proof/cpu_early_exit/results/sparse_repair_gate_pilot_20261005.json"
sys.path[:0] = [str(SNAPSHOT), str(UPSTREAM), str(ROOT / "proof/early_exit_vs_released")]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    from src.utils.transforms import ycbcr2rgb
    from mac_latency_audit import case_macs, CHANNELS, FEATURE_STRIDE
    import ctc_intra as C

    torch.set_num_threads(1)
    source = json.loads(SOURCE.read_text())
    all_cases = sorted(source["cases"], key=lambda r: (r["height"] * r["width"], r["sequence"]))
    cases = [all_cases[0], all_cases[len(all_cases) // 2], all_cases[-1]]
    assert len({r["sequence"] for r in cases}) == 3
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**checkpoint["config"])
    assert cfg.seam_repair == "grid" and cfg.full_frame_head
    net = FlexUFIntra(cfg).eval()
    net.load_state_dict(checkpoint.get("state_dict", checkpoint.get("net", checkpoint)), strict=True)
    gate = net.dec.seam_repair.gate.detach()
    assert tuple(gate.shape) == (1, 1, 32, 32)
    thresholds = [0.0, .24, .25, .30, .40, 1.0]
    sequences, _ = C.discover([])
    lookup = {Path(s["path"]).name: s for s in sequences}
    rows = []
    with torch.inference_mode():
        for case in cases:
            h, w = case["height"], case["width"]
            x, _ = C.read_frames(lookup[case["sequence"]]["path"], w, h, 1, 1)
            xp = F.pad(x, (0, (-w) % cfg.rgb_patch, 0, (-h) % cfg.rgb_patch), mode="replicate")
            y, q, _ = net._encode_to_latent(xp, torch.tensor([32], dtype=torch.int32))
            exit_map = torch.tensor(case["map"], dtype=torch.long)
            captured = []
            hook = net.dec.seam_repair.register_forward_pre_hook(
                lambda _module, args: captured.append(args[0].detach().clone()))
            try:
                baseline = net.dec(y, q, exit_map=exit_map)
            finally:
                hook.remove()
            assert len(captured) == 1
            feat = captured[0]
            module = net.dec.seam_repair
            correction = module.pw(module.act(module.dw(feat)))
            H, W = feat.shape[-2:]
            full_gate = gate.repeat(1, 1, -(-H // 32), -(-W // 32))[:, :, :H, :W]
            target = ycbcr2rgb(x + .5, clamp=True)
            variants = {row["variant"]: row for row in case["rows"]}
            counts = [case["map"].count(k) for k in range(6)]
            m = case_macs({"padded_shape": [H * FEATURE_STRIDE, W * FEATURE_STRIDE],
                           "tile_counts": counts})
            assert H * W == sum(counts) * 32**2
            for threshold in thresholds:
                active = (full_gate >= threshold)
                repaired = feat + full_gate * active * correction
                out = net.dec._apply_head(repaired, q)
                if threshold == 0:
                    assert torch.equal(out, baseline)
                rgb = ycbcr2rgb(out[:, :, :h, :w].clamp(-.5, .5) + .5, clamp=True)
                mse = float((rgb - target).square().mean())
                if threshold == 0:
                    assert abs(mse - variants["trained"]["mse_rgb"]) < 1e-9, (
                        case["sequence"], mse, variants["trained"]["mse_rgb"])
                if threshold == 1.0:
                    assert abs(mse - variants["repair_identity"]["mse_rgb"]) < 1e-9, (
                        case["sequence"], mse, variants["repair_identity"]["mse_rgb"])
                active_fraction = float(active.float().mean())
                pointwise_macs = H * W * CHANNELS**2
                projected_macs = m["e15_routed_conv_macs"] - (1 - active_fraction) * pointwise_macs
                rows.append({
                    "sequence": case["sequence"], "height": h, "width": w,
                    "threshold": threshold, "active_pointwise_fraction": active_fraction,
                    "rgb_mse": mse,
                    "rgb_psnr_loss_vs_dense_repair_db": 10 * math.log10(mse / variants["trained"]["mse_rgb"]),
                    "rgb_loss_vs_e15_dense_db": 10 * math.log10(mse /
                        (variants["trained"]["mse_rgb"] /
                         10**(variants["trained"]["loss_vs_dense_anchor_db"] / 10))),
                    "projected_exact_conv_mac_saving_pct": 100 * (1 - projected_macs / m["released_conv_macs"]),
                })
            print(json.dumps({"finished": case["sequence"], "rows": len(rows)}), flush=True)
    assert not torch.cuda.is_initialized()
    result = {
        "scope": "Three geometry-stratified QP32 frames from the frozen 53-frame intervention cohort; no outcome-based image selection. CPU-only gate truncation with dense computation for quality.",
        "limitations": "Projected MAC savings require an unimplemented sparse 1x1 kernel; no actual operation skipping, runtime, memory or full-codec result is measured. This is a development-cohort pilot, not a held-out result.",
        "thresholds": thresholds, "selected_sequences": [r["sequence"] for r in cases],
        "gate_shape": list(gate.shape),
        "gate_quantiles": np.quantile(gate.numpy(), [0,.1,.25,.5,.75,.9,1]).tolist(),
        "source_sha256": digest(SOURCE), "checkpoint_sha256": digest(CHECKPOINT),
        "snapshot_decoder_sha256": digest(SNAPSHOT / "flexuf/backbone/decoder.py"),
        "script_sha256": digest(Path(__file__)), "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(str(OUT), flush=True)


if __name__ == "__main__":
    main()
