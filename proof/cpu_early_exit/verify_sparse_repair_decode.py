"""Stream-to-reconstruction smoke test of the sparse repair wrapper on CPU."""
from __future__ import annotations

import hashlib
import json
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
sys.path[:0] = [str(SNAPSHOT), str(UPSTREAM), str(ROOT)]
SOURCE = ROOT / "cvpr2027/data/research20260927/component_interventions/analysis.json"
PILOT = ROOT / "proof/cpu_early_exit/results/sparse_repair_gate_pilot_20261005.json"
CHECKPOINT = ROOT / "runs/RECIPE512/ckpt_PIN_e15.pth.tar"
OUT = ROOT / "proof/cpu_early_exit/results/sparse_repair_decode_smoke_20261005.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    import torch
    import torch.nn.functional as F
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    from src.utils.transforms import ycbcr2rgb
    from proof.cpu_early_exit.sparse_grid_repair import SparseGridRepairWrapper
    import ctc_intra as C

    torch.set_num_threads(1)
    source = json.loads(SOURCE.read_text())
    case = next(r for r in source["cases"] if r["sequence"] == "BQSquare_416x240_60.yuv")
    pilot = json.loads(PILOT.read_text())
    expected = next(r for r in pilot["rows"] if r["sequence"] == case["sequence"]
                    and r["threshold"] == .25)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**checkpoint["config"])
    net = FlexUFIntra(cfg).eval()
    net.load_state_dict(checkpoint.get("state_dict", checkpoint.get("net", checkpoint)), strict=True)
    trained_repair = net.dec.seam_repair
    wrapper = SparseGridRepairWrapper(trained_repair, threshold=.25).eval()
    net.dec.seam_repair = wrapper
    captured = []
    hook = wrapper.register_forward_pre_hook(lambda _module, args: captured.append(args[0].detach().clone()))
    sequences, _ = C.discover([])
    path = next(Path(r["path"]) for r in sequences if Path(r["path"]).name == case["sequence"])
    h, w = case["height"], case["width"]
    with torch.inference_mode():
        x, _ = C.read_frames(path, w, h, 1, 1)
        xp = F.pad(x, (0, (-w) % cfg.rgb_patch, 0, (-h) % cfg.rgb_patch), mode="replicate")
        y, q, _ = net._encode_to_latent(xp, torch.tensor([32], dtype=torch.int32))
        exit_map = torch.tensor(case["map"], dtype=torch.long)
        decoded = net.dec(y, q, exit_map=exit_map)
        hook.remove()
        assert len(captured) == 1
        feat = captured[0]
        gate = trained_repair.gate.repeat(1, 1, -(-feat.shape[-2] // 32),
                                          -(-feat.shape[-1] // 32))[:, :, :feat.shape[-2], :feat.shape[-1]]
        correction = trained_repair.pw(trained_repair.act(trained_repair.dw(feat)))
        dense = net.dec._apply_head(feat + gate * (gate >= .25) * correction, q)
        max_abs = float((decoded - dense).abs().max())
        assert max_abs < 2e-6
        target = ycbcr2rgb(x + .5, clamp=True)
        rgb = ycbcr2rgb(decoded[:, :, :h, :w].clamp(-.5, .5) + .5, clamp=True)
        mse = float((rgb - target).square().mean())
        assert abs(mse - expected["rgb_mse"]) < 1e-9
        assert len(wrapper._plans) == 1
        active_fraction = next(iter(wrapper._plans.values())).active_fraction
        assert abs(active_fraction - expected["active_pointwise_fraction"]) < 1e-12
    assert not torch.cuda.is_initialized()
    result = {
        "scope": "Frozen e15 checkpoint, archived QP32 Q90 route, source-derived latent, CPU FP32; sparse wrapper installed after strict checkpoint load.",
        "sequence": case["sequence"], "threshold": .25,
        "dense_thresholded_vs_sparse_max_abs": max_abs,
        "cropped_rgb_mse": mse, "pilot_cropped_rgb_mse": expected["rgb_mse"],
        "active_pointwise_fraction": active_fraction, "cuda_initialized": False,
        "runtime_measured": False,
        "source_sha256": sha(SOURCE), "pilot_sha256": sha(PILOT),
        "checkpoint_sha256": sha(CHECKPOINT), "script_sha256": sha(Path(__file__)),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
