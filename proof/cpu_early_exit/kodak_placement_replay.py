"""External Kodak24 x QP5 router-placement intervention on pinned FUFREF2 bytes.

For each already measured Kodak case, compare the frozen routed map to three
permutations of its six exit assignments. The six tiles have equal valid
extent, so each permutation holds the exit histogram and synthesis MAC fixed.
This is an untimed CPU replay and does not retune the router or control.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / "proof/early_exit_vs_released"
SCAN = HERE / "results/kodak24_qp5"
OUT = HERE / "results/kodak24_qp5_placement_20261005"
SEEDS = (202610050, 202610051, 202610052)
sys.path[:0] = [str(ROOT), str(ROOT / "proof/depth_bitstream"), str(PROOF)]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def main() -> None:
    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from matched_routed_cpu import psnr
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map

    sys.path.insert(0, str((PROOF / ".local/DCVC").resolve()))
    from src.utils.transforms import rgb2ycbcr_np, ycbcr2rgb

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    scan_manifest_path = SCAN / "manifest.json"
    scan_manifest = json.loads(scan_manifest_path.read_text())
    assert scan_manifest["cases"] == 120
    assert scan_manifest["images"] == [f"kodim{i:02d}.png" for i in range(1, 25)]
    assert scan_manifest["qps"] == [0, 16, 32, 48, 63]
    release_path = PROOF / "artifacts/released_cvpr2026_image.pth.tar"
    e15_path = PROOF / "artifacts/e15_epoch15.pth.tar"
    assert digest(release_path) == scan_manifest["released_sha256"]
    assert digest(e15_path) == scan_manifest["e15_sha256"]
    manifest = {
        "scope": "All 24 Kodak images x QP 0/16/32/48/63, fixed released bitstreams and calibrated decoder-side router maps. Original versus three fixed spatial permutations with exactly the same exit histogram; no retuning, timing or new bitstreams.",
        "limitation": "Kodak has six equal-size 256x256 tiles; three permutations sample assignment uncertainty but do not exhaust the 6! possibilities. Quality gain is relative to the frozen map and e15 weights, not an end-to-end codec speedup.",
        "scan_manifest_sha256": digest(scan_manifest_path),
        "script_sha256": digest(Path(__file__)),
        "release_sha256": digest(release_path),
        "e15_sha256": digest(e15_path),
        "seeds": list(SEEDS),
        "thread_count": 1,
        "metric": "Equal-image/quality-point mean of 10log10(mean shuffled RGB MSE / original RGB MSE), with matched YCbCr444 MSE also saved. Source RGB is ycbcr2rgb(centered YCbCr + .5, clamp=True).",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "manifest.json"
    if path.exists():
        assert json.loads(path.read_text()) == manifest, "Run source changed"
    else:
        write(path, manifest)
    manifest_hash = digest(path)

    released, info = load_model(release_path, 12, PROOF / ".local/DCVC")
    codec = ReferenceCodec(released, info["sha256"], PROOF / ".local/entropy")
    checkpoint = torch.load(e15_path, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**checkpoint["config"])
    e15 = FlexUFIntra(cfg).eval()
    load_flexuf_state(e15, checkpoint)
    from dataclasses import replace
    e15.dec.cfg = replace(cfg, sorted_tiles=True)
    assert cfg.rgb_patch == 256 and cfg.num_exits == 6
    assert all(k in e15.state_dict() and torch.equal(v, e15.state_dict()[k])
               for k, v in released.state_dict().items() if not k.startswith("dec."))

    for image in scan_manifest["images"]:
        image_path = ROOT / "data/kodak" / image
        rgb_array = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255
        h, w = rgb_array.shape[:2]
        assert (h, w) == (512, 768) or (h, w) == (768, 512)
        source = torch.from_numpy(rgb2ycbcr_np(rgb_array) - .5).permute(2, 0, 1)[None].contiguous()
        target_rgb = ycbcr2rgb(source + .5, clamp=True)
        for qp in scan_manifest["qps"]:
            stem_name = f"{Path(image).stem}_qp{qp}"
            scan_path = SCAN / f"{stem_name}.json"
            stream_path = SCAN / f"{stem_name}.fufref2"
            scan = json.loads(scan_path.read_text())
            assert scan["manifest_sha256"] == digest(scan_manifest_path)
            assert scan["source_sha256"] == digest(image_path)
            assert scan["stream_sha256"] == digest(stream_path)
            result_path = OUT / f"{stem_name}.json"
            if result_path.exists():
                row = json.loads(result_path.read_text())
                assert row["manifest_sha256"] == manifest_hash
                assert row["scan_result_sha256"] == digest(scan_path)
                print(json.dumps({"resumed": stem_name}), flush=True)
                continue
            scores = torch.tensor(scan["route_log_probs"], dtype=torch.float64)
            costs = torch.tensor(scan["router_costs"], dtype=torch.float64)
            beta = float(scan_manifest["calibrated_beta"][str(qp)])
            original = (scores - beta * costs[cfg.split_depth:][None]).argmax(1) + cfg.split_depth
            assert len(original) == 6
            assert torch.bincount(original, minlength=cfg.num_exits).tolist() == scan["exit_counts"]
            y, q, _ = codec.decode_latent(stream_path.read_bytes())
            with torch.inference_mode():
                stem = e15.dec.upsample(y)
                for group in e15.dec.groups[:cfg.split_depth]:
                    stem = group(stem)
                cache = {}
                def measure(route: torch.Tensor) -> dict:
                    key = tuple(route.tolist())
                    if key not in cache:
                        out = forward_from_stem_with_cpu_map(e15.dec, stem, q, route)[:, :, :h, :w]
                        rgb = ycbcr2rgb(out.clamp(-.5, .5) + .5, clamp=True)
                        cache[key] = {"mse_rgb": float((rgb - target_rgb).square().mean()),
                                      "mse_444": float((out - source).square().mean()),
                                      "yuv611_db": psnr(source, out)["yuv_6_1_1"]}
                    return cache[key]
                baseline = measure(original)
                assert abs(baseline["yuv611_db"] - scan["routed_yuv611_db"]) < 1e-5
                permutations = []
                for seed in SEEDS:
                    rng = random.Random(int(hashlib.sha256(f"{stem_name}:{seed}".encode()).hexdigest(), 16))
                    order = list(range(6))
                    rng.shuffle(order)
                    permuted = original[order]
                    assert torch.equal(torch.bincount(permuted, minlength=cfg.num_exits),
                                       torch.bincount(original, minlength=cfg.num_exits))
                    permutations.append({"seed": seed, "order": order, "map": permuted.tolist(),
                                         **measure(permuted)})
            row = {"schema": 1, "image": image, "qp": qp, "shape": [h, w],
                   "manifest_sha256": manifest_hash,
                   "scan_result_sha256": digest(scan_path),
                   "stream_sha256": scan["stream_sha256"],
                   "exit_map": original.tolist(), "exit_counts": scan["exit_counts"],
                   "mac_saved_pct": scan["mac_saved_pct"],
                   "original": baseline, "permutations": permutations,
                   "distinct_maps_executed": len(cache), "cpu_timing": None}
            write(result_path, row)
            print(json.dumps({"finished": stem_name, "distinct_maps": len(cache)}), flush=True)
    assert not torch.cuda.is_initialized()
    assert sum(1 for _ in OUT.glob("kodim*_qp*.json")) == 120
    print(json.dumps({"completed": 120, "out": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
