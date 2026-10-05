"""Kodak matched-histogram Bayer control for the frozen decoder-side router.

This is a deterministic spatial-placement control, not the archived scalar
dither policy: it receives exactly the router's six depth counts for each
image/QP and assigns deeper exits to lower fixed Bayer ranks. No source
quality or router score influences the ordering after counts are fixed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from proof.cpu_early_exit.kodak_placement_replay import (
    ROOT, HERE, PROOF, UPSTREAM, EXTENSION, SCAN, OUT as PLACEMENT, digest, write,
)

OUT = HERE / "results/kodak24_qp5_bayer_histogram_20261005"
sys.path[:0] = [str(ROOT), str(ROOT / "proof/depth_bitstream"), str(PROOF), str(UPSTREAM)]


def main() -> None:
    import numpy as np
    from PIL import Image
    import torch
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map
    from flexplus.null_models_uf import bayer
    from src.utils.transforms import rgb2ycbcr_np, ycbcr2rgb

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    placement_manifest_path = PLACEMENT / "manifest.json"
    placement_manifest = json.loads(placement_manifest_path.read_text())
    assert len(list(PLACEMENT.glob("kodim*_qp*.json"))) == 120
    scan_manifest_path = SCAN / "manifest.json"
    scan_manifest = json.loads(scan_manifest_path.read_text())
    assert scan_manifest["cases"] == 120
    release_path = PROOF / "artifacts/released_cvpr2026_image.pth.tar"
    e15_path = PROOF / "artifacts/e15_epoch15.pth.tar"
    manifest = {
        "scope": "Kodak24 x QP5 fixed bitstreams; each Bayer-rank map uses the frozen router map's exact exit histogram and therefore its exact convolution MAC. Deeper exits fill lower Bayer ranks first. No tuning, timing, or new bitstreams.",
        "limitation": "The matched-histogram Bayer control is distinct from the archived scalar ordered-dither policy, which chooses its own adjacent-depth histogram. Results isolate placement within the router's existing depth counts.",
        "script_sha256": digest(Path(__file__)),
        "bayer_source_sha256": digest(ROOT / "flexplus/null_models_uf.py"),
        "placement_manifest_sha256": digest(placement_manifest_path),
        "scan_manifest_sha256": digest(scan_manifest_path),
        "release_sha256": digest(release_path),
        "e15_sha256": digest(e15_path),
        "upstream_head": placement_manifest["upstream_head"],
        "entropy_build_manifest_sha256": placement_manifest["entropy_build_manifest_sha256"],
        "python": sys.version,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    manifest_path = OUT / "manifest.json"
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, "Run source changed"
    else:
        write(manifest_path, manifest)
    manifest_hash = digest(manifest_path)

    released, info = load_model(release_path, 12, UPSTREAM)
    codec = ReferenceCodec(released, info["sha256"], EXTENSION)
    checkpoint = torch.load(e15_path, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**checkpoint["config"])
    e15 = FlexUFIntra(cfg).eval()
    load_flexuf_state(e15, checkpoint)
    from dataclasses import replace
    e15.dec.cfg = replace(cfg, sorted_tiles=True)
    assert cfg.rgb_patch == 256 and cfg.num_exits == 6

    for image in scan_manifest["images"]:
        path = ROOT / "data/kodak" / image
        rgb_array = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        h, w = rgb_array.shape[:2]
        assert h * w == 6 * 256**2
        source = torch.from_numpy(rgb2ycbcr_np(rgb_array) - .5).permute(2, 0, 1)[None].contiguous()
        target_rgb = ycbcr2rgb(source + .5, clamp=True)
        rank = bayer(h // 256, w // 256)
        assert len(rank) == 6
        positions = torch.argsort(rank, stable=True)
        for qp in scan_manifest["qps"]:
            stem_name = f"{Path(image).stem}_qp{qp}"
            scan_row = json.loads((SCAN / f"{stem_name}.json").read_text())
            assert scan_row["source_sha256"] == digest(path)
            assert scan_row["manifest_sha256"] == digest(scan_manifest_path)
            placement_path = PLACEMENT / f"{stem_name}.json"
            placement = json.loads(placement_path.read_text())
            assert placement["manifest_sha256"] == digest(placement_manifest_path)
            assert placement["stream_sha256"] == digest(SCAN / f"{stem_name}.fufref2")
            result_path = OUT / f"{stem_name}.json"
            if result_path.exists():
                row = json.loads(result_path.read_text())
                assert row["manifest_sha256"] == manifest_hash
                assert row["placement_result_sha256"] == digest(placement_path)
                print(json.dumps({"resumed": stem_name}), flush=True)
                continue
            original = torch.tensor(placement["exit_map"], dtype=torch.long)
            fixed = torch.empty_like(original)
            fixed[positions] = original.sort(descending=True).values
            assert torch.equal(torch.bincount(fixed, minlength=cfg.num_exits),
                               torch.bincount(original, minlength=cfg.num_exits))
            if torch.equal(fixed, original):
                measured = placement["original"]
                same_map = True
            else:
                stream = (SCAN / f"{stem_name}.fufref2").read_bytes()
                y, q, _ = codec.decode_latent(stream)
                with torch.inference_mode():
                    stem = e15.dec.upsample(y)
                    for group in e15.dec.groups[:cfg.split_depth]:
                        stem = group(stem)
                    def measure(route):
                        out = forward_from_stem_with_cpu_map(e15.dec, stem, q, route)[:, :, :h, :w]
                        rgb = ycbcr2rgb(out.clamp(-.5, .5) + .5, clamp=True)
                        return {"mse_rgb": float((rgb - target_rgb).square().mean()),
                                "mse_444": float((out - source).square().mean())}
                    reproduced = measure(original)
                    assert abs(reproduced["mse_rgb"] - placement["original"]["mse_rgb"]) < 1e-12
                    assert abs(reproduced["mse_444"] - placement["original"]["mse_444"]) < 1e-12
                    measured = measure(fixed)
                same_map = False
            row = {"schema": 1, "image": image, "qp": qp,
                   "manifest_sha256": manifest_hash,
                   "placement_result_sha256": digest(placement_path),
                   "stream_sha256": placement["stream_sha256"],
                   "router_map": original.tolist(), "bayer_map": fixed.tolist(),
                   "exit_counts": placement["exit_counts"],
                   "mac_saved_pct": placement["mac_saved_pct"],
                   "same_map": same_map,
                   "router": {k: placement["original"][k] for k in ("mse_rgb", "mse_444")},
                   "bayer": {k: measured[k] for k in ("mse_rgb", "mse_444")},
                   "cpu_timing": None}
            write(result_path, row)
            print(json.dumps({"finished": stem_name, "same_map": same_map}), flush=True)
    assert not torch.cuda.is_initialized()
    assert len(list(OUT.glob("kodim*_qp*.json"))) == 120
    print(json.dumps({"completed": 120, "out": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
