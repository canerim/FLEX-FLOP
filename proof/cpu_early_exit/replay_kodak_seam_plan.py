"""Untimed Kodak quality replay of an already frozen seam-regularized map plan."""
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

PLAN = HERE / "results/kodak_seam_regularized_plan_20261005.json"
OUT = HERE / "results/kodak_seam_regularized_quality_20261005"
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
    from src.utils.transforms import rgb2ycbcr_np, ycbcr2rgb

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    plan = json.loads(PLAN.read_text())
    assert plan["n_cases"] == len(plan["rows"]) == 120
    assert plan["changed_cases"] == 14 and plan["lambda_tv"] == .5
    scan_manifest_path = SCAN / "manifest.json"
    assert plan["scan_manifest_sha256"] == digest(scan_manifest_path)
    scan_manifest = json.loads(scan_manifest_path.read_text())
    placement_manifest_path = PLACEMENT / "manifest.json"
    placement_manifest = json.loads(placement_manifest_path.read_text())
    assert len(list(PLACEMENT.glob("kodim*_qp*.json"))) == 120
    release_path = PROOF / "artifacts/released_cvpr2026_image.pth.tar"
    e15_path = PROOF / "artifacts/e15_epoch15.pth.tar"
    manifest = {
        "scope": "Untimed CPU quality replay of 120 frozen, same-histogram Kodak map plans, 14 changed. No map selection or quality tuning occurs here.",
        "plan_sha256": digest(PLAN),
        "placement_manifest_sha256": digest(placement_manifest_path),
        "scan_manifest_sha256": digest(scan_manifest_path),
        "script_sha256": digest(Path(__file__)),
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

    for item in plan["rows"]:
        image, qp = item["image"], item["qp"]
        stem_name = f"{Path(image).stem}_qp{qp}"
        scan_path = SCAN / f"{stem_name}.json"
        assert item["scan_result_sha256"] == digest(scan_path)
        source_path = ROOT / "data/kodak" / image
        scan_row = json.loads(scan_path.read_text())
        assert scan_row["source_sha256"] == digest(source_path)
        placement_path = PLACEMENT / f"{stem_name}.json"
        placement = json.loads(placement_path.read_text())
        assert placement["manifest_sha256"] == digest(placement_manifest_path)
        assert placement["scan_result_sha256"] == digest(scan_path)
        assert placement["exit_map"] == item["original"]
        assert placement["exit_counts"] == [item["selected"].count(k) for k in range(6)]
        result_path = OUT / f"{stem_name}.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            assert result["manifest_sha256"] == manifest_hash
            assert result["placement_result_sha256"] == digest(placement_path)
            print(json.dumps({"resumed": stem_name}), flush=True)
            continue
        baseline = {k: placement["original"][k] for k in ("mse_rgb", "mse_444")}
        if not item["changed"]:
            measured = baseline
        else:
            rgb_array = np.asarray(Image.open(source_path).convert("RGB"), dtype=np.float32) / 255
            h, w = rgb_array.shape[:2]
            assert [h, w] == item["shape"]
            source = torch.from_numpy(rgb2ycbcr_np(rgb_array) - .5).permute(2, 0, 1)[None].contiguous()
            target_rgb = ycbcr2rgb(source + .5, clamp=True)
            stream_path = SCAN / f"{stem_name}.fufref2"
            assert placement["stream_sha256"] == digest(stream_path)
            y, q, _ = codec.decode_latent(stream_path.read_bytes())
            with torch.inference_mode():
                stem = e15.dec.upsample(y)
                for group in e15.dec.groups[:cfg.split_depth]:
                    stem = group(stem)
                def measure(mode):
                    out = forward_from_stem_with_cpu_map(
                        e15.dec, stem, q, torch.tensor(mode, dtype=torch.long))[:, :, :h, :w]
                    rgb = ycbcr2rgb(out.clamp(-.5, .5) + .5, clamp=True)
                    return {"mse_rgb": float((rgb - target_rgb).square().mean()),
                            "mse_444": float((out - source).square().mean())}
                reproduced = measure(item["original"])
                assert abs(reproduced["mse_rgb"] - baseline["mse_rgb"]) < 1e-12
                assert abs(reproduced["mse_444"] - baseline["mse_444"]) < 1e-12
                measured = measure(item["selected"])
        result = {"schema": 1, "image": image, "qp": qp,
                  "manifest_sha256": manifest_hash,
                  "placement_result_sha256": digest(placement_path),
                  "plan_sha256": digest(PLAN),
                  "changed": item["changed"],
                  "original_map": item["original"],
                  "selected_map": item["selected"],
                  "exit_counts": placement["exit_counts"],
                  "mac_saved_pct": placement["mac_saved_pct"],
                  "original": baseline, "seam_regularized": measured,
                  "cpu_timing": None}
        write(result_path, result)
        print(json.dumps({"finished": stem_name, "changed": item["changed"]}), flush=True)
    assert not torch.cuda.is_initialized()
    assert len(list(OUT.glob("kodim*_qp*.json"))) == 120
    print(json.dumps({"completed": 120, "out": str(OUT)}), flush=True)


if __name__ == "__main__":
    main()
