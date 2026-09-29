"""Twelve CPU neural-output smoke cases for prepared fragmentation controls.

Two smallest-area frames with a nonzero router geometry contrast, selected
without quality outcomes. Not an actual-stream, native timing, or cohort test.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
BASE = Path("/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research")
ORIGINAL = BASE / "shared_crossfit_qp32"
SNAPSHOT = BASE / "shared_metric_audit/source"
UPSTREAM = Path("/home/can_karsal/DCVC")
PLAN = REPO / "cvpr2027/data/ablation20260929/fragmentation_maps.json"
OUT = REPO / "cvpr2027/data/ablation20260929/fragmentation_smoke.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    if OUT.exists():
        raise ValueError("Refusing to overwrite an existing smoke result")
    torch.set_num_threads(2); torch.manual_seed(42)
    plan = json.loads(PLAN.read_text())
    if plan["exit_labels"] != [2, 3, 4, 5] or plan["n_scheduled_maps"] != 318:
        raise ValueError("Unexpected map protocol")
    manifest = json.loads((ORIGINAL / "manifest.json").read_text())
    for path, digest in manifest["source_sha256"].items():
        if sha(path) != digest:
            raise ValueError("Original replay dependency changed: " + path)
    pinned = "819c219b24db34310bbd15c51a720aaaf5eb2e7d"
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip() != pinned:
        raise ValueError("Shared upstream changed")
    if subprocess.check_output(["git", "-C", str(UPSTREAM), "diff", "HEAD", "--", "src"]):
        raise ValueError("Shared upstream sources dirty")
    checkpoint = REPO / "runs/RECIPE512/ckpt_PIN_e15.pth.tar"
    source_paths = [Path(__file__), PLAN, checkpoint, ORIGINAL / "manifest.json",
                    *sorted(SNAPSHOT.rglob("*.py")), *sorted((UPSTREAM / "src").rglob("*.py"))]
    frozen = {str(p): sha(p) for p in source_paths}
    candidates = [c for c in plan["cases"] if c["policy"] == "router" and
                  c["rows"][1]["cross_exit_interface_length_rgb_pixels"] != c["rows"][2]["cross_exit_interface_length_rgb_pixels"]]
    selected = sorted(candidates, key=lambda c: (c["height"] * c["width"], c["sequence"]))[:2]
    if len(selected) != 2:
        raise ValueError("Insufficient nontrivial engineering cases")
    sys.path.insert(0, str(SNAPSHOT)); sys.path.insert(0, str(UPSTREAM))
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra
    from src.utils.transforms import ycbcr2rgb
    import ctc_intra as C
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    if (cfg.num_exits, cfg.split_depth, cfg.rgb_patch) != (6, 2, 256):
        raise ValueError("Unexpected checkpoint geometry")
    net = FlexUFIntra(cfg).eval()
    net.load_state_dict(ck.get("state_dict", ck.get("net", ck)), strict=True)
    sequences, _ = C.discover([])
    lookup = {Path(s["path"]).name: s for s in sequences}
    rows = []
    with torch.inference_mode():
        for frame in selected:
            name, h, w = frame["sequence"], frame["height"], frame["width"]
            casefile = ORIGINAL / "cases" / f"{frame['sequence_index']:02d}.json"
            if sha(casefile) != plan["source_cases_sha256"][casefile.name]:
                raise ValueError("Source case changed")
            frozen[str(casefile)] = sha(casefile)
            original = json.loads(casefile.read_text())
            with Path(lookup[name]["path"]).open("rb") as stream:
                if hashlib.sha256(stream.read(h * w * 3 // 2)).hexdigest() != frame["frame_sha256"]:
                    raise ValueError("Source image changed")
            x, planes = C.read_frames(lookup[name]["path"], w, h, 1, 1)
            xp = F.pad(x, (0, (-w) % 256, 0, (-h) % 256), mode="replicate")
            y, q, _ = net._encode_to_latent(xp, torch.tensor([32], dtype=torch.int32))
            target = ycbcr2rgb(x + .5, clamp=True)
            cache = {}
            for case in [c for c in plan["cases"] if c["sequence"] == name]:
                baseline = next(r for r in original["rows"] if (r["criterion"], r["policy"]) == ("q90", case["policy"]))
                for variant in case["rows"]:
                    key = tuple(variant["map"])
                    if key not in cache:
                        output = net.dec(y, q, exit_map=torch.tensor(key, dtype=torch.long))[:, :, :h, :w]
                        rgb = ycbcr2rgb(output.clamp(-.5, .5) + .5, clamp=True)
                        if not torch.isfinite(output).all():
                            raise ValueError("Nonfinite reconstruction")
                        cache[key] = {"mse_rgb": float((rgb - target).square().mean()),
                                      "psnr_yuv611_420": float(C.psnr_611_420(output, planes[0]))}
                    metrics = cache[key]
                    baseline_ok = None
                    if variant["variant"] == "original":
                        baseline_ok = abs(metrics["mse_rgb"] - baseline["cropped_rgb_mse"]) <= 1e-12
                        loss = original["anchor_cropped_yuv611_psnr"] - metrics["psnr_yuv611_420"]
                        baseline_ok = baseline_ok and abs(loss - baseline["cropped_yuv611_loss_db"]) <= 1e-6
                        if not baseline_ok:
                            raise ValueError("Original map does not reproduce RGB/YUV baseline")
                    rows.append({"sequence": name, "policy": case["policy"], "variant": variant["variant"],
                                 "baseline_reproduced": baseline_ok, **metrics,
                                 "rgb_loss_vs_original_db": 10 * math.log10(metrics["mse_rgb"] / baseline["cropped_rgb_mse"]),
                                 "cross_exit_interface_length_rgb_pixels": variant["cross_exit_interface_length_rgb_pixels"]})
            print(json.dumps({"sequence": name, "completed_cases": len(rows), "distinct_outputs_for_frame": len(cache)}), flush=True)
    for path, digest in frozen.items():
        if sha(path) != digest:
            raise ValueError("Source changed while running")
    if torch.cuda.is_initialized() or len(rows) != 12:
        raise ValueError("Unexpected CUDA context or incomplete check")
    result = {"scope": __doc__, "state": "complete", "selection": "Two smallest-area router maps with a lower/higher interface contrast; ties by sequence name. Geometry-only selection.",
              "all_originals_reproduced": all(r["baseline_reproduced"] for r in rows if r["variant"] == "original"),
              "metric": "Original shared-exit RGB conversion and weighted per-plane YUV 4:2:0 PSNR. Does not change the bank's 4:4:4 metric.",
              "n_cases": len(rows), "cuda_initialized": False, "source_sha256": frozen, "rows": rows,
              "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
