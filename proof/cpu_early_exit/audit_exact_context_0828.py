"""CPU-only counterfactual: exact feature context for DIV2K 0828 routed tiles.

This is an ablation, not the shipped decoder. Windows are clipped to the true
frame boundary and use stock zero padding; each core has enough feature context
for every suffix depthwise convolution. No quality/latency claim is inferred
from the context-area proxy for compute.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT / 'proof/depth_bitstream'), str(PROOF)]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=PROOF / '.local/DCVC')
    parser.add_argument('--extension', type=Path, default=PROOF / '.local/entropy')
    parser.add_argument('--e15', type=Path, default=PROOF / 'artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--release', type=Path, default=PROOF / 'artifacts/released_cvpr2026_image.pth.tar')
    parser.add_argument('--source-dir', type=Path, default=ROOT / 'data/DIV2K_valid_HR')
    args = parser.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map

    sys.path.insert(0, str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    row = json.loads(args.case.read_text())
    policy = json.loads(args.policy.read_text())
    if digest(args.stream) != row['stream_sha256']:
        raise RuntimeError('Stream hash mismatch')
    source_path = args.source_dir / row['image']
    if digest(source_path) != row['source_sha256']:
        raise RuntimeError('Source hash mismatch')
    choice = [c for c in row['candidates'] if c['beta'] == policy['beta'][str(row['qp'])]]
    if len(choice) != 1:
        raise RuntimeError('Policy beta absent or ambiguous')
    chosen = choice[0]
    x0, y0, width, height = row['crop_xywh']
    with Image.open(source_path) as image:
        rgb = np.asarray(image.convert('RGB').crop((x0, y0, x0 + width, y0 + height)),
                         dtype=np.float32) / 255
    source = torch.from_numpy(rgb2ycbcr_np(rgb) - .5).permute(2, 0, 1)[None].contiguous()

    released, info = load_model(args.release, 12, args.upstream)
    codec = ReferenceCodec(released, info['sha256'], args.extension)
    ckpt = torch.load(args.e15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ckpt['config'])
    model = FlexUFIntra(cfg).eval()
    load_flexuf_state(model, ckpt)
    dec = model.dec
    latent, q, _ = codec.decode_latent(args.stream.read_bytes())
    ntiles = len(chosen['exit_map'])
    route = torch.tensor(chosen['exit_map'], dtype=torch.long)
    deep = torch.full_like(route, cfg.num_exits - 1)

    def quality(image, ref_mse):
        return 10 * math.log10(float((source - image[:, :, :height, :width]).square().mean()) / ref_mse)

    def context_reconstruct(stem, exit_map, *, repair):
        """Full-context suffix, per tile; each halo is exact for that exit."""
        b, c, h, w = stem.shape
        assert b == 1 and h % cfg.feature_patch == 0 and w % cfg.feature_patch == 0
        side = cfg.feature_patch
        nh, nw = h // side, w // side
        assert ntiles == nh * nw
        canvas = torch.empty_like(stem)
        group_areas = [0] * cfg.num_exits
        tile_windows = []
        for tile, mode in enumerate(exit_map.tolist()):
            top, left = (tile // nw) * side, (tile % nw) * side
            # Two 3x3 depthwise convolutions per exit group; radius sums.
            halo = (mode - cfg.split_depth + 1) * cfg.blocks_per_exit
            t, l = max(0, top - halo), max(0, left - halo)
            bottom, right = min(h, top + side + halo), min(w, left + side + halo)
            work = stem[:, :, t:bottom, l:right].contiguous()
            area = (bottom - t) * (right - l)
            for group in range(cfg.split_depth, mode + 1):
                group_areas[group] += area
                work = dec.groups[group](work)
            work = dec._at_exit(work, mode)
            canvas[:, :, top:top + side, left:left + side] = (
                work[:, :, top - t:top - t + side, left - l:left - l + side])
            tile_windows.append([t, l, bottom, right])
        if repair and dec.seam_repair is not None:
            canvas = dec.seam_repair(canvas)
        return dec._apply_head(canvas, q), group_areas, tile_windows

    with torch.inference_mode():
        reference = dec.forward_full(latent, q)
        ref_mse = float((source - reference[:, :, :height, :width]).square().mean())
        stem = dec.upsample(latent)
        for group in dec.groups[:cfg.split_depth]:
            stem = group(stem)
        controls = {}
        for name, em in [('all_deep', deep), ('primary_route', route)]:
            no_context = forward_from_stem_with_cpu_map(dec, stem, q, em)
            context, areas, windows = context_reconstruct(stem, em, repair=False)
            context_repair, _, _ = context_reconstruct(stem, em, repair=True)
            no_context_db = quality(no_context, ref_mse)
            context_db = quality(context, ref_mse)
            repaired_db = quality(context_repair, ref_mse)
            # Compare the exact context output against the relevant full-frame
            # control only for uniform all-deep routing.
            controls[name] = {
                'exit_map': em.tolist(),
                'zero_halo_deployed_delta444_db': no_context_db,
                'exact_context_no_repair_delta444_db': context_db,
                'exact_context_with_repair_delta444_db': repaired_db,
                'exact_context_vs_full_max_abs': float((context - reference).abs().max()) if name == 'all_deep' else None,
                'per_suffix_group_feature_cell_areas': areas[cfg.split_depth:],
                'zero_halo_suffix_feature_cell_areas': [
                    cfg.feature_patch ** 2 * sum(int(v >= g) for v in em.tolist())
                    for g in range(cfg.split_depth, cfg.num_exits)],
                'tile_windows_tlbr': windows,
            }
            if name == 'all_deep' and not torch.allclose(context, reference, atol=1e-4, rtol=1e-4):
                raise RuntimeError('Sufficient-context all-deep output did not match full-frame')
        # Existing decoder's depthwise-only canvas coupling is a much cheaper
        # mechanism than haloing every pointwise layer. It is a counterfactual
        # here: the e15 checkpoint was trained for the deployed uncoupled path.
        original_cfg, original_repair = dec.cfg, dec.seam_repair
        try:
            dec.cfg = replace(cfg, tile_coupling=True, sorted_tiles=False,
                              tile_pad_mode='zeros', trunk_halo=0)
            for name, em in [('all_deep', deep), ('primary_route', route)]:
                dec.seam_repair = None
                coupled = dec(latent, q, exit_map=em)
                dec.seam_repair = original_repair
                coupled_repair = dec(latent, q, exit_map=em)
                controls[name]['canvas_coupled_no_repair_delta444_db'] = quality(coupled, ref_mse)
                controls[name]['canvas_coupled_with_repair_delta444_db'] = quality(coupled_repair, ref_mse)
                controls[name]['canvas_coupled_no_repair_vs_full_max_abs'] = (
                    float((coupled-reference).abs().max()) if name == 'all_deep' else None)
                if name == 'all_deep' and not torch.allclose(coupled, reference, atol=1e-4, rtol=1e-4):
                    raise RuntimeError('Canvas-coupled all-deep output did not match full-frame')
        finally:
            dec.cfg, dec.seam_repair = original_cfg, original_repair
    result = {
        'schema': 1,
        'scope': 'CPU FP32, DIV2K 0828 crop, released analysis/entropy stream, e15 synthesis; exact-context counterfactual; no latency measured',
        'case_sha256': digest(args.case), 'stream_sha256': digest(args.stream),
        'source_sha256': digest(source_path), 'e15_sha256': digest(args.e15),
        'release_sha256': digest(args.release), 'policy_sha256': digest(args.policy),
        'image': row['image'], 'qp': row['qp'], 'crop_xywh': row['crop_xywh'],
        'feature_patch': cfg.feature_patch, 'split_depth': cfg.split_depth,
        'blocks_per_exit': cfg.blocks_per_exit,
        'zero_halo_cached_primary_delta444_db': chosen['delta444_db'],
        'controls': controls,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
