"""Measure where active context changes reconstruction error on frozen DIV2K.

Matched isolated/no-repair and active-replicate/no-repair outputs are replayed
from the same e15 model, frozen exit map and FUFREF2 bitstream. Pixel MSE is
binned by distance to an *internal* 256-pixel tile boundary. This is an
untimed exploratory mechanism audit, not a new routing evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT / 'proof/depth_bitstream'),
                str(PROOF), str(HERE)]
from audit_exact_context_pilot import sha
from active_canvas_replicate import ActiveCanvasReplicateCoupler

BINS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, None))


def masks_for_internal_seams(h: int, w: int, tile: int):
    import torch
    assert h % tile == w % tile == 0 and h >= 2*tile and w >= 2*tile
    x = torch.arange(w)
    y = torch.arange(h)
    x_boundaries = list(range(tile, w, tile))
    y_boundaries = list(range(tile, h, tile))
    assert x_boundaries and y_boundaries
    dx = torch.stack([(x-b).abs() for b in x_boundaries]).min(dim=0).values
    dy = torch.stack([(y-b).abs() for b in y_boundaries]).min(dim=0).values
    distance = torch.minimum(dy[:, None], dx[None, :])
    masks = []
    for lower, upper in BINS:
        masks.append((distance >= lower) &
                     ((distance < upper) if upper is not None else True))
    assert torch.stack(masks).sum(dim=0).eq(1).all()
    return masks


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-new-cases', type=int, default=120)
    ap.add_argument('--output', type=Path,
                    default=HERE / 'results/div2k_beta/quality_floor/context_seam_locality_validation24.json')
    args = ap.parse_args()
    assert 1 <= args.max_new_cases <= 120

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    import flexuf.backbone.coupling as coupling_module

    exact_path = HERE / 'results/div2k_beta/quality_floor/exact_context_validation24.json'
    isolated_path = HERE / 'results/div2k_beta/quality_floor/isolated_no_repair_validation24.json'
    active_path = HERE / 'results/div2k_beta/quality_floor/active_canvas_validation24_replicate.json'
    manifest_path = HERE / 'results/div2k_beta/manifest.json'
    cases = Path('/tmp/flexplus-div2k-beta-validation')
    source_dir = ROOT / 'data/DIV2K_valid_HR'
    upstream = Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC')
    extension = Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy')
    released_path = PROOF / 'artifacts/released_cvpr2026_image.pth.tar'
    e15_path = PROOF / 'artifacts/e15_epoch15.pth.tar'
    sys.path.insert(0, str(upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    exact = json.loads(exact_path.read_text())
    isolated = json.loads(isolated_path.read_text())
    active = json.loads(active_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    assert len(exact['rows']) == 120 and isolated['complete'] and active['complete']
    assert len(isolated['rows']) == len(active['rows']) == 120
    assert exact['qps'] == manifest['qps'] == [0, 16, 32, 48, 63]
    assert isolated['provenance']['exact_sha256'] == active['provenance']['exact_sha256'] == sha(exact_path)
    by_iso = {(r['image'], r['qp']): r for r in isolated['rows']}
    by_active = {(r['image'], r['qp']): r for r in active['rows']}
    assert len(by_iso) == len(by_active) == 120
    manifest_images = {r['image']: r for r in manifest['rows']['validation']}
    assert len(manifest_images) == 24
    critical = [Path(__file__), HERE / 'active_canvas_replicate.py',
                ROOT / 'flexuf/backbone/decoder.py',
                ROOT / 'flexuf/backbone/coupling.py',
                ROOT / 'flexuf/model.py', ROOT / 'flexuf/config.py']
    provenance = {
        'exact_sha256': sha(exact_path), 'isolated_sha256': sha(isolated_path),
        'active_sha256': sha(active_path), 'manifest_sha256': sha(manifest_path),
        'released_sha256': sha(released_path), 'e15_sha256': sha(e15_path),
        'source_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in critical},
    }
    result = {
        'scope': 'Exploratory DIV2K validation24 x five-QP pixel-error locality from matched frozen isolated/no-repair and active-replicate/no-repair outputs. Untimed CPU FP32, no native bitstream or rate claim.',
        'expected_cases': 120, 'complete': False,
        'metric': 'YCbCr444 per-pixel channel-mean MSE; positive PSNR gain favours active-replicate',
        'distance_unit': 'RGB pixels to nearest internal tile boundary',
        'bins_lower_inclusive_upper_exclusive': [[lo, hi] for lo, hi in BINS],
        'provenance': provenance, 'rows': [],
    }
    if args.output.exists():
        old = json.loads(args.output.read_text())
        assert old['provenance'] == provenance and old['expected_cases'] == 120
        result['rows'] = old['rows']
    done = {(r['image'], r['qp']) for r in result['rows']}
    assert len(done) == len(result['rows'])
    released, info = load_model(released_path, 12, upstream)
    codec = ReferenceCodec(released, info['sha256'], extension)
    checkpoint = torch.load(e15_path, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**checkpoint['config'])
    assert cfg.rgb_patch == 256 and not cfg.tile_coupling
    assert cfg.tile_pad_mode == 'replicate' and cfg.seam_repair == 'grid'
    model = FlexUFIntra(cfg).eval()
    load_flexuf_state(model, checkpoint)
    dec = model.dec
    original_cfg, original_repair = dec.cfg, dec.seam_repair
    original_class = coupling_module.CanvasCoupler
    dec.seam_repair = None
    source_cache = {}
    mask_cache = {}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    try:
        with torch.inference_mode():
            for archived in exact['rows']:
                image, qp = archived['image'], archived['qp']
                if (image, qp) in done:
                    continue
                if added >= args.max_new_cases:
                    break
                iso_ref = by_iso[image, qp]
                active_ref = by_active[image, qp]
                assert (archived['exit_map'] == iso_ref['exit_map'] == active_ref['exit_map'])
                assert (archived['stream_sha256'] == iso_ref['stream_sha256'] == active_ref['stream_sha256'])
                im = manifest_images[image]
                source_path = source_dir / image
                assert sha(source_path) == im['source_sha256']
                if image not in source_cache:
                    x, y, w, h = im['crop_xywh']
                    with Image.open(source_path) as file:
                        rgb = np.asarray(file.convert('RGB').crop((x, y, x+w, y+h)),
                                         dtype=np.float32) / 255
                    source_cache[image] = torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2, 0, 1)[None].contiguous()
                source = source_cache[image]
                stem = f'{Path(image).stem}_qp{qp}'
                case_path = cases / f'{stem}.json'
                stream_path = cases / f'{stem}.fufref2'
                assert sha(case_path) == archived['case_sha256']
                assert sha(stream_path) == archived['stream_sha256']
                route = torch.tensor(archived['exit_map'], dtype=torch.long)
                latent, quant_step, _ = codec.decode_latent(stream_path.read_bytes())
                dec.cfg = cfg
                iso_output = dec(latent, quant_step, exit_map=route)
                coupling_module.CanvasCoupler = ActiveCanvasReplicateCoupler
                dec.cfg = replace(cfg, tile_coupling=True, sorted_tiles=False,
                                  trunk_halo=0)
                active_output = dec(latent, quant_step, exit_map=route)
                coupling_module.CanvasCoupler = original_class
                h, w = source.shape[-2:]
                iso_error = (source-iso_output[:, :, :h, :w]).square().mean(dim=1)[0]
                active_error = (source-active_output[:, :, :h, :w]).square().mean(dim=1)[0]
                iso_mse = float(iso_error.mean())
                active_mse = float(active_error.mean())
                gain = 10*math.log10(iso_mse/active_mse)
                archived_gain = (iso_ref['isolated_no_repair_delta444_db']-
                                 active_ref['coupled']['no_repair']['delta444_db'])
                assert abs(gain-archived_gain) < 1e-5, (stem, gain, archived_gain)
                if (h, w) not in mask_cache:
                    mask_cache[h, w] = masks_for_internal_seams(h, w, cfg.rgb_patch)
                bins = []
                for (lower, upper), mask in zip(BINS, mask_cache[h, w]):
                    a = float(iso_error[mask].mean())
                    b = float(active_error[mask].mean())
                    bins.append({'lower_px': lower, 'upper_px': upper,
                                 'pixels': int(mask.sum()),
                                 'isolated_mse444': a, 'active_mse444': b,
                                 'active_gain_db': 10*math.log10(a/b)})
                assert sum(item['pixels'] for item in bins) == h*w
                result['rows'].append({
                    'image': image, 'qp': qp, 'case_sha256': sha(case_path),
                    'stream_sha256': sha(stream_path), 'exit_map': route.tolist(),
                    'crop_hw': [h, w], 'full_image_gain_db': gain,
                    'archived_full_image_gain_db': archived_gain, 'bins': bins,
                })
                added += 1
                result['complete'] = len(result['rows']) == 120
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print(f'{stem}: full +{gain:.5f} dB; near-seam +{bins[0]["active_gain_db"]:.5f} dB', flush=True)
    finally:
        dec.cfg, dec.seam_repair = original_cfg, original_repair
        coupling_module.CanvasCoupler = original_class
    assert provenance['source_code_sha256'] == {str(p.relative_to(ROOT)): sha(p)
                                                 for p in critical}
    print(f"complete={result['complete']}; cases={len(result['rows'])}")


if __name__ == '__main__':
    main()
