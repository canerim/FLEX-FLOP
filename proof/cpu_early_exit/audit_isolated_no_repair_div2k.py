"""Matched DIV2K no-repair control for active-neighbour context experiments.

Decode the same frozen validation24 x five-QP FUFREF2 streams and exit maps
with the deployed isolated-tile path, once with the trained seam repair and
once without it. The with-repair reconstruction is checked against the
archived deployed result. This isolates the effect of boundary context from
the effect of disabling repair; it is CPU FP32 and untimed.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT / 'proof/depth_bitstream'), str(PROOF), str(HERE)]
from audit_exact_context_pilot import sha


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-new-cases', type=int, default=120)
    ap.add_argument('--output', type=Path, default=HERE / 'results/div2k_beta/quality_floor/isolated_no_repair_validation24.json')
    args = ap.parse_args()
    assert 1 <= args.max_new_cases <= 120

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state

    exact_path = HERE / 'results/div2k_beta/quality_floor/exact_context_validation24.json'
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
    manifest = json.loads(manifest_path.read_text())
    assert len(exact['rows']) == 120 and len(exact['images']) == 24
    assert exact['qps'] == manifest['qps'] == [0, 16, 32, 48, 63]
    manifest_images = {r['image']: r for r in manifest['rows']['validation']}
    assert set(exact['images']) == set(manifest_images)
    critical = [Path(__file__), ROOT / 'flexuf/backbone/decoder.py',
                ROOT / 'flexuf/model.py', ROOT / 'flexuf/config.py']
    provenance = {
        'exact_sha256': sha(exact_path), 'manifest_sha256': sha(manifest_path),
        'released_sha256': sha(released_path), 'e15_sha256': sha(e15_path),
        'source_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in critical},
    }
    assert exact['provenance']['manifest_sha256'] == provenance['manifest_sha256']
    assert exact['provenance']['release_sha256'] == provenance['released_sha256']
    assert exact['provenance']['e15_sha256'] == provenance['e15_sha256']
    result = {
        'scope': 'DIV2K validation24 x QP5, isolated tiles with frozen replicate padding, repair on/off; exact frozen FUFREF2 streams/maps, untimed CPU FP32.',
        'expected_cases': 120, 'complete': False, 'provenance': provenance,
        'rows': [],
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
    assert not cfg.tile_coupling and cfg.tile_pad_mode == 'replicate'
    model = FlexUFIntra(cfg).eval()
    load_flexuf_state(model, checkpoint)
    dec = model.dec
    original_repair = dec.seam_repair
    assert original_repair is not None
    source_cache = {}
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
                case = json.loads(case_path.read_text())
                assert sha(case_path) == archived['case_sha256']
                assert sha(stream_path) == archived['stream_sha256'] == case['stream_sha256']
                route = torch.tensor(archived['exit_map'], dtype=torch.long)
                latent, quant_step, _ = codec.decode_latent(stream_path.read_bytes())
                full = dec.forward_full(latent, quant_step)
                h, w = source.shape[-2:]
                full_mse = float((source-full[:, :, :h, :w]).square().mean())
                assert math.isclose(full_mse, case['e15_full_mse444'], abs_tol=1e-11)
                values = {}
                for repair in (True, False):
                    dec.seam_repair = original_repair if repair else None
                    output = dec(latent, quant_step, exit_map=route)
                    mse = float((source-output[:, :, :h, :w]).square().mean())
                    values['with_repair' if repair else 'no_repair'] = 10*math.log10(mse/full_mse)
                assert abs(values['with_repair'] - archived['deployed_delta444_db']) < 1e-6, stem
                result['rows'].append({
                    'image': image, 'qp': qp,
                    'case_sha256': sha(case_path), 'stream_sha256': sha(stream_path),
                    'exit_map': route.tolist(),
                    'isolated_with_repair_delta444_db': values['with_repair'],
                    'isolated_no_repair_delta444_db': values['no_repair'],
                    'exact_context_delta444_db': archived['exact_context_delta444_db'],
                })
                added += 1
                result['complete'] = len(result['rows']) == 120
                args.output.write_text(json.dumps(result, indent=2) + '\n')
                print(f'{stem}: isolated no-repair {values["no_repair"]:.6f} dB', flush=True)
    finally:
        dec.seam_repair = original_repair
    assert provenance['source_code_sha256'] == {str(p.relative_to(ROOT)): sha(p)
                                                 for p in critical}
    print(f"complete={result['complete']}; cases={len(result['rows'])}")


if __name__ == '__main__':
    main()
