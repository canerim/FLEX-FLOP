"""Replay active-neighbour depthwise coupling on Kodak bitstreams.

Only neighbours still executing the same suffix group supply context. The
frozen-weight, frozen-route CPU counterfactual does not measure latency or
establish a retrained production decoder.
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
sys.path[:0] = [str(ROOT), str(ROOT / 'proof/depth_bitstream'), str(PROOF)]
from audit_exact_context_pilot import sha
from active_canvas_coupling import ActiveCanvasCoupler


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan-dir', type=Path, default=HERE/'results/kodak24_qp5')
    p.add_argument('--exact', type=Path, default=HERE/'results/div2k_beta/quality_floor/exact_context_kodak24.json')
    p.add_argument('--source-dir', type=Path, default=ROOT/'data/kodak')
    p.add_argument('--upstream', type=Path, default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC'))
    p.add_argument('--extension', type=Path, default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy'))
    p.add_argument('--release', type=Path, default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    p.add_argument('--e15', type=Path, default=PROOF/'artifacts/e15_epoch15.pth.tar')
    p.add_argument('--output', type=Path, default=HERE/'results/kodak_active_canvas_20261005.json')
    p.add_argument('--max-new-cases', type=int, default=120)
    args = p.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from matched_routed_cpu import psnr
    import flexuf.backbone.coupling as coupling_module
    sys.path.insert(0, str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    exact = json.loads(args.exact.read_text())
    scan_manifest = json.loads((args.scan_dir/'manifest.json').read_text())
    assert len(exact['rows']) == 120 and scan_manifest['cases'] == 120
    assert exact['qps'] == scan_manifest['qps'] == [0, 16, 32, 48, 63]
    assert exact['images'] == scan_manifest['images']
    critical = [ROOT/'flexuf/backbone/decoder.py', ROOT/'flexuf/backbone/coupling.py',
                ROOT/'flexuf/model.py', ROOT/'flexuf/config.py', Path(__file__)]
    critical.append(HERE/'active_canvas_coupling.py')
    provenance = {'exact_sha256': sha(args.exact),
                  'scan_manifest_sha256': sha(args.scan_dir/'manifest.json'),
                  'released_sha256': sha(args.release), 'e15_sha256': sha(args.e15),
                  'source_code_sha256': {str(x.relative_to(ROOT)): sha(x) for x in critical}}
    assert provenance['scan_manifest_sha256'] == exact['provenance']['scan_manifest_sha256']
    assert provenance['released_sha256'] == exact['provenance']['release_sha256']
    assert provenance['e15_sha256'] == exact['provenance']['e15_sha256']
    result = {'schema': 1,
              'scope': 'Kodak24 x QP5 frozen FUFREF2 streams, DIV2K-locked early-exit maps, untimed CPU FP32 active-neighbour canvas coupling; no retraining',
              'quality_unit': 'Delta444 dB against full-frame e15; positive gain is lower MSE than deployed zero-halo decoder',
              'expected_cases': 120, 'complete': False, 'provenance': provenance, 'rows': []}
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if old['provenance'] != provenance or old['expected_cases'] != 120:
            raise RuntimeError('Existing output has different provenance')
        result['rows'] = old['rows']
    done = {(r['image'], r['qp']) for r in result['rows']}
    assert len(done) == len(result['rows'])

    released, info = load_model(args.release, 12, args.upstream)
    codec = ReferenceCodec(released, info['sha256'], args.extension)
    ckpt = torch.load(args.e15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ckpt['config'])
    model = FlexUFIntra(cfg).eval()
    load_flexuf_state(model, ckpt)
    dec = model.dec
    original_repair = dec.seam_repair
    original_cfg = dec.cfg
    original_coupler_class = coupling_module.CanvasCoupler
    coupling_module.CanvasCoupler = ActiveCanvasCoupler
    source_cache = {}
    added = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for archived in exact['rows']:
            image, qp = archived['image'], archived['qp']
            if (image, qp) in done:
                continue
            if added >= args.max_new_cases:
                break
            source_path = args.source_dir/image
            scan_path = args.scan_dir/f'{Path(image).stem}_qp{qp}.json'
            stream_path = args.scan_dir/f'{Path(image).stem}_qp{qp}.fufref2'
            scan = json.loads(scan_path.read_text())
            assert archived['scan_sha256'] == sha(scan_path)
            assert scan['stream_sha256'] == archived['stream_sha256'] == sha(stream_path)
            assert scan['source_sha256'] == sha(source_path)
            if image not in source_cache:
                rgb = np.asarray(Image.open(source_path).convert('RGB'), dtype=np.float32)/255
                source_cache[image] = torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
            source = source_cache[image]
            latent, q, _ = codec.decode_latent(stream_path.read_bytes())
            full = dec.forward_full(latent, q)
            h,w = scan['shape']
            full_mse = float((source-full[:,:,:h,:w]).square().mean())
            route = torch.tensor(archived['exit_map'], dtype=torch.long)
            dec.cfg = replace(cfg, tile_coupling=True, sorted_tiles=False, trunk_halo=0)
            outputs = {}
            for repair in (False, True):
                dec.seam_repair = original_repair if repair else None
                output = dec(latent, q, exit_map=route)
                crop = output[:,:,:h,:w]
                mse = float((source-crop).square().mean())
                outputs['with_repair' if repair else 'no_repair'] = {
                    'delta444_db': 10*math.log10(mse/full_mse),
                    'yuv611_db': psnr(source,crop)['yuv_6_1_1']}
                if all(x == cfg.num_exits-1 for x in route.tolist()) and not repair:
                    if not torch.allclose(output, full, atol=1e-4, rtol=1e-4):
                        raise RuntimeError(f'All-deep coupled/full mismatch: {image} QP{qp}')
            dec.cfg, dec.seam_repair = original_cfg, original_repair
            result['rows'].append({
                'image': image, 'qp': qp, 'scan_sha256': sha(scan_path),
                'stream_sha256': sha(stream_path), 'exit_map': route.tolist(),
                'deployed_delta444_db': archived['deployed_delta444_db'],
                'exact_context_delta444_db': archived['exact_context_delta444_db'],
                'deployed_yuv611_db': archived['deployed_yuv611_db'],
                'exact_context_yuv611_db': archived['exact_context_yuv611_db'],
                'coupled': outputs})
            added += 1
            result['complete'] = len(result['rows']) == 120
            args.output.write_text(json.dumps(result, indent=2)+'\n')
            print(f'{image} QP{qp}: coupled no-repair {outputs["no_repair"]["delta444_db"]:.4f}, '
                  f'with-repair {outputs["with_repair"]["delta444_db"]:.4f} dB', flush=True)
    assert provenance['source_code_sha256'] == {str(x.relative_to(ROOT)): sha(x) for x in critical}
    coupling_module.CanvasCoupler = original_coupler_class
    print(f'complete={result["complete"]}; cases={len(result["rows"])}')


if __name__ == '__main__':
    main()
