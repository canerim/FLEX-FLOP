"""Full frozen beta-map frontier under active-replicate, repair-off decoding.

Replay every distinct archived candidate map on DIV2K calibration24 x QP5
using the same FUFREF2 stream and e15 weights. Candidate betas and maps were
recorded before this experiment. This CPU FP32 run does not select a policy;
it produces the calibration measurements needed for a later locked choice.
No GPU, new encoder, retraining or latency claim is involved.
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
sys.path[:0] = [str(ROOT), str(ROOT / 'proof/depth_bitstream'), str(PROOF), str(HERE)]
from audit_exact_context_pilot import sha
from active_canvas_replicate import ActiveCanvasReplicateCoupler


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-new-cases', type=int, default=120)
    ap.add_argument('--output', type=Path, default=HERE / 'results/div2k_beta/quality_floor/active_replicate_beta_calibration24.json')
    args = ap.parse_args()
    assert 1 <= args.max_new_cases <= 120

    import numpy as np
    import torch
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.cost import (_BLOCK_MACPX, _C, N_TRUNK_BLOCKS, SHARE_TRUNK,
                             frame_relative_cost, seam_repair_share)
    import flexuf.backbone.coupling as coupling_module

    exact_path = HERE / 'results/div2k_beta/quality_floor/exact_context_calibration24.json'
    manifest_path = HERE / 'results/div2k_beta/manifest.json'
    cases = Path('/tmp/flexplus-div2k-beta-calibration')
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
    manifest_images = {r['image']: r for r in manifest['rows']['calibration']}
    assert set(exact['images']) == set(manifest_images)
    critical = [Path(__file__), HERE / 'active_canvas_replicate.py',
                ROOT / 'flexuf/backbone/decoder.py',
                ROOT / 'flexuf/backbone/coupling.py', ROOT / 'flexuf/cost.py',
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
        'scope': 'DIV2K calibration24 x QP5 full archived beta-map frontier, active-replicate depthwise context and no old seam repair; FUFREF2 CPU FP32, untimed.',
        'expected_cases': 120, 'complete': False,
        'quality_unit': 'Delta444 dB against full-frame e15',
        'cost_unit': 'Analytical full-synthesis convolution MAC saving %, excluding router/control/memory/entropy',
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
    assert not cfg.tile_coupling and cfg.tile_pad_mode == 'replicate'
    assert cfg.trunk_halo == 0 and cfg.seam_repair == 'grid'
    model = FlexUFIntra(cfg).eval()
    load_flexuf_state(model, checkpoint)
    dec = model.dec
    original_cfg, original_repair = dec.cfg, dec.seam_repair
    original_class = coupling_module.CanvasCoupler
    coupling_module.CanvasCoupler = ActiveCanvasReplicateCoupler
    dec.cfg = replace(cfg, tile_coupling=True, sorted_tiles=False, trunk_halo=0)
    dec.seam_repair = None
    per_block_extra = (9*_C/_BLOCK_MACPX)*(SHARE_TRUNK/N_TRUNK_BLOCKS)*(
        (cfg.feature_patch+2)**2/cfg.feature_patch**2-1)
    repair_share = seam_repair_share(cfg.seam_repair)
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
                assert case['split'] == 'calibration' and case['qp'] == qp
                latent, quant_step, _ = codec.decode_latent(stream_path.read_bytes())
                full = dec.forward_full(latent, quant_step)
                h, w = source.shape[-2:]
                full_mse = float((source-full[:, :, :h, :w]).square().mean())
                assert math.isclose(full_mse, case['e15_full_mse444'], abs_tol=1e-11)

                map_ids = {}
                map_rows = []
                candidate_rows = []
                for candidate in case['candidates']:
                    route_key = tuple(candidate['exit_map'])
                    assert len(route_key) == 6 and min(route_key) >= cfg.split_depth
                    assert max(route_key) < cfg.num_exits
                    if route_key not in map_ids:
                        map_ids[route_key] = len(map_rows)
                        route = torch.tensor(route_key, dtype=torch.long)
                        output = dec(latent, quant_step, exit_map=route)
                        mse = float((source-output[:, :, :h, :w]).square().mean())
                        delta = 10*math.log10(mse/full_mse)
                        blocks = float(((route-cfg.split_depth+1)*cfg.blocks_per_exit).float().mean())
                        deployed_cost = frame_relative_cost(route, cfg)
                        active_saving = 100*(1-deployed_cost+repair_share-blocks*per_block_extra)
                        map_rows.append({'exit_map': list(route_key),
                                         'active_delta444_db': delta,
                                         'active_no_repair_conv_saving_pct': active_saving,
                                         'mean_executed_suffix_blocks_per_tile': blocks})
                    map_id = map_ids[route_key]
                    route = torch.tensor(route_key, dtype=torch.long)
                    deployed_saving = 100*(1-frame_relative_cost(route, cfg))
                    assert abs(deployed_saving-candidate['mac_saved_pct']) < 1e-4
                    candidate_rows.append({'beta': candidate['beta'], 'map_id': map_id,
                                           'cached_deployed_delta444_db': candidate['delta444_db'],
                                           'cached_deployed_conv_saving_pct': candidate['mac_saved_pct']})
                assert len(candidate_rows) == len(manifest['candidates'][str(qp)])
                assert [c['beta'] for c in candidate_rows] == manifest['candidates'][str(qp)]
                result['rows'].append({
                    'image': image, 'qp': qp,
                    'case_sha256': sha(case_path), 'stream_sha256': sha(stream_path),
                    'source_sha256': im['source_sha256'],
                    'maps': map_rows, 'candidates': candidate_rows,
                })
                added += 1
                result['complete'] = len(result['rows']) == 120
                args.output.write_text(json.dumps(result, indent=2) + '\n')
                print(f'{stem}: {len(map_rows)} unique active maps', flush=True)
    finally:
        dec.cfg, dec.seam_repair = original_cfg, original_repair
        coupling_module.CanvasCoupler = original_class
    assert provenance['source_code_sha256'] == {str(p.relative_to(ROOT)): sha(p)
                                                 for p in critical}
    print(f"complete={result['complete']}; cases={len(result['rows'])}")


if __name__ == '__main__':
    main()
