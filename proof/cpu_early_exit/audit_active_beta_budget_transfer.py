"""Replay predeclared calibration-only beta budgets on DIV2K validation24.

The original FUFREF2 bitstream and e15 model remain frozen. The 0.1-dB arm
must match the separate primary beta replay; other feasible budgets are
decoded at their frozen calibration-selected maps. This is untimed CPU FP32
and reports analytical synthesis-convolution MAC, not native codec latency.
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--max-new-cases', type=int, default=120)
    ap.add_argument('--output', type=Path,
                    default=HERE / 'results/div2k_beta/quality_floor/active_replicate_budget_transfer_validation24.json')
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

    base = HERE / 'results/div2k_beta'
    exact_path = base / 'quality_floor/exact_context_validation24.json'
    manifest_path = base / 'manifest.json'
    policy_path = base / 'quality_floor/active_replicate_budget_sweep_policy.json'
    primary_path = base / 'quality_floor/active_replicate_beta_validation24.json'
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
    policy = json.loads(policy_path.read_text())
    primary = json.loads(primary_path.read_text())
    assert len(exact['rows']) == 120 and primary['complete'] and len(primary['rows']) == 120
    assert policy['targets'] == [0.075, 0.1, 0.125, 0.15]
    assert policy['qps'] == exact['qps'] == manifest['qps'] == [0, 16, 32, 48, 63]
    assert policy['source_sha256']['manifest'] == sha(manifest_path)
    assert primary['provenance']['policy_sha256'] == policy['source_sha256']['primary_policy']
    assert policy['validation_status'] == 'not evaluated by this selector'
    primary_by_key = {(r['image'], r['qp']): r for r in primary['rows']}
    assert len(primary_by_key) == 120
    manifest_images = {r['image']: r for r in manifest['rows']['validation']}
    assert set(exact['images']) == set(manifest_images)
    critical = [Path(__file__), HERE / 'active_canvas_replicate.py',
                ROOT / 'flexuf/backbone/decoder.py',
                ROOT / 'flexuf/backbone/coupling.py', ROOT / 'flexuf/cost.py',
                ROOT / 'flexuf/model.py', ROOT / 'flexuf/config.py']
    provenance = {
        'exact_sha256': sha(exact_path), 'manifest_sha256': sha(manifest_path),
        'policy_sha256': sha(policy_path), 'primary_sha256': sha(primary_path),
        'released_sha256': sha(released_path), 'e15_sha256': sha(e15_path),
        'source_code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in critical},
    }
    result = {
        'scope': 'Predeclared calibration-only active-replicate/no-repair beta budgets transferred to DIV2K validation24 x QP5, frozen e15/FUFREF2, CPU FP32 untimed.',
        'expected_cases': 120, 'complete': False,
        'quality_unit': 'Delta444 dB against full-frame e15',
        'cost_unit': 'Analytical full-synthesis convolution MAC saving %, excluding router/control/memory/entropy',
        'targets': policy['targets'],
        'feasible_targets': [str(target) for target in policy['targets']
                             if policy['policies'][str(target)]['complete_five_qp_policy']],
        'provenance': provenance, 'rows': [],
    }
    if args.output.exists():
        old = json.loads(args.output.read_text())
        assert old['provenance'] == provenance and old['expected_cases'] == 120
        result['rows'] = old['rows']
    done = {(r['image'], r['qp']) for r in result['rows']}
    assert len(done) == len(result['rows'])
    assert '0.1' in result['feasible_targets']

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
                                         dtype=np.float32)/255
                    source_cache[image] = torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2, 0, 1)[None].contiguous()
                source = source_cache[image]
                stem = f'{Path(image).stem}_qp{qp}'
                case_path = cases / f'{stem}.json'
                stream_path = cases / f'{stem}.fufref2'
                case = json.loads(case_path.read_text())
                assert sha(case_path) == archived['case_sha256']
                assert sha(stream_path) == archived['stream_sha256'] == case['stream_sha256']
                assert case['split'] == 'validation' and case['qp'] == qp
                latent, quant_step, _ = codec.decode_latent(stream_path.read_bytes())
                full = dec.forward_full(latent, quant_step)
                h, w = source.shape[-2:]
                full_mse = float((source-full[:, :, :h, :w]).square().mean())
                assert math.isclose(full_mse, case['e15_full_mse444'], abs_tol=1e-11)
                primary_row = primary_by_key[image, qp]
                assert primary_row['stream_sha256'] == archived['stream_sha256']
                target_rows = {}
                decoded_maps = {}
                for target in result['feasible_targets']:
                    beta = policy['policies'][target]['beta'][str(qp)]
                    selected = [c for c in case['candidates'] if c['beta'] == beta]
                    assert len(selected) == 1
                    candidate = selected[0]
                    route_map = tuple(candidate['exit_map'])
                    route = torch.tensor(route_map, dtype=torch.long)
                    blocks = float(((route-cfg.split_depth+1)*cfg.blocks_per_exit).float().mean())
                    cost = frame_relative_cost(route, cfg)
                    saving = 100*(1-cost+repair_share-blocks*per_block_extra)
                    assert abs(100*(1-cost)-candidate['mac_saved_pct'])<1e-4
                    if target == '0.1':
                        assert route_map == tuple(primary_row['exit_map'])
                        delta = primary_row['new_delta444_db']
                        assert abs(saving-primary_row['new_conv_mac_saving_pct'])<1e-7
                    else:
                        if route_map not in decoded_maps:
                            output = dec(latent, quant_step, exit_map=route)
                            mse = float((source-output[:, :, :h, :w]).square().mean())
                            decoded_maps[route_map] = 10*math.log10(mse/full_mse)
                        delta = decoded_maps[route_map]
                        if route_map == tuple(primary_row['exit_map']):
                            assert abs(delta-primary_row['new_delta444_db'])<1e-6
                    target_rows[target] = {'beta': beta, 'exit_map': list(route_map),
                                           'delta444_db': delta,
                                           'conv_mac_saving_pct': saving}
                result['rows'].append({
                    'image': image, 'qp': qp,
                    'case_sha256': sha(case_path), 'stream_sha256': sha(stream_path),
                    'targets': target_rows,
                })
                added += 1
                result['complete'] = len(result['rows']) == 120
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print(f'{stem}: '+', '.join(f'{target}dB={target_rows[target]["conv_mac_saving_pct"]:.2f}%'
                                               for target in result['feasible_targets']), flush=True)
    finally:
        dec.cfg, dec.seam_repair = original_cfg, original_repair
        coupling_module.CanvasCoupler = original_class
    assert provenance['source_code_sha256'] == {str(p.relative_to(ROOT)): sha(p)
                                                 for p in critical}
    print(f"complete={result['complete']}; cases={len(result['rows'])}")


if __name__ == '__main__':
    main()
