"""CPU-only exact-context quality pilot on a fixed DIV2K validation prefix.

Measures reconstructed quality; the separately computed ideal shared-context
MAC is an optimistic arithmetic bound, never a measured runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT / 'proof/depth_bitstream'), str(PROOF)]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_reconstruct(dec, cfg, stem, q, route):
    """Sufficient clipped context for a prescribed tile exit map."""
    import torch
    _, _, h, w = stem.shape
    side = cfg.feature_patch
    assert h % side == 0 and w % side == 0
    nh, nw = h // side, w // side
    assert len(route) == nh * nw
    canvas = torch.empty_like(stem)
    for tile, mode in enumerate(route):
        top, left = (tile // nw) * side, (tile % nw) * side
        halo = (mode - cfg.split_depth + 1) * cfg.blocks_per_exit
        t, l = max(0, top - halo), max(0, left - halo)
        b, r = min(h, top + side + halo), min(w, left + side + halo)
        work = stem[:, :, t:b, l:r].contiguous()
        for group in range(cfg.split_depth, mode + 1):
            work = dec.groups[group](work)
        work = dec._at_exit(work, mode)
        canvas[:, :, top:top + side, left:left + side] = work[:, :, top-t:top-t+side, left-l:left-l+side]
    return dec._apply_head(canvas, q)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=HERE/'results/div2k_beta/manifest.json')
    p.add_argument('--policy', type=Path, default=HERE/'results/div2k_beta/locked_policy.json')
    p.add_argument('--cases', type=Path, default=Path('/tmp/flexplus-div2k-beta-validation'))
    p.add_argument('--split', choices=['calibration','validation'], default='validation')
    p.add_argument('--source-dir', type=Path, default=ROOT/'data/DIV2K_valid_HR')
    p.add_argument('--upstream', type=Path, default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC'))
    p.add_argument('--extension', type=Path, default=Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy'))
    p.add_argument('--e15', type=Path, default=PROOF/'artifacts/e15_epoch15.pth.tar')
    p.add_argument('--release', type=Path, default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    p.add_argument('--ideal', type=Path, default=HERE/'results/div2k_beta/quality_floor/ideal_shared_context_cohort.json')
    p.add_argument('--output', type=Path, default=HERE/'results/div2k_beta/quality_floor/exact_context_pilot.json')
    p.add_argument('--images', type=int, default=6)
    args = p.parse_args()

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
    manifest, policy = json.loads(args.manifest.read_text()), json.loads(args.policy.read_text())
    if policy['manifest_sha256'] != sha(args.manifest):
        raise RuntimeError('Policy / manifest mismatch')
    fixed = manifest['rows'][args.split][:args.images]
    qps = manifest['qps']
    ideal = json.loads(args.ideal.read_text())
    if ideal['policy_sha256'] != sha(args.policy):
        raise RuntimeError('Ideal-cost / policy mismatch')
    ideal_lookup = {(r['image'], r['qp']): r for r in ideal['rows'][args.split]}
    provenance = {'manifest_sha256': sha(args.manifest), 'policy_sha256': sha(args.policy),
                  'e15_sha256': sha(args.e15), 'release_sha256': sha(args.release),
                  'ideal_sha256': sha(args.ideal)}
    result = {'schema': 1, 'scope': f'Fixed first {args.split} images by SHA256 filename manifest; CPU FP32 released FUFREF2 stream + e15 synthesis; no retraining, no latency',
              'images': [r['image'] for r in fixed], 'qps': qps,
              'quality_unit': 'Delta dB = 10log10(MSE_policy/MSE_e15_full), centred YCbCr444',
              'ideal_cost_unit': 'optimistic unique feature-cell MAC with zero overhead; not implemented sparse runtime',
              'provenance': provenance, 'rows': []}
    if args.output.exists():
        old = json.loads(args.output.read_text())
        if (any(old[k] != result[k] for k in ['schema', 'qps', 'provenance']) or
            old['images'] != result['images'][:len(old['images'])]):
            raise RuntimeError('Existing output is from a different experiment')
        result['rows'] = old['rows']
    done = {(r['image'], r['qp']) for r in result['rows']}

    released, info = load_model(args.release, 12, args.upstream)
    codec = ReferenceCodec(released, info['sha256'], args.extension)
    ckpt = torch.load(args.e15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ckpt['config'])
    model = FlexUFIntra(cfg).eval()
    load_flexuf_state(model, ckpt)
    dec = model.dec

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for image in fixed:
            source_path = args.source_dir / image['image']
            if sha(source_path) != image['source_sha256']:
                raise RuntimeError(f'Source hash mismatch: {source_path}')
            x, y, width, height = image['crop_xywh']
            with Image.open(source_path) as im:
                rgb = np.asarray(im.convert('RGB').crop((x, y, x + width, y + height)), dtype=np.float32) / 255
            source = torch.from_numpy(rgb2ycbcr_np(rgb) - .5).permute(2, 0, 1)[None].contiguous()
            for qp in qps:
                if (image['image'], qp) in done:
                    continue
                stemname = Path(image['image']).stem
                case_path = args.cases / f'{stemname}_qp{qp}.json'
                stream_path = args.cases / f'{stemname}_qp{qp}.fufref2'
                case = json.loads(case_path.read_text())
                if (case['split'] != args.split or case['image'] != image['image'] or case['qp'] != qp or
                    case['source_sha256'] != sha(source_path) or
                    case['stream_sha256'] != sha(stream_path) or
                    case['policy_sha256'] != (sha(args.policy) if args.split=='validation' else None)):
                    raise RuntimeError(f'Case provenance mismatch: {case_path}')
                selected = [c for c in case['candidates'] if c['beta'] == policy['beta'][str(qp)]]
                if len(selected) != 1:
                    raise RuntimeError(f'No unambiguous frozen route: {case_path}')
                chosen = selected[0]
                latent, q, _ = codec.decode_latent(stream_path.read_bytes())
                reference = dec.forward_full(latent, q)
                ref_mse = float((source-reference[:, :, :height, :width]).square().mean())
                if not math.isclose(ref_mse, case['e15_full_mse444'], abs_tol=1e-11):
                    raise RuntimeError(f'Reference MSE mismatch: {case_path}')
                stem = dec.upsample(latent)
                for group in dec.groups[:cfg.split_depth]:
                    stem = group(stem)
                route = chosen['exit_map']
                exact_output = exact_reconstruct(dec, cfg, stem, q, route)
                if all(mode == cfg.num_exits - 1 for mode in route):
                    if not torch.allclose(exact_output, reference, atol=1e-4, rtol=1e-4):
                        raise RuntimeError(f'Uniform deep exact context differs from full frame: {case_path}')
                deployed = forward_from_stem_with_cpu_map(dec, stem, q, torch.tensor(route))
                def delta(output):
                    mse = float((source-output[:, :, :height, :width]).square().mean())
                    return 10*math.log10(mse/ref_mse)
                deployed_db = delta(deployed)
                if abs(deployed_db-chosen['delta444_db']) > 1e-3:
                    raise RuntimeError(f'Cached/deployed mismatch: {case_path}: {deployed_db}, {chosen["delta444_db"]}')
                exact_db = delta(exact_output)
                ideal_row = ideal_lookup[(image['image'], qp)]
                result['rows'].append({'image': image['image'], 'qp': qp,
                                       'case_sha256': sha(case_path), 'stream_sha256': sha(stream_path),
                                       'exit_map': route, 'deployed_delta444_db': deployed_db,
                                       'exact_context_delta444_db': exact_db,
                                       'ideal_shared_saving_pct': ideal_row['ideal_shared_context_mac_saved_pct']})
                args.output.write_text(json.dumps(result, indent=2)+'\n')
                print(f'{image["image"]} QP{qp}: deployed {deployed_db:.4f}, exact {exact_db:.4f} dB', flush=True)


if __name__ == '__main__':
    main()
