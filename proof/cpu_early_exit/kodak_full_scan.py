"""Untimed Kodak24 x QP5 bitstream/router audit on one CPU core.

FUFREF2 is the pinned research stream, not Microsoft's native CUDA format.
The fixed-beta counterfactual recomputes decisions and MAC only; it does not
claim quality for reconstructions that have not been synthesized.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT/'proof/depth_bitstream'), str(PROOF)]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=PROOF/'.local/DCVC')
    parser.add_argument('--extension', type=Path, default=PROOF/'.local/entropy')
    parser.add_argument('--release', type=Path, default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    parser.add_argument('--e15', type=Path, default=PROOF/'artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--router', type=Path, default=PROOF/'artifacts/router_stem_qp.pth')
    parser.add_argument('--calibration', type=Path, default=PROOF/'router_calibration.json')
    parser.add_argument('--source-dir', type=Path, default=ROOT/'data/kodak')
    parser.add_argument('--pilot-stream-dir', type=Path,
                        default=PROOF/'results/bitstream_kodak3x3')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--images', type=int, default=24, choices=range(1, 25),
                        help='24 for the preregistered full scan; 1 for smoke')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from model_io import load_model
    from reference_codec import ReferenceCodec, parse_container, tensor_hash
    from bitstream_benchmark import load_router
    from matched_routed_cpu import psnr
    from flexuf.config import FlexUFConfig
    from flexuf.cost import frame_relative_cost
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map

    sys.path.insert(0, str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    released, info = load_model(args.release, 12, args.upstream)
    codec = ReferenceCodec(released, info['sha256'], args.extension)
    checkpoint = torch.load(args.e15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**checkpoint['config'])
    e15 = FlexUFIntra(cfg).eval()
    load_flexuf_state(e15, checkpoint)
    shared = {k: v for k, v in released.state_dict().items() if not k.startswith('dec.')}
    if any(k not in e15.state_dict() or not torch.equal(v, e15.state_dict()[k])
           for k, v in shared.items()):
        raise RuntimeError('Released/e15 analysis and entropy weights differ')
    from dataclasses import replace
    e15.dec.cfg = replace(cfg, sorted_tiles=True)
    head, cost, beta, calibration_scope = load_router(args, cfg, torch.device('cpu'))
    qps = [0, 16, 32, 48, 63]
    if set(qps) != set(beta):
        raise RuntimeError('Expected five pinned QP controls')
    fixed_beta = beta[16]
    images = [f'kodim{i:02d}.png' for i in range(1, args.images + 1)]
    identity = {
        'schema': 1, 'scope': 'Kodak RGB -> centred YCbCr444 -> released D12 FUFREF2 bytes -> independently decoded latent -> released/routed CPU reconstruction; no latency',
        'images': images, 'qps': qps, 'cases': len(images)*len(qps),
        'calibrated_beta': {str(k): v for k, v in beta.items()}, 'fixed_beta': fixed_beta,
        'fixed_beta_note': 'QP16 beta held constant for route/MAC counterfactual only; no fixed-beta quality measured',
        'calibration_scope': calibration_scope,
        'released_sha256': digest(args.release), 'e15_sha256': digest(args.e15),
        'router_sha256': digest(args.router), 'calibration_sha256': digest(args.calibration),
        'cost_model_sha256': digest(ROOT/'flexuf/cost.py'),
        'script_sha256': digest(Path(__file__)),
        'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        'backend': 'PyTorch CPU FP32, one thread, no latency; analytical synthesis MAC includes adapters/seam repair but omits router',
    }
    manifest_path = args.out_dir/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != identity:
            raise RuntimeError('Output directory belongs to a different audit configuration')
    else:
        write_json(manifest_path, identity)

    def route_from(scores, price):
        selected = scores - price * cost[cfg.split_depth:].double()[None, :]
        return (selected.argmax(1) + cfg.split_depth).cpu()

    for image_name in images:
        source_path = args.source_dir/image_name
        source_sha = digest(source_path)
        source_rgb = np.asarray(Image.open(source_path).convert('RGB'), dtype=np.float32)/255
        h, w = source_rgb.shape[:2]
        if h % 256 or w % 256:
            raise RuntimeError(f'Unsupported geometry: {image_name} {h}x{w}')
        source = torch.from_numpy(rgb2ycbcr_np(source_rgb)-.5).permute(2, 0, 1)[None].contiguous()
        for qp in qps:
            stem_name = f'{Path(image_name).stem}_qp{qp}'
            result_path = args.out_dir/f'{stem_name}.json'
            if result_path.exists():
                old = json.loads(result_path.read_text())
                if (old['manifest_sha256'] == digest(manifest_path) and
                        old['source_sha256'] == source_sha and old['qp'] == qp):
                    print(f'skip {stem_name}', flush=True)
                    continue
                raise RuntimeError(f'Existing result has incompatible provenance: {result_path}')
            pilot_stream = args.pilot_stream_dir/f'{stem_name}.fufref2'
            stream_path = pilot_stream if pilot_stream.exists() else args.out_dir/f'{stem_name}.fufref2'
            if stream_path.exists():
                stream = stream_path.read_bytes()
            else:
                encoded = codec.encode(source, qp, audit=False, reconstruct=False)
                stream = encoded.stream
                stream_path.write_bytes(stream)
            meta = parse_container(stream, 12, info['sha256'])
            if [meta['height'], meta['width'], meta['qp']] != [h, w, qp]:
                raise RuntimeError(f'Stream header disagrees with source: {stem_name}')
            y, q, audit = codec.decode_latent(stream)
            with torch.inference_mode():
                released_image = released.dec(y, q)
                decoder = e15.dec
                stem = decoder.upsample(y)
                for group in decoder.groups[:cfg.split_depth]:
                    stem = group(stem)
                qp_tensor = torch.tensor([qp], dtype=torch.int32)
                logits = head(stem, y, y, qp_tensor, cfg.feature_patch, cfg.latent_patch)
                scores = F.log_softmax(logits[:, cfg.split_depth:], dim=1).double()
                selected = route_from(scores, beta[qp])
                fixed = route_from(scores, fixed_beta)
                routed_image = forward_from_stem_with_cpu_map(decoder, stem, q, selected)
            if selected.numel() != (h//256)*(w//256):
                raise RuntimeError('Unexpected tile count')
            ref_quality = psnr(source, released_image[:, :, :h, :w])
            routed_quality = psnr(source, routed_image[:, :, :h, :w])
            row = {
                'schema': 1, 'image': image_name, 'qp': qp, 'shape': [h, w],
                'source_sha256': source_sha, 'stream_sha256': hashlib.sha256(stream).hexdigest(),
                'stream_bytes': len(stream), 'latent_sha256': tensor_hash(y),
                'manifest_sha256': digest(manifest_path),
                'released_yuv611_db': ref_quality['yuv_6_1_1'],
                'routed_yuv611_db': routed_quality['yuv_6_1_1'],
                'quality_loss_db': ref_quality['yuv_6_1_1']-routed_quality['yuv_6_1_1'],
                'exit_counts': torch.bincount(selected, minlength=cfg.num_exits).tolist(),
                'fixed_beta_exit_counts': torch.bincount(fixed, minlength=cfg.num_exits).tolist(),
                'mac_saved_pct': 100*(1-frame_relative_cost(selected, cfg, 'head')),
                'fixed_beta_mac_saved_pct': 100*(1-frame_relative_cost(fixed, cfg, 'head')),
                'route_log_probs': scores.tolist(),
                'router_costs': cost.tolist(),
                'cpu_timing': None,
            }
            write_json(result_path, row)
            print(json.dumps({'case': stem_name, 'loss_db': row['quality_loss_db'],
                              'mac_saved_pct': row['mac_saved_pct'],
                              'fixed_beta_mac_saved_pct': row['fixed_beta_mac_saved_pct'],
                              'exit_counts': row['exit_counts']}), flush=True)


if __name__ == '__main__':
    main()
