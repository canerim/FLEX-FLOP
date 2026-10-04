"""Transfer a DIV2K-locked beta policy to Kodak from the same bitstreams.

Uses saved decoder-side router log probabilities and measures actual assembled
image quality. No Kodak data or metrics enter beta fitting.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT/'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT/'proof/depth_bitstream'), str(PROOF)]


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, obj):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(obj, indent=2)+'\n')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan-dir', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--upstream', type=Path, default=PROOF/'.local/DCVC')
    parser.add_argument('--extension', type=Path, default=PROOF/'.local/entropy')
    parser.add_argument('--release', type=Path, default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    parser.add_argument('--e15', type=Path, default=PROOF/'artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--source-dir', type=Path, default=ROOT/'data/kodak')
    parser.add_argument('--pilot-stream-dir', type=Path, default=PROOF/'results/bitstream_kodak3x3')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.scan_dir/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['cases'] != 120 or manifest['images'] != [f'kodim{i:02d}.png' for i in range(1,25)]:
        raise RuntimeError('This counterfactual requires the complete Kodak24 scan')
    manifest_hash = digest(manifest_path)
    policy = json.loads(args.policy.read_text())
    policy_hash = digest(args.policy)
    if policy['schema'] != 1 or set(policy['beta']) != set(map(str, manifest['qps'])):
        raise RuntimeError('The locked policy must contain exactly the five scan QPs')
    if not policy['scope'].startswith('Locked beta chosen on DIV2K calibration images only'):
        raise RuntimeError('Expected the independently fitted DIV2K policy')
    if manifest['released_sha256'] != digest(args.release) or manifest['e15_sha256'] != digest(args.e15):
        raise RuntimeError('Checkpoint identity differs from the scan')

    import numpy as np
    from PIL import Image
    import torch
    from model_io import load_model
    from reference_codec import ReferenceCodec
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
           for k,v in shared.items()):
        raise RuntimeError('Released/e15 entropy weights differ')
    from dataclasses import replace
    e15.dec.cfg = replace(cfg, sorted_tiles=True)
    for image_name in manifest['images']:
        source_path = args.source_dir/image_name
        source_rgb = np.asarray(Image.open(source_path).convert('RGB'), dtype=np.float32)/255
        source = torch.from_numpy(rgb2ycbcr_np(source_rgb)-.5).permute(2,0,1)[None].contiguous()
        for qp in manifest['qps']:
            stem_name = f'{Path(image_name).stem}_qp{qp}'
            scan_path = args.scan_dir/f'{stem_name}.json'
            scan = json.loads(scan_path.read_text())
            if scan['manifest_sha256'] != manifest_hash or scan['source_sha256'] != digest(source_path):
                raise RuntimeError(f'Scan provenance mismatch: {scan_path}')
            output_path = args.out_dir/f'{stem_name}.json'
            if output_path.exists():
                old = json.loads(output_path.read_text())
                if (old['scan_result_sha256'] == digest(scan_path) and
                        old['policy_sha256'] == policy_hash):
                    print(f'skip {stem_name}', flush=True)
                    continue
                raise RuntimeError(f'Conflicting output: {output_path}')
            scores = torch.tensor(scan['route_log_probs'], dtype=torch.float64)
            costs = torch.tensor(scan['router_costs'], dtype=torch.float64)
            calibrated_beta = float(manifest['calibrated_beta'][str(qp)])
            locked_beta = float(policy['beta'][str(qp)])
            calibrated = (scores-calibrated_beta*costs[cfg.split_depth:][None,:]).argmax(1)+cfg.split_depth
            selected = (scores-locked_beta*costs[cfg.split_depth:][None,:]).argmax(1)+cfg.split_depth
            if torch.bincount(calibrated, minlength=cfg.num_exits).tolist() != scan['exit_counts']:
                raise RuntimeError(f'Saved router scores disagree with decisions: {stem_name}')
            same_route = torch.equal(calibrated, selected)
            stream_path = args.scan_dir/f'{stem_name}.fufref2'
            if not stream_path.exists():
                stream_path = args.pilot_stream_dir/f'{stem_name}.fufref2'
            if digest(stream_path) != scan['stream_sha256']:
                raise RuntimeError(f'Stream hash mismatch: {stem_name}')
            y, q, _ = codec.decode_latent(stream_path.read_bytes())
            with torch.inference_mode():
                reference = e15.dec.forward_full(y, q)
                stem = e15.dec.upsample(y)
                for group in e15.dec.groups[:cfg.split_depth]:
                    stem = group(stem)
                calibrated_image = forward_from_stem_with_cpu_map(e15.dec, stem, q, calibrated)
                selected_image = (calibrated_image if same_route else
                                  forward_from_stem_with_cpu_map(e15.dec, stem, q, selected))
            h,w = scan['shape']
            reference = reference[:, :, :h, :w]
            calibrated_image = calibrated_image[:, :, :h, :w]
            selected_image = selected_image[:, :, :h, :w]
            calibrated_quality = psnr(source, calibrated_image)['yuv_6_1_1']
            if abs(calibrated_quality-scan['routed_yuv611_db']) > 1e-5:
                raise RuntimeError(f'Calibrated replay differs from first pass: {stem_name}')
            selected_quality = psnr(source, selected_image)['yuv_6_1_1']
            dref = float((source-reference).square().mean())
            dcal = float((source-calibrated_image).square().mean())
            dselected = float((source-selected_image).square().mean())
            if min(dref,dcal,dselected) <= 0:
                raise RuntimeError(f'Nonpositive source distortion: {stem_name}')
            row = {
                'schema': 1, 'image': image_name, 'qp': qp,
                'scan_manifest_sha256': manifest_hash,
                'scan_result_sha256': digest(scan_path),
                'stream_sha256': scan['stream_sha256'],
                'policy_sha256': policy_hash,
                'calibrated_beta': calibrated_beta, 'locked_beta': locked_beta,
                'same_route_as_calibrated': same_route,
                'released_yuv611_db': scan['released_yuv611_db'],
                'calibrated_yuv611_db': scan['routed_yuv611_db'],
                'locked_beta_yuv611_db': selected_quality,
                'calibrated_loss_db': scan['quality_loss_db'],
                'locked_beta_loss_db': scan['released_yuv611_db']-selected_quality,
                'e15_full_mse444': dref,
                'calibrated_delta444_db': 10*math.log10(dcal/dref),
                'locked_beta_delta444_db': 10*math.log10(dselected/dref),
                'calibrated_mac_saved_pct': scan['mac_saved_pct'],
                'locked_beta_mac_saved_pct': 100*(1-frame_relative_cost(selected,cfg,'head')),
                'locked_exit_counts': torch.bincount(selected,minlength=cfg.num_exits).tolist(),
                'cpu_timing': None,
            }
            write_json(output_path, row)
            print(json.dumps({'case': stem_name, 'same_route': same_route,
                              'locked_loss_db': row['locked_beta_loss_db'],
                              'locked_mac_saved_pct': row['locked_beta_mac_saved_pct']}), flush=True)


if __name__ == '__main__':
    main()
