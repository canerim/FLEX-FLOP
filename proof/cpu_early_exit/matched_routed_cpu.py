"""Matched CPU synthesis control for released D12 and routed e15.

The two arms consume one independently decoded FUFREF2 latent. The e15 arm
includes decoder-side routing, patch scheduling, adapters and assembly.
This script contains no custom kernel; it establishes the shared-backend
quality baseline before oneDNN or AVX2 fusion is compared.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT/'proof/depth_bitstream'), str(PROOF)]


def psnr(source, recon):
    mse = (source - recon).square().mean(dim=(0, 2, 3))
    channels = [-10 * math.log10(max(float(x), 1e-20)) for x in mse]
    return {'y_cb_cr': channels,
            'yuv_6_1_1': (6 * channels[0] + channels[1] + channels[2]) / 8}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local = PROOF/'.local'
    parser.add_argument('--upstream', type=Path, default=local/'DCVC')
    parser.add_argument('--extension', type=Path, default=local/'entropy')
    parser.add_argument('--release', type=Path, default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    parser.add_argument('--e15', type=Path, default=PROOF/'artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--router', type=Path, default=PROOF/'artifacts/router_stem_qp.pth')
    parser.add_argument('--calibration', type=Path, default=PROOF/'router_calibration.json')
    parser.add_argument('--stream', type=Path, default=PROOF/'results/kodim01_qp32.fufref2')
    parser.add_argument('--source', type=Path, default=ROOT/'data/kodak/kodim01.png')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--blocks', type=int, default=0,
                        help='0 runs correctness only; timing requires an idle CPU')
    parser.add_argument('--max-load', type=float, default=12.0)
    parser.add_argument('--out', type=Path, default=HERE/'results/matched_routed_smoke.json')
    args = parser.parse_args()
    if args.threads < 1 or args.blocks < 0:
        parser.error('threads must be >= 1 and blocks >= 0')
    load_before = os.getloadavg()[0]
    if args.blocks and load_before > args.max_load:
        raise RuntimeError('CPU host loaded; refusing latency benchmark')

    import numpy as np
    from PIL import Image
    import torch
    import torch.nn.functional as F
    from model_io import load_model
    from reference_codec import ReferenceCodec, parse_container, tensor_hash
    from bitstream_benchmark import digest, load_router
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    released, info = load_model(args.release, 12, args.upstream)
    codec = ReferenceCodec(released, info['sha256'], args.extension)
    stream = args.stream.read_bytes()
    meta = parse_container(stream, 12, info['sha256'])
    y, q, trace = codec.decode_latent(stream)

    ckpt = torch.load(args.e15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ckpt['config'])
    e15 = FlexUFIntra(cfg).eval()
    load_flexuf_state(e15, ckpt)
    shared = {name: value for name, value in released.state_dict().items()
              if not name.startswith('dec.')}
    unequal = [name for name, value in shared.items()
               if name not in e15.state_dict() or
               not torch.equal(value, e15.state_dict()[name])]
    if unequal:
        raise RuntimeError(f'Shared encoder and entropy tensors differ: {unequal[:5]}')
    from dataclasses import replace
    e15.dec.cfg = replace(cfg, sorted_tiles=True)
    head, cost, beta, calibration_scope = load_router(args, cfg, torch.device('cpu'))
    if meta['qp'] not in beta:
        raise ValueError(f'No pinned beta for QP {meta["qp"]}')
    if meta['height'] % 256 or meta['width'] % 256:
        raise ValueError('Pinned e15 router requires dimensions divisible by 256')

    def run(kind):
        with torch.inference_mode():
            if kind == 'released_cpu':
                return released.dec(y, q), None
            decoder = e15.dec
            stem = decoder.upsample(y)
            for group in decoder.groups[:cfg.split_depth]:
                stem = group(stem)
            qp = torch.tensor([meta['qp']], dtype=torch.int32)
            logits = head(stem, y, y, qp, cfg.feature_patch, cfg.latent_patch)
            scores = F.log_softmax(logits[:, cfg.split_depth:], dim=1).double()
            scores = scores - beta[meta['qp']] * cost[cfg.split_depth:].double()[None, :]
            route = (scores.argmax(1) + cfg.split_depth).cpu()
            if kind == 'e15_all_deep_cpu':
                route.fill_(cfg.num_exits - 1)
            elif kind != 'e15_routed_cpu':
                raise ValueError(kind)
            image = forward_from_stem_with_cpu_map(decoder, stem, q, route)
            return image, route

    kinds = ('released_cpu', 'e15_routed_cpu', 'e15_all_deep_cpu')
    outputs = {kind: run(kind) for kind in kinds}
    source_rgb = np.asarray(Image.open(args.source).convert('RGB'), dtype=np.float32) / 255
    if list(source_rgb.shape[:2]) != [meta['height'], meta['width']]:
        raise ValueError('Source geometry differs from stream')
    from src.utils.transforms import rgb2ycbcr_np
    source = torch.from_numpy(rgb2ycbcr_np(source_rgb)-.5).permute(2, 0, 1)[None]
    quality = {kind: psnr(source, image[:, :, :meta['height'], :meta['width']])
               for kind, (image, _) in outputs.items()}
    route = outputs['e15_routed_cpu'][1]
    if route.numel() != (meta['height']//256)*(meta['width']//256):
        raise RuntimeError('Unexpected router tile count')
    max_abs = float((outputs['e15_routed_cpu'][0] - outputs['released_cpu'][0]).abs().max())
    tracked_files = ['proof/cpu_early_exit/matched_routed_cpu.py',
                     'proof/early_exit_vs_released/bitstream_benchmark.py',
                     'proof/depth_bitstream/reference_codec.py',
                     'proof/depth_bitstream/model_io.py',
                     'flexuf/backbone/decoder.py', 'flexuf/router/head2.py',
                     'flexuf/model.py', 'flexuf/kernels/planned_decoder.py']
    tracked_changes = subprocess.check_output(
        ['git', '-C', str(ROOT), 'status', '--porcelain', '--', *tracked_files],
        text=True).strip()
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                     text=True).strip()
    samples = {kind: [] for kind in kinds}
    orders = []
    if args.blocks:
        rng = random.Random(20261004)
        for _ in range(3):
            for kind in kinds:
                run(kind)
        for block in range(args.blocks):
            if os.getloadavg()[0] > args.max_load:
                raise RuntimeError(f'Host became loaded at block {block}; aborting')
            order = list(kinds)
            rng.shuffle(order)
            orders.append(order)
            for kind in order:
                start = time.perf_counter_ns()
                _, new_route = run(kind)
                samples[kind].append((time.perf_counter_ns() - start)/1e6)
                if kind == 'e15_routed_cpu' and not torch.equal(new_route, route):
                    raise RuntimeError('Router decision changed between paired runs')

    load_after = os.getloadavg()[0]
    report = {
        'schema': 1,
        'scope': 'Same decoded FUFREF2 latent -> CPU synthesis; e15 includes router and patch assembly; entropy, model loading and disk I/O excluded',
        'backend': 'PyTorch CPU FP32 for all arms; no custom fusion or ONNX EP',
        'claim_eligible': bool(args.blocks >= 30 and
                               load_before <= args.max_load and
                               load_after <= args.max_load and
                               not tracked_changes),
        'git_commit': commit, 'tracked_code_changes': tracked_changes,
        'script_sha256': digest(Path(__file__)),
        'loadavg_1m_before_after': [load_before, load_after],
        'affinity_cpus': sorted(os.sched_getaffinity(0)), 'threads': args.threads,
        'stream_sha256': digest(args.stream), 'latent_sha256': tensor_hash(y),
        'source_sha256': digest(args.source),
        'released_sha256': info['sha256'], 'e15_sha256': digest(args.e15),
        'router_sha256': digest(args.router), 'calibration_sha256': digest(args.calibration),
        'shared_nondecoder_tensors_exact': len(shared),
        'qp': meta['qp'], 'shape': [meta['height'], meta['width']],
        'route_counts': torch.bincount(route, minlength=cfg.num_exits).tolist(),
        'e15_released_max_abs': max_abs, 'quality': quality,
        'timing_ms_raw': samples,
        'timing_ms_median': {kind: statistics.median(values) if values else None
                             for kind, values in samples.items()},
        'block_orders': orders,
        'limitations': 'FUFREF2 research stream; fixed-QP beta; CPU smoke or synthesis timing only',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
