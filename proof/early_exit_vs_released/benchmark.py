"""Paired GPU proof: released DCVC-UF D12 versus the trained e15 early exit.

The measured scope is synthesis from one identical latent, not the entropy coder,
encoder, or router decision. A claimed result requires an otherwise idle GPU.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/reference_d12/released_cvpr2026_image.pth.tar')
E15 = ROOT / 'runs/RECIPE512/ckpt_PIN_e15.pth.tar'
UPSTREAM = Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927/upstream')
ARCHIVE = Path(__file__).resolve().parent / 'archived_maps.json'


def emit(obj):
    print(json.dumps(obj, allow_nan=False), flush=True)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def first_frame_sha(path, width, height):
    with Path(path).open('rb') as stream:
        return hashlib.sha256(stream.read(width * height * 3 // 2)).hexdigest()


def gpu_occupants(gpu):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    uuid = lines[gpu].strip()
    out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], text=True)
    return sorted({int(line.split(',')[0]) for line in out.splitlines() if uuid in line})


def timing(fn, torch):
    torch.cuda.synchronize()
    first = torch.cuda.Event(enable_timing=True)
    last = torch.cuda.Event(enable_timing=True)
    wall = time.perf_counter_ns()
    first.record()
    fn()
    last.record()
    last.synchronize()
    return {'cuda_ms': first.elapsed_time(last), 'wall_ms': (time.perf_counter_ns() - wall) / 1e6}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--sequence', choices=('videoSRC05', 'FourPeople', 'BQMall'), default='videoSRC05')
    p.add_argument('--qp', type=int, choices=(0, 16, 32, 48, 63), default=32)
    p.add_argument('--budget', choices=('0.1',), default='0.1')
    p.add_argument('--blocks', type=int, default=20)
    p.add_argument('--warmup', type=int, default=4)
    p.add_argument('--allow-shared-diagnostic', action='store_true')
    p.add_argument('--matched-kernels', action='store_true',
                   help='Also benchmark released D12 with the same fused Triton operators')
    p.add_argument('--out', type=Path, default=ROOT / 'proof/early_exit_vs_released/results/latest.json')
    a = p.parse_args()
    if a.blocks < 5 or a.warmup < 2:
        p.error('At least 5 paired blocks and 2 warmups required')
    occupied = gpu_occupants(a.gpu)
    if occupied and not a.allow_shared_diagnostic:
        emit({'type': 'blocked', 'reason': 'GPU has active compute processes', 'pids': occupied})
        return 3
    diagnostic = bool(occupied)
    sys.path[:0] = [str(ROOT), str(UPSTREAM)]
    import torch
    import torch.nn.functional as F
    import ctc_intra as C
    from src.models.image_model import DMCI
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_with_cpu_map
    from flexuf.kernels import enable_fast_inference
    from flexuf.kernels.released_decoder import enable_fast_released_inference
    torch.set_num_threads(2)
    torch.cuda.set_device(a.gpu)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{a.gpu}')
    release_state = torch.load(RELEASE, map_location='cpu', weights_only=False)
    release_state = release_state.get('state_dict', release_state)
    ck = torch.load(E15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ck['config'])
    e15 = FlexUFIntra(cfg).to(device).eval()
    load_flexuf_state(e15, ck)
    released = DMCI().to(device).eval()
    released.load_state_dict(release_state, strict=True)
    shared = {k: v for k, v in release_state.items() if not k.startswith('dec.')}
    unequal = [k for k, v in shared.items() if k not in e15.state_dict() or not torch.equal(v, e15.state_dict()[k].cpu())]
    if unequal:
        raise RuntimeError(f'Analysis/entropy weights differ: {unequal[:5]}')
    seq = next(s for s in C.discover([])[0] if s['name'].startswith(a.sequence))
    x, planes = C.read_frames(seq['path'], seq['w'], seq['h'], 1, 1)
    x = x.to(device)
    hp = (seq['h'] + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
    wp = (seq['w'] + cfg.rgb_patch - 1) // cfg.rgb_patch * cfg.rgb_patch
    xp = F.pad(x, (0, wp-seq['w'], 0, hp-seq['h']), mode='replicate')
    archive = json.loads(ARCHIVE.read_text())
    row = next(r for r in archive['rows'] if r['seq'] == seq['name'] and r['qp'] == a.qp and r['budget'] == a.budget)
    map_cpu = torch.tensor(row['map'], dtype=torch.long)
    with torch.inference_mode():
        y, q, _ = e15._encode_to_latent(xp, torch.tensor([a.qp], dtype=torch.int32, device=device))
        expected = (wp // cfg.rgb_patch) * (hp // cfg.rgb_patch)
        if map_cpu.numel() != expected:
            raise RuntimeError('Archived router map does not match tile geometry')
        stock = copy.deepcopy(e15.dec).eval()
        fast = copy.deepcopy(e15.dec).eval()
        stock.cfg = replace(cfg, sorted_tiles=True)
        patches = enable_fast_inference(fast, sort_tiles=True)
        arms = {
            'released_d12': lambda: released.dec(y, q),
            'e15_stock': lambda: forward_with_cpu_map(stock, y, q, map_cpu),
            'e15_triton': lambda: forward_with_cpu_map(fast, y, q, map_cpu),
        }
        released_patches = None
        if a.matched_kernels:
            released_fast = copy.deepcopy(released.dec).eval()
            released_patches = enable_fast_released_inference(released_fast)
            all_deep_cpu = torch.full_like(map_cpu, cfg.num_exits - 1)
            arms['released_d12_triton'] = lambda: released_fast(y, q)
            arms['e15_all_deep_stock'] = lambda: forward_with_cpu_map(stock, y, q, all_deep_cpu)
            arms['e15_all_deep_triton'] = lambda: forward_with_cpu_map(fast, y, q, all_deep_cpu)
        outputs = {name: fn() for name, fn in arms.items()}
        torch.cuda.synchronize()
        max_abs = float((outputs['e15_stock']-outputs['e15_triton']).abs().max())
        if max_abs > 1e-4:
            raise RuntimeError(f'Optimized e15 output differs by {max_abs}')
        release_max_abs = None
        all_deep_max_abs = None
        if a.matched_kernels:
            release_max_abs = float((outputs['released_d12']-outputs['released_d12_triton']).abs().max())
            if release_max_abs > 1e-4:
                raise RuntimeError(f'Optimized released D12 output differs by {release_max_abs}')
            all_deep_max_abs = float((outputs['e15_all_deep_stock']-
                                      outputs['e15_all_deep_triton']).abs().max())
            if all_deep_max_abs > 1e-4:
                raise RuntimeError(f'Optimized all-deep e15 output differs by {all_deep_max_abs}')
        quality = {name: C.psnr_611_420(out[:, :, :seq['h'], :seq['w']], planes[0]) for name, out in outputs.items()}
        for _ in range(a.warmup):
            for fn in arms.values():
                fn()
        torch.cuda.synchronize()
        emit({'type': 'ready', 'gpu': a.gpu, 'diagnostic_shared_gpu': diagnostic,
              'quality_yuv611_db': quality, 'shared_tensors_exact': len(shared),
              'e15_stock_vs_triton_max_abs': max_abs,
              'released_stock_vs_triton_max_abs': release_max_abs,
              'e15_all_deep_stock_vs_triton_max_abs': all_deep_max_abs})
        samples = {name: [] for name in arms}
        rng = random.Random(20261003)
        interference = []
        for block in range(a.blocks):
            others = [pid for pid in gpu_occupants(a.gpu) if pid != os.getpid()]
            if others:
                if not a.allow_shared_diagnostic:
                    raise RuntimeError(f'GPU became shared during block {block+1}: {others}')
                diagnostic = True
                interference.append({'block': block+1, 'pids': others})
            order = list(arms)
            rng.shuffle(order)
            times = {name: timing(arms[name], torch) for name in order}
            for name, value in times.items():
                samples[name].append(value)
            emit({'type': 'block', 'block': block+1, 'total': a.blocks, 'order': order,
                  'other_gpu_pids': others, 'timing_ms': times,
                  'paired_speedup': times['released_d12']['wall_ms']/times['e15_triton']['wall_ms']})
    post_occupants = [pid for pid in gpu_occupants(a.gpu) if pid != os.getpid()]
    if post_occupants:
        diagnostic = True
    pairs = [samples['released_d12'][i]['wall_ms']/samples['e15_triton'][i]['wall_ms'] for i in range(a.blocks)]
    stock_pairs = [samples['released_d12'][i]['wall_ms']/samples['e15_stock'][i]['wall_ms'] for i in range(a.blocks)]
    matched_pairs = ([samples['released_d12_triton'][i]['wall_ms']/samples['e15_triton'][i]['wall_ms']
                      for i in range(a.blocks)] if a.matched_kernels else None)
    skip_stock_pairs = ([samples['e15_all_deep_stock'][i]['wall_ms']/samples['e15_stock'][i]['wall_ms']
                         for i in range(a.blocks)] if a.matched_kernels else None)
    skip_fast_pairs = ([samples['e15_all_deep_triton'][i]['wall_ms']/samples['e15_triton'][i]['wall_ms']
                        for i in range(a.blocks)] if a.matched_kernels else None)
    result = {
        'schema': 3 if a.matched_kernels else 1,
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'claim_eligible': not diagnostic,
        'scope': 'FP32 PyTorch synthesis from identical released/e15 latent; wall time includes host tile planning, CUDA time also recorded; no encoder, entropy coder, router inference or transfers',
        'device': torch.cuda.get_device_name(device), 'gpu_index': a.gpu,
        'torch': torch.__version__, 'triton_patches': patches,
        'released_triton_patches': released_patches,
        'benchmark_sha256': sha(__file__),
        'sequence': seq['name'], 'qp': a.qp, 'budget': a.budget,
        'first_frame_sha256': first_frame_sha(seq['path'], seq['w'], seq['h']),
        'archived_maps_sha256': sha(ARCHIVE),
        'source_shape': [seq['h'], seq['w']],
        'padded_shape': [hp, wp], 'tile_counts': torch.bincount(map_cpu, minlength=cfg.num_exits).tolist(),
        'checkpoint_sha256': {'released_d12': sha(RELEASE), 'e15': sha(E15)},
        'shared_tensors_exact': len(shared), 'quality_yuv611_db': quality,
        'e15_stock_vs_triton_max_abs': max_abs,
        'released_stock_vs_triton_max_abs': release_max_abs,
        'e15_all_deep_stock_vs_triton_max_abs': all_deep_max_abs,
        'samples': samples, 'speedup_wall_paired': pairs,
        'speedup_wall_released_vs_e15_stock_paired': stock_pairs,
        'speedup_wall_released_triton_vs_e15_triton_paired': matched_pairs,
        'speedup_wall_e15_all_deep_vs_routed_stock_paired': skip_stock_pairs,
        'speedup_wall_e15_all_deep_vs_routed_triton_paired': skip_fast_pairs,
        'gpu_interference_during_blocks': interference,
        'median_speedup_wall': statistics.median(pairs),
        'median_speedup_released_vs_e15_stock_wall': statistics.median(stock_pairs),
        'median_speedup_released_triton_vs_e15_triton_wall':
            statistics.median(matched_pairs) if matched_pairs is not None else None,
        'median_speedup_e15_all_deep_vs_routed_stock_wall':
            statistics.median(skip_stock_pairs) if skip_stock_pairs is not None else None,
        'median_speedup_e15_all_deep_vs_routed_triton_wall':
            statistics.median(skip_fast_pairs) if skip_fast_pairs is not None else None,
        'median_wall_ms': {k: statistics.median(vv['wall_ms'] for vv in vals) for k, vals in samples.items()},
        'gpu_occupants_before': occupied, 'gpu_occupants_after': post_occupants,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix('.tmp')
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    tmp.replace(a.out)
    emit({'type': 'complete', 'result': str(a.out), 'claim_eligible': result['claim_eligible'],
          'median_speedup_wall': result['median_speedup_wall'], 'median_wall_ms': result['median_wall_ms']})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
