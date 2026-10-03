"""Paired bitstream-to-image benchmark for released D12 and routed e15.

FUFREF2 is a CPU FP32 *research* rANS stream, not Microsoft's native CUDA
format. Each timed arm independently entropy-decodes the exact same bytes,
copies its latent to the GPU, and synthesizes an image. The e15 arm additionally
runs its decoder-side stem router; no encoder-side map or image is passed in.
Model loading, warmup, disk I/O and optional image encoding are outside timing.
"""
from __future__ import annotations

import argparse
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

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEPTH_PROOF = REPO / 'proof/depth_bitstream'
DEFAULT_TRAIN = Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
LOCAL = HERE/'.local'
DEFAULT_UPSTREAM = LOCAL/'DCVC' if (LOCAL/'DCVC').exists() else DEFAULT_TRAIN/'upstream'
DEFAULT_EXTENSION = LOCAL/'entropy' if (LOCAL/'entropy').exists() else DEFAULT_TRAIN/'research/reference_entropy_v1'
DEFAULT_RELEASE = HERE/'artifacts/released_cvpr2026_image.pth.tar'
DEFAULT_E15 = HERE/'artifacts/e15_epoch15.pth.tar'
DEFAULT_ROUTER = HERE/'artifacts/router_stem_qp.pth'
sys.path[:0] = [str(REPO), str(DEPTH_PROOF)]

CODE_FILES = (
    'proof/early_exit_vs_released/bitstream_benchmark.py',
    'proof/depth_bitstream/reference_codec.py',
    'proof/depth_bitstream/model_io.py',
    'flexuf/kernels/planned_decoder.py',
    'flexuf/kernels/released_decoder.py',
    'flexuf/kernels/fused_ffn.py',
    'flexuf/kernels/fused_pwout.py',
    'flexuf/kernels/depthwise3x3.py',
    'flexuf/kernels/wsilu_chunkadd.py',
    'flexuf/backbone/decoder.py',
    'flexuf/router/head2.py',
    'flexuf/model.py',
    'flexuf/cost.py',
    'flexuf/config.py',
)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def code_digests():
    return {name: digest(REPO/name) for name in CODE_FILES}


def tracked_code_state():
    commit = subprocess.check_output(
        ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output(
        ['git', '-C', str(REPO), 'status', '--porcelain',
         '--untracked-files=no', '--', *CODE_FILES], text=True).strip()
    return commit, changes


def gpu_occupants(gpu):
    uuids = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).splitlines()
    uuid = uuids[gpu].strip()
    raw = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'],
        text=True)
    return sorted({int(line.split(',')[0]) for line in raw.splitlines() if uuid in line})


def load_codec(args):
    from model_io import load_model
    from reference_codec import ReferenceCodec
    model, info = load_model(args.release, 12, args.upstream)
    return ReferenceCodec(model, info['sha256'], args.extension), info


def prepare(args):
    import numpy as np
    from PIL import Image
    import torch
    from reference_codec import tensor_hash

    torch.set_num_threads(1)
    codec, info = load_codec(args)
    from src.utils.transforms import rgb2ycbcr_np
    source = np.asarray(Image.open(args.image).convert('RGB'))
    if source.shape[0] % 256 or source.shape[1] % 256:
        raise ValueError('Early-exit inference requires image height and width divisible by 256')
    x = torch.from_numpy(rgb2ycbcr_np(source.astype(np.float32) / 255) - .5)
    x = x.permute(2, 0, 1)[None].contiguous()
    encoded = codec.encode(x, args.qp)
    y, _, audit = codec.decode_latent(encoded.stream)
    if audit['y_hat_sha256'] != encoded.diagnostics['y_hat_sha256']:
        raise RuntimeError('Bitstream latent does not match encoder latent')
    if tensor_hash(y) != audit['y_hat_sha256']:
        raise RuntimeError('Decoded latent hash is inconsistent')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(encoded.stream)
    record = {
        'schema': 1, 'scope': 'FUFREF2 CPU FP32 research stream; not Microsoft native CUDA wire',
        'image': str(args.image), 'image_sha256': digest(args.image),
        'stream': str(args.out), 'stream_sha256': digest(args.out),
        'source_shape': list(source.shape[:2]), 'qp': args.qp,
        'stream_bytes': len(encoded.stream), 'payload_bytes': encoded.diagnostics['payload_bytes'],
        'checkpoint': info, 'latent_sha256': audit['y_hat_sha256'],
        'cpu_encode_with_diagnostics_seconds': encoded.diagnostics['cpu_encode_and_diagnostic_seconds'],
    }
    args.out.with_suffix(args.out.suffix + '.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def load_router(args, cfg, device):
    import torch
    from flexuf.config import LATENT_CH, TRUNK_CH
    from flexuf.router.head2 import StemRouterHeadV2
    from flexuf.cost import exit_costs

    checkpoint = torch.load(args.router, map_location='cpu', weights_only=False)
    if checkpoint.get('context', 0) != 0:
        raise ValueError('This proof is pinned to the context-free stem,qp router')
    head = StemRouterHeadV2(
        TRUNK_CH, LATENT_CH, cfg.num_exits, min_exit=cfg.split_depth,
        inputs=checkpoint.get('inputs'), with_bits=checkpoint.get('with_bits'),
        r_stem=checkpoint.get('r_stem', 48), r_lat=checkpoint.get('r_lat', 32),
        hidden=checkpoint.get('hidden', 256),
    ).to(device).eval()
    head.load_state_dict(checkpoint.get('router_head_v2', checkpoint.get('state_dict', checkpoint)))
    if head.gate['latent'] or head.gate['scales'] or head.gate['bits']:
        raise ValueError('This protocol supports the stem,qp router checkpoint only')
    calibration = json.loads(Path(args.calibration).read_text())
    if calibration.get('router_checkpoint_sha256') != digest(args.router):
        raise RuntimeError('Router checkpoint differs from the beta calibration')
    rows = calibration['rows']
    beta = {int(r['qp']): float(r['beta']) for r in rows if r['kind'] == 'B'}
    return head, exit_costs(cfg, 'head').to(device), beta


def benchmark(args):
    sys.path.insert(0, str(args.upstream.resolve()))
    import torch
    import torch.nn.functional as F
    from src.models.image_model import DMCI
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map
    from flexuf.kernels import enable_fast_inference
    from flexuf.kernels.released_decoder import enable_fast_released_inference
    from reference_codec import parse_container, tensor_hash

    before = gpu_occupants(args.gpu)
    if before:
        raise RuntimeError(f'GPU {args.gpu} is occupied by compute PIDs {before}; refusing benchmark')
    torch.set_num_threads(1)
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(f'cuda:{args.gpu}')
    codec, info = load_codec(args)
    data = args.stream.read_bytes()
    meta = parse_container(data, 12, info['sha256'])
    source_cpu = None
    if args.include_encoder:
        if args.source is None:
            raise ValueError('--include-encoder requires --source')
        import numpy as np
        from PIL import Image
        from src.utils.transforms import rgb2ycbcr_np
        raw = np.asarray(Image.open(args.source).convert('RGB'))
        if list(raw.shape[:2]) != [meta['height'], meta['width']]:
            raise ValueError('Source image geometry differs from the stream')
        source_cpu = torch.from_numpy(rgb2ycbcr_np(raw.astype(np.float32)/255)-.5)
        source_cpu = source_cpu.permute(2,0,1)[None].contiguous()
        if codec.encode(source_cpu, meta['qp'], audit=False, reconstruct=False).stream != data:
            raise RuntimeError('Source encoder did not reproduce the supplied stream')
    if meta['qp'] not in (0, 16, 32, 48, 63):
        raise ValueError('Router beta is calibrated only at QPs 0,16,32,48,63')
    if meta['height'] % 256 or meta['width'] % 256:
        raise ValueError('Current e15 route protocol requires dimensions divisible by 256; crop or pad before encoding')
    release_state = torch.load(args.release, map_location='cpu', weights_only=False)
    release_state = release_state.get('state_dict', release_state)
    ckpt = torch.load(args.e15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ckpt['config'])
    if cfg.rgb_patch != 256:
        raise ValueError(f'Unexpected e15 patch geometry: {cfg.rgb_patch}')
    released = DMCI().to(device).eval()
    released.load_state_dict(release_state, strict=True)
    e15 = FlexUFIntra(cfg).to(device).eval()
    load_flexuf_state(e15, ckpt)
    shared = {k: v for k, v in release_state.items() if not k.startswith('dec.')}
    unequal = [k for k, v in shared.items()
               if k not in e15.state_dict() or not torch.equal(v, e15.state_dict()[k].cpu())]
    if unequal:
        raise RuntimeError(f'Analysis/entropy checkpoint mismatch: {unequal[:5]}')
    from dataclasses import replace
    import copy
    stock_e15 = copy.deepcopy(e15.dec).eval()
    stock_e15.cfg = replace(cfg, sorted_tiles=True)
    fast_e15 = copy.deepcopy(e15.dec).eval()
    e15_patches = enable_fast_inference(fast_e15, sort_tiles=True)
    fast_released = copy.deepcopy(released.dec).eval()
    released_patches = enable_fast_released_inference(fast_released)
    head, cost, beta = load_router(args, cfg, device)

    def run(kind, *, audit=False):
        # Each arm independently executes the requested boundary. No latent or
        # encoder-side tensor is shared, and output is always synthesized.
        stream = (codec.encode(source_cpu, meta['qp'], audit=False,
                               reconstruct=False).stream
                  if args.include_encoder else data)
        y_cpu, q_cpu, trace = codec.decode_latent(stream, audit=audit)
        y = y_cpu.to(device)
        q = q_cpu.to(device)
        route = None
        stem = None
        if kind.startswith('e15_'):
            decoder = stock_e15 if kind.endswith('_stock') else fast_e15
            stem = decoder.upsample(y)
            for group in decoder.groups[:cfg.split_depth]:
                stem = group(stem)
            qp = torch.tensor([meta['qp']], dtype=torch.int32, device=device)
            # The checkpoint gates latent, scales and bits off. The existing
            # latent supplies the scales tensor's shape without allocation.
            logits = head(stem, y, y, qp,
                          cfg.feature_patch, cfg.latent_patch)
            # Match the archived calibration's FP32 softmax followed by FP64
            # beta/cost comparison; a near-tie can otherwise flip an exit.
            scores = F.log_softmax(logits[:, cfg.split_depth:], dim=1).double()
            scores = scores - beta[meta['qp']] * cost[cfg.split_depth:].double()[None, :]
            route = (scores.argmax(1) + cfg.split_depth).cpu()
            if kind.startswith('e15_all_deep_'):
                route = torch.full_like(route, cfg.num_exits - 1)
        if kind == 'released_stock':
            image = released.dec(y, q)
        elif kind == 'released_triton':
            image = fast_released(y, q)
        elif kind in ('e15_stock', 'e15_all_deep_stock'):
            image = forward_from_stem_with_cpu_map(stock_e15, stem, q, route)
        elif kind in ('e15_triton', 'e15_all_deep_triton'):
            image = forward_from_stem_with_cpu_map(fast_e15, stem, q, route)
        else:
            raise ValueError(kind)
        image = image[:, :, :meta['height'], :meta['width']]
        if args.host_output:
            # Include the transfer needed by a caller consuming decoded pixels
            # outside CUDA. The returned CPU tensor is a materialized output.
            image = image.contiguous().cpu()
        return image, trace, route

    kinds = ('released_stock', 'released_triton', 'e15_stock', 'e15_triton',
             'e15_all_deep_stock', 'e15_all_deep_triton')
    with torch.inference_mode():
        outputs = {kind: run(kind, audit=True) for kind in kinds}
        torch.cuda.synchronize()
        latent_hashes = {v[1]['y_hat_sha256'] for v in outputs.values()}
        if len(latent_hashes) != 1:
            raise RuntimeError('Arms decoded different latents from the same stream')
        errors = {
            'released_stock_vs_triton': float((outputs['released_stock'][0]-outputs['released_triton'][0]).abs().max()),
            'e15_stock_vs_triton': float((outputs['e15_stock'][0]-outputs['e15_triton'][0]).abs().max()),
            'e15_all_deep_stock_vs_triton': float((outputs['e15_all_deep_stock'][0]-outputs['e15_all_deep_triton'][0]).abs().max()),
        }
        if max(errors.values()) > 1e-4:
            raise RuntimeError(f'Triton output equivalence failed: {errors}')
        if not torch.equal(outputs['e15_stock'][2], outputs['e15_triton'][2]):
            raise RuntimeError('Router made different mode decisions')
        if not torch.equal(outputs['e15_all_deep_stock'][2], outputs['e15_all_deep_triton'][2]):
            raise RuntimeError('All-deep controls used different mode maps')
        route_counts = torch.bincount(outputs['e15_stock'][2], minlength=cfg.num_exits).tolist()
        ref = outputs['released_stock'][0]
        mse = float((outputs['e15_stock'][0]-ref).square().mean())
        saved_reconstructions = None
        if args.save_recon_dir is not None:
            from PIL import Image
            from src.utils.transforms import ycbcr2rgb
            args.save_recon_dir.mkdir(parents=True, exist_ok=True)
            saved_reconstructions = {}
            for kind in ('released_stock', 'e15_stock', 'e15_triton',
                         'e15_all_deep_stock'):
                rgb = ycbcr2rgb(outputs[kind][0].clamp(-.5,.5)+.5,
                                clamp=True)[0]
                pixels = (rgb.permute(1,2,0).mul(255).add(.5)
                          .clamp(0,255).byte().cpu().numpy())
                path = args.save_recon_dir/f'{args.stream.stem}_{kind}.png'
                Image.fromarray(pixels).save(path)
                saved_reconstructions[kind] = {'path': str(path),
                                               'sha256': digest(path)}
        quality = None
        if args.source is not None:
            import math
            import numpy as np
            from PIL import Image
            from src.utils.transforms import rgb2ycbcr_np
            raw = np.asarray(Image.open(args.source).convert('RGB'))
            if list(raw.shape[:2]) != [meta['height'], meta['width']]:
                raise ValueError('Source image geometry differs from the stream')
            target = torch.from_numpy(rgb2ycbcr_np(raw.astype(np.float32)/255)-.5)
            target = target.permute(2,0,1)[None].to(
                'cpu' if args.host_output else device)
            quality = {}
            for kind in kinds:
                per_channel = (outputs[kind][0]-target).square().mean(dim=(0,2,3))
                psnr = [-10*math.log10(max(float(v), 1e-12)) for v in per_channel]
                # Kodak RGB -> YCbCr 4:4:4, matching the depth-bitstream study.
                # CTC YUV420 uses a different source/chroma sampling protocol.
                quality[kind] = (6*psnr[0]+psnr[1]+psnr[2])/8
        for _ in range(args.warmup):
            for kind in kinds:
                run(kind)
        torch.cuda.synchronize()
        samples = {kind: [] for kind in kinds}
        orders = []
        rng = random.Random(20261003)
        for block in range(args.blocks):
            interference = [pid for pid in gpu_occupants(args.gpu) if pid != os.getpid()]
            if interference:
                raise RuntimeError(f'GPU contention at block {block + 1}: {interference}')
            order = list(kinds)
            rng.shuffle(order)
            orders.append(order)
            for kind in order:
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                run(kind)
                torch.cuda.synchronize()
                samples[kind].append((time.perf_counter_ns()-start)/1e6)
            print(json.dumps({'block': block + 1, 'total': args.blocks,
                              'ms': {k: samples[k][-1] for k in kinds}}), flush=True)
    after = [pid for pid in gpu_occupants(args.gpu) if pid != os.getpid()]
    if after:
        raise RuntimeError(f'GPU contention after benchmark: {after}')
    git_commit, tracked_changes = tracked_code_state()
    result = {
        'schema': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'claim_eligible': not bool(tracked_changes),
        'git_commit': git_commit,
        'tracked_code_clean': not bool(tracked_changes),
        'tracked_code_changes': tracked_changes,
        'scope': ('RGB image -> CPU research encoder -> FUFREF2 bytes -> independent CPU rANS/hyperprior decode -> GPU copy -> decoder-side router (e15) -> GPU synthesis -> output tensor; excludes disk I/O and model loading; not Microsoft native CUDA stream'
                  if args.include_encoder else
                  'In-memory FUFREF2 bitstream -> independent CPU rANS/hyperprior decode -> GPU copy -> decoder-side router (e15) -> GPU synthesis -> output tensor; excludes encoder, disk I/O, model loading; not Microsoft native CUDA stream') +
                  ('; includes GPU-to-CPU image transfer' if args.host_output else
                   '; output remains on GPU'),
        'include_encoder': args.include_encoder,
        'host_output': args.host_output,
        'output_boundary': ('CPU YCbCr image tensor, GPU-to-CPU transfer timed'
                            if args.host_output else 'GPU YCbCr image tensor'),
        'stream_sha256': digest(args.stream), 'stream_bytes': len(data),
        'source_sha256': digest(args.source) if args.source else None,
        'shape': [meta['height'], meta['width']], 'qp': meta['qp'],
        'checkpoint_sha256': {'released': digest(args.release), 'e15': digest(args.e15),
                              'router': digest(args.router)},
        'calibration_sha256': digest(args.calibration),
        'shared_nondecoder_tensors_exact': len(shared),
        'latent_sha256': next(iter(latent_hashes)),
        'triton_max_abs_errors': errors, 'e15_vs_released_mse': mse,
        'quality_yuv611_444_db': quality,
        'saved_reconstructions': saved_reconstructions,
        'released_minus_e15_yuv611_444_db':
            (quality['released_stock']-quality['e15_stock']) if quality else None,
        'route_counts': route_counts, 'gpu': torch.cuda.get_device_name(device),
        'gpu_index': args.gpu, 'gpu_occupants_before': before,
        'gpu_occupants_after': after,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'triton': __import__('triton').__version__,
        'fp32_tf32_disabled': True, 'threads': torch.get_num_threads(),
        'blocks': args.blocks, 'warmup': args.warmup,
        'upstream_commit': info['upstream_commit'],
        'extension_manifest_sha256': digest(args.extension/'build_manifest.json'),
        'code_sha256': code_digests(),
        'e15_patches': e15_patches,
        'released_patches': released_patches,
        'arm_orders': orders,
        'samples_wall_ms': samples,
        'median_ms': {k: statistics.median(v) for k, v in samples.items()},
        'paired_speedups': {
            'stock_vs_stock': [a/b for a,b in zip(samples['released_stock'],samples['e15_stock'])],
            'matched_triton': [a/b for a,b in zip(samples['released_triton'],samples['e15_triton'])],
            'same_model_early_exit_stock': [a/b for a,b in zip(samples['e15_all_deep_stock'],samples['e15_stock'])],
            'same_model_early_exit_triton': [a/b for a,b in zip(samples['e15_all_deep_triton'],samples['e15_triton'])],
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'result': str(args.out), 'median_ms': result['median_ms'],
                      'paired_speedup_median': {k: statistics.median(v)
                                               for k,v in result['paired_speedups'].items()}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--extension', type=Path, default=DEFAULT_EXTENSION)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare', help='Encode a PNG and verify bytes -> latent')
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--qp', type=int, choices=(0,16,32,48,63), default=32)
    p.add_argument('--out', type=Path, required=True)
    b = commands.add_parser('benchmark', help='Time independent bitstream -> image decodes')
    b.add_argument('--stream', type=Path, required=True)
    b.add_argument('--source', type=Path, help='Original RGB PNG for YUV 6:1:1 PSNR in 4:4:4')
    b.add_argument('--include-encoder', action='store_true',
                   help='Time a fresh CPU image encoder in every paired arm')
    b.add_argument('--host-output', action='store_true',
                   help='Include GPU-to-CPU transfer of the decoded image tensor')
    b.add_argument('--save-recon-dir', type=Path,
                   help='Write released/e15 reconstruction PNGs outside timed blocks')
    b.add_argument('--e15', type=Path, default=DEFAULT_E15)
    b.add_argument('--router', type=Path, default=DEFAULT_ROUTER)
    b.add_argument('--calibration', type=Path, default=HERE/'router_calibration.json')
    b.add_argument('--gpu', type=int, default=0)
    b.add_argument('--blocks', type=int, default=20)
    b.add_argument('--warmup', type=int, default=3)
    b.add_argument('--out', type=Path, default=HERE/'results/bitstream_latest.json')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    else:
        if args.blocks < 5 or args.warmup < 1:
            parser.error('Need at least 5 paired blocks and 1 warmup')
        benchmark(args)


if __name__ == '__main__':
    main()
