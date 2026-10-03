"""CPU-only, paired DCVC-UF synthesis export and codec-quality audit.

The byte streams are fixed FUFREF2 research streams. Timing starts after
entropy decoding and excludes disk I/O. It never opens a CUDA device.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / 'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT / 'proof/depth_bitstream'), str(ROOT)]


def sha(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def hardware():
    cpuinfo = Path('/proc/cpuinfo').read_text().splitlines()
    flags = next((line.split(':', 1)[1].strip().split() for line in cpuinfo
                  if line.startswith('flags')), [])
    model = next((line.split(':', 1)[1].strip() for line in cpuinfo
                  if line.startswith('model name')), platform.processor())
    return {'cpu': model, 'machine': platform.platform(),
            'avx2': 'avx2' in flags, 'avx512f': 'avx512f' in flags,
            'loadavg': list(os.getloadavg()), 'logical_cpus': os.cpu_count(),
            'affinity_cpus': sorted(os.sched_getaffinity(0))}


def setup(threads: int):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    return torch


def load(upstream: Path, release: Path, extension: Path, stream: Path):
    from model_io import load_model
    from reference_codec import ReferenceCodec
    net, identity = load_model(release, 12, upstream)
    codec = ReferenceCodec(net, identity['sha256'], extension)
    y, q, trace = codec.decode_latent(stream.read_bytes(), audit=False)
    return net, identity, y, q, trace


def synthesis(args, released):
    if args.variant == 'released':
        return released.dec, {'variant': 'released'}
    import torch
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    checkpoint = torch.load(args.e15, map_location='cpu', weights_only=False)
    flex = FlexUFIntra(FlexUFConfig(**checkpoint['config'])).eval()
    load_flexuf_state(flex, checkpoint)
    exit_idx = 0 if args.variant == 'e15_exit0' else flex.cfg.num_exits - 1

    class UniformExit(torch.nn.Module):
        def __init__(self, decoder, index):
            super().__init__()
            self.decoder = decoder
            self.index = index

        def forward(self, latent, q_dec):
            return self.decoder.forward_full(latent, q_dec, exit_idx=self.index)

    return UniformExit(flex.dec, exit_idx), {'variant': args.variant,
                                            'e15_sha256': sha(args.e15),
                                            'exit_idx': exit_idx}


def export(args):
    import onnx
    torch = setup(1)
    net, identity, y, q, trace = load(args.upstream, args.release, args.extension, args.stream)
    decoder, variant = synthesis(args, net)
    if args.torch_channels_last:
        decoder = decoder.to(memory_format=torch.channels_last)
        y = y.contiguous(memory_format=torch.channels_last)
    args.model.parent.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        expected = decoder(y, q).numpy()
        torch.onnx.export(decoder, (y, q), str(args.model), dynamo=False,
                          opset_version=17, input_names=['latent', 'q_dec'],
                          output_names=['recon_ycbcr_centered'],
                          dynamic_axes={'latent': {2: 'latent_h', 3: 'latent_w'},
                                        'recon_ycbcr_centered': {2: 'image_h', 3: 'image_w'}})
    onnx.checker.check_model(str(args.model))
    import onnxruntime as ort
    session = ort.InferenceSession(str(args.model), providers=['CPUExecutionProvider'],
                                   sess_options=ort.SessionOptions())
    got = session.run(None, {'latent': y.numpy(), 'q_dec': q.numpy()})[0]
    delta = float(abs(got - expected).max())
    report = {'model': str(args.model), 'model_sha256': sha(args.model),
              'stream': str(args.stream), 'stream_sha256': sha(args.stream),
              'checkpoint': identity, 'variant': variant, 'latent_shape': list(y.shape),
              'q_shape': list(q.shape), 'output_shape': list(got.shape),
              'max_abs_ort_vs_torch': delta,
              'ort_providers': session.get_providers(),
              'note': 'CPUExecutionProvider is MLAS; this is not the oneDNN EP',
              'torch': torch.__version__, 'ort': ort.__version__,
              'onnx': onnx.__version__, 'hardware': hardware()}
    args.model.with_suffix('.audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    if delta > args.tolerance:
        raise RuntimeError(f'ONNX output delta {delta} exceeds {args.tolerance}')


class StreamReader:
    def __init__(self, rows):
        self.rows = rows
        self.index = 0

    def get_next(self):
        if self.index == len(self.rows):
            return None
        row = self.rows[self.index]
        self.index += 1
        return row

    def rewind(self):
        self.index = 0


def quantize(args):
    import onnx
    from onnxruntime.quantization import quantize_static, QuantFormat, QuantType, CalibrationMethod
    setup(1)
    rows = []
    identities = []
    for stream in args.calibration_stream:
        _, _, y, q, _ = load(args.upstream, args.release, args.extension, stream)
        rows.append({'latent': y.numpy(), 'q_dec': q.numpy()})
        identities.append({'path': str(stream), 'sha256': sha(stream)})
    args.quant_model.parent.mkdir(parents=True, exist_ok=True)
    nodes = None
    if args.quant_selection != 'all_conv':
        model = onnx.load(str(args.model), load_external_data=False)
        nodes = [node.name for node in model.graph.node
                 if node.op_type == 'Conv' and '/dec_1/dec_1.' in node.name
                 and '/up/' not in node.name and not node.name.endswith('/dc.2/Conv')
                 and (args.quant_selection == 'trunk_pointwise' or
                      int(node.name.split('/dec_1/dec_1.')[1].split('/')[0]) <= 6)]
        if not nodes:
            raise RuntimeError('Selected no Conv nodes; inspect exported ONNX names')
    quantize_static(str(args.model), str(args.quant_model), StreamReader(rows),
                    quant_format=QuantFormat.QDQ, activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=args.per_channel,
                    reduce_range=args.reduce_range, op_types_to_quantize=['Conv'],
                    nodes_to_quantize=nodes,
                    calibrate_method=CalibrationMethod.MinMax)
    report = {'source_model_sha256': sha(args.model),
              'quant_model_sha256': sha(args.quant_model),
              'calibration_streams': identities, 'format': 'QDQ S8S8',
              'ops': ['Conv'], 'per_channel': args.per_channel,
              'selection': args.quant_selection, 'selected_nodes': nodes,
              'reduce_range': args.reduce_range,
              'warning': 'Kodak smoke calibration is not a representative production calibration set.'}
    args.quant_model.with_suffix('.audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def psnr(source, out):
    import numpy as np
    mse = np.mean((source - out) ** 2, axis=(0, 2, 3), dtype=np.float64)
    return [float(-10 * np.log10(max(v, 1e-20))) for v in mse]


def benchmark(args):
    import numpy as np
    from PIL import Image
    import onnxruntime as ort
    torch = setup(args.threads)
    load_before = os.getloadavg()[0]
    if args.enforce_idle and load_before > args.max_load:
        raise RuntimeError('CPU host is loaded; refusing claim-eligible timing')
    net, identity, y, q, trace = load(args.upstream, args.release, args.extension, args.stream)
    decoder, variant = synthesis(args, net)
    model_audit = args.model.with_suffix('.audit.json')
    if not model_audit.exists():
        raise RuntimeError(f'Missing export audit: {model_audit}')
    exported = json.loads(model_audit.read_text())
    if exported['model_sha256'] != sha(args.model) or exported['variant'] != variant:
        raise RuntimeError('ONNX model hash or variant differs from export audit')
    if args.quant_model.exists():
        quant_audit = args.quant_model.with_suffix('.audit.json')
        if not quant_audit.exists():
            raise RuntimeError(f'Missing quantization audit: {quant_audit}')
        quantized = json.loads(quant_audit.read_text())
        if (quantized['source_model_sha256'] != sha(args.model) or
                quantized['quant_model_sha256'] != sha(args.quant_model)):
            raise RuntimeError('Quantized model does not match the selected FP32 ONNX model')
    if args.torch_channels_last:
        decoder = decoder.to(memory_format=torch.channels_last)
        y = y.contiguous(memory_format=torch.channels_last)
    from src.utils.transforms import rgb2ycbcr_np
    raw = np.asarray(Image.open(args.source).convert('RGB'), dtype=np.float32) / 255
    source = rgb2ycbcr_np(raw).transpose(2, 0, 1)[None]
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sessions = {name: ort.InferenceSession(str(path), sess_options=options,
                                          providers=[args.provider])
                for name, path in [('onnx_fp32', args.model)] +
                ([('onnx_int8', args.quant_model)] if args.quant_model.exists() else [])}
    for name, session in sessions.items():
        if session.get_providers()[0] != args.provider:
            raise RuntimeError(f'{name} silently fell back to {session.get_providers()}')
    inputs = {'latent': y.contiguous().numpy(), 'q_dec': q.numpy()}
    with torch.inference_mode():
        funcs = {'torch_cpu_fp32': lambda: decoder(y, q).numpy()}
        funcs.update({name: (lambda session=session: session.run(None, inputs)[0])
                      for name, session in sessions.items()})
        outputs = {}
        times = {}
        for name, fn in funcs.items():
            for _ in range(args.warmup):
                fn()
            samples = []
            for _ in range(args.repeats):
                start = time.perf_counter_ns()
                out = fn()
                samples.append((time.perf_counter_ns() - start) / 1e6)
            outputs[name] = out
            times[name] = samples
    ref = outputs['torch_cpu_fp32']
    results = {}
    for name, out in outputs.items():
        cropped = out[:, :, :source.shape[2], :source.shape[3]] + .5
        channel_psnr = psnr(source, cropped)
        results[name] = {'yuv_psnr_db': channel_psnr,
                         'yuv_6_1_1_psnr_db': (6 * channel_psnr[0] + channel_psnr[1] + channel_psnr[2]) / 8,
                         'max_abs_vs_torch': float(np.max(np.abs(out - ref))),
                         'latency_ms_raw': times[name],
                         'median_ms': statistics.median(times[name])}
    load_after = os.getloadavg()[0]
    report = {'scope': 'Fixed FUFREF2 stream -> already entropy-decoded latent -> YCbCr synthesis; excludes entropy and IO',
              'claim_eligible': bool(args.enforce_idle and args.warmup >= 5 and
                                     args.repeats >= 30 and load_after <= args.max_load),
              'load_1m_before_after': [load_before, load_after],
              'stream_sha256': sha(args.stream), 'checkpoint': identity,
              'variant': variant,
              'model_sha256': sha(args.model),
              'quant_model_sha256': sha(args.quant_model) if args.quant_model.exists() else None,
              'source_sha256': sha(args.source), 'threads': args.threads,
              'torch_channels_last': args.torch_channels_last,
              'torch_mkldnn_available': torch.backends.mkldnn.is_available(),
              'providers': {name: session.get_providers() for name, session in sessions.items()},
              'hardware': hardware(), 'results': results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def profile(args):
    torch = setup(args.threads)
    net, identity, y, q, _ = load(args.upstream, args.release, args.extension, args.stream)
    decoder, variant = synthesis(args, net)
    if args.torch_channels_last:
        decoder = decoder.to(memory_format=torch.channels_last)
        y = y.contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        decoder(y, q)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                                    record_shapes=True) as prof:
            decoder(y, q)
    rows = [{'operator': event.key, 'count': event.count,
             'self_cpu_time_ms': event.self_cpu_time_total / 1000,
             'total_cpu_time_ms': event.cpu_time_total / 1000}
            for event in prof.key_averages()]
    rows.sort(key=lambda row: row['self_cpu_time_ms'], reverse=True)
    report = {'scope': 'One instrumented PyTorch CPU decoder synthesis; operator priority only, not latency',
              'variant': variant, 'checkpoint': identity, 'stream_sha256': sha(args.stream),
              'threads': args.threads, 'hardware': hardware(), 'operators': rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'top_operators': rows[:12], 'out': str(args.out)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export', 'quantize', 'benchmark', 'profile'])
    local = PROOF/'.local'
    parser.add_argument('--upstream', type=Path, default=local/'DCVC')
    parser.add_argument('--extension', type=Path, default=local/'entropy')
    parser.add_argument('--release', type=Path, default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    parser.add_argument('--e15', type=Path, default=PROOF/'artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--variant', choices=['released', 'e15_exit0', 'e15_deep'], default='released')
    parser.add_argument('--stream', type=Path, default=PROOF/'results/kodim01_qp32.fufref2')
    parser.add_argument('--source', type=Path, default=ROOT/'data/kodak/kodim01.png')
    parser.add_argument('--model', type=Path, default=Path('/tmp/flexplus-cpu/released_d12.onnx'))
    parser.add_argument('--quant-model', type=Path, default=Path('/tmp/flexplus-cpu/released_d12_int8.onnx'))
    parser.add_argument('--calibration-stream', type=Path, action='append', default=[])
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--torch-channels-last', action='store_true')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--provider', choices=['CPUExecutionProvider', 'DnnlExecutionProvider'], default='CPUExecutionProvider')
    parser.add_argument('--out', type=Path, default=Path('/tmp/flexplus-cpu/benchmark.json'))
    parser.add_argument('--tolerance', type=float, default=1e-3)
    parser.add_argument('--per-channel', action='store_true')
    parser.add_argument('--reduce-range', action='store_true')
    parser.add_argument('--quant-selection', choices=['all_conv', 'trunk_pointwise', 'first6_pointwise'], default='all_conv')
    parser.add_argument('--enforce-idle', action='store_true')
    parser.add_argument('--max-load', type=float, default=12)
    args = parser.parse_args()
    if args.command == 'export':
        export(args)
    elif args.command == 'quantize':
        if not args.calibration_stream:
            args.calibration_stream = sorted((PROOF/'results/bitstream_kodak3x3').glob('*.fufref2'))
        quantize(args)
    elif args.command == 'benchmark':
        benchmark(args)
    else:
        profile(args)


if __name__ == '__main__':
    main()
