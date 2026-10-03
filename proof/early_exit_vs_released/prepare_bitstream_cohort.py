"""Pin nine Kodak byte streams before measuring their GPU runtime cohort."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from bitstream_benchmark import (DEFAULT_EXTENSION, DEFAULT_RELEASE,
                                 DEFAULT_UPSTREAM, digest, load_codec)

IMAGES = ('kodim01.png', 'kodim13.png', 'kodim24.png')
QPS = (16, 32, 48)
EXPECTED_KODIM01_QP32 = 'ae06007c9ebef84894b8aacaaf7d551a9a062076aac97590586a42652d609ceb'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument('--release', type=Path, default=DEFAULT_RELEASE)
    parser.add_argument('--extension', type=Path, default=DEFAULT_EXTENSION)
    parser.add_argument('--out', type=Path,
                        default=HERE/'results/bitstream_kodak3x3')
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(1)
    codec, info = load_codec(args)
    from src.utils.transforms import rgb2ycbcr_np
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in IMAGES:
        source = REPO/'data/kodak'/name
        raw = np.asarray(Image.open(source).convert('RGB'))
        if raw.shape[0] % 256 or raw.shape[1] % 256:
            raise RuntimeError(f'Image is not aligned to e15 tiles: {name}')
        x = torch.from_numpy(rgb2ycbcr_np(raw.astype(np.float32)/255)-.5)
        x = x.permute(2,0,1)[None].contiguous()
        for qp in QPS:
            stream = args.out/f'{source.stem}_qp{qp}.fufref2'
            coded = codec.encode(x, qp, audit=False, reconstruct=False)
            latent, _, trace = codec.decode_latent(coded.stream)
            if list(latent.shape[-2:]) != [raw.shape[0]//16, raw.shape[1]//16]:
                raise RuntimeError(f'Unexpected latent geometry: {name} QP{qp}')
            if stream.exists() and stream.read_bytes() != coded.stream:
                raise RuntimeError(f'Existing stream differs: {stream}')
            stream.write_bytes(coded.stream)
            if name == 'kodim01.png' and qp == 32 and digest(stream) != EXPECTED_KODIM01_QP32:
                raise RuntimeError('Pinned Kodim01 QP32 stream hash changed')
            row = {
                'image': name, 'image_sha256': digest(source), 'qp': qp,
                'shape': list(raw.shape[:2]),
                'stream': str(stream.relative_to(REPO)),
                'stream_sha256': digest(stream), 'stream_bytes': len(coded.stream),
                'payload_bytes': coded.diagnostics['payload_bytes'],
                'latent_sha256': trace['y_hat_sha256'],
                'cpu_encoder_seconds_single_pass': coded.diagnostics['cpu_encode_and_diagnostic_seconds'],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    manifest = {
        'schema': 1, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'Predeclared Kodak3x3 real FUFREF2 research byte streams, not native Microsoft CUDA format',
        'images': IMAGES, 'qps': QPS, 'cases': len(rows),
        'released_checkpoint_sha256': info['sha256'],
        'extension_manifest_sha256': digest(args.extension/'build_manifest.json'),
        'source_script_sha256': digest(__file__),
        'rows': rows,
    }
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'manifest': str(args.out/'manifest.json'), 'cases': len(rows)}))


if __name__ == '__main__':
    main()
