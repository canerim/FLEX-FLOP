"""Verify public checkpoint assets and build the pinned CPU entropy extension."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
COMMIT = 'cbdae87a5445114cdc7f48816da63ea80bdeac40'
EXPECTED = {
    'released_cvpr2026_image.pth.tar': 'b3b900de23f30e4fc437010ddffe6a5413a56d6cfd97fb6235adbcd2b6973302',
    'e15_epoch15.pth.tar': '91556d30bca41988755ff43dee3bc6d0cd2a0a4eda1d18d2891b168dc0a04ce9',
    'router_stem_qp.pth': '5b832c639fa7f857ef071276fd683e8e6150ca35e56b4470008a7d9dd79dda0f',
}


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path,
                        default=HERE/'.local/DCVC',
                        help='Existing pinned DCVC checkout, or destination for clone')
    parser.add_argument('--extension', type=Path, default=HERE/'.local/entropy')
    parser.add_argument('--skip-cpu-smoke', action='store_true',
                        help='Skip deterministic Kodak stream/latent re-encoding check')
    args = parser.parse_args()
    for name, digest in EXPECTED.items():
        path = HERE/'artifacts'/name
        if not path.exists() or sha(path) != digest:
            raise RuntimeError(f'Missing or incorrect {path}; install Git LFS and run git lfs pull')
    if not args.upstream.exists():
        args.upstream.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['git', 'clone', '--filter=blob:none', '--sparse',
                        'https://github.com/microsoft/DCVC.git', str(args.upstream)], check=True)
        subprocess.run(['git', '-C', str(args.upstream), 'checkout', COMMIT], check=True)
        subprocess.run(['git', '-C', str(args.upstream), 'sparse-checkout',
                        'set', 'src'], check=True)
    head = subprocess.check_output(['git', '-C', str(args.upstream),
                                    'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise RuntimeError(f'DCVC must be at {COMMIT}; found {head}')
    subprocess.run([sys.executable, str(REPO/'proof/depth_bitstream/build_entropy.py'),
                    '--upstream', str(args.upstream), '--out', str(args.extension)], check=True)
    smoke = None
    if not args.skip_cpu_smoke:
        import numpy as np
        from PIL import Image
        import torch
        sys.path.insert(0, str(REPO/'proof/depth_bitstream'))
        from model_io import load_model
        from reference_codec import ReferenceCodec
        torch.set_num_threads(1)
        net, info = load_model(HERE/'artifacts/released_cvpr2026_image.pth.tar',
                               12, args.upstream)
        from src.utils.transforms import rgb2ycbcr_np
        codec = ReferenceCodec(net, info['sha256'], args.extension)
        image = REPO/'data/kodak/kodim01.png'
        raw = np.asarray(Image.open(image).convert('RGB'))
        x = torch.from_numpy(rgb2ycbcr_np(raw.astype(np.float32)/255)-.5)
        x = x.permute(2,0,1)[None].contiguous()
        sample = HERE/'results/kodim01_qp32.fufref2'
        encoded = codec.encode(x, 32, audit=False, reconstruct=False)
        if encoded.stream != sample.read_bytes():
            raise RuntimeError('Kodak re-encoding differs from tracked bitstream')
        _, _, trace = codec.decode_latent(encoded.stream)
        expected_latent = '3bb29024361fa50510ecf5b746e333d1e885b215d1da0c21e83837aca336f21c'
        if trace['y_hat_sha256'] != expected_latent:
            raise RuntimeError('Decoded Kodak latent differs from reference hash')
        smoke = {'sample_stream_sha256': sha(sample),
                 'latent_sha256': trace['y_hat_sha256'], 'passed': True}
    record = {
        'upstream': str(args.upstream.resolve()), 'commit': head,
        'extension': str(args.extension.resolve()),
        'checkpoint_sha256': EXPECTED,
        'cpu_smoke': smoke,
    }
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
