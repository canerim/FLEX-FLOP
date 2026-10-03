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
    record = {
        'upstream': str(args.upstream.resolve()), 'commit': head,
        'extension': str(args.extension.resolve()),
        'checkpoint_sha256': EXPECTED,
    }
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
