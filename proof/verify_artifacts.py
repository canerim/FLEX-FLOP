"""CPU-only checkpoint identity and released/e15 shared-front-end proof."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TRAIN = Path('/data10/shareddata/can_karsal/dcvcuf_depth_20260927')
PATHS = {
    'released_d12': TRAIN/'reference_d12/released_cvpr2026_image.pth.tar',
    'e15_shared_early_exit': ROOT/'runs/RECIPE512/ckpt_PIN_e15.pth.tar',
    'independent_d2': TRAIN/'runs/d2/ckpt.pth.tar',
    'independent_d4': TRAIN/'runs/d4/ckpt.pth.tar',
    'independent_d6': TRAIN/'runs/d6/ckpt.pth.tar',
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    states = {}
    identities = {}
    for name, path in PATHS.items():
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        state = checkpoint.get('state_dict', checkpoint.get('net', checkpoint))
        states[name] = state
        identities[name] = {'filename': path.name, 'sha256': sha(path),
                            'epoch': checkpoint.get('epoch'),
                            'global_step': checkpoint.get('global_step'),
                            'state_tensors': len(state)}
    released = states['released_d12']
    e15 = states['e15_shared_early_exit']
    shared = {key: tensor for key,tensor in released.items() if not key.startswith('dec.')}
    unequal = [key for key,value in shared.items() if key not in e15 or not torch.equal(value, e15[key])]
    if unequal:
        raise RuntimeError(f'Released/e15 shared-front-end mismatch: {unequal[:5]}')
    for depth in (2, 4, 6):
        state = states[f'independent_d{depth}']
        block_keys = [k for k in state if k.startswith('dec.dec_1.') and k.split('.')[2].isdigit()]
        highest = max(int(k.split('.')[2]) for k in block_keys)
        if highest != depth:
            raise RuntimeError(f'D{depth} decoder block count differs')
    result = {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'checkpoint_identities': identities,
              'released_e15_shared_nondecoder_tensors_exact': len(shared),
              'released_e15_shared_nondecoder_mismatches': unequal,
              'claim': 'e15 and released D12 have exact analysis/hyperprior/entropy/quality-scale tensors; synthesis differs',
              'upstream_commit': subprocess.check_output(
                  ['git','-C',str(TRAIN/'upstream'),'rev-parse','HEAD'], text=True).strip(),
              'dataset': 'Kodak 24 original PNGs, source hashes in bitstream manifest'}
    (HERE/'artifact_manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__ == '__main__':
    main()
