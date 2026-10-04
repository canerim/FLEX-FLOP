"""Replay one DIV2K router case through the uncached direct decoder."""
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


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--source-dir', type=Path, default=ROOT/'data/DIV2K_valid_HR')
    parser.add_argument('--upstream', type=Path, default=PROOF/'.local/DCVC')
    parser.add_argument('--extension', type=Path, default=PROOF/'.local/entropy')
    parser.add_argument('--release', type=Path,
                        default=PROOF/'artifacts/released_cvpr2026_image.pth.tar')
    parser.add_argument('--e15', type=Path,
                        default=PROOF/'artifacts/e15_epoch15.pth.tar')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from dataclasses import replace
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    from flexuf.kernels.planned_decoder import forward_from_stem_with_cpu_map

    sys.path.insert(0, str(args.upstream.resolve()))
    from src.utils.transforms import rgb2ycbcr_np

    row = json.loads(args.case.read_text())
    policy = json.loads(args.policy.read_text())
    if sha(args.stream) != row['stream_sha256']:
        raise RuntimeError('Stream does not match the selected case')
    source_path = args.source_dir/row['image']
    if sha(source_path) != row['source_sha256']:
        raise RuntimeError('Source image does not match the selected case')
    selected = [c for c in row['candidates']
                if c['beta'] == policy['beta'][str(row['qp'])]]
    if len(selected) != 1:
        raise RuntimeError('The policy beta is absent from the case grid')
    candidate = selected[0]
    left, top, width, height = row['crop_xywh']
    with Image.open(source_path) as image:
        rgb = np.asarray(image.convert('RGB').crop(
            (left, top, left+width, top+height)), dtype=np.float32)/255
    source = torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2,0,1)[None].contiguous()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    released, info = load_model(args.release, 12, args.upstream)
    codec = ReferenceCodec(released, info['sha256'], args.extension)
    ckpt = torch.load(args.e15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ckpt['config'])
    e15 = FlexUFIntra(cfg).eval()
    load_flexuf_state(e15, ckpt)
    e15.dec.cfg = replace(cfg, sorted_tiles=True)
    latent, q, _ = codec.decode_latent(args.stream.read_bytes())
    with torch.inference_mode():
        reference = e15.dec.forward_full(latent,q)[:,:,:height,:width]
        stem = e15.dec.upsample(latent)
        for group in e15.dec.groups[:cfg.split_depth]:
            stem = group(stem)
        route = torch.tensor(candidate['exit_map'], dtype=torch.long)
        output = forward_from_stem_with_cpu_map(e15.dec,stem,q,route)[:,:,:height,:width]
        delta = 10*math.log10(float((source-output).square().mean()) /
                              float((source-reference).square().mean()))
    result = {'schema':1, 'case_sha256':sha(args.case),
              'stream_sha256':sha(args.stream), 'source_sha256':sha(source_path),
              'policy_sha256':sha(args.policy), 'release_sha256':sha(args.release),
              'e15_sha256':sha(args.e15), 'image':row['image'],'qp':row['qp'],
              'beta':candidate['beta'],'exit_map':candidate['exit_map'],
              'cached_delta444_db':candidate['delta444_db'],
              'direct_delta444_db':delta,
              'absolute_difference_db':abs(delta-candidate['delta444_db'])}
    if result['absolute_difference_db'] > 1e-6:
        raise RuntimeError(f'Cached/direct quality mismatch: {result}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
