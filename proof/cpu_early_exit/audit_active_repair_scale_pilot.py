"""Frozen six-case CPU pilot: attenuate old seam repair under active-zero context.

This is a post-transfer exploratory calibration study, not a validation or
latency result. Three lexicographically indexed DIV2K calibration images
(indices 0, 8, 16), each at QP 16 and 63, are selected before reading their
outcomes. At every case the stream, exit map and e15 weights stay fixed.
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROOF = ROOT / 'proof/early_exit_vs_released'
sys.path[:0] = [str(ROOT), str(ROOT / 'proof/depth_bitstream'), str(PROOF), str(HERE)]
from audit_exact_context_pilot import sha
from active_canvas_coupling import ActiveCanvasCoupler

EXACT = HERE / 'results/div2k_beta/quality_floor/exact_context_calibration24.json'
MANIFEST = HERE / 'results/div2k_beta/manifest.json'
CASES = Path('/tmp/flexplus-div2k-beta-calibration')
SOURCES = ROOT / 'data/DIV2K_valid_HR'
UPSTREAM = Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC')
EXTENSION = Path('/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy')
RELEASED = PROOF / 'artifacts/released_cvpr2026_image.pth.tar'
E15 = PROOF / 'artifacts/e15_epoch15.pth.tar'
OUT = HERE / 'results/div2k_beta/quality_floor/active_repair_scale_pilot.json'
ALPHAS = (0., .25, .5, .75, 1.)
IMAGE_INDICES = (0, 8, 16)
QPS = (16, 63)


class ScaledRepair(nn.Module):
    def __init__(self, base: nn.Module, alpha: float):
        super().__init__()
        self.base = base
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x if self.alpha == 0 else x + self.alpha * (self.base(x) - x)


def main() -> None:
    import numpy as np
    from PIL import Image
    from model_io import load_model
    from reference_codec import ReferenceCodec
    from flexuf.config import FlexUFConfig
    from flexuf.model import FlexUFIntra, load_flexuf_state
    import flexuf.backbone.coupling as coupling_module

    sys.path.insert(0, str(UPSTREAM.resolve()))
    from src.utils.transforms import rgb2ycbcr_np

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    exact = json.loads(EXACT.read_text())
    manifest = json.loads(MANIFEST.read_text())
    assert len(exact['rows']) == 120 and len(exact['images']) == 24
    assert exact['qps'] == manifest['qps'] == [0, 16, 32, 48, 63]
    images = sorted(exact['images'])
    chosen = {images[i] for i in IMAGE_INDICES}
    selected = [row for row in exact['rows']
                if row['image'] in chosen and row['qp'] in QPS]
    assert len(selected) == len(chosen) * len(QPS) == 6
    manifest_images = {r['image']: r for r in manifest['rows']['calibration']}
    assert chosen <= set(manifest_images)
    provenance = {
        'exact_sha256': sha(EXACT), 'manifest_sha256': sha(MANIFEST),
        'released_sha256': sha(RELEASED), 'e15_sha256': sha(E15),
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                          (Path(__file__), HERE / 'active_canvas_coupling.py',
                           ROOT / 'flexuf/backbone/decoder.py',
                           ROOT / 'flexuf/backbone/coupling.py')},
    }
    assert exact['provenance']['manifest_sha256'] == provenance['manifest_sha256']
    assert exact['provenance']['release_sha256'] == provenance['released_sha256']
    assert exact['provenance']['e15_sha256'] == provenance['e15_sha256']
    if OUT.exists():
        result = json.loads(OUT.read_text())
        assert result['provenance'] == provenance and result['alphas'] == list(ALPHAS)
    else:
        result = {
            'scope': 'Exploratory DIV2K calibration 3 images x QP16/63; frozen streams/maps, active-zero context, scaled trained seam repair, CPU FP32; no GPU, training or latency.',
            'selection': {'image_indices_sorted': list(IMAGE_INDICES),
                          'images': sorted(chosen), 'qps': list(QPS)},
            'alphas': list(ALPHAS), 'provenance': provenance,
            'complete': False, 'rows': [],
        }
    completed = {(r['image'], r['qp']) for r in result['rows']}
    assert len(completed) == len(result['rows'])

    released, info = load_model(RELEASED, 12, UPSTREAM)
    codec = ReferenceCodec(released, info['sha256'], EXTENSION)
    ckpt = torch.load(E15, map_location='cpu', weights_only=False)
    cfg = FlexUFConfig(**ckpt['config'])
    model = FlexUFIntra(cfg).eval()
    load_flexuf_state(model, ckpt)
    dec = model.dec
    original_cfg, original_repair = dec.cfg, dec.seam_repair
    assert original_repair is not None
    original_class = coupling_module.CanvasCoupler
    coupling_module.CanvasCoupler = ActiveCanvasCoupler
    dec.cfg = replace(cfg, tile_coupling=True, sorted_tiles=False, trunk_halo=0)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        with torch.inference_mode():
            for archived in selected:
                image, qp = archived['image'], archived['qp']
                if (image, qp) in completed:
                    continue
                im = manifest_images[image]
                source_path = SOURCES / image
                assert sha(source_path) == im['source_sha256']
                x, y, w, h = im['crop_xywh']
                with Image.open(source_path) as file:
                    rgb = np.asarray(file.convert('RGB').crop((x, y, x+w, y+h)),
                                     dtype=np.float32) / 255
                source = torch.from_numpy(rgb2ycbcr_np(rgb)-.5).permute(2, 0, 1)[None].contiguous()
                stem = f'{Path(image).stem}_qp{qp}'
                case_path = CASES / f'{stem}.json'
                stream_path = CASES / f'{stem}.fufref2'
                case = json.loads(case_path.read_text())
                assert sha(case_path) == archived['case_sha256']
                assert sha(stream_path) == archived['stream_sha256'] == case['stream_sha256']
                route = torch.tensor(archived['exit_map'], dtype=torch.long)
                latent, quant_step, _ = codec.decode_latent(stream_path.read_bytes())
                full = dec.forward_full(latent, quant_step)
                full_mse = float((source-full[:, :, :h, :w]).square().mean())
                assert math.isclose(full_mse, case['e15_full_mse444'], abs_tol=1e-11)
                losses = {}
                for alpha in ALPHAS:
                    dec.seam_repair = None if alpha == 0 else ScaledRepair(original_repair, alpha)
                    output = dec(latent, quant_step, exit_map=route)
                    mse = float((source-output[:, :, :h, :w]).square().mean())
                    losses[str(alpha)] = 10 * math.log10(mse/full_mse)
                dec.seam_repair = original_repair
                result['rows'].append({
                    'image': image, 'qp': qp,
                    'case_sha256': sha(case_path), 'stream_sha256': sha(stream_path),
                    'exit_map': route.tolist(),
                    'deployed_delta444_db': archived['deployed_delta444_db'],
                    'exact_context_delta444_db': archived['exact_context_delta444_db'],
                    'alpha_delta444_db': losses,
                })
                result['complete'] = len(result['rows']) == 6
                OUT.write_text(json.dumps(result, indent=2) + '\n')
                print(f'{stem}: {losses}', flush=True)
    finally:
        dec.cfg, dec.seam_repair = original_cfg, original_repair
        coupling_module.CanvasCoupler = original_class
    assert provenance['source_sha256'] == {str(p.relative_to(ROOT)): sha(p) for p in
                                          (Path(__file__), HERE / 'active_canvas_coupling.py',
                                           ROOT / 'flexuf/backbone/decoder.py',
                                           ROOT / 'flexuf/backbone/coupling.py')}
    print(f"complete={result['complete']}; cases={len(result['rows'])}")


if __name__ == '__main__':
    main()
