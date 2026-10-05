"""Analyze the frozen CLIC39 beta transfer without fitting on CLIC outcomes."""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.interpolate import PchipInterpolator

HERE = Path(__file__).resolve().parent
RAW = HERE / 'results/clic39_active_beta/raw.json'
OUT = HERE / 'results/clic39_active_beta/summary.json'
QPS = (0, 16, 32, 48, 63)
ARMS = ('released_d12', 'e15_full', 'deployed_old_beta',
        'active_old_beta', 'active_new_beta')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_summary(values: list[float], rng: np.random.Generator) -> dict:
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {'n': 0}
    indices = rng.integers(0, len(x), size=(10000, len(x)))
    return {'n': len(x), 'mean': float(x.mean()), 'median': float(np.median(x)),
            'ci95_mean_image_bootstrap': np.quantile(x[indices].mean(axis=1),
                                                    [.025, .975]).tolist(),
            'min': float(x.min()), 'max': float(x.max())}


def bd_rate(rates: list[float], anchor: list[float], test: list[float]) -> tuple[float | None, str | None]:
    """PCHIP in log-rate/PSNR on measured common support; no extrapolation."""
    r = np.asarray(rates, dtype=np.float64)
    a = np.asarray(anchor, dtype=np.float64)
    t = np.asarray(test, dtype=np.float64)
    if not (len(r) == len(a) == len(t) == 5 and
            np.isfinite(r).all() and np.isfinite(a).all() and np.isfinite(t).all()):
        return None, 'nonfinite_or_incomplete_curve'
    if not np.all(np.diff(r) > 0):
        return None, 'rate_not_strictly_increasing'
    if not np.all(np.diff(a) > 0):
        return None, 'anchor_quality_not_strictly_increasing'
    if not np.all(np.diff(t) > 0):
        return None, 'test_quality_not_strictly_increasing'
    lo, hi = max(a[0], t[0]), min(a[-1], t[-1])
    if hi <= lo:
        return None, 'no_common_quality_support'
    reference = PchipInterpolator(a, np.log(r))
    candidate = PchipInterpolator(t, np.log(r))
    value = 100.0 * math.expm1(float(candidate.integrate(lo, hi) -
                                    reference.integrate(lo, hi)) / (hi - lo))
    if not math.isfinite(value):
        return None, 'nonfinite_integral'
    return value, None


def main() -> None:
    raw = json.loads(RAW.read_text())
    manifest_path = HERE / 'results/clic39_active_beta/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert raw['complete'] and raw['expected_cases'] == len(raw['rows']) == 195
    assert raw['provenance']['manifest_sha256'] == sha(manifest_path)
    assert len(manifest['included']) == 39 and len(manifest['excluded']) == 2
    assert manifest['qps'] == list(QPS)
    old_policy = json.loads((HERE / 'results/div2k_beta/locked_policy.json').read_text())
    new_policy = json.loads((HERE / 'results/div2k_beta/quality_floor/active_replicate_beta_locked_policy.json').read_text())
    assert raw['provenance']['old_policy_sha256'] == sha(HERE / 'results/div2k_beta/locked_policy.json')
    assert raw['provenance']['new_policy_sha256'] == sha(HERE / 'results/div2k_beta/quality_floor/active_replicate_beta_locked_policy.json')
    rows = {(r['image'], r['qp']): r for r in raw['rows']}
    assert len(rows) == 195
    assert set(rows) == {(m['image'], q) for m in manifest['included'] for q in QPS}
    for row in rows.values():
        assert set(row['yuv611_psnr_db']) == set(ARMS)
        assert set(row['conv_mac_saving_pct']) == set(ARMS[2:])
        assert row['old_beta'] == old_policy['beta'][str(row['qp'])]
        assert row['new_beta'] == new_policy['beta'][str(row['qp'])]
        assert row['stream_bytes'] > 0 and row['rate_bpp'] > 0
    rng = np.random.default_rng(20261005)
    per_qp = {}
    for qp in QPS:
        r = [rows[m['image'], qp] for m in manifest['included']]
        def metric(fn):
            return paired_summary([fn(x) for x in r], rng)
        per_qp[str(qp)] = {
            'n_images': len(r),
            'new_delta444_db': metric(lambda x: x['delta444_db']['active_new_beta']),
            'new_minus_deployed_yuv611_db': metric(
                lambda x: x['yuv611_psnr_db']['active_new_beta'] -
                          x['yuv611_psnr_db']['deployed_old_beta']),
            'new_minus_old_active_yuv611_db': metric(
                lambda x: x['yuv611_psnr_db']['active_new_beta'] -
                          x['yuv611_psnr_db']['active_old_beta']),
            'new_conv_mac_saving_pct': metric(
                lambda x: x['conv_mac_saving_pct']['active_new_beta']),
            'new_minus_deployed_conv_mac_saving_pp': metric(
                lambda x: x['conv_mac_saving_pct']['active_new_beta'] -
                          x['conv_mac_saving_pct']['deployed_old_beta']),
            'new_above_0p1db_delta444_count': sum(
                x['delta444_db']['active_new_beta'] > .1 for x in r),
            'new_exit_histogram': {str(exit_idx): sum(
                x['new_exit_map'].count(exit_idx) for x in r)
                for exit_idx in range(2, 6)},
        }
    per_image = []
    reasons: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in manifest['included']:
        image = item['image']
        r = [rows[image, qp] for qp in QPS]
        assert [x['stream_sha256'] for x in r] == [
            rows[image, qp]['stream_sha256'] for qp in QPS]
        rate = [x['rate_bpp'] for x in r]
        qualities = {arm: [x['yuv611_psnr_db'][arm] for x in r] for arm in ARMS}
        report = {'image': image, 'qps': list(QPS), 'rate_bpp': rate,
                  'yuv611_psnr_db': qualities, 'bd_rate_pct': {},
                  'bd_rate_unavailable_reason': {}}
        for anchor_name in ('released_d12', 'deployed_old_beta'):
            for arm in ARMS:
                if arm == anchor_name:
                    continue
                key = f'{arm}_vs_{anchor_name}'
                value, reason = bd_rate(rate, qualities[anchor_name], qualities[arm])
                if reason:
                    report['bd_rate_unavailable_reason'][key] = reason
                    reasons[key][reason] += 1
                else:
                    report['bd_rate_pct'][key] = value
        per_image.append(report)
    keys = sorted({key for row in per_image for key in row['bd_rate_pct']}
                  | {key for row in per_image for key in row['bd_rate_unavailable_reason']})
    bd_summary = {key: {'available': paired_summary(
        [row['bd_rate_pct'][key] for row in per_image if key in row['bd_rate_pct']], rng),
        'unavailable_reasons': dict(reasons[key])} for key in keys}
    result = {
        'scope': 'CLIC39 geometry-selected crops x five QPs, frozen DIV2K beta policies, same FUFREF2 stream and e15/router. CPU FP32 untimed.',
        'limitations': 'Not blind external data; local CLIC files appeared in prior other-decoder work. Analytical synthesis-conv MAC excludes router/control/memory/entropy. BD-rate is within FUFREF2 research streams, not native codec bitrate or measured latency.',
        'bd_rate_method': 'PCHIP log(rate) versus YUV 6:1:1 PSNR, integral on measured common support only; per-image then image-bootstrap mean. Missing/nonmonotonic curves excluded with explicit reason.',
        'script_sha256': sha(Path(__file__)), 'input_sha256': sha(RAW),
        'manifest_sha256': sha(manifest_path),
        'n_images': 39, 'n_cases': 195, 'geometry_excluded_images': manifest['excluded'],
        'per_qp': per_qp, 'bd_rate_summary': bd_summary, 'per_image': per_image,
    }
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'cases': result['n_cases'], 'per_qp': {
        q: {'loss_db': v['new_delta444_db']['mean'],
            'mac_saving_pct': v['new_conv_mac_saving_pct']['mean']}
        for q, v in per_qp.items()},
        'bd_rate_availability': {k: v['available']['n'] for k, v in bd_summary.items()}}, indent=2))


if __name__ == '__main__':
    main()
