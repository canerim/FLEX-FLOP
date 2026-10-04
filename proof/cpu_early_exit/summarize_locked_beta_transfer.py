"""Audit and summarize a frozen DIV2K beta policy on DIV2K and Kodak."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--cal-dir', type=Path, required=True)
    parser.add_argument('--val-dir', type=Path, required=True)
    parser.add_argument('--kodak-dir', type=Path, required=True)
    parser.add_argument('--kodak-scan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    manifest = json.loads(args.manifest.read_text())
    policy = json.loads(args.policy.read_text())
    if policy['manifest_sha256'] != sha(args.manifest):
        raise RuntimeError('Policy and manifest differ')
    policy_sha = sha(args.policy)
    scan_manifest_path = args.kodak_scan/'manifest.json'
    scan_manifest = json.loads(scan_manifest_path.read_text())
    if scan_manifest['cases'] != 120:
        raise RuntimeError('Kodak cohort is incomplete')
    scan_sha = sha(scan_manifest_path)
    qps = manifest['qps']

    def div_rows(split: str, folder: Path) -> list[dict]:
        output = []
        for item in manifest['rows'][split]:
            for qp in qps:
                path = folder/f'{Path(item["image"]).stem}_qp{qp}.json'
                row = json.loads(path.read_text())
                if (row['manifest_sha256'] != sha(args.manifest) or
                    row['split'] != split or row['image'] != item['image'] or
                    row['qp'] != qp or row['source_sha256'] != item['source_sha256'] or
                    row['policy_sha256'] != (policy_sha if split == 'validation' else None)):
                    raise RuntimeError(f'Invalid DIV2K result: {path}')
                matches = [r for r in row['candidates']
                           if r['beta'] == policy['beta'][str(qp)]]
                if len(matches) != 1:
                    raise RuntimeError(f'Locked beta absent from result: {path}')
                selected = matches[0]
                output.append({'image': item['image'], 'qp': qp,
                               'delta444_db': selected['delta444_db'],
                               'mac_saved_pct': selected['mac_saved_pct'],
                               'exit_map': selected['exit_map']})
        if len(output) != 120:
            raise RuntimeError(f'Incomplete {split} cohort')
        return output

    def kodak_rows() -> list[dict]:
        output = []
        for image in scan_manifest['images']:
            for qp in qps:
                stem = f'{Path(image).stem}_qp{qp}'
                path = args.kodak_dir/f'{stem}.json'
                scan_path = args.kodak_scan/f'{stem}.json'
                row = json.loads(path.read_text())
                scan = json.loads(scan_path.read_text())
                if (row['policy_sha256'] != policy_sha or
                    row['scan_manifest_sha256'] != scan_sha or
                    row['scan_result_sha256'] != sha(scan_path) or
                    row['stream_sha256'] != scan['stream_sha256'] or
                    row['image'] != image or row['qp'] != qp):
                    raise RuntimeError(f'Invalid Kodak result: {path}')
                output.append({'image': image, 'qp': qp,
                               'delta444_db': row['locked_beta_delta444_db'],
                               'mac_saved_pct': row['locked_beta_mac_saved_pct'],
                               'calibrated_delta444_db': row['calibrated_delta444_db'],
                               'calibrated_mac_saved_pct': row['calibrated_mac_saved_pct'],
                               'released_relative_yuv611_loss_db': row['locked_beta_loss_db'],
                               'exit_counts': row['locked_exit_counts']})
        if len(output) != 120:
            raise RuntimeError('Incomplete Kodak transfer cohort')
        return output

    rng = np.random.default_rng(20261004)

    def stats(rows: list[dict]) -> dict:
        images = list(dict.fromkeys(row['image'] for row in rows))
        if len(images) != 24:
            raise RuntimeError('Expected 24 distinct images')
        def metric(group: list[dict], field: str) -> dict:
            values = np.asarray([row[field] for row in group], dtype=float)
            if len(values) != len(images):
                raise RuntimeError('Expected one observation per image and QP')
            boot = rng.choice(values, size=(10000, len(values)), replace=True).mean(axis=1)
            return {'mean': float(values.mean()), 'median': float(np.median(values)),
                    'bootstrap_image_95': [float(np.quantile(boot, .025)),
                                           float(np.quantile(boot, .975))],
                    'over_0p1_count': int((values > .1).sum()) if field == 'delta444_db' else None}
        per_qp = {}
        for qp in qps:
            subset = [row for row in rows if row['qp'] == qp]
            per_qp[str(qp)] = {field: metric(subset, field)
                               for field in ['delta444_db', 'mac_saved_pct']}
            if 'released_relative_yuv611_loss_db' in subset[0]:
                per_qp[str(qp)]['released_relative_yuv611_loss_db'] = metric(
                    subset, 'released_relative_yuv611_loss_db')
        by_image = {image: [r for r in rows if r['image'] == image] for image in images}
        overall = {}
        for field in ['delta444_db', 'mac_saved_pct']:
            image_means = np.asarray([np.mean([r[field] for r in by_image[image]])
                                      for image in images])
            boot = rng.choice(image_means, size=(10000, len(images)), replace=True).mean(axis=1)
            overall[field] = {'mean': float(image_means.mean()),
                              'bootstrap_image_95': [float(np.quantile(boot, .025)),
                                                     float(np.quantile(boot, .975))]}
        overall['delta444_over_0p1_cases'] = sum(r['delta444_db'] > .1 for r in rows)
        if 'calibrated_mac_saved_pct' in rows[0]:
            overall['old_ctc_calibrated_mac_saved_pct'] = float(np.mean(
                [r['calibrated_mac_saved_pct'] for r in rows]))
            overall['old_ctc_calibrated_delta444_db'] = float(np.mean(
                [r['calibrated_delta444_db'] for r in rows]))
            overall['released_relative_yuv611_loss_db'] = float(np.mean(
                [r['released_relative_yuv611_loss_db'] for r in rows]))
        return {'cases': len(rows), 'images': len(images), 'per_qp': per_qp,
                'equal_qp_mean': overall}

    result = {'schema': 1, 'policy_sha256': policy_sha,
              'manifest_sha256': sha(args.manifest), 'kodak_scan_manifest_sha256': scan_sha,
              'beta': policy['beta'],
              'calibration': stats(div_rows('calibration', args.cal_dir)),
              'validation': stats(div_rows('validation', args.val_dir)),
              'kodak_transfer': stats(kodak_rows()),
              'metric_note': 'Delta444 = 10 log10(MSE policy / MSE e15 full), centred YCbCr444. Equal image and QP weight. Analytical synthesis MAC includes adapters and seam repair, excludes router; no latency measured.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key]['equal_qp_mean']
                      for key in ['calibration', 'validation', 'kodak_transfer']}, indent=2))


if __name__ == '__main__':
    main()
