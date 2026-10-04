"""Build a 120-case Kodak QP table against released D12 without GPU execution."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'proof/early_exit_vs_released'))
from mac_latency_audit import case_macs

QPS = (0, 16, 32, 48, 63)
FIELDS = ('image', 'qp', 'bpp_actual', 'released_yuv611_db',
          'router_yuv611_db', 'released_minus_router_db', 'delta444_vs_e15_db',
          'released_relative_conv_mac_saved_pct', 'legacy_model_mac_saved_pct',
          'exit_map', 'stream_sha256')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(values: list[float], rng: np.random.Generator) -> dict:
    x = np.asarray(values, dtype=float)
    samples = x[rng.integers(0, len(x), size=(10000, len(x)))].mean(axis=1)
    return {'mean': float(x.mean()),
            'ci95': [float(y) for y in np.quantile(samples, (.025, .975))]}


def fmt(x: float, digits: int = 3) -> str:
    return f'{x:.{digits}f}'


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scan', type=Path, default=HERE/'results/kodak24_qp5')
    p.add_argument('--transfer', type=Path, default=Path('/tmp/flexplus-div2k-beta-kodak'))
    p.add_argument('--exact', type=Path,
                   default=HERE/'results/div2k_beta/quality_floor/exact_context_kodak24.json')
    p.add_argument('--output', type=Path,
                   default=ROOT/'docs/research/2026-10-05-kodak-qp-released')
    a = p.parse_args()
    exact = json.loads(a.exact.read_text())
    exact_by_key = {(r['image'], r['qp']): r for r in exact['rows']}
    assert len(exact_by_key) == 120
    a.output.mkdir(parents=True, exist_ok=True)
    snapshot_path = a.output/'transfer_snapshot.json'
    previous_snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    transfer_snapshot = {}
    rows = []
    provenance = {'exact_context_sha256': sha(a.exact), 'scan': {}, 'transfer': {}}
    for index in range(1, 25):
        image = f'kodim{index:02d}.png'
        for qp in QPS:
            stem = f'kodim{index:02d}_qp{qp}.json'
            scan_path = a.scan/stem
            transfer_path = a.transfer/stem
            scan = json.loads(scan_path.read_text())
            if transfer_path.exists():
                transfer = json.loads(transfer_path.read_text())
                transfer_sha = sha(transfer_path)
            else:
                saved = previous_snapshot[stem]
                transfer = saved['row']
                transfer_sha = saved['sha256']
            transfer_snapshot[stem] = {'sha256': transfer_sha, 'row': transfer}
            diagnosis = exact_by_key[(image, qp)]
            scan_sha = sha(scan_path)
            assert transfer['scan_result_sha256'] == scan_sha
            assert scan['image'] == transfer['image'] == diagnosis['image'] == image
            assert scan['qp'] == transfer['qp'] == diagnosis['qp'] == qp
            assert scan['stream_sha256'] == transfer['stream_sha256'] == diagnosis['stream_sha256']
            assert abs(scan['released_yuv611_db']-transfer['released_yuv611_db']) < 1e-9
            assert abs(transfer['locked_beta_yuv611_db']-diagnosis['deployed_yuv611_db']) < 1e-6
            assert abs(transfer['locked_beta_delta444_db']-diagnosis['deployed_delta444_db']) < 1e-6
            counts = [diagnosis['exit_map'].count(k) for k in range(6)]
            assert counts == transfer['locked_exit_counts']
            assert sum(counts) == 6
            mac = case_macs({'padded_shape':scan['shape'], 'tile_counts':counts})
            pixels = scan['shape'][0]*scan['shape'][1]
            row = {
                'image': image, 'qp': qp,
                'bpp_actual': 8*scan['stream_bytes']/pixels,
                'released_yuv611_db': scan['released_yuv611_db'],
                'router_yuv611_db': transfer['locked_beta_yuv611_db'],
                'released_minus_router_db': transfer['locked_beta_loss_db'],
                'delta444_vs_e15_db': diagnosis['deployed_delta444_db'],
                'released_relative_conv_mac_saved_pct':
                    100*mac['e15_routed_conv_mac_saving_fraction'],
                'legacy_model_mac_saved_pct': transfer['locked_beta_mac_saved_pct'],
                'released_conv_gmac': mac['released_conv_macs']/1e9,
                'routed_conv_gmac': mac['e15_routed_conv_macs']/1e9,
                'exit_map': diagnosis['exit_map'],
                'exit_counts': counts,
                'stream_sha256': scan['stream_sha256'],
            }
            assert abs(row['released_yuv611_db']-row['router_yuv611_db']-
                       row['released_minus_router_db']) < 1e-8
            provenance['scan'][stem] = scan_sha
            provenance['transfer'][stem] = transfer_sha
            rows.append(row)
    assert len(rows) == 120 and all(r['released_conv_gmac'] == rows[0]['released_conv_gmac'] for r in rows)
    rng = np.random.default_rng(20261005)
    metrics = ('bpp_actual', 'released_yuv611_db', 'router_yuv611_db',
               'released_minus_router_db', 'delta444_vs_e15_db',
               'released_relative_conv_mac_saved_pct', 'legacy_model_mac_saved_pct')
    summary = {}
    for qp in QPS:
        group = [r for r in rows if r['qp'] == qp]
        summary[str(qp)] = {
            'n': 24,
            **{metric: mean_ci([r[metric] for r in group], rng) for metric in metrics},
            'over_0p1_delta444_count': sum(r['delta444_vs_e15_db'] > .1 for r in group),
            'over_0p1_released_yuv_gap_count': sum(r['released_minus_router_db'] > .1 for r in group),
            'deepest_tile_fraction': sum(r['exit_counts'][5] for r in group)/(24*6),
        }
    # Image clusters keep all five quality points together.
    images = [[r for r in rows if r['image'] == f'kodim{i:02d}.png'] for i in range(1,25)]
    overall = {metric: mean_ci([sum(r[metric] for r in image)/5 for image in images], rng)
               for metric in metrics}
    result = {'schema': 1,
              'scope': 'Locked DIV2K beta Kodak24 x 5 QP; source-informed beta calibration, actual FUFREF2, released D12 reference',
              'conv_mac_definition': 'Exact convolution-only count from mac_latency_audit.case_macs, including routed adapters and seam repair, divided by released D12 full-synthesis convolution MAC; router and entropy excluded',
              'quality_definition': 'Released minus routed YUV 6:1:1 PSNR on the same actual bitstream; positive is worse',
              'provenance': provenance, 'summary_by_qp': summary,
              'overall_image_cluster_mean': overall, 'rows': rows}
    snapshot_path.write_text(json.dumps(transfer_snapshot, indent=2)+'\n')
    (a.output/'analysis.json').write_text(json.dumps(result, indent=2)+'\n')
    with (a.output/'all_120_cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: '/'.join(map(str,row[k])) if k=='exit_map' else row[k] for k in FIELDS})
    report = [
        '# Kodak24 × 5 QP: early-exit router versus released D12', '',
        'Frozen DIV2K-fitted beta, 120 actual FUFREF2 streams. Positive YUV gap means the routed output is worse than the released decoder. All quality comparisons use the **same stream** and YUV 6:1:1 PSNR. The MAC percentage counts **convolutions only**, including exit adapters and seam repair, relative to released D12 full synthesis on the same 512 × 768 geometry. Router evaluation, entropy, signalling, scheduling and memory costs are omitted; it is not runtime or a full-codec BD-rate result. Kodak had been inspected earlier, so this is a transfer diagnostic rather than an untouched prospective test.', '',
        'The older normalized MAC model slightly overprices the repeated block. Its value is kept in the case table for audit, while the headline uses the exact convolution count. The diagnostic exact-context/no-repair variant has a different, unimplemented cost model and is not mixed into this deployed-path table.', '',
        '## QP means (24 images each)', '',
        '| QP | Actual bpp | Released YUV | Router YUV | Gap to released | Gap >0.1 dB | Conv-MAC saved | Old model saved | Δ444 vs full e15 | Δ444 >0.1 dB | Deepest tiles |',
        '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for qp in QPS:
        v = summary[str(qp)]
        m = lambda name: v[name]['mean']
        report.append(f'| {qp} | {fmt(m("bpp_actual"),4)} | {fmt(m("released_yuv611_db"))} | {fmt(m("router_yuv611_db"))} | {fmt(m("released_minus_router_db"),4)} | {v["over_0p1_released_yuv_gap_count"]}/24 | {fmt(m("released_relative_conv_mac_saved_pct"),2)}% | {fmt(m("legacy_model_mac_saved_pct"),2)}% | {fmt(m("delta444_vs_e15_db"),4)} | {v["over_0p1_delta444_count"]}/24 | {fmt(100*v["deepest_tile_fraction"],1)}% |')
    report.extend(['', f'Across all 120 cases: released/router YUV {fmt(overall["released_yuv611_db"]["mean"],4)}/{fmt(overall["router_yuv611_db"]["mean"],4)} dB; gap {fmt(overall["released_minus_router_db"]["mean"],4)} dB; exact convolution MAC saving {fmt(overall["released_relative_conv_mac_saved_pct"]["mean"],2)}% versus old model {fmt(overall["legacy_model_mac_saved_pct"]["mean"],2)}%. The full-geometry released denominator is {fmt(rows[0]["released_conv_gmac"],2)} GMAC per image.', '',
                   'The 95% image-bootstrap intervals for each mean and all case-level input hashes are in [analysis.json](analysis.json). [CSV](all_120_cases.csv) is suitable for plotting or sorting. D2/D4/D6 independent decoder-bank BD-rate results are a separate experiment. A negative MAC saving is possible: two QP63 maps send all six tiles to the deepest exit and still run seam repair, giving −1.09% relative to released full-frame synthesis.', '',
                   '## All 120 image–QP cases', '',
                   '| Image | QP | Actual bpp | Released YUV | Router YUV | Gap (dB) | Conv-MAC saved | Old model | Δ444 | Exits (six tiles) |',
                   '|:--|--:|--:|--:|--:|--:|--:|--:|--:|:--|'])
    for row in rows:
        report.append(f'| {row["image"][:-4]} | {row["qp"]} | {fmt(row["bpp_actual"],4)} | {fmt(row["released_yuv611_db"])} | {fmt(row["router_yuv611_db"])} | {fmt(row["released_minus_router_db"],4)} | {fmt(row["released_relative_conv_mac_saved_pct"],2)}% | {fmt(row["legacy_model_mac_saved_pct"],2)}% | {fmt(row["delta444_vs_e15_db"],4)} | {"/".join(map(str,row["exit_map"]))} |')
    report.extend(['', 'Reproduce with `python3 proof/cpu_early_exit/build_kodak_released_qp_table.py`. It reads the archived [transfer snapshot](transfer_snapshot.json) when the original `/tmp/flexplus-div2k-beta-kodak` scratch files are absent, checks scan and exact-context stream hashes, and performs no GPU work. The snapshot preserves the original transfer-file SHA256 values.', ''])
    (a.output/'REPORT.md').write_text('\n'.join(report))
    print('wrote', a.output, 'cases', len(rows), 'mean exact MAC saving', overall['released_relative_conv_mac_saved_pct']['mean'])


if __name__ == '__main__':
    main()
