"""Recompute GPU-free router-budget and whole-frame seam findings from records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROUTING = HERE/'offline_data/routing_summary_e15.json'
RAW_ROUTING = HERE/'offline_data/map_coding_e15_perframe3.json'
ROUTING_SOURCE = HERE/'offline_data/routing_summary_source.py'
SEAM = HERE/'offline_data/seam_audit_ctc_e15_all.json'
SEAM_SOURCE = HERE/'offline_data/seam_audit_ctc_source.py'
N_FRAMES = 53 * 5
ROUTING_SHA = 'b4f1c074c78b802234acfafccf218b85fcf3d6d704c327dbb27c18578bfd711f'
RAW_ROUTING_SHA = '1e13761e2ef4ba9cc5fce32598dbf6989ebe58274d4f81f8d80c6ca70aa5be39'
ROUTING_SOURCE_SHA = 'e7743a2113f56d8d0211c8481e813b587f01aa6ab7d49dc04167823aa88f17c6'
SEAM_SHA = '8278e4f31dd26d6b627533f83be870f42a55eb00207b16adefa28ff691eb98e0'
SEAM_SOURCE_SHA = '452368a7a878917b717fe1606e9179dcf77521bec4d0a168c95e814fd7979054'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    if (sha(ROUTING) != ROUTING_SHA or sha(RAW_ROUTING) != RAW_ROUTING_SHA or
            sha(ROUTING_SOURCE) != ROUTING_SOURCE_SHA or
            sha(SEAM) != SEAM_SHA or sha(SEAM_SOURCE) != SEAM_SOURCE_SHA):
        raise RuntimeError('Pinned offline routing/seam evidence or generator differs')
    routing = json.loads(ROUTING.read_text())
    raw_routing = json.loads(RAW_ROUTING.read_text())
    seam = json.loads(SEAM.read_text())
    if len(seam) != 2*N_FRAMES:
        raise RuntimeError('Expected 53 frames x five QPs x two seam arms')
    specs = {
        'B_signalled_beta': ('B_perframe_beta', None, None, 8.0),
        'D2': ('D2_perframe', None, 'bits_kt_per_frame', None),
        'D2_fallback_A': ('D2_fallbackA_perframe', None, 'bits_per_frame', None),
        'D3': ('D3_perframe', None, 'bits_kt_per_frame', None),
        'A_kt4': ('A_perframe', None, 'bits_kt4_per_frame', None),
        'C_rho0.1': ('C_perframe', 0.1, 'bits_per_frame', None),
        'C_rho0.2': ('C_perframe', 0.2, 'bits_per_frame', None),
        'C_rho0.35': ('C_perframe', 0.35, 'bits_per_frame', None),
    }
    points = {}
    for name, (kind, rho, bit_key, fixed_bits) in specs.items():
        rows = [r for r in raw_routing['rows'] if r['kind'] == kind and
                (rho is None or r.get('rho') == rho)]
        if len(rows) != 5 or sorted(r['qp'] for r in rows) != [0, 16, 32, 48, 63]:
            raise RuntimeError(f'Expected five raw QP rows for {name}')
        if any(r['n_frames'] != 53 for r in rows):
            raise RuntimeError(f'Expected 53 frames per QP for {name}')
        if name == 'A_kt4':
            bit_rows = [r for r in raw_routing['rows'] if r['kind'] == 'A_perframe_ktbits']
            if len(bit_rows) != 5:
                raise RuntimeError('Expected five raw KT bit rows for A')
        else:
            bit_rows = rows
        saving = statistics.mean(r['saving_pct_vs_release'] for r in rows)
        bits = fixed_bits if fixed_bits is not None else statistics.mean(r[bit_key] for r in bit_rows)
        frames_over = sum(r['frames_over_budget'] for r in rows)
        row = routing['per_frame_budget_rule'][name]
        if (abs(saving - row['saving']) > 1e-9 or
                abs(bits - row['bits']) > 1e-9 or frames_over != row['frames_over']):
            raise RuntimeError(f'Published routing summary differs from raw QP rows: {name}')
        points[name] = {
            'analytical_saving_pct_vs_released': saving,
            'estimated_side_bits_per_frame': bits,
            'frames_over_0_1db': frames_over,
            'violation_fraction': frames_over/N_FRAMES,
        }
    # Dominance here means no more side bits or budget violations and at least
    # as much analytical saving. It says nothing about measured GPU latency.
    dominated = []
    for name, row in points.items():
        rivals = [other for other, candidate in points.items() if other != name and
                  candidate['estimated_side_bits_per_frame'] <= row['estimated_side_bits_per_frame'] and
                  candidate['frames_over_0_1db'] <= row['frames_over_0_1db'] and
                  candidate['analytical_saving_pct_vs_released'] >= row['analytical_saving_pct_vs_released'] and
                  (candidate['estimated_side_bits_per_frame'] < row['estimated_side_bits_per_frame'] or
                   candidate['frames_over_0_1db'] < row['frames_over_0_1db'] or
                   candidate['analytical_saving_pct_vs_released'] > row['analytical_saving_pct_vs_released'])]
        if rivals:
            dominated.append({'point': name, 'dominated_by': sorted(rivals)})
    seam_rows = []
    for qp in (0,16,32,48,63):
        for arm in ('deepest','routed'):
            subset = [row for row in seam if row['qp'] == qp and row['arm'] == arm]
            if len(subset) != 53:
                raise RuntimeError(f'Incomplete seam subset: {qp}/{arm}')
            seam_rows.append({
                'qp': qp, 'arm': ('random_mixed' if arm == 'routed' else arm),
                'source_arm_label': arm, 'n_frames': len(subset),
                'map_source': ('all tiles at deepest exit' if arm == 'deepest'
                               else 'random uniform exit IDs, seeded by frame and QP; NOT learned router'),
                'median_full_frame_gap_to_released_db':
                    statistics.median(row['gap_on_db'] for row in subset),
                'median_repair_gain_db':
                    statistics.median(row['repair_gain_db'] for row in subset),
            })
    output = {
        'schema': 1, 'gpu_used': False,
        'routing_source_sha256': sha(ROUTING),
        'routing_raw_qp_rows_sha256': sha(RAW_ROUTING),
        'routing_generator_sha256': sha(ROUTING_SOURCE),
        'seam_source_sha256': sha(SEAM),
        'seam_generator_sha256': sha(SEAM_SOURCE),
        'budget_rule': 'per-frame 0.1 dB proxy from existing true-decode verified routing sweep; 265 frame-QP cases',
        'side_bits_scope': 'estimated adaptive code lengths, not emitted map bytes',
        'saving_scope': 'analytical cost model, not measured latency',
        'points': points, 'dominated': dominated,
        'seam_scope': 'whole-frame PSNR gap under all-deep or RANDOM mixed exits; routed is only a source-file label and does NOT mean a trained router; cannot isolate border-band distortion',
        'seam': seam_rows,
    }
    target = HERE/'offline_data/offline_audit.json'
    target.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(target), 'dominated': dominated,
                      'random_mixed_qp63_gap_db': next(r['median_full_frame_gap_to_released_db']
                                                       for r in seam_rows if r['qp']==63 and r['arm']=='random_mixed')}))


if __name__ == '__main__':
    main()
