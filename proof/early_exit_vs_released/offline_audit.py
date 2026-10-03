"""Recompute GPU-free router-budget and whole-frame seam findings from records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROUTING = HERE/'offline_data/routing_summary_e15.json'
SEAM = HERE/'offline_data/seam_audit_ctc_e15_all.json'
N_FRAMES = 53 * 5
ROUTING_SHA = 'b4f1c074c78b802234acfafccf218b85fcf3d6d704c327dbb27c18578bfd711f'
SEAM_SHA = '8278e4f31dd26d6b627533f83be870f42a55eb00207b16adefa28ff691eb98e0'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    if sha(ROUTING) != ROUTING_SHA or sha(SEAM) != SEAM_SHA:
        raise RuntimeError('Pinned offline routing/seam evidence differs')
    routing = json.loads(ROUTING.read_text())
    seam = json.loads(SEAM.read_text())
    if len(seam) != 2*N_FRAMES:
        raise RuntimeError('Expected 53 frames x five QPs x two seam arms')
    points = {}
    for name, row in routing['per_frame_budget_rule'].items():
        if not {'saving','bits','frames_over'} <= set(row):
            raise RuntimeError(f'Missing budget evidence: {name}')
        points[name] = {
            'analytical_saving_pct_vs_released': row['saving'],
            'estimated_side_bits_per_frame': row['bits'],
            'frames_over_0_1db': row['frames_over'],
            'violation_fraction': row['frames_over']/N_FRAMES,
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
                'qp': qp, 'arm': arm, 'n_frames': len(subset),
                'median_full_frame_gap_to_released_db':
                    statistics.median(row['gap_on_db'] for row in subset),
                'median_repair_gain_db':
                    statistics.median(row['repair_gain_db'] for row in subset),
            })
    output = {
        'schema': 1, 'gpu_used': False,
        'routing_source_sha256': sha(ROUTING), 'seam_source_sha256': sha(SEAM),
        'budget_rule': 'per-frame 0.1 dB proxy from existing true-decode verified routing sweep; 265 frame-QP cases',
        'side_bits_scope': 'estimated adaptive code lengths, not emitted map bytes',
        'saving_scope': 'analytical cost model, not measured latency',
        'points': points, 'dominated': dominated,
        'seam_scope': 'whole-frame PSNR gap; cannot isolate border-band distortion from these records',
        'seam': seam_rows,
    }
    target = HERE/'offline_data/offline_audit.json'
    target.write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(target), 'dominated': dominated,
                      'routed_qp63_gap_db': next(r['median_full_frame_gap_to_released_db']
                                                 for r in seam_rows if r['qp']==63 and r['arm']=='routed')}))


if __name__ == '__main__':
    main()
