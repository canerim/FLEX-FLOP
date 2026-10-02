"""Check that cohort claims require full coverage and sequence-level sampling."""
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    'triton_cohort_summary', ROOT / 'scripts/summarize_triton_ctc_cohort.py')
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def _row(sequence, qp, stock, fast):
    return {'sequence': sequence, 'qp': qp, 'stock_median_ms': stock,
            'fast_median_ms': fast, 'dense_fallback': False,
            'max_abs_output_error': 1e-7, 'delta_yuv611_db': 1e-8}


def test_complete_cohort_rejects_partial_or_duplicate_records():
    rows = [_row('a', 0, 2.0, 1.0)]
    with pytest.raises(ValueError, match='53 sequences'):
        summary.summarize(rows)
    with pytest.raises(ValueError, match='duplicate'):
        summary.summarize(rows * 2, complete=False)
    with pytest.raises(ValueError, match='empty'):
        summary.summarize([], complete=False)


def test_sequence_cluster_summary_and_qp_strata():
    rows = [_row(f's{i:02d}', qp, 2.0 + i, 1.0 + i / 2)
            for i in range(53) for qp in summary.QPS]
    result = summary.summarize(rows)
    assert result['n_frame_qp'] == 265
    assert result['n_sequences'] == 53
    assert result['overall']['ratio'] == pytest.approx(2.0)
    assert result['overall']['ci95'] == pytest.approx([2.0, 2.0])
    assert set(result['by_qp']) == {'0', '16', '32', '48', '63'}
    assert all(val['ratio'] == pytest.approx(2.0)
               for val in result['by_qp'].values())


def test_partial_summary_keeps_each_sequence_as_one_bootstrap_cluster():
    rows = [_row('small', 0, 4, 2), _row('small', 16, 4, 2),
            _row('large', 0, 8, 2)]
    result = summary.summarize(rows, complete=False)
    assert result['overall']['ratio'] == pytest.approx(16 / 6)
    assert result['overall']['n_sequences'] == 2
    assert result['by_qp']['0']['ratio'] == pytest.approx(12 / 4)


def test_dense_fallbacks_are_not_misreported_as_router_cases():
    rows=[_row('routed',0,2,1),_row('dense',0,9,3)]
    rows[1]['dense_fallback']=True
    result=summary.summarize(rows,complete=False)
    assert result['dense_fallbacks']==1
    assert result['router_only']['ratio']==pytest.approx(2)
    assert result['dense_fallback_only']['ratio']==pytest.approx(3)
    assert result['overall']['ratio']==pytest.approx(11/4)


def test_wall_clock_ratio_is_reported_separately_from_cuda_events():
    rows = [_row('one', 0, 2, 1), _row('two', 0, 4, 2)]
    for row in rows:
        row['stock_wall_median_ms'] = row['stock_median_ms'] + 2
        row['fast_wall_median_ms'] = row['fast_median_ms'] + 2
    result = summary.summarize(rows, complete=False)
    assert result['overall']['ratio'] == pytest.approx(2)
    assert result['overall_wall']['ratio'] == pytest.approx(10 / 7)
