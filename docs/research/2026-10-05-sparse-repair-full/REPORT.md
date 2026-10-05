# Frozen grid-repair gate: complete QP32 scan

We measured all 53 CTC first frames with the frozen e15 checkpoint and the same QP32/Q90 route used for the repair-identity intervention. Thresholds 0.25 and 0.30 were chosen on three geometry-stratified pilot frames before scanning the other 50. For reconstruction quality, the script executes the dense pointwise convolution and masks its contribution, so its outputs are actual neural reconstructions. The lower MAC figures **project** what an exact sparse 1×1 kernel would skip; they are not measured runtime or executed-operation counts.

| Frozen repair variant, 50-image extension | Active 1×1 locations | Projected synthesis-conv MAC saving vs released D12 | Added RGB PSNR loss vs trained repair | Cases exceeding 0.1 dB vs dense e15 |
| --- | ---: | ---: | ---: | ---: |
| Trained, dense | 100% | 16.8025% | 0 | 4/50 |
| Gate ≥ 0.25 | 13.5742% | 17.7196% | −0.00060 dB [−0.00139, 0.00003] | 5/50 |
| Gate ≥ 0.30 | 12.1094% | 17.7352% | −0.00060 dB [−0.00140, 0.00004] | 5/50 |
| Repair off | 0% | 17.8637% | +0.00211 dB [0.00057, 0.00379] | 6/50 |

At 0.25, the projected **additional saving is 0.9171 percentage points**, with worst per-image additional RGB loss 0.00287 dB. The 50 extension images split 25 improved / 25 worsened in RGB PSNR; the mean interval includes zero. `YachtRide` crosses the 0.1-dB line narrowly, from 0.09958 to 0.10077 dB against dense e15. Across all 53 images, projected saving rises from 16.2602% to 17.1774%, mean added loss is −0.00055 dB, and threshold violations rise from 4 to 5. The fact that disabling repair entirely gains only 1.0612 extra points on the extension bounds this optimization's arithmetic upside.

The threshold extension is separate from the three pilot images, but the codec and route were developed on CTC; it is not an untouched external set. Only QP32 and one frozen route/weight set were measured. The packed implementation in [`sparse_grid_repair.py`](../../../proof/cpu_early_exit/sparse_grid_repair.py) is inference-only and reproduces one decoded case, but no quiet-GPU repair-stage or full-decoder latency has been measured. Avoid a speedup claim until the packed path beats the matched fused dense-thresholded baseline with identical output on a free GPU.

Inputs, hashes, full rows and paired 5,000-draw image intervals are in [`sparse_repair_gate_full_20261005.json`](../../../proof/cpu_early_exit/results/sparse_repair_gate_full_20261005.json) and [`sparse_repair_gate_full_summary_20261005.json`](../../../proof/cpu_early_exit/results/sparse_repair_gate_full_summary_20261005.json). The replay is [`audit_sparse_repair_gate_full.py`](../../../proof/cpu_early_exit/audit_sparse_repair_gate_full.py); the summary is [`summarize_sparse_repair_full.py`](../../../proof/cpu_early_exit/summarize_sparse_repair_full.py).
