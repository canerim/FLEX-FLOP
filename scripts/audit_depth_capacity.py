"""Instantiate all six DCVC-UF depth controls on CPU and verify capacity.

Uses the frozen training constructor. This checks architecture and retained
initial tensors only; it is not training, reconstruction quality, or timing.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
EXPERIMENT = Path("/data10/shareddata/can_karsal/dcvcuf_depth_20260927")
sys.path.insert(0, str(EXPERIMENT / "code_snapshot_v2"))
with contextlib.redirect_stdout(sys.stderr):
    import torch
    from runtime import check_upstream, model_for_depth


def main():
    torch.set_num_threads(2)
    check_upstream()
    baseline = model_for_depth(12)
    baseline_state = baseline.state_dict()
    rows = []
    for depth in (2, 4, 6, 8, 10, 12):
        model = model_for_depth(depth)
        state = model.state_dict()
        changed = [key for key, value in state.items()
                   if not torch.equal(value, baseline_state[key])]
        assert not changed, (depth, changed)
        decoder_count = sum(p.numel() for p in model.dec.parameters())
        full_count = sum(p.numel() for p in model.parameters())
        assert decoder_count == 1767168 + 1038720 * depth
        assert full_count == decoder_count + 27947520
        assert len(model.dec.dec_1) == depth + 1  # upsampler plus retained blocks
        rows.append(dict(
            depth=depth, decoder_parameters=decoder_count,
            total_parameters=full_count, matched_initial_tensors=len(state),
            differing_initial_tensors=changed,
            all_parameters_trainable=all(p.requires_grad for p in model.parameters()),
        ))
        del state, model
    assert not torch.cuda.is_initialized()
    sources = [Path(__file__), EXPERIMENT / "code_snapshot_v2/runtime.py",
               EXPERIMENT / "upstream/src/models/image_model.py",
               EXPERIMENT / "upstream/src/models/common_model.py"]
    report = dict(
        scope="CPU instantiation of D2/D4/D6/D8/D10/D12 with the frozen training constructor. Exact capacity and equal retained initialization only; no trained RD, inference runtime, or new training run.",
        seed=42, torch_version=torch.__version__, gpu_initialized=False,
        checks="All retained tensors exactly match independently reconstructed D12 initialization; parameter counts agree with the architectural formula; retained block count is depth plus one upsampler.",
        rows=rows,
        bank_parameters=sum(row["total_parameters"] for row in rows),
        bank_fp32_weights_mib=sum(row["total_parameters"] for row in rows) * 4 / 2**20,
        source_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
