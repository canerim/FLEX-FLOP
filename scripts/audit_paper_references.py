"""Trace the two reconstruction anchors used by the archived e15 evaluation.

CPU-only checkpoint inspection; does not rerun a benchmark or alter raw records.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flexuf.config import FlexUFConfig
from flexuf.reference import reference_for


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    torch.set_num_threads(2)
    p = ROOT / "runs/RECIPE512/ckpt_PIN_e15.pth.tar"
    ck = torch.load(p, map_location="cpu", weights_only=False, mmap=True)
    ref_path = Path(reference_for(FlexUFConfig(**ck["config"]), None))
    ref = torch.load(ref_path, map_location="cpu", weights_only=False, mmap=True)
    a = ck["state_dict"]; b = ref.get("state_dict", ref.get("net", ref))
    groups = {}
    for name, predicate in [
        ("encoder_entropy", lambda k: not k.startswith(("dec.", "router_head."))),
        ("decoder_inherited", lambda k: k.startswith(("dec.groups.", "dec.upsample.", "dec.head."))),
    ]:
        keys = [k for k in a if k in b and predicate(k) and a[k].shape == b[k].shape]
        changed = [k for k in keys if not torch.equal(a[k], b[k])]
        groups[name] = dict(compared=len(keys), changed=len(changed), changed_keys=changed,
                            max_absolute_difference=max((a[k] - b[k]).abs().max().item() for k in keys))
    assert groups["encoder_entropy"]["compared"] > 0
    assert groups["encoder_entropy"]["changed"] == 0
    assert groups["decoder_inherited"]["changed"] > 0
    files = [p, ref_path, ROOT / "flexplus/dump_router_lp.py",
             ROOT / "flexplus/eval_rules_ctc.py", ROOT / "flexuf/backbone/decoder.py"]
    result = dict(
        inspection="CPU state-dict comparison and source trace; no GPU or new image-quality measurement",
        checkpoint_config=ck["config"], checkpoint_epoch=ck["epoch"], groups=groups,
        selection_reference="dump_router_lp.py loads reference_for(cfg) into a separate ref model; R is reference_frame_mse(ref.dec, y, q, padded_source)",
        reporting_reference="eval_rules_ctc.py uses net.dec.forward_full(y,q), after loading the fine-tuned e15 model, and crops it; raw field psnr_release is misleading",
        implications=[
            "255 shared encoder/entropy tensors are equal in the two inspected checkpoints.",
            "Inherited synthesis weights are changed; full-frame e15 is not released synthesis.",
            "Reported RGB and YUV losses are relative to full-frame fine-tuned e15.",
            "Counts above the nominal target combine reference and support differences; they do not isolate a budget-enforcement defect.",
            "Paired MAC differences between recorded policies do not depend on renaming the reporting anchor.",
            "Exact cropped released-reference losses cannot be reconstructed from the exported JSON alone; rerun with both anchors.",
        ],
        hashes={str(f.relative_to(ROOT)): sha(f) for f in files},
        caveat="Hashes and weight comparison describe files present at audit time; historical evaluator code/checkpoint hashes were not recorded during execution.",
    )
    out = ROOT / "paper/data/refresh20260927/reference_audit.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {s: v for s, v in x.items() if s != "changed_keys"} for k, x in groups.items()}))


if __name__ == "__main__":
    main()
