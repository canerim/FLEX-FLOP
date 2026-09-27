"""Save auditable source locations for the historical runtime scope review."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    path = ROOT / "flexplus/runtime_full_uf.py"
    source = path.read_text()
    tree = ast.parse(source)
    evidence = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Call, ast.Assign)):
            continue
        segment = ast.get_source_segment(source, node)
        labels = []
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func)
            if name == "ec.decoder.decode_y": labels.append("entropy_decode_uses_encoder_indices")
            if name == "decoder_side": labels.append("neural_decode_reads_encoder_capture")
            if name == "net.dec.forward_full": labels.append("full_synthesis_uses_e15_network")
            if name == "kt_bits": labels.append("ideal_map_length_evaluation")
        elif isinstance(node, ast.Assign) and 'Q["saving_R1"]' in segment:
            labels.append("r1_quality_copied_from_r2")
        for label in labels:
            evidence.append(dict(label=label, line=node.lineno, end_line=node.end_lineno, source=segment))
    expected = {"entropy_decode_uses_encoder_indices", "neural_decode_reads_encoder_capture",
                "full_synthesis_uses_e15_network", "ideal_map_length_evaluation", "r1_quality_copied_from_r2"}
    assert {e["label"] for e in evidence} == expected
    files = [path, ROOT/"flexuf/decodability.py", ROOT/"flexplus/results/runtime_full_uf.json",
             ROOT/"flexplus/results/encoder_time_v2.json", Path(__file__)]
    result = dict(scope="Static current-source inspection; not a new timing or entropy correctness test",
        implication="The archived full-runtime program times real entropy calls and neural replay stages, but does not feed decoded stream outputs into the neural reconstruction. Map parse is an ideal-length calculation.",
        evidence=sorted(evidence,key=lambda e:e["line"]),
        caveat="The historical executable's hash was not saved at run time. These hashes identify files inspected on 27 September 2026.",
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    out = ROOT/"paper/data/refresh20260927/runtime_scope_audit.json"
    out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(dict(evidence_items=len(evidence),output=str(out))))


if __name__=="__main__":main()
