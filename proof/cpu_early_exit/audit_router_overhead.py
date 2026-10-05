"""Add measured router arithmetic to frozen shared-exit MAC comparisons.

This is accounting, not a latency benchmark. Dither is a deterministic map
and needs no learned router; routed maps pay the hook-counted router MACs.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "proof/early_exit_vs_released"))
from mac_latency_audit import case_macs

ARCHIVE = ROOT / "docs/research/2026-10-05-ctc-exact-mac/analysis.json"
CONTROLLER = ROOT / "cvpr2027/data/research20260927/shared_crossfit_qp32/router_logit_audit.json"
FIXED = ROOT / "cvpr2027/data/research20260927/shared_crossfit_qp32/analysis.json"
OUT = ROOT / "docs/research/2026-10-05-router-overhead"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    archive = json.loads(ARCHIVE.read_text())
    control = json.loads(CONTROLLER.read_text())
    fixed = json.loads(FIXED.read_text())
    by_seq = {r["sequence"]: r for r in control["rows"]}
    assert len(by_seq) == 53 and len(fixed["rows"]) == 318
    for r in by_seq.values():
        h, w = r["padded_shape"][-2:]
        assert r["router_conv_linear_macs"] / (h*w) == 289.71484375
    by_key = {(r["seq"], r["qp"], r["budget"], r["policy"]): r
              for r in archive["per_case"]}
    keys = sorted({(seq, qp) for seq, qp, budget, policy in by_key
                   if budget == "0.1" and all((seq, qp, "0.1", p) in by_key
                                                for p in ("uniform", "dither", "router", "oracle"))})
    assert len(keys) == 263
    nominal = []
    for seq, qp in keys:
        base = by_key[(seq, qp, "0.1", "router")]
        ref = base["released_conv_gmac"] * 1e9
        overhead_pp = 100 * by_seq[seq]["router_conv_linear_macs"] / ref
        nominal.append({"sequence": seq, "qp": qp,
                        "router_excluded_saving_pct": base["exact_conv_saving_pct"],
                        "dither_saving_pct": by_key[(seq, qp, "0.1", "dither")]["exact_conv_saving_pct"],
                        "router_included_saving_pct": base["exact_conv_saving_pct"] - overhead_pp,
                        "controller_cost_pct_points": overhead_pp})
    fixed_rows = []
    for row in fixed["rows"]:
        if row["policy"] not in ("router", "dither"):
            continue
        src = by_seq[row["sequence"]]
        h, w = src["padded_shape"][-2:]
        counts = [row["map"].count(i) for i in range(6)]
        mac = case_macs({"padded_shape": [h, w], "tile_counts": counts})
        overhead_pp = 100 * src["router_conv_linear_macs"] / mac["released_conv_macs"]
        fixed_rows.append({"sequence": row["sequence"], "criterion": row["criterion"],
                           "policy": row["policy"], "excluded_saving_pct": 100 * mac["e15_routed_conv_mac_saving_fraction"],
                           "included_saving_pct": 100 * mac["e15_routed_conv_mac_saving_fraction"] -
                                                  (overhead_pp if row["policy"] == "router" else 0),
                           "controller_cost_pct_points": overhead_pp if row["policy"] == "router" else 0})
    assert len(fixed_rows) == 212

    def mean(x): return sum(x) / len(x)
    def summarize(rows, criterion=None):
        if criterion is not None:
            rows = [r for r in rows if r.get("criterion") == criterion]
        result = {"n": len(rows),
                  "mean_controller_cost_pct_points": mean([r["controller_cost_pct_points"] for r in rows if
                                                            "policy" not in r or r["policy"] == "router"]),
                  "mean_router_included_saving_pct": mean([r["router_included_saving_pct"] if "policy" not in r
                                                             else r["included_saving_pct"]
                                                             for r in rows if "policy" not in r or r["policy"] == "router"]),
                  "mean_dither_saving_pct": mean([r["dither_saving_pct"] if "policy" not in r
                                                   else r["included_saving_pct"]
                                                   for r in rows if "policy" not in r or r["policy"] == "dither"])}
        result["router_minus_dither_included_points"] = (result["mean_router_included_saving_pct"] -
                                                          result["mean_dither_saving_pct"])
        return result

    result = {
        "scope": "Frozen CTC shared-exit maps; exact synthesis convolution MAC plus measured router Conv2d/Linear MAC, relative to released D12 synthesis convolutions. No latency, control dispatch, entropy or memory estimate.",
        "inputs_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (ARCHIVE, CONTROLLER, FIXED)},
        "nominal_0p1_all_feasible_263": summarize(nominal),
        "fixed_qp32_mean_53": summarize(fixed_rows, "mean"),
        "fixed_qp32_q90_53": summarize(fixed_rows, "q90"),
        "controller_cost_range_pct_points": [min(r["controller_cost_pct_points"] for r in nominal),
                                              max(r["controller_cost_pct_points"] for r in nominal)],
        "cases_nominal": nominal,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("cases_nominal", "inputs_sha256")}, indent=2))


if __name__ == "__main__":
    main()
