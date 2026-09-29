"""Freeze outcome-blind EE map interventions for a later decode/timing ablation.

Original, lower-interface and higher-interface maps preserve exit counts within
valid-tile-size strata. Local search is a heuristic, not a global optimum.
Preparation measures geometry only; it does not decode or estimate runtime.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
PATCH = 256
SEEDS = (202609290, 202609291)
LABELS = (2, 3, 4, 5)  # Archived global exit IDs, NOT reindexed 0..3.


def geometry(h, w):
    ny, nx = (h + PATCH - 1) // PATCH, (w + PATCH - 1) // PATCH
    extents = [(min(PATCH, h - r * PATCH), min(PATCH, w - c * PATCH))
               for r in range(ny) for c in range(nx)]
    edges = []
    for r in range(ny):
        for c in range(nx):
            i = r * nx + c
            if c + 1 < nx:
                edges.append((i, i + 1, extents[i][0]))
            if r + 1 < ny:
                edges.append((i, i + nx, extents[i][1]))
    groups = defaultdict(list)
    for i, extent in enumerate(extents):
        groups[extent].append(i)
    return extents, edges, list(groups.values())


def boundary_length(values, edges):
    return sum(length for a, b, length in edges if values[a] != values[b])


def components(values, edges):
    parent = list(range(len(values)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, _ in edges:
        if values[a] == values[b]:
            parent[find(a)] = find(b)
    return len({find(i) for i in range(len(values))})


def optimize(start, edges, groups, direction):
    """Steepest swap search; direction=-1 reduces, +1 increases interface."""
    values = start.copy()
    neighbours = [[] for _ in values]
    for a, b, length in edges:
        neighbours[a].append((b, length)); neighbours[b].append((a, length))
    swaps = [(a, b) for group in groups for i, a in enumerate(group) for b in group[i + 1:]]
    steps = 0
    while steps < 256:
        best, pair = 0, None
        for a, b in swaps:
            va, vb = values[a], values[b]
            if va == vb:
                continue
            # The a--b interface is invariant under their exchange.
            delta = sum(length * ((vb != values[j]) - (va != values[j]))
                        for j, length in neighbours[a] if j != b)
            delta += sum(length * ((va != values[j]) - (vb != values[j]))
                         for j, length in neighbours[b] if j != a)
            if direction * delta > best:
                best, pair = direction * delta, (a, b)
        if pair is None:
            break
        a, b = pair
        values[a], values[b] = values[b], values[a]
        steps += 1
    assert direction * (boundary_length(values, edges) - boundary_length(start, edges)) >= 0
    return values, steps


def make_variants(values, h, w, identity):
    extents, edges, groups = geometry(h, w)
    if len(values) != len(extents) or not set(values) <= set(LABELS):
        raise ValueError("Unexpected shared-exit map geometry or labels")
    starts = [values.copy()]
    for seed in SEEDS:
        rng = random.Random(int(hashlib.sha256(f"{identity}:{seed}".encode()).hexdigest(), 16))
        shuffled = values.copy()
        for group in groups:
            labels = [values[i] for i in group]; rng.shuffle(labels)
            for i, value in zip(group, labels):
                shuffled[i] = value
        starts.append(shuffled)
    result = [("original", values.copy(), 0)]
    for name, direction in (("lower_interface", -1), ("higher_interface", 1)):
        candidates = [optimize(start, edges, groups, direction) for start in starts]
        best = max(candidates, key=lambda pair: direction * boundary_length(pair[0], edges))
        result.append((name, *best))
    rows = []
    for name, mapping, steps in result:
        for group in groups:
            assert Counter(values[i] for i in group) == Counter(mapping[i] for i in group)
        area = [sum(hh * ww for i, (hh, ww) in enumerate(extents) if mapping[i] == e) for e in LABELS]
        rows.append({"variant": name, "map": mapping,
                     "exit_tile_counts": [mapping.count(e) for e in LABELS],
                     "exit_valid_pixel_counts": area,
                     "cross_exit_interface_length_rgb_pixels": boundary_length(mapping, edges),
                     "same_exit_connected_components": components(mapping, edges),
                     "selected_start_local_search_steps": steps})
    assert rows[1]["cross_exit_interface_length_rgb_pixels"] <= rows[0]["cross_exit_interface_length_rgb_pixels"] <= rows[2]["cross_exit_interface_length_rgb_pixels"]
    assert all(r["exit_valid_pixel_counts"] == rows[0]["exit_valid_pixel_counts"] for r in rows)
    return rows


def self_check():
    # Ragged edges must not exchange labels with full-size tiles.
    examples = [(513, 769, [2, 3, 4, 5, 4, 3, 2, 5, 2, 3, 4, 5]),
                (768, 768, [2, 3, 2, 3, 2, 3, 2, 3, 2]),
                (512, 512, [5, 5, 5, 5])]
    for h, w, values in examples:
        rows = make_variants(values, h, w, "synthetic")
        assert rows == make_variants(values, h, w, "synthetic")
        if len(set(values)) == 1:
            assert all(r["map"] == values and r["same_exit_connected_components"] == 1 for r in rows)
    return "Ragged-stratum conservation, deterministic selection, constant-map identity, and monotone interface controls passed."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-cases", type=Path, default=Path("/data10/shareddata/can_karsal/dcvcuf_depth_20260927/research/shared_crossfit_qp32/cases"))
    parser.add_argument("--output", type=Path, default=ROOT / "data/ablation20260929/fragmentation_maps.json")
    args = parser.parse_args()
    check = self_check()
    sources = sorted(args.source_cases.glob("*.json"))
    if len(sources) != 53:
        raise ValueError("Expected the complete frozen 53-sequence development corpus")
    cases, hashes = [], {}
    for path in sources:
        payload = path.read_bytes(); hashes[path.name] = hashlib.sha256(payload).hexdigest()
        source = json.loads(payload)
        if source["qp"] != 32:
            raise ValueError("Expected QP32")
        for policy in ("router", "dither"):
            matches = [r for r in source["rows"] if (r["criterion"], r["policy"]) == ("q90", policy)]
            if len(matches) != 1:
                raise ValueError("Ambiguous source map")
            # Only these fields reach map generation. No quality field is used.
            values = matches[0]["map"]
            case = {"sequence": source["sequence"], "sequence_index": source["sequence_index"],
                    "policy": policy, "height": source["height"], "width": source["width"],
                    "frame_sha256": source["first_frame_bytes_sha256"]}
            case["rows"] = make_variants(values, case["height"], case["width"], case["sequence"])
            cases.append(case)
    result = {"state": "prepared_not_decoded_or_timed", "scope": __doc__,
              "source": "Frozen QP32/Q90 maps from the already inspected shared-exit development replay, all 53 sequences and both policies; no external-test claim.",
              "exit_labels": list(LABELS), "exit_depths": [6, 8, 10, 12], "rgb_patch": PATCH,
              "map_rule": "Original plus two seeded starts; steepest strict interface-length-improving within-valid-extent swaps, at most256 accepted swaps/start/direction; choose by geometry alone.",
              "seeds": list(SEEDS), "source_cases_sha256": hashes,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checks": check, "n_sequences": 53, "n_policy_cases": len(cases),
              "n_scheduled_maps": sum(len(c["rows"]) for c in cases),
              "n_maps_with_geometry_contrast": sum(c["rows"][1]["cross_exit_interface_length_rgb_pixels"] != c["rows"][2]["cross_exit_interface_length_rgb_pixels"] for c in cases),
              "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("state", "checks", "n_scheduled_maps", "n_maps_with_geometry_contrast")}, indent=2))


if __name__ == "__main__":
    main()
