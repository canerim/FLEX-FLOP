"""Reconcile exact convolution MACs with the nine recorded latency workloads.

Counts output pixels times convolution kernel taps and input channels per group.
Bias, activations, patch assembly and host planning are outside MAC accounting.
The arithmetic follows Microsoft's IntraDecoder and the pinned e15 geometry.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
DATA = HERE / "results/cohort_20261003"
CHANNELS = 384
HEAD_CHANNELS = 192
FEATURE_STRIDE = 8
TILE_FEATURE = 32


def block_macs_per_feature_pixel(channels: int) -> int:
    return 7 * channels * channels + 9 * channels


def case_macs(row: dict) -> dict:
    height, width = row["padded_shape"]
    assert height % 256 == 0 and width % 256 == 0
    feature_pixels = height * width // FEATURE_STRIDE**2
    tile_pixels = TILE_FEATURE**2
    counts = row["tile_counts"]
    assert sum(counts) * tile_pixels == feature_pixels
    assert len(counts) == 6 and counts[:2] == [0, 0]

    # The 256->1536 upsample convolution runs on the latent grid, whose
    # resolution is 1/4 of the post-PixelShuffle feature grid.
    upsample_per_feature = 256 * (CHANNELS * 4) // 4
    stem = upsample_per_feature + block_macs_per_feature_pixel(CHANNELS)
    head = CHANNELS * HEAD_CHANNELS + block_macs_per_feature_pixel(HEAD_CHANNELS)
    block = block_macs_per_feature_pixel(CHANNELS)
    released = feature_pixels * (stem + 12 * block + head)
    # The deepest exit has no adapter. Exits 2/3 use a 5C² FFN adapter;
    # exit 4 uses a C² adapter. Grid seam repair runs on the whole canvas.
    adapters = tile_pixels * (counts[2] * 5 * CHANNELS**2 +
                              counts[3] * 5 * CHANNELS**2 +
                              counts[4] * CHANNELS**2)
    seam = feature_pixels * (CHANNELS**2 + 9 * CHANNELS)
    trunk = tile_pixels * sum(count * 2 * (mode + 1) * block
                              for mode, count in enumerate(counts))
    routed = feature_pixels * (stem + head) + trunk + adapters + seam
    all_deep_tiled = released + seam
    return {"released_conv_macs": released, "e15_routed_conv_macs": routed,
            "e15_all_deep_tiled_conv_macs": all_deep_tiled,
            "e15_routed_conv_mac_saving_fraction": 1 - routed / released,
            "e15_routed_vs_all_deep_conv_mac_saving_fraction":
                1 - routed / all_deep_tiled,
            "average_trunk_blocks_per_tile":
                sum(count * 2 * (mode + 1) for mode, count in enumerate(counts)) /
                sum(counts)}


def main() -> None:
    rows = []
    for source in sorted(DATA.glob("*qp*.json")):
        record = json.loads(source.read_text())
        measured = record["median_wall_ms"]
        row = {"sequence": record["sequence"], "qp": record["qp"],
               "padded_shape": record["padded_shape"],
               "tile_counts": record["tile_counts"],
               **case_macs(record),
               "released_vs_e15_stock_time_ratio":
                   record["median_speedup_released_vs_e15_stock_wall"],
               "released_vs_e15_triton_time_ratio": record["median_speedup_wall"],
               "released_stock_wall_ms": measured["released_d12"],
               "e15_stock_wall_ms": measured["e15_stock"],
               "e15_triton_wall_ms": measured["e15_triton"]}
        rows.append(row)
    result = {"definition": "convolution MACs only, exact padded frame and archived tile map",
              "block_macs_per_feature_pixel": block_macs_per_feature_pixel(CHANNELS),
              "incorrect_8c2_plus_9c_per_feature_pixel": 8*CHANNELS**2+9*CHANNELS,
              "cost_model_block_overpricing_fraction":
                  (8*CHANNELS**2+9*CHANNELS)/block_macs_per_feature_pixel(CHANNELS)-1,
              "median_routed_conv_mac_saving_fraction": statistics.median(
                  x["e15_routed_conv_mac_saving_fraction"] for x in rows),
              "rows": rows}
    output = DATA / "mac_latency_audit.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("per-block MACs", result["block_macs_per_feature_pixel"])
    print("cost-model overpricing", round(100*result["cost_model_block_overpricing_fraction"], 2), "%")
    print("median routed MAC saving", round(100*result["median_routed_conv_mac_saving_fraction"], 2), "%")
    for x in rows:
        print(x["sequence"].split("_")[0], "QP", x["qp"],
              "padded", "x".join(map(str, x["padded_shape"])),
              "released", round(x["released_conv_macs"]/1e9, 2), "GMAC",
              "saving", round(100*x["e15_routed_conv_mac_saving_fraction"], 1), "%",
              "stock speedup", round(x["released_vs_e15_stock_time_ratio"], 2))


if __name__ == "__main__":
    main()
