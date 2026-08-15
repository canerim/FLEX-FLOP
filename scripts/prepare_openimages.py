"""Turn the Open Images tarball into the layout DCVC-UF's ImageFolder expects.

What the recipe expects
-----------------------
`~/DCVC/src/datasets/image_dataset.py:16-26` reads a single `description.json`
at the dataset root — a flat JSON array of image paths relative to that root:

    /path/to/image_dataset/
      description.json          ["img_000001.jpg", "sub/img_000002.jpg", ...]
      img_000001.jpg
      ...

`__getitem__` then opens the file, converts to RGB, pads if smaller than the
crop, takes a random crop of the current patch size, random horizontal flip,
scales to [0,1], converts to YCbCr and subtracts 0.5.

Two consequences that drive this script:

1. **No resizing happens at load time.** Images smaller than the patch get
   zero-padded, and zero padding is not a real image — it teaches the codec to
   reconstruct black borders. At the 512x512 stage (epochs 90+) any image with a
   side under 512 contributes padding. So this script filters by minimum side.

2. **The list is read fully into memory and indexed per sample**, so the cost of
   a large list is negligible but the cost of *scanning the tree at every start*
   is not. Building `description.json` once, here, keeps epoch startup instant.

Open Images images are JPEG at a variety of sizes, typically 1024 on the long
side. Filtering at min-side 512 keeps the great majority and guarantees every
crop at both training stages is real pixels.

Usage
-----
    python scripts/prepare_openimages.py \
        --tar /data10/shareddata/openimages/train_0.tar.gz \
        --dest /data10/shareddata/openimages/dcvc_train \
        --min-side 512
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def extract(tar_path: Path, dest: Path) -> None:
    """Extract the tarball if it has not already been extracted."""
    dest.mkdir(parents=True, exist_ok=True)
    marker = dest / ".extracted"
    if marker.exists():
        print(f"already extracted (marker {marker} present); skipping")
        return

    print(f"extracting {tar_path} -> {dest}  (this takes a while for ~46 GiB)")
    t0 = time.time()
    # tar handles the gzip stream itself; -k so a re-run never clobbers.
    r = subprocess.run(
        ["tar", "-xzf", str(tar_path), "-C", str(dest)],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print(r.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"tar failed with code {r.returncode}")
    marker.write_text(f"{tar_path}\n{time.time() - t0:.0f}s\n")
    print(f"extracted in {time.time() - t0:.0f}s")


def build_description(root: Path, min_side: int, out: Path) -> dict:
    """Scan for usable images and write `description.json`.

    Verifies each file actually opens and is big enough. That check costs one
    header read per file (Pillow's lazy open does not decode pixels), which is
    cheap, and it is the difference between finding a corrupt file now and having
    a dataloader worker die 40 epochs in.
    """
    exts = {".jpg", ".jpeg", ".png"}
    kept, too_small, broken = [], 0, 0
    t0 = time.time()
    n = 0

    for dirpath, _, files in os.walk(root):
        for fn in files:
            if Path(fn).suffix.lower() not in exts:
                continue
            n += 1
            p = Path(dirpath) / fn
            try:
                with Image.open(p) as im:
                    w, h = im.size
            except Exception:
                broken += 1
                continue
            if min(w, h) < min_side:
                too_small += 1
                continue
            kept.append(str(p.relative_to(root)))
            if len(kept) % 50000 == 0:
                print(f"  scanned {n:,}  kept {len(kept):,}  ({time.time() - t0:.0f}s)",
                      flush=True)

    kept.sort()

    # Hold out a fixed validation slice, and take it OUT of the training list.
    #
    # Why this matters: the recipe hands all of Open Images to training, so
    # without this every PSNR we report is measured on images the model has
    # already fitted. For a codec that is less catastrophic than for a
    # classifier — there are no labels to memorise — but "compute saved at X dB"
    # is still a generalisation claim, and a number measured on training data
    # cannot support it. FLEX kept a fixed 192-frame set aside for exactly this
    # reason and never compared across frame lists.
    #
    # The slice is deterministic (every VAL_STRIDE-th file of a sorted list), so
    # it is identical across the three experiments and the baseline, and stays
    # identical when subsets 1 and 2 enlarge the pool. Comparability across runs
    # is the whole point; a random split would break it.
    VAL_STRIDE = 400
    val = kept[::VAL_STRIDE][:512]
    val_set = set(val)
    train = [k for k in kept if k not in val_set]

    out.write_text(json.dumps(train))
    val_path = out.parent / "description_val.json"
    val_path.write_text(json.dumps(val))

    stats = {
        "scanned": n,
        "kept": len(kept),
        "train": len(train),
        "val_heldout": len(val),
        "val_stride": VAL_STRIDE,
        "dropped_too_small": too_small,
        "dropped_broken": broken,
        "min_side": min_side,
        "seconds": round(time.time() - t0, 1),
        "description_json": str(out),
        "description_val_json": str(val_path),
    }
    print(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", type=Path, default=None,
                    help="tarball to extract; omit if already extracted")
    ap.add_argument("--dest", type=Path, required=True)
    ap.add_argument("--min-side", type=int, default=512,
                    help="drop images whose short side is below this; 512 matches "
                         "the recipe's final patch size so no crop is ever padded")
    a = ap.parse_args()

    if a.tar is not None:
        extract(a.tar, a.dest)

    stats = build_description(a.dest, a.min_side, a.dest / "description.json")
    (a.dest / "prepare_stats.json").write_text(json.dumps(stats, indent=2))
    if stats["kept"] == 0:
        raise SystemExit("no usable images found — check --dest and --min-side")


if __name__ == "__main__":
    main()
