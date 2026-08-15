#!/bin/bash
# Fetch Open Images train subsets 0,1,2 — the exact data the DCVC-UF recipe
# names for the intra (image) model (training.md: "subsets 0, 1, and 2 of the
# Open Images training set").
#
# Source: CVDF public S3 mirror (https://github.com/cvdfoundation/open-images-dataset).
# Verified 2026-08-15: HTTP 200 with no credentials.
#   train_0.tar.gz  45.9 GiB
#   train_1.tar.gz  34.4 GiB
#   train_2.tar.gz  33.0 GiB
#   ------------------------
#   total          113.3 GiB
#
# --continue makes this resumable: re-running after an interruption picks up
# where it stopped instead of restarting a 46 GB file.
set -u

DEST=/data10/shareddata/openimages
BASE=https://open-images-dataset.s3.amazonaws.com/tar
mkdir -p "$DEST"
cd "$DEST" || exit 1

for i in 0 1 2; do
    f="train_${i}.tar.gz"
    echo "=== [$(date '+%H:%M:%S')] fetching ${f} ==="
    wget --continue --progress=dot:giga --tries=10 --waitretry=15 \
         --read-timeout=60 "${BASE}/${f}" 2>&1 | grep -vE "^\s*$" | tail -5
    echo "=== [$(date '+%H:%M:%S')] done ${f}: $(du -h "${f}" 2>/dev/null | cut -f1) ==="
done

echo "=== ALL OPEN IMAGES SUBSETS FETCHED ==="
ls -lh "$DEST"
df -h /data10 | tail -1
