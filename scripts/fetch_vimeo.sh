#!/bin/bash
# Fetch Vimeo-90k septuplet — the data the DCVC-UF recipe names for the video
# model (training.md: "The septuplet subset of Vimeo-90k").
#
# Source: MIT CSAIL toflow mirror. Verified alive 2026-08-15 (HTTP 200,
# Content-Length 87,930,374,183 = 81.9 GiB, content-type application/zip).
#
# Runs on a different host from the Open Images S3 fetch, so the two downloads
# are launched in parallel — they do not contend for the same upstream.
set -u

DEST=/data10/shareddata/vimeo90k
URL=http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip
mkdir -p "$DEST"
cd "$DEST" || exit 1

echo "=== [$(date '+%H:%M:%S')] fetching vimeo_septuplet.zip (81.9 GiB) ==="
wget --continue --progress=dot:giga --tries=10 --waitretry=15 \
     --read-timeout=60 "$URL" 2>&1 | grep -vE "^\s*$" | tail -5
echo "=== [$(date '+%H:%M:%S')] done: $(du -h vimeo_septuplet.zip 2>/dev/null | cut -f1) ==="

# The recipe's second video source: the ORIGINAL videos, used to build long
# sequences for the later training stages. We take the list here; fetching the
# videos themselves is a separate, much larger job handled after the septuplet
# pipeline is proven.
echo "=== fetching original_video_list.txt ==="
wget --continue -q "http://data.csail.mit.edu/tofu/dataset/original_video_list.txt" \
     -O original_video_list.txt 2>&1 | tail -2
wc -l original_video_list.txt 2>/dev/null

df -h /data10 | tail -1
