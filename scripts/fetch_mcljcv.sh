#!/bin/bash
# MCL-JCV: the 30 CTC sequences ctc_intra.py has been reporting as NOT MEASURED.
#
# Sequential and rate-limited for the same reason the UVG fetch was: this is a
# shared machine on a shared uplink, and saturating it is interfering with other
# people's work as surely as taking their GPU.
#
# Source is the authors' own Hugging Face mirror, linked from mcl.usc.edu.
set -u
DST=/data10/shareddata/test_sequences/YUV/MCL-JCV
TMP=/data10/shareddata/test_sequences/.mcl_tmp
mkdir -p "$DST" "$TMP" && cd "$TMP"
BASE="https://huggingface.co/datasets/uscmcl/MCL-JCV_Dataset/resolve/main"
LIST=$(curl -sL --max-time 60 \
  "https://huggingface.co/api/datasets/uscmcl/MCL-JCV_Dataset/tree/main" \
  | python3 -c "import json,sys;[print(f['path']) for f in json.load(sys.stdin) if f['path'].endswith('.zip')]")
n=0
for f in $LIST; do
  n=$((n+1))
  [ -s "$f" ] && { echo "[skip] $f"; continue; }
  echo "[get ] ($n) $f"
  curl -L -C - --limit-rate 40M --retry 5 --retry-delay 10 -o "$f" "$BASE/$f" \
    || { echo "[FAIL] $f"; continue; }
done
echo "[unzip]"
for f in *.zip; do unzip -o -q "$f" -d "$TMP/x" && rm -f "$f"; done
find "$TMP/x" -name '*.yuv' -exec mv -n {} "$DST/" \;
echo "[done] $(ls -1 "$DST"/*.yuv 2>/dev/null | wc -l) sekans, $(du -sh "$DST" | cut -f1)"
