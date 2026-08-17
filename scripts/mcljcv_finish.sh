#!/bin/bash
# Finish the MCL-JCV fetch, and extract only what the test set actually uses.
#
# Replaces the unpack step of scripts/fetch_mcljcv.sh, which would have expanded
# every archive whole. Surveyed, the 40 parts hold:
#
#     MCL_JCV/1080P/videos          3025 entries   encoded videos, not sources
#     MCL_JCV/720P/videos           2512 entries
#     MCL_JCV/1080P/JND_raw_samples   92
#     MCL_JCV/720P/JND_raw_samples    90
#     MCL_JCV/1080P/YUV_source        30   <-- the only ones we need
#     MCL_JCV/720P/YUV_source         30
#
# ~/DCVC/test_cfg/all_yuv420.json names the 1080p sources exactly
# (videoSRC01_1920x1080_30.yuv ...), so the 720p copies would not even be found
# by ctc_intra.py's discover() -- they would just occupy a shared disk. And the
# old flatten step (`find -name '*.yuv' -exec mv`) would have moved both.
#
# Rate-limited and sequential for the reason the UVG fetch was: this is a shared
# uplink, and saturating it interferes with other people's work as surely as
# taking their GPU would.
set -u
DST=/data10/shareddata/test_sequences/YUV/MCL-JCV
TMP=/data10/shareddata/test_sequences/.mcl_tmp
BASE="https://huggingface.co/datasets/uscmcl/MCL-JCV_Dataset/resolve/main"
mkdir -p "$DST" "$TMP" && cd "$TMP"

# Part list WITH sizes, so "already downloaded" means complete rather than
# merely present -- a truncated part is the failure that would otherwise
# surface much later as a corrupt archive.
curl -sL --max-time 120 \
  "https://huggingface.co/api/datasets/uscmcl/MCL-JCV_Dataset/tree/main" \
  > /tmp/mcl_tree.json || { echo "[FAIL] cannot list dataset"; exit 1; }

python3 - <<'PY' > /tmp/mcl_parts.txt
import json
for f in json.load(open("/tmp/mcl_tree.json")):
    if f["path"].endswith(".zip"):
        print(f["path"], f.get("size", 0))
PY

n=0
while read -r f sz; do
  n=$((n+1))
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$have" = "$sz" ]; then echo "[ok  ] ($n) $f"; continue; fi
  echo "[get ] ($n) $f  ($have / $sz)"
  curl -L -C - --limit-rate 40M --retry 5 --retry-delay 10 -o "$f" "$BASE/$f" \
    || { echo "[FAIL] $f"; continue; }
done < /tmp/mcl_parts.txt

echo "[extract] 1080p YUV sources only"
python3 - "$TMP" "$DST" <<'PY'
import sys, zipfile, pathlib, re
tmp, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
want = re.compile(r"1080P/YUV_source/.*1920x1080.*\.yuv$")
got = 0
for z in sorted(tmp.glob("*.zip")):
    try:
        zf = zipfile.ZipFile(z)
    except Exception as e:
        print(f"  [skip] {z.name}: {e}")
        continue
    for e in zf.namelist():
        if not want.search(e):
            continue
        out = dst / pathlib.Path(e).name
        # Size check rather than existence: a run interrupted mid-copy leaves a
        # short file, and silently keeping it would corrupt the test set.
        if out.exists() and out.stat().st_size == zf.getinfo(e).file_size:
            continue
        print(f"  {out.name}  ({zf.getinfo(e).file_size / 2**20:.0f} MB)")
        with zf.open(e) as src, open(out, "wb") as fh:
            while chunk := src.read(1 << 22):
                fh.write(chunk)
        got += 1
print(f"[extract] {got} new file(s)")
PY

echo "[done] $(ls -1 "$DST"/*.yuv 2>/dev/null | wc -l) sequences, $(du -sh "$DST" | cut -f1)"
echo "[clean] removing archives"
rm -f "$TMP"/*.zip
rmdir "$TMP" 2>/dev/null || true
