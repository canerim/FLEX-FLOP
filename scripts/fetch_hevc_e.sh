#!/bin/bash
# HEVC Class E, the three sequences DCVC's test_cfg names, from xiph.org's derf
# collection (the only clean public source found for them). Sequential and
# rate-limited for the same shared-uplink reason as the UVG fetch.
set -u
DST=/data10/shareddata/test_sequences/YUV/HEVC_E
mkdir -p "$DST" && cd "$DST"
for F in FourPeople_1280x720_60 Johnny_1280x720_60 KristenAndSara_1280x720_60; do
    [ -f "$F.yuv" ] && { echo "[skip] $F.yuv"; continue; }
    echo "[get ] $F"
    curl -L -C - --limit-rate 40M --retry 5 --retry-delay 10 \
         -o "$F.y4m" "https://media.xiph.org/video/derf/y4m/$F.y4m" || { echo "[FAIL] $F"; continue; }
    ~/FLEX-UF/.venv/bin/python ~/FLEX-UF/scripts/y4m2yuv.py "$F.y4m" "$F.yuv" && rm -f "$F.y4m"
done
echo "[done] $(ls -1 *.yuv 2>/dev/null|wc -l)/3, $(du -sh .|cut -f1)"
