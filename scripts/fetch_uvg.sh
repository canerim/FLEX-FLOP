#!/bin/bash
# UVG, the 7 sequences DCVC's own test_cfg/all_yuv420.json names, in the exact
# 1920x1080 / 120fps / 4:2:0 / 8-bit form that config expects.
#
# Sequential, not parallel: this is a shared server and a shared uplink, and
# saturating it would be interfering with other people's work just as surely as
# taking their GPU. One stream, resumable (-C -), rate-limited.
set -u
DST=/data10/shareddata/test_sequences/YUV/UVG
mkdir -p "$DST" && cd "$DST"
for S in Beauty Bosphorus HoneyBee Jockey ReadySteadyGo ShakeNDry YachtRide; do
    F="${S}_1920x1080_120fps_420_8bit_YUV"
    [ -f "$F.yuv" ] && { echo "[skip] $F.yuv var"; continue; }
    echo "[get ] $F"
    curl -L -C - --limit-rate 40M --retry 5 --retry-delay 10 \
         -o "${F}_RAW.7z" "https://ultravideo.fi/video/${F}_RAW.7z" || { echo "[FAIL] $F"; continue; }
    echo "[unz ] $F"
    ~/FLEX-UF/.venv/bin/python -c "
import py7zr,sys; py7zr.SevenZipFile('${F}_RAW.7z','r').extractall('.')" && rm -f "${F}_RAW.7z"
    ls -la "$F.yuv" 2>/dev/null || echo "[WARN] $F.yuv cikmadi"
done
echo "[done] $(ls -1 *.yuv 2>/dev/null | wc -l)/7 sekans, $(du -sh . | cut -f1)"
