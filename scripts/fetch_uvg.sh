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
# Ultra Video Group publishes the fourth sequence as "ReadySetGo"; DCVC's
# test_cfg (following the JVET convention) calls the same sequence
# "ReadySteadyGo". Downloaded under the site's name, stored under the config's,
# so test_cfg matches without editing Microsoft's file. Left as a visible pair
# rather than a silent rename because two names for one sequence is exactly the
# kind of thing that later looks like a missing measurement.
for S in Beauty Bosphorus HoneyBee Jockey ReadySetGo:ReadySteadyGo ShakeNDry YachtRide; do
    SRC="${S%%:*}"; DST="${S##*:}"
    F="${SRC}_1920x1080_120fps_420_8bit_YUV"
    G="${DST}_1920x1080_120fps_420_8bit_YUV"
    [ -f "$G.yuv" ] && { echo "[skip] $G.yuv var"; continue; }
    echo "[get ] $F"
    curl -L -C - --limit-rate 40M --retry 5 --retry-delay 10 \
         -o "${F}_RAW.7z" "https://ultravideo.fi/video/${F}_RAW.7z" || { echo "[FAIL] $F"; continue; }
    echo "[unz ] $F"
    ~/FLEX-UF/.venv/bin/python -c "
import py7zr,sys; py7zr.SevenZipFile('${F}_RAW.7z','r').extractall('.')" && rm -f "${F}_RAW.7z"
    [ "$F" != "$G" ] && [ -f "$F.yuv" ] && mv "$F.yuv" "$G.yuv"
    ls -la "$G.yuv" 2>/dev/null || echo "[WARN] $G.yuv cikmadi"
done
echo "[done] $(ls -1 *.yuv 2>/dev/null | wc -l)/7 sekans, $(du -sh . | cut -f1)"
