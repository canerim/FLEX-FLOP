#!/bin/bash
# HEVC Classes B, C and D -- the 13 CTC sequences missing from this machine.
#
# Without them the test set is UVG + MCL-JCV + HEVC-E, which is 40 of the 53
# sequences DCVC's own test_cfg names, and no result measured on it can honestly
# be called "on CTC". Class B especially matters: five 1080p sequences with
# content unlike UVG's, and Class C and D add the low-resolution end where a
# 256 px tile covers a much larger fraction of the frame.
#
# Source: huggingface.co/datasets/sjcalan/HEVC, whose filenames match DCVC's
# test_cfg exactly. Sizes were checked against W*H*1.5*frames before downloading;
# a few files carry one extra frame, which is harmless -- nothing here reads past
# what the config asks for.
#
# Rate-limited and sequential: this is a shared uplink on a shared machine.
set -u
BASE="https://huggingface.co/datasets/sjcalan/HEVC/resolve/main"
ROOT=/data10/shareddata/test_sequences/YUV

get () {   # get <class-dir-on-hf> <local-class-dir> <file>
    local dst="$ROOT/$2"
    mkdir -p "$dst"
    if [ -f "$dst/$3" ]; then echo "[skip] $2/$3"; return; fi
    echo "[get ] $2/$3"
    curl -L -C - --limit-rate 40M --retry 5 --retry-delay 10 --fail \
         -o "$dst/$3.part" "$BASE/$1/$3" \
      && mv "$dst/$3.part" "$dst/$3" \
      || { echo "[FAIL] $2/$3"; rm -f "$dst/$3.part"; }
}

for F in BQTerrace_1920x1080_60 BasketballDrive_1920x1080_50 \
         Cactus_1920x1080_50 Kimono1_1920x1080_24 ParkScene_1920x1080_24; do
    get ClassB HEVC_B "$F.yuv"
done
for F in BQMall_832x480_60 BasketballDrill_832x480_50 \
         PartyScene_832x480_50 RaceHorses_832x480_30; do
    get ClassC HEVC_C "$F.yuv"
done
for F in BasketballPass_416x240_50 BlowingBubbles_416x240_50 \
         BQSquare_416x240_60 RaceHorses_416x240_30; do
    get ClassD HEVC_D "$F.yuv"
done
echo "[done] $(du -sh $ROOT | cut -f1) total in $ROOT"
