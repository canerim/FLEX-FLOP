#!/bin/bash
# One status line every INTERVAL seconds, for the terminal.
#
# Reports only what changes decisions: how far the fix list has got, whether the
# paper still builds and still agrees with its data, how long it is, which cards
# are ours and which are other people's, and whether training is still
# advancing. Everything is read from disk; nothing here touches a GPU.
set -u
cd "$HOME/FLEX-UF"
INTERVAL=${1:-300}
while true; do
  TS=$(date '+%H:%M')
  FIX=$(grep -m1 '^Status:' REVIEW_FIXES.txt 2>/dev/null | sed 's/Status: //')
  CHK=$(./.venv/bin/python scripts/check_paper.py 2>/dev/null | tail -1 | sed 's/^ *//;s/ prose claims match the data//')
  MAIN=$(pdfinfo paper/FLEX-UF.pdf 2>/dev/null | awk '/Pages/{print $2}')
  SUPP=$(pdfinfo paper/FLEX-UF-supp.pdf 2>/dev/null | awk '/Pages/{print $2}')
  OURS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
         | while read -r p; do ps -o user= -p "$p" 2>/dev/null; done | grep -c can_karsal)
  OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
          | while read -r p; do ps -o user= -p "$p" 2>/dev/null; done | grep -vc can_karsal)
  NEW=$(find runs -name 'ckpt_*.pth.tar' -newermt '-40 minutes' 2>/dev/null | wc -l)
  EP=$(for t in RECIPE512 BEST COUPLED512 FINE12; do
         f=$(ls -t runs/$t/*.log 2>/dev/null | head -1)
         [ -n "$f" ] && tail -1 "$f" 2>/dev/null | grep -o '"epoch": [0-9]*' | head -1 \
           | grep -o '[0-9]*$' | sed "s/^/${t:0:4}/"
       done | tr '\n' ' ')
  echo "$TS  fix $FIX | check $CHK | paper ${MAIN}p supp ${SUPP}p | gpu ours=$OURS others=$OTHER | new ckpts 40m: $NEW | ep $EP"
  sleep "$INTERVAL"
done
