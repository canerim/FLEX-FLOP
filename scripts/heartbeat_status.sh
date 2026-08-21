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
  # grep for the score line, not tail -1: check_paper now prints the twin and
  # prose audits after it, so the last line is one of those.
  OUT=$(./.venv/bin/python scripts/check_paper.py 2>/dev/null)
  CHK=$(echo "$OUT" | grep -oE '[0-9]+/[0-9]+ prose claims' | head -1 | sed 's/ prose claims//')
  TW=$(echo "$OUT" | grep -oE 'check_twins: [A-Za-z0-9 ()]+' | head -1 | sed 's/check_twins: //')
  PR=$(echo "$OUT" | grep -oE 'prose_audit: [A-Za-z0-9 ()]+' | head -1 | sed 's/prose_audit: //')
  MAIN=$(pdfinfo paper/FLEX-UF.pdf 2>/dev/null | awk '/Pages/{print $2}')
  SUPP=$(pdfinfo paper/FLEX-UF-supp.pdf 2>/dev/null | awk '/Pages/{print $2}')
  OURS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
         | while read -r p; do ps -o user= -p "$p" 2>/dev/null; done | grep -c can_karsal)
  OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
          | while read -r p; do ps -o user= -p "$p" 2>/dev/null; done | grep -vc can_karsal)
  # runs is a symlink, so find needs -L or it never descends. stat over a glob
  # is simpler and does not care.
  NEWEST=$(stat -c %Y runs/*/ckpt_eval.pth.tar runs/*/ckpt_step.pth.tar \
           2>/dev/null | sort -rn | head -1)
  AGE=$([ -n "${NEWEST:-}" ] && echo "$(( ($(date +%s) - NEWEST) / 60 ))m" || echo "-")
  EP=$(for t in RECIPE512 BEST COUPLED512 FINE12; do
         f=$(ls -t runs/$t/*.log 2>/dev/null | head -1)
         [ -n "$f" ] && tail -1 "$f" 2>/dev/null | grep -o '"epoch": [0-9]*' | head -1 \
           | grep -o '[0-9]*$' | sed "s/^/${t:0:4}/"
       done | tr '\n' ' ')
  echo "$TS  fix $FIX | claims $CHK twins $TW prose $PR | paper ${MAIN}p supp ${SUPP}p | gpu ours=$OURS others=$OTHER | newest ckpt $AGE | ep $EP"
  sleep "$INTERVAL"
done
