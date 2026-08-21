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
  TX=$(echo "$OUT" | grep -oE 'check_tex: [A-Za-z0-9 ()]+' | head -1 | sed 's/check_tex: //')
  # The three checks added on 21 August read the artefact rather than the
  # source, which is where the last few defects were only visible.
  LY=$(echo "$OUT" | grep -oE 'check_layout: [A-Za-z0-9 ()]+' | head -1 | sed 's/check_layout: //')
  FR=$(echo "$OUT" | grep -oE 'check_figs_fresh: [A-Za-z0-9 ()]+' | head -1 | sed 's/check_figs_fresh: //')
  RN=$(echo "$OUT" | grep -oE 'check_render: [A-Za-z0-9 ()]+' | head -1 | sed 's/check_render: //')
  FP=$(echo "$OUT" | grep -oE 'check_fig_prose: [A-Za-z0-9 ()]+' | head -1 | sed 's/check_fig_prose: //')
  FO=$(echo "$OUT" | grep -oE 'check_fig_overlap: [A-Za-z0-9 ()/,]+' | head -1 | sed 's/check_fig_overlap: //')
  CT=$(echo "$OUT" | grep -oE 'check_cites: [A-Za-z0-9 ()]+' | head -1 | sed 's/check_cites: //')
  NU=$(echo "$OUT" | grep -oE 'check_numbers: [A-Za-z0-9 ()]+' | head -1 | sed 's/check_numbers: //')
  # A page count that comes back empty is the PDF being rewritten as we read
  # it, not a missing paper. Saying "building" costs one word and stops a
  # blank field looking like a broken artefact.
  MAIN=$(pdfinfo paper/FLEX-UF.pdf 2>/dev/null | awk '/Pages/{print $2}')
  SUPP=$(pdfinfo paper/FLEX-UF-supp.pdf 2>/dev/null | awk '/Pages/{print $2}')
  MAIN=${MAIN:-building}
  SUPP=${SUPP:-building}
  OURS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
         | while read -r p; do ps -o user= -p "$p" 2>/dev/null; done | grep -c can_karsal)
  OTHER=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
          | while read -r p; do ps -o user= -p "$p" 2>/dev/null; done | grep -vc can_karsal)
  # runs is a symlink, so find needs -L or it never descends. stat over a glob
  # is simpler and does not care.
  NEWEST=$(stat -c %Y runs/*/ckpt_eval.pth.tar runs/*/ckpt_step.pth.tar \
           2>/dev/null | sort -rn | head -1)
  AGE=$([ -n "${NEWEST:-}" ] && echo "$(( ($(date +%s) - NEWEST) / 60 ))m" || echo "-")
  EP=$(for t in RECIPE512 BEST FINE12 SCRATCH105; do
         f=$(ls -t runs/$t/*.log 2>/dev/null | head -1)
         [ -n "$f" ] && tail -1 "$f" 2>/dev/null | grep -o '"epoch": [0-9]*' | head -1 \
           | grep -o '[0-9]*$' | sed "s/^/${t:0:4}/"
       done | tr '\n' ' ')
  SC=$(tail -1 runs/SCRATCH105/train.log 2>/dev/null \
       | ./.venv/bin/python -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
except Exception:
    print('-'); raise SystemExit
print(f\"ep{d['epoch']} {100*d['seen']/d['total']:.1f}% spread {d['spread_dB']:+.2f}dB {d['sec']:.0f}s/200\")
" 2>/dev/null)
  SCALIVE=$(pgrep -fc "save_dir.*SCRATCH105" 2>/dev/null || echo 0)
  echo "$TS  fix $FIX | claims $CHK twins $TW tex $TX prose $PR layout $LY figs $FR render $RN figprose $FP overlap "$FO" cites $CT numbers $NU | paper ${MAIN}p supp ${SUPP}p | gpu ours=$OURS others=$OTHER | newest ckpt $AGE | ep $EP | scratch ${SC:--} pids=$SCALIVE"
  sleep "$INTERVAL"
done
