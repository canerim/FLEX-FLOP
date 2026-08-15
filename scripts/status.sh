#!/bin/bash
# One-screen status of the three runs. Written because the thing that actually
# needs watching is not the loss — it is whether the exits are DIFFERENTIATING.
#
# FLEX-FLOP's from-scratch attempt failed in a specific, recognisable way: every
# exit converged to about the same quality (~26 dB) with no spread between them,
# so the ladder existed on paper but there was nothing to route. `spread_dB`
# below is exactly that diagnostic — the gap between the deepest and shallowest
# exit's PSNR. It should grow and stay clearly positive. If it sits near zero
# while the loss falls, the run is failing even though it looks healthy.

ROOT="$HOME/FLEX-UF"
RUNS="$ROOT/runs"

printf '\n=== GPU (mine only; others left alone) ===\n'
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader \
  | awk -F', ' '{printf "  GPU %s  util %-5s  mem %s / %s\n",$1,$2,$3,$4}'

printf '\n=== runs ===\n'
for d in "$RUNS"/*/; do
    [ -d "$d" ] || continue
    tag=$(basename "$d")
    log="$d/train_log.jsonl"
    printf '\n--- %s ---\n' "$tag"
    if [ ! -f "$log" ]; then
        printf '  no log yet\n'
        [ -f "$d/stdout.log" ] && tail -3 "$d/stdout.log" | sed 's/^/  /'
        continue
    fi
    python3 - "$log" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
if not rows:
    print("  log empty"); raise SystemExit
r = rows[-1]
print(f"  epoch {r['epoch']:>3}  step {r['step']:>6}  "
      f"{100*r['seen']/max(r['total'],1):5.1f}% of epoch  lr {r['lr']:.0e}  patch {r['patch']}")
print(f"  loss {r['loss']:10.4f}   bpp {r['bpp']:.4f}   grad {r.get('grad_norm',0):.4f}"
      f"   skipped {r.get('skipped',0)}")
print(f"  PSNR/exit  {r['psnr_per_exit']}")
sp = r['spread_dB']
flag = "OK" if sp > 0.5 else ("!! COLLAPSE RISK" if sp < 0.15 else "watch")
print(f"  spread     {sp:+.3f} dB   <- {flag}")
if len(rows) > 8:
    first = rows[max(0,len(rows)-40)]
    print(f"  trend      loss {first['loss']:.4f} -> {r['loss']:.4f}   "
          f"spread {first['spread_dB']:+.3f} -> {sp:+.3f}")
PY
done
printf '\n'
