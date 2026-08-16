#!/bin/bash
# Real routers across the frontier. lam is the operating point; each value gives
# one deployable router, and the sweep is what turns a point into a curve.
set -u
cd "$HOME/FLEX-UF"
CK=${1:?usage: sweep_router_heads.sh <ckpt> [gpu] [steps]}
GPU=${2:-2}; ST=${3:-1200}
OUT="$(dirname "$CK")/routers"; mkdir -p "$OUT"
for lam in 1e-4 3e-4 1e-3 3e-3; do
  f="$OUT/head_lam${lam}.pth"
  [ -s "$f" ] && { echo "  lam=$lam var, atlaniyor"; continue; }
  CUDA_VISIBLE_DEVICES=$GPU ./.venv/bin/python scripts/train_router_head.py \
    --ckpt "$CK" --lam "$lam" --steps "$ST" --device cuda:0 --out "$f" \
    2>&1 | grep -E "regret|wrote" | tail -4
done
