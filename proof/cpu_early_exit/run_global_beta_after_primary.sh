#!/usr/bin/env bash
# Sequential CPU-only transfer of the exploratory calibration-only global rule.
set -euo pipefail

root=/home/can_karsal/FLEX-PLUS
python_bin=/data10/shareddata/can_karsal/dcvcuf_depth_20260927/venv/bin/python
capture_script=/tmp/flexplus-div2k-beta/proof/cpu_early_exit/div2k_beta_transfer.py
manifest="$root/proof/cpu_early_exit/results/div2k_beta/manifest.json"
policy="$root/proof/cpu_early_exit/results/div2k_beta/locked_global_policy.json"
val_dir=/tmp/flexplus-div2k-beta-global-validation
kodak_dir=/tmp/flexplus-div2k-beta-global-kodak
proof="$root/proof/early_exit_vs_released"
upstream=/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/DCVC
extension=/tmp/flexplus-proof-clean/proof/early_exit_vs_released/.local/entropy

while tmux has-session -t reglic_div2k_beta_pipeline 2>/dev/null; do
    sleep 30
done

echo "Exploratory global policy SHA256: $(sha256sum "$policy")"
echo 'Evaluating disjoint DIV2K validation cohort'
"$python_bin" "$capture_script" capture --split validation \
    --source-dir "$root/data/DIV2K_valid_HR" --manifest "$manifest" \
    --out-dir "$val_dir" --policy "$policy" --upstream "$upstream" \
    --extension "$extension" \
    --release "$proof/artifacts/released_cvpr2026_image.pth.tar" \
    --e15 "$proof/artifacts/e15_epoch15.pth.tar" \
    --router "$proof/artifacts/router_stem_qp.pth"

echo 'Transferring exploratory policy to Kodak24 x QP5'
"$python_bin" "$root/proof/cpu_early_exit/kodak_locked_beta_transfer.py" \
    --scan-dir "$root/proof/cpu_early_exit/results/kodak24_qp5" \
    --out-dir "$kodak_dir" --policy "$policy" \
    --upstream "$upstream" --extension "$extension" \
    --release "$proof/artifacts/released_cvpr2026_image.pth.tar" \
    --e15 "$proof/artifacts/e15_epoch15.pth.tar" \
    --source-dir "$root/data/kodak" \
    --pilot-stream-dir "$proof/results/bitstream_kodak3x3"

echo 'Exploratory global-policy validation and Kodak transfer complete'
