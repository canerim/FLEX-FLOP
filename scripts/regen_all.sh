#!/bin/bash
# Everything derived, in dependency order, after a repin.
#
# Step 10 of repin.sh, pulled out so it can be run on its own once the
# measurements are in place -- and extended with the figures that were added
# after repin.sh was written. Nothing here needs a card: every script reads a
# results file.
#
# Order matters. paper_metrics writes results/bdrate.json from the anchor and
# the rate file; make_paper_tables writes every macro and .tex table from the
# results; the figure scripts read both. Running them out of order leaves a
# figure drawn from the previous checkpoint beside a table drawn from this one,
# which is the failure check_figs_fresh exists to catch.
set -uo pipefail
cd "$HOME/FLEX-UF"
PY=./.venv/bin/python
LOG=results/regen.log
echo "=== regenerate everything, $(date '+%F %T') ===" | tee -a "$LOG"

run () {
  echo "  $(date '+%H:%M:%S')  $*" | tee -a "$LOG"
  "$@" >>"$LOG" 2>&1 || echo "    FAILED rc=$?" | tee -a "$LOG"
}

# 1. derived results
run $PY scripts/paper_metrics.py
run $PY scripts/make_paper_tables.py

# 2. figures that read results files.
#
# This list was short, and short meant that after a repin most figures kept
# whatever data they were last drawn from while the tables moved. Everything
# here reads a results file and needs no card; the ones that need a
# checkpoint are in repin_stage3.sh.
run $PY scripts/motivation_figs.py
run $PY scripts/perclass_figure.py
run $PY scripts/contamination_figure.py
run $PY scripts/theory_figure.py
run $PY scripts/transfer_figure.py
run $PY scripts/adapter_figure.py
run $PY scripts/power_figure.py
run $PY scripts/rd_figure.py
run $PY scripts/exit_vs_rate.py
run $PY scripts/saturation_figure.py
run $PY scripts/system_figs.py
run $PY scripts/build_router_pdf.py
run $PY scripts/nature_plots.py
run $PY scripts/spread_figs.py
run $PY scripts/tradeoff_figure.py
run $PY scripts/tradeoff_figure.py results/signalled_BEST_grid.json \
        results/saturation_BEST_ctc53.json \
        docs/figures/budget_band_BEST.png results/band_collapse_BEST.json
run $PY scripts/rd_vs_uf_figure.py
run $PY scripts/rd_yuv_figure.py
run $PY scripts/hybrid_figure.py
run $PY scripts/raterank_figure.py
run $PY scripts/bd_figure.py --layout wide   --out docs/figures/bdrate_wide.png
run $PY scripts/bd_figure.py --layout column --out docs/figures/bdrate.png
run $PY scripts/training_plateau.py --layout wide \
        --out docs/figures/training_plateau_wide.png
run $PY scripts/training_plateau.py --layout column \
        --out docs/figures/training_plateau.png
run $PY scripts/paper_figures.py

# 3. the documents, then everything that checks them
run $PY scripts/build_pdf.py
run $PY scripts/build_supp_pdf.py
$PY scripts/check_paper.py 2>&1 | tee -a "$LOG" | grep -E "claims match|check_|FAILING"
echo "=== regenerate done $(date '+%F %T') ===" | tee -a "$LOG"
