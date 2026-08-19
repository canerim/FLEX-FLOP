# Rebuild every artefact this project publishes, in dependency order.
#
# Nothing here measures anything -- measurement scripts need a GPU, take
# minutes to hours, and write into results/. This turns results/ into the
# paper, the report and the figures, and it fails if a number in the prose
# has drifted from the file it was read out of.
#
#   make paper     tables -> PDF -> claim check
#   make figures   every figure the paper and the docs reference
#   make check     the claim check alone
#   make test      the test suite
#   make all       all of it

PY := ./.venv/bin/python

.PHONY: all paper figures check test tables report clean-figs

all: figures paper report test

tables:
	$(PY) scripts/make_paper_tables.py

paper: tables
	$(PY) scripts/paper_figures.py
	$(PY) scripts/build_pdf.py
	$(PY) scripts/build_router_pdf.py
	$(PY) scripts/check_paper.py

check:
	$(PY) scripts/check_paper.py

report:
	$(PY) scripts/make_report.py

# Figures that read only results/ and docs/ -- no GPU, no checkpoints.
figures:
	$(PY) scripts/system_figs.py
	$(PY) scripts/baseline_fig.py
	$(PY) scripts/pipeline_detail.py
	$(PY) scripts/hybrid_figure.py
	$(PY) scripts/raterank_figure.py
	$(PY) scripts/blend_figure.py
	$(PY) scripts/tradeoff_figure.py   # budget_band.png
	$(PY) scripts/tradeoff_figure.py results/signalled_BEST_grid.json \
		results/saturation_BEST_ctc53.json docs/figures/budget_band_BEST.png \
		results/band_collapse_BEST.json
	$(PY) scripts/saturation_figure.py
	$(PY) scripts/perclass_figure.py
	$(PY) scripts/contamination_figure.py
	$(PY) scripts/transfer_figure.py
	$(PY) scripts/theory_figure.py
	$(PY) scripts/adapter_figure.py
	$(PY) scripts/rd_figure.py
	$(PY) scripts/seam_vs_qp_figure.py
	$(PY) scripts/paper_metrics.py

test:
	$(PY) -m pytest tests/ -q
