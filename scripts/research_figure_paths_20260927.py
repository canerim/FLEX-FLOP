"""Resolve identical research plots in the parent repo and portable paper bundle."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BUNDLED=(ROOT/'data/refresh20260927/analysis.json').exists()

def paths(name):
    data=ROOT/('data/research20260927' if BUNDLED else 'docs/research/2026-09-27-six-hour')/name
    out=ROOT/'figs/research20260927'/name if BUNDLED else data
    out.mkdir(parents=True,exist_ok=True)
    return data,out
