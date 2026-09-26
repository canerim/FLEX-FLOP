# CVPR author kit, latest main (see kit_reference/KIT_VERSION), taken verbatim from github.com/cvpr-org/author-kit
all: main.pdf
main.pdf: main.tex preamble.tex cvpr.sty main.bib sec/*.tex fig/*.tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
rebuttal.pdf: rebuttal.tex cvpr.sty
	latexmk -pdf -interaction=nonstopmode -halt-on-error rebuttal.tex
clean:
	latexmk -C main.tex rebuttal.tex; rm -f *.bbl *.blg *.brf *.fdb_latexmk *.fls
.PHONY: all clean
