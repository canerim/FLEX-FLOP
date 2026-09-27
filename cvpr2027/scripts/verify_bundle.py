"""Check the published paper bundle without GPU or training checkpoints."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/refresh20260927"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-pdfs",action="store_true",help="Also inspect compiled PDFs with Poppler tools")
    args=ap.parse_args()
    checked=[]
    for name, expected in json.loads((DATA/"bundle_manifest.json").read_text()).items():
        assert digest(DATA/name)==expected,f"Changed bundled source: {name}"
        checked.append(str((DATA/name).relative_to(ROOT)))
    d=json.loads((DATA/"analysis.json").read_text())
    with (DATA/"samples.csv").open() as f:samples=list(csv.DictReader(f))
    for row in d["summary"]:
        group=[r for r in samples if float(r["budget"])==row["budget"] and r["rule"]==row["rule"]]
        assert len(group)==row["n"]
        assert abs(sum(float(r["saving"]) for r in group)/len(group)-row["mean"])<1e-10
        assert sum(float(r["rgb_loss"])>row["budget"]+1e-4 for r in group)==row["rgb_over"]
        assert sum(float(r["yuv_loss"])>row["budget"]+1e-4 for r in group)==row["yuv_over"]
    figure_count=0
    min_font=float("inf")
    for folder in ["refresh20260927","extended20260927"]:
        path=ROOT/"figs"/folder
        for name, expected in json.loads((path/"artifact_manifest.json").read_text()).items():
            assert digest(path/name)==expected,f"Changed figure: {folder}/{name}"
            checked.append(str((path/name).relative_to(ROOT)))
        for record in json.loads((path/"layout_audit.json").read_text()):
            assert not record["outside_canvas"],record["figure"]
            assert abs(record["width_mm"]-183)<1e-6
            assert record["height_mm"]<=170
            assert "<text" in (path/(record["figure"]+".svg")).read_text()
            min_font=min(min_font,*(t["size_pt"] for t in record["text"]))
            figure_count+=1
    assert figure_count==9
    reports={}
    if args.check_pdfs:
        for name in ["main","supplement"]:
            content=subprocess.check_output(["pdftotext",str(ROOT/(name+".pdf")),"-"],text=True)
            pages=[page for page in content.split("\f") if page.strip()]
            assert all(len(page.split())>30 for page in pages),f"Near-blank PDF page in {name}"
            assert "??" not in content,f"Unresolved reference in {name}"
            log=(ROOT/(name+".log")).read_text()
            assert not re.search(r"undefined (?:references|citations)|Citation .*undefined|^!|Overfull",log,re.M)
            fonts=subprocess.check_output(["pdffonts",str(ROOT/(name+".pdf"))],text=True).splitlines()
            pos=fonts[0].index("emb")
            assert all(line[pos:pos+3]=="yes" for line in fonts[2:] if line.strip()),f"Unembedded font in {name}"
            assert not any("Type 3" in line for line in fonts[2:]),f"Type 3 font in {name}"
            reports[name]=dict(pages=len(pages),all_fonts_embedded=True,unresolved_references=0,near_blank_pages=0,
                               sha256=digest(ROOT/(name+".pdf")))
    result=dict(bundled_files_checked=len(checked),summary_rows_checked=len(d["summary"]),
                paired_samples=len(samples),vector_figure_sets=figure_count,min_figure_font_pt=min_font,
                text_outside_canvas=0,compiled_documents=reports,
                scope="Integrity and internal consistency of the supplied publication bundle, not historical-run reproduction or a new codec evaluation")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
