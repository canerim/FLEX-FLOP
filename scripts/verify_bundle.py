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
    cap_data=json.loads((DATA/"delivered_frontier_audit.json").read_text())
    for summary in cap_data["summary"]:
        group=[r for r in cap_data["rows"] if r["cap"]==summary["cap"] and r["rule"]==summary["rule"]]
        assert len(group)==summary["n"]==265
        assert len({(r["sequence"],r["qp"]) for r in group})==265
        assert abs(sum(r["saving"] for r in group)/265-summary["mean"])<1e-10
        assert abs(sum(r["loss"] for r in group)/265-summary["mean_loss"])<1e-10
        assert sum(r["fallback"] for r in group)==summary["fallbacks"]
        assert all(r["loss"]<=r["cap"]+cap_data["tolerance_db"] for r in group)
        assert all(r["loss"]==r["saving"]==0 for r in group if r["fallback"])
    influence=json.loads((DATA/"cap_influence_audit.json").read_text())
    assert influence["source_sha256"]==digest(DATA/"delivered_frontier_audit.json")
    assert influence["script_sha256"]==digest(ROOT/"scripts/audit_cap_influence.py")
    for row in influence["rows"]:
        reference_contrast=next(r for r in cap_data["contrasts"]
            if r["cap"]==row["cap_db"] and r["contrast"]=="router_minus_dither")
        assert abs(row["paired_frames"]["mean_points"]-reference_contrast["mean"])<1e-10
        for group, n in [(row["paired_frames"],265),(row["sequence_means"],53)]:
            assert group["n"]==n
            assert sum(group[k] for k in ("router_higher","dither_higher","tied"))==n
        assert len(row["leave_one_sequence_out"])==53
        for left in row["leave_one_sequence_out"]:
            values=[r["saving_difference_points"] for r in row["paired_rows"]
                    if r["sequence"]!=left["removed"]]
            assert len(values)==260
            assert abs(sum(values)/len(values)-left["mean_points"])<1e-10
    reference=json.loads((DATA/"reference_audit.json").read_text())["warmstart_matches_official_release"]
    assert reference["compared"]==398 and not reference["missing"] and not reference["changed"]
    design=json.loads((DATA/"design_audit.json").read_text())
    assert design["router"]["inputs"]=="stem,qp"
    assert "NOT MEASURED" in design["projection"]["label"]
    for row in design["parameters"]["rows"]:
        assert row["decoder_parameters"]==1767168+1038720*row["depth"]
        assert row["total_parameters"]==row["decoder_parameters"]+27947520
    architecture=json.loads((DATA/"depth_architecture_check.json").read_text())
    assert not architecture["gpu_initialized"]
    assert len(architecture["rows"])==6
    for row in architecture["rows"]:
        claimed=next(r for r in design["parameters"]["rows"] if r["depth"]==row["depth"])
        assert all(row[k]==claimed[k] for k in ("decoder_parameters","total_parameters"))
        assert not row["differing_initial_tensors"] and row["all_parameters_trainable"]
    for row in design["projection"]["rows"]:
        f=row["assumed_trunk_time_fraction"];h=row["assumed_extra_overhead_fraction"]
        assert abs(row["relative_time"]-((1-f)+f*row["depth"]/12+h))<1e-12
        assert abs(row["speedup"]*row["relative_time"]-1)<1e-12
    depth=json.loads((DATA/"exit_depth_profile.json").read_text())
    for summary in depth["uniform_summary"]:
        group=[r for r in depth["uniform_rows"] if r["depth"]==summary["depth"] and
               (summary["qp"] is None or r["qp"]==summary["qp"])]
        assert len(group)==summary["n"]
        assert abs(sum(r["padded_rgb_loss_db"] for r in group)/len(group)-summary["mean"])<1e-10
    assert "other_decoders" not in d and "perception" not in d
    figure_count=0
    min_font=float("inf")
    for folder in ["refresh20260927","extended20260927","crossfit20260927","depthmacs20260927"]:
        path=ROOT/"figs"/folder
        for name, expected in json.loads((path/"artifact_manifest.json").read_text()).items():
            assert digest(path/name)==expected,f"Changed figure: {folder}/{name}"
            checked.append(str((path/name).relative_to(ROOT)))
        audit=(json.loads((path/'figure_evidence.json').read_text())['layout_audit'] if folder=='depthmacs20260927'
               else json.loads((path/"layout_audit.json").read_text()))
        for record in audit:
            assert not record["outside_canvas"],record["figure"]
            assert abs(record["width_mm"]-183)<1e-6
            assert record["height_mm"]<=170
            assert "<text" in (path/(record["figure"]+".svg")).read_text()
            min_font=min(min_font,*(t["size_pt"] for t in record["text"]))
            figure_count+=1
    assert figure_count==16
    calibration_source=json.loads((ROOT/'figs/crossfit20260927/source_manifest.json').read_text())
    assert calibration_source['analysis_sha256']==digest(ROOT/'data/crossfit20260927/analysis.json')
    assert calibration_source['plot_script_sha256']==digest(ROOT/'scripts/plot_crossfit_control_20260927.py')
    mac_source=json.loads((ROOT/'figs/depthmacs20260927/figure_evidence.json').read_text())
    assert mac_source['analysis_sha256']==digest(ROOT/'data/depthmacs20260927/analysis.json')
    assert mac_source['script_sha256']==digest(ROOT/'scripts/plot_depth_macs_20260927.py')
    mac=json.loads((ROOT/'data/depthmacs20260927/analysis.json').read_text())
    assert [r['depth'] for r in mac['rows']]==[2,4,6,8,10,12]
    assert not mac['cuda_initialized']
    for row in mac['rows']:
        assert sum(r['macs'] for r in row['layers'])==row['encoder_with_reconstruction_macs']
        assert row['neural_decoder_macs']==row['entropy_neural_macs']+row['synthesis_macs']
        assert row['spatial_prior_calls']==3
        assert row['entropy_neural_macs']==mac['rows'][0]['entropy_neural_macs']
    illustration=json.loads((ROOT/"figs/adaptive20260927/provenance.json").read_text())
    asset=ROOT/illustration["asset"]
    assert digest(asset)==illustration["sha256"],"Changed conceptual illustration"
    checked.append(str(asset.relative_to(ROOT)))
    reports={}
    if args.check_pdfs:
        for name in ["main","supplement"]:
            content=subprocess.check_output(["pdftotext",str(ROOT/(name+".pdf")),"-"],text=True)
            pages=[page for page in content.split("\f") if page.strip()]
            assert all(len(page.split())>30 for page in pages),f"Near-blank PDF page in {name}"
            assert "??" not in content,f"Unresolved reference in {name}"
            assert not any(s in content for s in ["NoGAN", "MS-ILLM"]),f"Out-of-scope decoder in {name}"
            log=(ROOT/(name+".log")).read_text()
            assert not re.search(r"undefined (?:references|citations)|Citation .*undefined|^!|Overfull",log,re.M)
            fonts=subprocess.check_output(["pdffonts",str(ROOT/(name+".pdf"))],text=True).splitlines()
            pos=fonts[0].index("emb")
            assert all(line[pos:pos+3]=="yes" for line in fonts[2:] if line.strip()),f"Unembedded font in {name}"
            assert not any("Type 3" in line for line in fonts[2:]),f"Type 3 font in {name}"
            reference_page=next((i+1 for i,page in enumerate(pages) if "\nReferences\n" in page),None)
            assert reference_page is not None,f"Missing bibliography in {name}"
            if name=="main":
                assert reference_page<=9,f"Main text exceeds eight pages: bibliography starts on {reference_page}"
            reports[name]=dict(pages=len(pages),all_fonts_embedded=True,unresolved_references=0,near_blank_pages=0,
                               reference_start_page=reference_page,
                               sha256=digest(ROOT/(name+".pdf")))
    result=dict(bundled_files_checked=len(checked),summary_rows_checked=len(d["summary"]),
                paired_samples=len(samples),delivered_cap_summary_rows_checked=len(cap_data["summary"]),
                delivered_cap_influence_contrasts_checked=len(influence["rows"]),
                released_warmstart_tensors_matched=reference["compared"],
                uniform_depth_summary_rows_checked=len(depth["uniform_summary"]),
                analytical_scenarios_checked=len(design["projection"]["rows"]),
                independently_instantiated_depths_checked=len(architecture["rows"]),
                vector_figure_sets=figure_count,conceptual_ai_illustrations=1,min_figure_font_pt=min_font,
                text_outside_canvas=0,compiled_documents=reports,
                scope="Integrity and internal consistency of the supplied publication bundle, not historical-run reproduction or a new codec evaluation")
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
