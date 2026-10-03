"""One file for the paper's routing tables: every head's agreement, every
on-budget B/D set, and the per-frame comparison. Re-run after any router job."""
import json, glob, os, statistics as st
R = "flexplus/results/"
out = {"checkpoint": "runs/RECIPE512/ckpt_PIN_e15.pth.tar", "frames": "53 CTC intra, 1080p, YUV 6:1:1", "budget_db": 0.1}
heads = {}
for f in sorted(glob.glob("results/router_e15/*.json")):
    d = json.load(open(f)); e = d.get("eval", {})
    heads[d.get("label") or d.get("inputs")] = {"agree": e.get("agree"), "stderr": e.get("stderr_across_frames"),
                                                "inputs": d.get("inputs"), "context": d.get("context", 0)}
out["decoder_side_agreement"] = heads
def summ(f):
    d = json.load(open(f)); rs = d["router_compute_share_pct"]
    S = {(r["qp"], r["top_m"]): r["saving_pct_vs_release"] for r in d["rows"] if r["kind"] == "search"}
    D = {r["qp"]: r for r in d["rows"] if r["kind"] == "binary_refine"}
    X = {r["qp"]: r for r in d["rows"] if r["kind"] == "binary_refine_ctx"}
    qs = sorted(D)
    return {"router_share_pct": rs, "B": st.mean(S[(q, 1)] - rs for q in qs), "A": st.mean(S[(q, 4)] for q in qs),
            "D2": st.mean(D[q]["saving_pct_vs_release"] - rs for q in qs),
            "D2_bits_kt": st.mean(X[q]["bits_kt_iid"] for q in qs) if X else None}
mean_rule = {}
for tag, f in (("v3_seed0", "map_coding_e15.json"), ("seed1", "map_coding_e15_seed1.json"), ("seed2", "map_coding_e15_seed2.json"),
               ("seed3", "map_coding_e15_seed3.json"), ("cap_small_stem", "map_coding_e15_cap_small_stem.json"),
               ("cap_small_stem_s1", "map_coding_e15_cap_small_stem_s1.json"), ("cap_tiny_stem", "map_coding_e15_cap_tiny_stem.json"),
               ("inputs_stem_qp", "map_coding_e15_inputs_stem_qp.json"),
               ("cap_small_6k", "map_coding_e15_cap_small_6k.json"), ("cap_base_6k", "map_coding_e15_cap_base_6k.json"),
               ("cap_base_6k_s1", "map_coding_e15_cap_base_6k_s1.json"), ("base_12k", "map_coding_e15_base_12k.json"),
               ("ce_soft_6k", "map_coding_e15_ce_soft_6k.json"), ("ce_bits_6k", "map_coding_e15_ce_bits_6k.json"),
               ("bits_only_6k", "map_coding_e15_bits_only_6k.json"), ("crop1024_6k", "map_coding_e15_crop1024_6k.json"),
               ("ce_soft_6k_s1", "map_coding_e15_ce_soft_6k_s1.json"), ("ce_soft_tau0.05_6k", "map_coding_e15_ce_soft_tau0.05_6k.json"),
               ("ce_soft_tau0.3_6k", "map_coding_e15_ce_soft_tau0.3_6k.json"), ("ce_soft_12k", "map_coding_e15_ce_soft_12k.json")):
    if os.path.exists(R + f): mean_rule[tag] = summ(R + f)
# the parameter-free predictor (Sec 5.6 rank-1 bit surrogate), two bit sources
rr = {}
for tag, f in (("raterank_noisy_bits", "hybrid_e15_raterank_b01.json"), ("raterank_decoder_bits", "hybrid_e15_raterank_decbits_b01.json"),
               ("raterank_decoder_bits_TRAINFIT", "hybrid_e15_raterank_trainfit_b01.json")):
    if os.path.exists(R + f):
        d = json.load(open(R + f)); r0 = [x for x in d["rows"] if x["rho"] == 0.0 and x.get("budget_reachable")]
        r1 = [x for x in d["rows"] if x["rho"] == 1.0 and x.get("budget_reachable")]
        rr[tag] = {"B": st.mean(x["saving_pct_vs_release"] for x in r0), "A": st.mean(x["saving_pct_vs_release"] for x in r1),
                   "params": 0, "router_share_pct": 0.0,
                   "footing": ("alpha,c,lphi fitted on 300 openimages training crops, tested on CTC -- the head's own footing"
                               if "TRAINFIT" in tag else
                               "alpha,c fitted leave-one-sequence-out on the CTC oracle tables; the trained heads never saw CTC")}
mean_rule["_parameter_free"] = rr
out["mean_budget_rule"] = mean_rule
d = json.load(open(R + "map_coding_e15_perframe3.json")); rows = d["rows"]
def ro(k, **kw): return [x for x in rows if x["kind"] == k and all(x.get(a) == b for a, b in kw.items())]
def m(rs, key): return st.mean(x[key] for x in rs)
def ov(rs): return sum(x["frames_over_budget"] for x in rs)
pf = {"B_signalled_beta": {"saving": m(ro("B_perframe_beta"), "saving_pct_vs_release"), "bits": 8.0, "frames_over": ov(ro("B_perframe_beta"))},
      "D2": {"saving": m(ro("D2_perframe"), "saving_pct_vs_release"), "bits": m(ro("D2_perframe"), "bits_kt_per_frame"), "frames_over": ov(ro("D2_perframe"))},
      "D2_fallback_A": {"saving": m(ro("D2_fallbackA_perframe"), "saving_pct_vs_release"), "bits": m(ro("D2_fallbackA_perframe"), "bits_per_frame"), "frames_over": ov(ro("D2_fallbackA_perframe"))},
      "D3": {"saving": m(ro("D3_perframe"), "saving_pct_vs_release"), "bits": m(ro("D3_perframe"), "bits_kt_per_frame"), "frames_over": ov(ro("D3_perframe"))},
      "A_kt4": {"saving": m(ro("A_perframe"), "saving_pct_vs_release"), "bits": m(ro("A_perframe_ktbits"), "bits_kt4_per_frame"), "bits_flat": 66.6, "frames_over": ov(ro("A_perframe"))}}
for rho in (0.1, 0.2, 0.35):
    C = ro("C_perframe", rho=rho); pf[f"C_rho{rho}"] = {"saving": m(C, "saving_pct_vs_release"), "bits": m(C, "bits_per_frame"), "frames_over": ov(C)}
# decoder-side hedge: on unsure tiles take the deeper of the head's two best; beta re-bisected to budget
hedge = {}
for f in sorted(glob.glob(R + "map_coding_e15_hedge*.json")):
    d = json.load(open(f)); rows = [x for x in d["rows"] if x["kind"] == "B"]
    if rows:
        hedge[os.path.basename(f)[len("map_coding_e15_"):-5]] = {
            "B": st.mean(x["saving_pct_vs_release"] for x in rows), "all_feasible": all(x["feasible"] for x in rows),
            "per_qp": {str(x["qp"]): x["saving_pct_vs_release"] for x in rows}}
if hedge:
    out["decoder_side_hedge"] = hedge
if os.path.exists(R + "ensemble_eval.json"):
    out["ensemble_two_seeds_agreement"] = json.load(open(R + "ensemble_eval.json"))
out["per_frame_budget_rule"] = pf
# the same per-frame table with the parameter-free predictor in the head's place
pfr = R + "map_coding_e15_raterank_perframe.json"
if os.path.exists(pfr):
    d2 = json.load(open(pfr)); rows2 = d2["rows"]
    def ro2(k, **kw): return [x for x in rows2 if x["kind"] == k and all(x.get(a) == b for a, b in kw.items())]
    pf2 = {"B_signalled_beta": {"saving": m(ro2("B_perframe_beta"), "saving_pct_vs_release"), "bits": 8.0, "frames_over": ov(ro2("B_perframe_beta"))},
           "D3": {"saving": m(ro2("D3_perframe"), "saving_pct_vs_release"), "bits": m(ro2("D3_perframe"), "bits_kt_per_frame"), "frames_over": ov(ro2("D3_perframe"))},
           "A_kt4": {"saving": m(ro2("A_perframe"), "saving_pct_vs_release"), "bits": m(ro2("A_perframe_ktbits"), "bits_kt4_per_frame"), "frames_over": ov(ro2("A_perframe"))}}
    for rho in (0.1, 0.2, 0.35):
        C = ro2("C_perframe", rho=rho); pf2[f"C_rho{rho}"] = {"saving": m(C, "saving_pct_vs_release"), "bits": m(C, "bits_per_frame"), "frames_over": ov(C)}
    out["per_frame_budget_rule_raterank_trainfit"] = pf2
out["findings"] = [
 "stem alone equals all inputs (0.766 vs 0.765); latent and scales add nothing",
 "regret weight irrelevant: alpha 0/1/3 -> 0.7654/0.7650/0.7646",
 "smaller head better on two seeds: 47k params 0.776/0.782 > 148k 0.766 > large(1.2M) 0.752 > ctx3(430k) 0.746; on budget all within the seed band; router share 0.163 -> 0.067%",
 "spatial context helps neither the router (ctx3 -0.02) nor the flag coder (KT-ctx 101-105% of iid)",
 "D2 beats C under the MEAN budget rule (28.65 @ 20 bits vs C 27.5 at equal bits) but collapses to B under the PER-FRAME rule (23/265 frames unreachable)",
 "under the per-frame rule: A adaptively coded (KT-4) costs 41.5 bits, not 66.6, and is the best point; D3 is the only useful intermediate (28.72 @ 36.8); C is dominated",
 "D is 3.4x less seed-sensitive than B (0.07 vs 0.24)",
 "regression heads (log-MSE, relative log-MSE) collapse to one exit (0.34 / 0.11 agreement): the decision lives in differences ~1e-5 that a 10% regression error swamps; CE learns the decision directly",
 "two-seed ensemble of the paper-size head: +0.1 agreement, within noise",
 "decoder-side uncertainty hedge (deeper of top-2 when max prob < 0.5), beta re-bisected: B 35.01/30.76/25.25/22.76/19.32 vs 35.02/30.82/25.34/22.77/19.33 -- nothing",
 "the parameter-free rank-1 bit surrogate, computed from DECODER-side symbols round(y_res)=y_q, delivers B=28.08% (mean rule) -- above every trained head (26.3-26.7) -- with zero parameters; its two scalars per qp are fitted LOSO on CTC, which the heads never saw",
 "6000 steps lift the 47k head to 0.789 (from 0.776/0.782 at 3000); the base head at 6000 steps decides whether the capacity ordering was optimisation",
 "6000 steps: base head 0.800 > small head 0.789 -> the 3000-step capacity ordering was optimisation, not size; on budget all heads sit in the seed band regardless",
 "per-frame rule, parameter-free predictor: B_pf 27.51 (head 27.42) -- a table replaces the router for configuration B; for D3 the head's sharper shortlist still helps (28.72 @ 36.8 vs 28.29 @ 44.7)",
 "on the head's own footing (fitted on training crops, tested on CTC) the parameter-free rule gives B=27.51%: still 0.85 above the trained head (26.66); the LOSO figure (28.08) carried 0.57 of test-set advantage; the shortfall is at q63 (19.52 vs 21.56)"]
json.dump(out, open(R + "routing_summary_e15.json", "w"), indent=2)
print(f"routing_summary_e15.json: {len(heads)} baslik, {len(mean_rule)} on-budget set, {len(pf)} per-frame satir")
