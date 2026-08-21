"""Choose beta on held-out data, then evaluate with it fixed on the test set.

Configuration B bisects beta against the quality budget. A deployed decoder
cannot do that: the budget is a distortion against a source it never receives,
which is the same missing variable that stopped it running the encoder's argmin.
Bisecting on the test set and reporting the result would be reading the answer
off the exam.

So beta is calibrated once, offline, on the 512 Open Images validation frames
the training run never sees, and the resulting per-rate table is then applied
unchanged to the CTC test frames. The gap between that and bisecting on the test
set itself is what tells us whether the table transfers, and both are reported.

    python scripts/beta_calibration.py --stage calibrate
    python scripts/beta_calibration.py --stage evaluate

Three numbers per rate, not two
-------------------------------
The held-out beta and the test-bisected beta land at DIFFERENT delivered
qualities, so their savings are not comparable on their own: a table that
overshoots the budget is buying saving with quality it was not allowed to spend.
The comparison is therefore reported three ways.

  held out      beta from the validation frames, applied unchanged. Its saving
                and its delivered dB are both what a deployment would get.
  test bisected beta bisected on the test set itself, which is the number
                configuration B has been quoted at until now. On budget by
                construction.
  matched       beta bisected on the test set to the dB the held-out beta
                actually DELIVERED. Same quality as the held-out row, so the
                difference in saving is the transfer cost with the quality
                difference taken back out of it.

Why two stages
--------------
The card is shared and every evaluation on it runs behind one lock
(scripts/watch_ckpts.sh), so a job holding that lock for hours is a job that
stops the checkpoint watchers measuring. Calibration and evaluation are
therefore separate invocations, writing to and reading from the same file. That
is also the shape the deployment has: the table is produced once, offline, and
the decoder only ever reads it.

The bisection itself is `flexuf.beta`, which is the code
`scripts/router_curve.py` runs; nothing about the search is re-implemented here.
"""

from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path.home() / "DCVC"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu import pick as _gpu  # noqa: E402

import ctc_intra as C                                     # noqa: E402
from flexuf.beta import (at_beta, bisect_beta, build_cache,  # noqa: E402
                         floor_db, measured_saving, true_db)
from flexuf.config import FlexUFConfig                    # noqa: E402
from flexuf.cost import exit_costs                        # noqa: E402
from flexuf.model import FlexUFIntra, load_flexuf_state   # noqa: E402
from flexuf.reference import reference_for                # noqa: E402
from why_qp_val import load_val, VAL                      # noqa: E402


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/RECIPE512/ckpt_PAPER.pth.tar")
    ap.add_argument("--router2",
                    default="runs/RECIPE512/routers2/v2_lam1.3e-5.pth")
    ap.add_argument("--qps", type=int, nargs="+", default=[0, 16, 32, 48, 63])
    ap.add_argument("--budget", type=float, default=0.1)
    ap.add_argument("--n_val", type=int, default=512,
                    help="held-out images to calibrate on. description_val.json "
                         "holds 512 and every one of them is at least 512px, so "
                         "the default is the whole split.")
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--max_seqs", type=int, default=0)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--per_frame_scale", action="store_true")
    ap.add_argument("--stage", default="both",
                    choices=["both", "calibrate", "evaluate"],
                    help="'calibrate' writes the beta table and stops; "
                         "'evaluate' reads it back and applies it to the test "
                         "set. Two invocations rather than one long one, so the "
                         "shared card's lock is released in between.")
    ap.add_argument("--store", default="cpu",
                    help="where the cached latents and padded frames live. The "
                         "calibration set is 512 images; on the card that is "
                         "gigabytes for no reason, since the bisection only "
                         "ever reads the small tables.")
    ap.add_argument("--compare",
                    default="results/router_RECIPE512_b01_PAPER.json",
                    help="a router_curve.py output on the same checkpoint, head "
                         "and budget. Its test-set bisection is recomputed here "
                         "through the shared code, so the two are printed side "
                         "by side and a divergence is visible rather than "
                         "assumed away.")
    ap.add_argument("--device", default=_gpu("cuda:0"))
    ap.add_argument("--out", default="results/beta_calibration.json")
    a = ap.parse_args(argv)

    dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location="cpu", weights_only=False)
    cfg = FlexUFConfig(**ck["config"])
    net = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(net, ck)
    sd0 = ck.get("state_dict", ck)
    if not any(k.startswith("router_head.") for k in sd0):
        raise SystemExit(f"{a.ckpt} carries no router head; nothing to evaluate")

    head2, r2meta = None, None
    if a.router2:
        from flexuf.router.head2 import StemRouterHeadV2
        from flexuf.config import LATENT_CH, TRUNK_CH
        h = torch.load(ROOT / a.router2, map_location="cpu", weights_only=False)
        hsd = h.get("router_head_v2", h.get("state_dict", h))
        head2 = StemRouterHeadV2(TRUNK_CH, LATENT_CH, cfg.num_exits,
                                 min_exit=cfg.split_depth).to(dev).eval()
        head2.load_state_dict(hsd)
        r2meta = {k: v for k, v in h.items() if k != "router_head_v2"}
        print(f"  router: V2 head from {a.router2}\n"
              f"          trained against a FROZEN decoder at lam="
              f"{r2meta.get('lam')}, held-out agreement "
              f"{r2meta.get('heldout_agree')}")
    ref = FlexUFIntra(cfg).to(dev).eval()
    load_flexuf_state(ref, torch.load(reference_for(cfg, a.ref),
                                      map_location="cpu", weights_only=False))
    sa, sb = net.enc.state_dict(), ref.enc.state_dict()
    worst = max((sa[k] - sb[k]).abs().max().item() for k in sa)
    assert worst == 0.0, f"encoders differ by {worst}"

    cost = exit_costs(cfg, "head").to(dev)
    j = cfg.split_depth
    out = ROOT / a.out

    # The router's share of a decode is measured on the first frame of each set,
    # not once globally: it is charged per padded pixel and the two sets have
    # different padded sizes. Keeping them apart is what makes the test numbers
    # below directly comparable with scripts/router_curve.py, which measures it
    # on a CTC frame.
    rshare_val = rshare_test = 0.0
    beta_table, val_rows = {}, []
    cal_meta = {"split": "description_val.json", "crop": a.crop,
                "root": str(VAL)}

    # ---- 1. calibrate, on frames the run never trained or tested on --------
    if a.stage in ("both", "calibrate"):
        val = load_val(a.n_val, a.crop, "cpu")
        if not val:
            raise SystemExit("no usable validation images")
        cal_meta["n_images"] = len(val)
        print(f"  calibrating on {len(val)} held-out OpenImages at {a.crop}px, "
              f"{cfg.rgb_patch}px tile, budget {a.budget} dB\n")
        print(f"  {'qp':>4}{'beta':>12}{'dB':>10}{'saving':>10}{'floor dB':>10}")
        for qp_v in a.qps:
            t0 = time.time()
            cache, rshare_val = build_cache(
                net, ref, head2, cfg, val, qp_v, dev,
                per_frame_scale=a.per_frame_scale, store=a.store,
                rshare=rshare_val,
                on_rshare=lambda r: print(
                    f"  router costs {100*r:.4f}% of a {a.crop}px decode; "
                    f"charged against every calibration saving\n"))
            fl = floor_db(net.dec, cache, cost, j, dev)
            if fl > a.budget:
                # The budget is unreachable on the calibration set, so there is
                # no beta to ship for this rate and the deepest allocation is
                # what a deployment falls back to.
                beta_table[qp_v] = None
                val_rows.append({"qp": qp_v, "beta": None, "floor_db": fl,
                                 "budget_reachable": False})
                print(f"  {qp_v:>4}{'-':>12}{'-':>10}{'-':>10}{fl:>10.4f}"
                      f"   floor > budget")
                del cache
                continue
            beta, td = bisect_beta(net.dec, cache, cost, j, rshare_val,
                                   a.budget, dev)
            sv, db_t, svr = at_beta(cache, cost, j, rshare_val, beta)
            svm = measured_saving(net.dec, ref.dec, cache, cost, j, rshare_val,
                                  beta, dev)
            beta_table[qp_v] = beta
            val_rows.append({"qp": qp_v, "beta": beta, "db_vs_uf": td,
                             "db_vs_uf_table": db_t, "saving_pct": sv,
                             "saving_pct_vs_release": svr,
                             "saving_pct_measured": svm, "floor_db": fl,
                             "budget_reachable": True})
            print(f"  {qp_v:>4}{beta:>12.3f}{td:>10.4f}{svm:>9.2f}%{fl:>10.4f}"
                  f"   ({time.time()-t0:.0f} s)")
            del cache

        # Written here, before a single test frame has been decoded with it, so
        # the table exists as its own artefact and the evaluation below can only
        # read it.
        out.write_text(json.dumps(
            {"stage": "calibration only", "ckpt": a.ckpt, "budget_db": a.budget,
             "router2": a.router2, "router2_meta": r2meta,
             "router_compute_share_pct_calibration": 100 * rshare_val,
             "calibration_set": cal_meta,
             "beta_table": {str(k): v for k, v in beta_table.items()},
             "calibration_rows": val_rows}, indent=2))
        print(f"\n  wrote the beta table to {a.out}")
        if a.stage == "calibrate":
            return 0
    else:
        prev = json.loads(out.read_text())
        if prev.get("ckpt") not in (None, a.ckpt):
            raise SystemExit(f"{a.out} was calibrated on {prev['ckpt']}, not "
                             f"{a.ckpt}")
        if abs(prev.get("budget_db", a.budget) - a.budget) > 1e-9:
            raise SystemExit(f"{a.out} holds a table for a "
                             f"{prev['budget_db']} dB budget, not {a.budget}")
        beta_table = {int(k): v for k, v in prev["beta_table"].items()}
        val_rows = prev["calibration_rows"]
        rshare_val = prev.get("router_compute_share_pct_calibration", 0.0) / 100
        cal_meta = prev.get("calibration_set", cal_meta)
        print(f"  beta table read back from {a.out}: "
              f"{cal_meta.get('n_images')} held-out images at "
              f"{cal_meta.get('crop')}px, budget {prev.get('budget_db')} dB")

    # ---- 2. apply it, unchanged, to the test set ---------------------------
    seqs, missing = C.discover([])
    if a.max_seqs:
        missing = missing + [{"name": s["name"]} for s in seqs[a.max_seqs:]]
        seqs = seqs[:a.max_seqs]
    test, measured = [], []
    for s in seqs:
        x, pl = C.read_frames(s["path"], s["w"], s["h"], a.frames, 1)
        if x is not None:
            for i in range(x.shape[0]):
                test.append(x[i:i + 1])
            measured.append(s["name"])
    print(f"\n  evaluating on {len(test)} CTC frames from {len(measured)} "
          f"sequences\n")
    print(f"  {'qp':>4}{'held save':>11}{'held dB':>10}{'test save':>11}"
          f"{'test dB':>10}{'beta held':>11}{'beta test':>11}")

    rows = []
    for qp_v in a.qps:
        t0 = time.time()
        cache, rshare_test = build_cache(
            net, ref, head2, cfg, test, qp_v, dev,
            per_frame_scale=a.per_frame_scale, store=a.store,
            rshare=rshare_test,
            on_rshare=lambda r: print(
                f"  router costs {100*r:.4f}% of a CTC decode; charged against "
                f"every test saving below\n"))
        vr = next(r for r in val_rows if r["qp"] == qp_v)
        b_held = beta_table.get(qp_v)
        row = {"qp": qp_v, "beta_heldout": b_held,
               "val_db": vr.get("db_vs_uf"),
               "val_saving_pct_measured": vr.get("saving_pct_measured")}

        # (a) the held-out beta, unchanged. Nothing in it is fitted to these
        # frames, so this is what a decoder that reads the quality index out of
        # the bitstream and looks beta up would deliver.
        if b_held is not None:
            db_h = true_db(net.dec, cache, cost, j, b_held, dev)
            sv_h, dbt_h, svr_h = at_beta(cache, cost, j, rshare_test, b_held)
            svm_h = measured_saving(net.dec, ref.dec, cache, cost, j,
                                    rshare_test, b_held, dev)
            row.update(heldout_db=db_h, heldout_db_table=dbt_h,
                       heldout_saving_pct=sv_h,
                       heldout_saving_pct_vs_release=svr_h,
                       heldout_saving_pct_measured=svm_h)

        # (b) bisecting on the test set itself: the number configuration B has
        # been quoted at until now, and the one a reviewer would object to.
        fl = floor_db(net.dec, cache, cost, j, dev)
        row["floor_db"] = fl
        if fl > a.budget:
            row["budget_reachable"] = False
        else:
            row["budget_reachable"] = True
            b_t, td_t = bisect_beta(net.dec, cache, cost, j, rshare_test,
                                    a.budget, dev)
            sv_t, dbt_t, svr_t = at_beta(cache, cost, j, rshare_test, b_t)
            svm_t = measured_saving(net.dec, ref.dec, cache, cost, j,
                                    rshare_test, b_t, dev)
            row.update(beta_test=b_t, test_db=td_t, test_saving_pct=sv_t,
                       test_saving_pct_vs_release=svr_t,
                       test_saving_pct_measured=svm_t)

            # (c) the same test-set bisection taken to the quality the held-out
            # beta actually delivered, so the two can be differenced at equal
            # quality instead of across a quality difference.
            if b_held is not None:
                if row["heldout_db"] > fl:
                    b_m, td_m = bisect_beta(net.dec, cache, cost, j,
                                            rshare_test, row["heldout_db"], dev)
                    svm_m = measured_saving(net.dec, ref.dec, cache, cost, j,
                                            rshare_test, b_m, dev)
                    row.update(matched_beta=b_m, matched_db=td_m,
                               matched_saving_pct_measured=svm_m,
                               transfer_cost_pts=(
                                   svm_m - row["heldout_saving_pct_measured"]))
                else:
                    # The held-out beta asks for better quality than the deepest
                    # allocation delivers, so no beta can match it.
                    row.update(matched_beta=None, matched_db=fl,
                               matched_saving_pct_measured=None,
                               transfer_cost_pts=None)
                row["d_saving_pts"] = (row["heldout_saving_pct_measured"]
                                       - svm_t)
                row["d_db"] = row["heldout_db"] - td_t
                row["db_over_budget"] = row["heldout_db"] - a.budget
        rows.append(row)

        def _n(k):
            # A rate the calibration set could not reach carries beta_heldout
            # as an explicit None, so dict.get's default never fires and the
            # format spec meets a None. Map missing and null to the same thing.
            v = row.get(k)
            return float("nan") if v is None else v

        print(f"  {qp_v:>4}"
              f"{_n('heldout_saving_pct_measured'):>10.2f}%"
              f"{_n('heldout_db'):>10.4f}"
              f"{_n('test_saving_pct_measured'):>10.2f}%"
              f"{_n('test_db'):>10.4f}"
              f"{_n('beta_heldout'):>11.2f}"
              f"{_n('beta_test'):>11.2f}   "
              f"({time.time()-t0:.0f} s)")
        del cache

    # ---- 3. does the shared bisection still reproduce router_curve? --------
    cmp_ = None
    p = ROOT / a.compare if a.compare else None
    if p is not None and p.exists():
        d = json.loads(p.read_text())
        prev_rows = {r["qp"]: r for r in d.get("rows", [])}
        cmp_rows = []
        for r in rows:
            q = r["qp"]
            if q in prev_rows and prev_rows[q].get("beta") is not None:
                cmp_rows.append({
                    "qp": q, "file_beta": prev_rows[q]["beta"],
                    "here_beta": r.get("beta_test"),
                    "file_saving_measured":
                        prev_rows[q].get("saving_pct_measured"),
                    "here_saving_measured": r.get("test_saving_pct_measured"),
                    "file_db": prev_rows[q].get("db_vs_uf"),
                    "here_db": r.get("test_db")})
        cmp_ = {"file": a.compare, "ckpt": d.get("ckpt"),
                "budget_db": d.get("budget_db"), "rows": cmp_rows}
        print(f"\n  the same test-set bisection, as scripts/router_curve.py "
              f"wrote it into {a.compare}:")
        for c in cmp_rows:
            if c["here_saving_measured"] is None:
                continue
            print(f"  {c['qp']:>4}  beta {c['file_beta']:>10.3f} vs "
                  f"{c['here_beta']:>10.3f}   saving "
                  f"{c['file_saving_measured']:>6.2f}% vs "
                  f"{c['here_saving_measured']:>6.2f}%")

    out.write_text(json.dumps(
        {"ckpt": a.ckpt, "ckpt_epoch": ck.get("epoch"),
         "ckpt_step": ck.get("step"),
         "decision": "beta calibrated on held-out images, applied unchanged to "
                     "the test set",
         "budget_db": a.budget,
         "router2": a.router2, "router2_meta": r2meta,
         "bits_added_per_frame": 0,
         "router_compute_share_pct": 100 * rshare_test,
         "router_compute_share_pct_calibration": 100 * rshare_val,
         "deepest_exit_cost": float(cost[-1]),
         "calibration_set": cal_meta,
         "test_set": {"n_sequences": len(measured), "n_frames": len(test),
                      "frames_per_seq": a.frames, "measured": measured,
                      "not_measured": [m["name"] for m in missing]},
         "beta_table": {str(k): v for k, v in beta_table.items()},
         "calibration_rows": val_rows,
         "rows": rows,
         "crosscheck": cmp_}, indent=2))
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
