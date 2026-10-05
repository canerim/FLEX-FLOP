"""Exploratory decoder-visible per-frame risk cap on frozen CTC tile tables.

The head, threshold and each selected map are fitted without labels from the
held-out sequence. The CTC development cohort is not an external test and
table losses are not actual mixed-map reconstruction losses.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from audit_decoder_risk_head import (CROSSFIT, DUMP, OUT, PRICES, ROOT,
                                     TARGET_DB, fit_ridge, predict, sha)

CAPS = np.r_[-0.1, np.linspace(-0.01, 0.15, 401), 1.0]


def candidates(frame: dict, predicted: np.ndarray, cost: np.ndarray) -> dict:
    maps = (predicted[None] + PRICES[:, None, None] * cost[None, None]).argmin(2)
    mse = frame["M"].double().numpy()[:, 2:]
    actual = mse[np.arange(len(mse))[None], maps].mean(1)
    return {"pred_delta": predicted[np.arange(len(mse))[None], maps].mean(1),
            "loss": 10 * np.log10(actual / frame["R"]),
            "saving": 100 * (1 - cost[maps].mean(1))}


def select(c: dict, cap: float) -> int:
    feasible = np.flatnonzero(c["pred_delta"] <= cap)
    if not len(feasible):
        # λ=0 is the highest-fidelity predicted map; no source-based fallback.
        return 0
    return int(max(feasible, key=lambda i: (c["saving"][i], -c["pred_delta"][i], -i)))


def choose_cap(candidates_train: list[dict]) -> tuple[int, bool, float]:
    loss = np.empty((len(CAPS), len(candidates_train)))
    saving = np.empty_like(loss)
    for ci, cap in enumerate(CAPS):
        for fi, frame in enumerate(candidates_train):
            selected = select(frame, cap)
            loss[ci, fi] = frame["loss"][selected]
            saving[ci, fi] = frame["saving"][selected]
    risk = np.quantile(loss, .9, axis=1, method="linear")
    feasible = np.flatnonzero(risk <= TARGET_DB + 1e-12)
    if len(feasible):
        best = min(feasible, key=lambda i: (-saving[i].mean(), risk[i], i))
        return int(best), True, float(risk[best])
    best = min(range(len(risk)), key=lambda i: (risk[i], -saving[i].mean(), i))
    return int(best), False, float(risk[best])


def main() -> None:
    torch.set_num_threads(1)
    dump = torch.load(DUMP, map_location="cpu", weights_only=False)
    cross = json.loads(CROSSFIT.read_text())
    assert dump["names"] == cross["names"]
    cost = dump["cost"].double().numpy()[2:]
    rows = []
    selected_controls = []
    for qp, frames in sorted(dump["frames"].items()):
        for fold in range(5):
            train = np.flatnonzero(np.asarray(cross["folds"]) != fold)
            test = np.flatnonzero(np.asarray(cross["folds"]) == fold)
            model = fit_ridge(frames, train, 2)
            cache = {int(i): candidates(frames[int(i)], predict(frames[int(i)], model), cost)
                     for i in np.r_[train, test]}
            index, feasible, training_risk = choose_cap([cache[int(i)] for i in train])
            cap = float(CAPS[index])
            selected_controls.append({"qp": int(qp), "fold": fold, "cap": cap,
                                      "training_feasible": feasible, "train_q90_loss_db": training_risk})
            for i in test:
                c = cache[int(i)]
                choice = select(c, cap)
                rows.append({"qp": int(qp), "sequence": dump["names"][int(i)], "fold": fold,
                             "cap": cap, "training_feasible": feasible,
                             "loss_db": float(c["loss"][choice]),
                             "saving_points": float(c["saving"][choice]),
                             "exceeds_0p1_db": bool(c["loss"][choice] > TARGET_DB + 1e-12)})
    assert len(rows) == 265
    baseline = [r for r in cross["rows"] if r["criterion"] == "q90" and
                r["budget"] == TARGET_DB and r["policy"] == "router"]
    a = {(r["qp"], r["sequence"]): r for r in rows}
    b = {(r["qp"], r["sequence"]): r for r in baseline}
    assert a.keys() == b.keys()

    def summary(keys):
        def stats(table):
            loss = np.asarray([table[k]["loss_db"] for k in keys])
            sav = np.asarray([table[k]["saving_points"] for k in keys])
            return {"mean_loss_db": float(loss.mean()), "q90_loss_db": float(np.quantile(loss, .9)),
                    "mean_saving_points": float(sav.mean()),
                    "over_0p1_db": int((loss > TARGET_DB + 1e-12).sum())}
        return {"n": len(keys), "adaptive_risk": stats(a), "baseline_logit": stats(b),
                "paired_saving_delta_points": float(np.mean([a[k]["saving_points"] -
                                                                b[k]["saving_points"] for k in keys]))}

    result = {"scope": "Exploratory sequence-disjoint decoder-visible per-frame risk cap on CTC development tile tables; no actual reconstruction or latency result.",
              "mechanism": "Fit a tile excess-MSE predictor from router logits on four sequence folds; a global predicted-risk cap is chosen by training Q90 loss. At inference, select the cheapest candidate map under that cap per frame. Candidate maps are generated by a dense price grid offline; deployable search cost is not measured.",
              "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (DUMP, CROSSFIT)},
              "all_qp": summary(sorted(a)),
              "qp32": summary(sorted(k for k in a if k[0] == 32)),
              "per_qp": {str(q): summary(sorted(k for k in a if k[0] == q)) for q in sorted(dump["frames"])},
              "selected_controls": selected_controls, "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "adaptive_analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("all_qp", "qp32", "per_qp")}, indent=2))
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
