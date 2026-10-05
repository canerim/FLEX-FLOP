"""CPU-only exploratory cross-fit of a decoder-visible early-exit risk head.

This uses frozen CTC tile-error labels only for fitting and train-fold
calibration. Held-out decisions see router logits, QP and a scalar price, not
source pixels or errors. The cohort informed prior model development; this is
a feasibility diagnostic and cannot establish external generalisation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "flexplus/results/router_dump_e15_ce_soft.pt"
CROSSFIT = ROOT / "docs/research/2026-09-27-six-hour/crossfit_control/analysis.json"
OUT = ROOT / "docs/research/2026-10-05-decoder-risk-head"
PRICES = np.r_[0.0, np.geomspace(1e-5, 1.0, 250)]
TARGET_DB = 0.1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def features(lp: np.ndarray) -> np.ndarray:
    p = np.exp(lp)
    top = np.sort(lp, axis=1)
    return np.column_stack((np.ones(len(lp)), lp, -(p * lp).sum(1),
                            top[:, -1] - top[:, -2]))


def fit_ridge(frames: list[dict], ids: np.ndarray, first: int) -> np.ndarray:
    xs, ys, ws = [], [], []
    for i in ids:
        f = frames[int(i)]
        lp = f["lp"].double().numpy()
        mse = f["M"].double().numpy()[:, first:]
        x = features(lp)
        xs.append(x)
        ys.append((mse - mse[:, -1:]) / f["R"])
        ws.append(np.full(len(x), 1 / len(x)))
    x, y, w = np.concatenate(xs), np.concatenate(ys), np.concatenate(ws)
    # Normalize nonconstant features on training data only; each image gets
    # equal total weight regardless of geometry.
    mu = np.average(x[:, 1:], axis=0, weights=w)
    scale = np.sqrt(np.average((x[:, 1:] - mu)**2, axis=0, weights=w))
    scale = np.maximum(scale, 1e-6)
    z = np.column_stack((x[:, :1], (x[:, 1:] - mu) / scale))
    xtwx = (z * w[:, None]).T @ z
    penalty = np.eye(z.shape[1]) * 1e-3
    penalty[0, 0] = 0
    coef = np.linalg.solve(xtwx + penalty, (z * w[:, None]).T @ y)
    coef[:, -1] = 0
    return mu, scale, coef


def predict(f: dict, model: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
    mu, scale, coef = model
    x = features(f["lp"].double().numpy())
    z = np.column_stack((x[:, :1], (x[:, 1:] - mu) / scale))
    return z @ coef


def evaluate(f: dict, predicted: np.ndarray, cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    maps = (predicted[None] + PRICES[:, None, None] * cost[None, None]).argmin(2)
    mse = f["M"].double().numpy()[:, 2:]
    measured = mse[np.arange(len(mse))[None], maps].mean(1)
    loss = 10 * np.log10(measured / f["R"])
    saving = 100 * (1 - cost[maps].mean(1))
    return loss, saving


def choose(loss: np.ndarray, saving: np.ndarray) -> tuple[int, bool]:
    risk = np.quantile(loss, 0.9, axis=1, method="linear")
    feasible = np.flatnonzero(risk <= TARGET_DB + 1e-12)
    if len(feasible):
        index = min(feasible, key=lambda i: (-saving[i].mean(), risk[i], i))
        return int(index), True
    return int(np.argmin(risk)), False


def main() -> None:
    torch.set_num_threads(1)
    dump = torch.load(DUMP, map_location="cpu", weights_only=False)
    cross = json.loads(CROSSFIT.read_text())
    assert dump["names"] == cross["names"] and len(cross["folds"]) == 53
    cost = dump["cost"].double().numpy()[2:]
    assert len(cost) == 4 and np.all(np.diff(cost) > 0)
    rows = []
    diagnostics = {}
    for qp, frames in sorted(dump["frames"].items()):
        assert len(frames) == 53
        truth_by_exit = [[] for _ in range(3)]
        pred_by_exit = [[] for _ in range(3)]
        for fold in range(5):
            train = np.flatnonzero(np.asarray(cross["folds"]) != fold)
            test = np.flatnonzero(np.asarray(cross["folds"]) == fold)
            assert not set(train) & set(test)
            model = fit_ridge(frames, train, 2)
            train_loss, train_saving = [], []
            for i in train:
                loss, saving = evaluate(frames[int(i)], predict(frames[int(i)], model), cost)
                train_loss.append(loss); train_saving.append(saving)
            selected, feasible = choose(np.asarray(train_loss).T, np.asarray(train_saving).T)
            for i in test:
                frame = frames[int(i)]
                predicted = predict(frame, model)
                true_mse = frame["M"].double().numpy()[:, 2:]
                true_excess = (true_mse - true_mse[:, -1:]) / frame["R"]
                for k in range(3):
                    truth_by_exit[k].extend(true_excess[:, k].tolist())
                    pred_by_exit[k].extend(predicted[:, k].tolist())
                loss, saving = evaluate(frame, predicted, cost)
                rows.append({"qp": int(qp), "sequence": dump["names"][int(i)], "fold": fold,
                             "price": float(PRICES[selected]), "training_feasible": feasible,
                             "loss_db": float(loss[selected]), "saving_points": float(saving[selected]),
                             "exceeds_0p1_db": bool(loss[selected] > TARGET_DB + 1e-12)})
        diagnostics[str(qp)] = {}
        for k, depth in enumerate((6, 8, 10)):
            actual = np.asarray(truth_by_exit[k]); estimated = np.asarray(pred_by_exit[k])
            diagnostics[str(qp)][str(depth)] = {
                "tiles": len(actual),
                "pearson": float(np.corrcoef(actual, estimated)[0, 1]),
                "r2": float(1 - np.mean((actual - estimated)**2) / np.var(actual)),
                "mean_bias_pred_minus_true": float(np.mean(estimated - actual)),
            }
    assert len(rows) == 265 and len({(r["qp"], r["sequence"]) for r in rows}) == 265
    baseline = [r for r in cross["rows"] if r["criterion"] == "q90" and
                r["budget"] == TARGET_DB and r["policy"] == "router"]
    assert len(baseline) == 265
    key = lambda r: (r["qp"], r["sequence"])
    a = {key(r): r for r in rows}
    b = {key(r): r for r in baseline}
    assert a.keys() == b.keys()

    def describe(subset: list[tuple[int, str]]) -> dict:
        def stats(lookup):
            loss = np.asarray([lookup[k]["loss_db"] for k in subset])
            sav = np.asarray([lookup[k]["saving_points"] for k in subset])
            return {"n": len(subset), "mean_loss_db": float(loss.mean()),
                    "q90_loss_db": float(np.quantile(loss, .9)),
                    "mean_saving_points": float(sav.mean()),
                    "over_0p1_db": int((loss > TARGET_DB + 1e-12).sum())}
        return {"risk_head": stats(a), "baseline_logit": stats(b),
                "paired_saving_delta_points": float(np.mean([a[k]["saving_points"] -
                                                                b[k]["saving_points"] for k in subset]))}

    result = {
        "scope": "Exploratory five-fold sequence-disjoint QP-specific ridge risk head on frozen CTC tile tables; labels never enter held-out decisions. The cohort influenced prior development, and tile-table quality is not mixed-image replay.",
        "target_db": TARGET_DB, "folds": cross["folds"],
        "features": "Four frozen router log-probabilities, their entropy and top-two margin; separate regression for each reachable exit's MSE difference from D12, normalized by released reference MSE.",
        "selection": "Training-fold Q90 table loss <= 0.1 dB; among feasible prices maximize training mean nominal MAC saving. All held-out cases retained.",
        "input_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (DUMP, CROSSFIT)},
        "all_qp": describe(sorted(a)),
        "qp32": describe(sorted(k for k in a if k[0] == 32)),
        "per_qp": {str(q): describe(sorted(k for k in a if k[0] == q)) for q in sorted(dump["frames"])},
        "heldout_tile_excess_error_diagnostics": diagnostics,
        "rows": rows,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("all_qp", "qp32", "per_qp")}, indent=2))
    assert not torch.cuda.is_initialized()


if __name__ == "__main__":
    main()
