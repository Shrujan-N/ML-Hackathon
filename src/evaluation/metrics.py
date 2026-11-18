"""Metrics for the candidate-ranking formulation.

Each GPS observation (trip_id, obs_idx) has K candidate arcs. A model scores
every candidate with P(label=1); the predicted arc for the observation is the
candidate with the highest score.

  top1_acc    : % observations whose predicted arc == ground-truth arc.
                (If the true arc is not among the K candidates, it counts as a miss.)
  macro_f1    : macro-averaged F1 over arc classes (true_arc_id vs predicted arc),
                over every arc id that appears as truth or prediction.
  log_loss    : binary log loss over all candidate rows (label vs P(label=1)).
  continuity  : % of consecutive predicted observations (within a trip) that are
                the same arc or topologically connected (to_node[t-1] == from_node[t]).
  infer_ms_per_1k_rows / infer_ms_per_obs : wall-clock predict_proba time.

!! If the hackathon problem statement defines Macro-F1 / Continuity differently,
   change ONLY this file - the harness, tuning and plots all call compute_metrics().
"""
import numpy as np
import pandas as pd


def predict_arcs(meta: pd.DataFrame, proba: np.ndarray) -> pd.DataFrame:
    """One row per (trip_id, obs_idx): predicted arc, its from/to node, true arc."""
    df = meta[["trip_id", "obs_idx", "candidate_arc_id", "from_node", "to_node", "true_arc_id"]].copy()
    df["p"] = proba
    df = df.reset_index(drop=True)
    idx = df.groupby(["trip_id", "obs_idx"], sort=True)["p"].idxmax()
    pred = df.loc[idx.values].reset_index(drop=True)
    return pred.rename(columns={"candidate_arc_id": "pred_arc_id"})


def macro_f1_arcs(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ids = np.unique(np.concatenate([y_true, y_pred]))
    t = np.searchsorted(ids, y_true)
    p = np.searchsorted(ids, y_pred)
    n = len(ids)
    tp = np.bincount(t[t == p], minlength=n).astype(np.float64)
    t_cnt = np.bincount(t, minlength=n)
    p_cnt = np.bincount(p, minlength=n)
    denom = t_cnt + p_cnt
    f1 = np.where(denom > 0, 2.0 * tp / np.maximum(denom, 1), 0.0)
    return float(f1.mean())


def route_continuity(pred: pd.DataFrame) -> float:
    trip = pred["trip_id"].values
    arc = pred["pred_arc_id"].values
    frm = pred["from_node"].values
    to = pred["to_node"].values
    same_trip = trip[1:] == trip[:-1]
    ok = (arc[1:] == arc[:-1]) | (to[:-1] == frm[1:])
    n = same_trip.sum()
    return float(ok[same_trip].sum() / n) if n else float("nan")


def binary_log_loss(y, p, eps=1e-15):
    p = np.clip(p, eps, 1 - eps)
    y = np.asarray(y)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def compute_metrics(meta: pd.DataFrame, y: np.ndarray, proba: np.ndarray, infer_seconds: float = float("nan")) -> dict:
    pred = predict_arcs(meta, proba)
    n_obs = len(pred)
    return {
        "top1_acc": float((pred["pred_arc_id"].values == pred["true_arc_id"].values).mean()),
        "macro_f1": macro_f1_arcs(pred["true_arc_id"].values, pred["pred_arc_id"].values),
        "log_loss": binary_log_loss(y, proba),
        "continuity": route_continuity(pred),
        "infer_ms_per_1k_rows": 1000.0 * infer_seconds / len(y) * 1000.0 if len(y) else float("nan"),
        "infer_ms_per_obs": 1000.0 * infer_seconds / n_obs if n_obs else float("nan"),
        "n_obs": int(n_obs),
        "n_rows": int(len(y)),
    }


METRIC_COLS = ["top1_acc", "macro_f1", "log_loss", "continuity", "infer_ms_per_1k_rows", "infer_ms_per_obs"]
HIGHER_IS_BETTER = {"top1_acc": True, "macro_f1": True, "log_loss": False, "continuity": True}
