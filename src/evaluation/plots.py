"""Sec 3.1 Master Comparison Table + Sec 3.3 model-comparison charts.

Reads outputs/evaluation/model_metrics.csv (written by every member via the harness) and builds:
  outputs/evaluation/master_comparison_cv.csv / .md        (mean ± std over folds)
  outputs/evaluation/master_comparison_student_test.csv / .md
  outputs/evaluation/plots/metric_<name>.png               (one bar chart per metric)
  outputs/evaluation/plots/model_comparison_overview.png   (2x3 panel)
  outputs/evaluation/plots/cv_fold_boxplot_top1.png        (fold variance per model)

  python -m src.evaluation.plots
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .base import METRICS_CSV, RESULTS_DIR

PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
PANELS = [("top1_acc", "Top-1 Accuracy (higher is better)"), ("macro_f1", "Macro F1 (higher is better)"),
          ("log_loss", "Log Loss (lower is better)"), ("continuity", "Route Continuity (higher is better)"),
          ("infer_ms_per_1k_rows", "Inference ms / 1k rows (lower is better)"),
          ("fit_seconds", "Training time, s (lower is better)")]


def _latest(df):
    """Keep the most recent row per (model, split) so re-runs overwrite old numbers."""
    return df.sort_values("timestamp").groupby(["model", "split"], as_index=False).tail(1)


def _md_table(t):
    cols = list(t.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.values]
    return "\n".join(lines)


def build(metrics_csv=METRICS_CSV):
    os.makedirs(PLOT_DIR, exist_ok=True)
    raw = pd.read_csv(metrics_csv)
    allrows = raw.copy()
    last = _latest(raw)

    cv = last[last["split"] == "cv_mean"].set_index("model")
    ho = last[last["split"] == "student_test"].set_index("model")

    if len(cv):
        t = pd.DataFrame({
            "Model": cv.index,
            "Top-1": [f"{r.top1_acc:.4f} ± {r.top1_acc_std:.4f}" for r in cv.itertuples()],
            "Macro F1": [f"{r.macro_f1:.4f} ± {r.macro_f1_std:.4f}" for r in cv.itertuples()],
            "Log Loss": [f"{r.log_loss:.4f} ± {r.log_loss_std:.4f}" for r in cv.itertuples()],
            "Continuity": [f"{r.continuity:.4f} ± {r.continuity_std:.4f}" for r in cv.itertuples()],
            "Infer ms/1k rows": [f"{r.infer_ms_per_1k_rows:.2f}" for r in cv.itertuples()],
            "Fit s/fold": [f"{r.fit_seconds:.1f}" for r in cv.itertuples()],
        }).sort_values("Top-1", ascending=False)
        t.to_csv(os.path.join(RESULTS_DIR, "master_comparison_cv.csv"), index=False)
        open(os.path.join(RESULTS_DIR, "master_comparison_cv.md"), "w").write(_md_table(t))
    if len(ho):
        t = ho[["top1_acc", "macro_f1", "log_loss", "continuity", "infer_ms_per_1k_rows", "fit_seconds"]].round(4)
        t.reset_index().to_csv(os.path.join(RESULTS_DIR, "master_comparison_student_test.csv"), index=False)
        open(os.path.join(RESULTS_DIR, "master_comparison_student_test.md"), "w").write(_md_table(t.reset_index()))

    src = cv if len(cv) else ho
    if not len(src):
        print("No cv_mean / student_test rows yet."); return
    order = src.sort_values("top1_acc", ascending=False).index
    colors = plt.cm.tab10.colors

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    for ax, (col, title) in zip(axes.ravel(), PANELS):
        vals = src.loc[order, col]
        err = src.loc[order, col + "_std"] if (col + "_std") in src.columns and len(cv) else None
        ax.bar(range(len(order)), vals, yerr=err, color=colors[:len(order)], capsize=3)
        ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10); ax.grid(axis="y", alpha=0.3)
        fig2, ax2 = plt.subplots(figsize=(7, 4.2))
        ax2.bar(range(len(order)), vals, yerr=err, color=colors[:len(order)], capsize=3)
        ax2.set_xticks(range(len(order))); ax2.set_xticklabels(order, rotation=30, ha="right")
        ax2.set_title(title); ax2.grid(axis="y", alpha=0.3); fig2.tight_layout()
        fig2.savefig(os.path.join(PLOT_DIR, f"metric_{col}.png"), dpi=180); plt.close(fig2)
    fig.suptitle("PathSnap - Model Comparison (Group-aware CV, mean over folds)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(PLOT_DIR, "model_comparison_overview.png"), dpi=180); plt.close(fig)

    folds = _latest_folds(allrows)
    if len(folds):
        groups = [folds[folds["model"] == m]["top1_acc"].values for m in order if m in set(folds["model"])]
        names = [m for m in order if m in set(folds["model"])]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.boxplot(groups); ax.set_xticklabels(names); ax.set_ylabel("Top-1 per fold")
        ax.set_title("Fold-to-fold variability (GroupKFold by trip)"); ax.grid(axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right"); fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "cv_fold_boxplot_top1.png"), dpi=180); plt.close(fig)
    print("Tables + plots written to", RESULTS_DIR)


def _latest_folds(raw):
    """Fold rows of the most recent CV run of each model (last n_splits fold rows)."""
    f = raw[raw["split"].astype(str).str.startswith("cv_fold_")].sort_values("timestamp")
    if not len(f):
        return f
    parts = [g.tail(int(g["n_splits"].iloc[-1])) for _, g in f.groupby("model")]
    return pd.concat(parts)


if __name__ == "__main__":
    build()
