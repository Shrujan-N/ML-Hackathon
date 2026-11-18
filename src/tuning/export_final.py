"""Fit the tuned model on ALL 80 train trips, score student_test, export artifacts.

  python -m src.tuning.export_final            # uses outputs/tuning/best_params.json
  python -m src.tuning.export_final --defaults # export untuned defaults (fallback)

Writes:
  artifacts/final_model.joblib   bundle {model, feature_cols, needs_scaling, scaler, meta}
  artifacts/final_scaler.joblib  StandardScaler fit on train only (copy of Member 1's scaler)
  artifacts/final_metrics.json   student_test metrics + params + inference time
  outputs/evaluation/boosting_feature_importance.png
"""
import argparse
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.evaluation.base import REPO_ROOT, RESULTS_DIR, load_data
from src.evaluation.cv_harness import evaluate_holdout
from src.models.boosting.boosting_model import make_model, save_bundle

ART = os.path.join(REPO_ROOT, "artifacts")
BEST = os.path.join(REPO_ROOT, "outputs", "tuning", "best_params.json")


def main(defaults=False):
    train_df, test_df, feature_cols = load_data()
    backend, params, name = "lightgbm", {}, "boosting_lightgbm_default"
    if not defaults:
        with open(BEST) as f:
            best = json.load(f)
        backend, params, name = best["backend"], best["best_params"], f"boosting_{best['backend']}_tuned"
    m, model = evaluate_holdout(name, lambda: make_model(backend=backend, **params), train_df, test_df,
                                feature_cols, params=params)
    scaler = joblib.load(os.path.join(ART, "feature_scaler.joblib"))  # fit on train only (Member 1)
    joblib.dump(scaler, os.path.join(ART, "final_scaler.joblib"))
    save_bundle(model, feature_cols, os.path.join(ART, "final_model.joblib"), scaler=scaler,
                meta={"backend": backend, "params": params, "name": name})
    with open(os.path.join(ART, "final_metrics.json"), "w") as f:
        json.dump({"model": name, "split": "student_test", "params": params, **m}, f, indent=2)

    imp = model.feature_importances()
    if imp is not None:
        order = sorted(range(len(feature_cols)), key=lambda i: imp[i])
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh([feature_cols[i] for i in order], [imp[i] for i in order], color="#16a34a")
        ax.set_title("Boosting Model - Feature Importance"); fig.tight_layout()
        os.makedirs(RESULTS_DIR, exist_ok=True)
        fig.savefig(os.path.join(RESULTS_DIR, "boosting_feature_importance.png"), dpi=180); plt.close(fig)
    print("Exported artifacts/final_model.joblib, final_scaler.joblib, final_metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", action="store_true")
    main(ap.parse_args().defaults)
