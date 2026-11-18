"""Unified comparison across all 6 model families (Models 1-6 + Boosting variants + HMM).

Loads every saved model artifact from `models/` (and compares against Member 2's
5-fold GroupKFold CV records in `outputs/evaluation/model_metrics.csv`), evaluates
all models identically on `student_test` (`19,160` GPS observations / `191,600` rows),
runs HMM/Viterbi decoding on the top boosting model to compare full-pipeline accuracy
against `MLP + HMM`, and exports unified tables and comparison plots.

Run from repo root:
    py -m src.models.compare_all_models
"""
import json
import os
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data.data_loader import load_trip
from src.evaluation.base import META_COLS, METRICS_CSV, REPO_ROOT, RESULTS_DIR, load_data, positive_proba
from src.evaluation.cv_harness import log_row
from src.evaluation.metrics import compute_metrics
from src.hmm import HMMConfig, HMMViterbiDecoder, RoadNetworkGraph, ShortestPathCache, compare_classifier_vs_hmm
from src.models.boosting.boosting_model import bundle_predict_proba, load_bundle
from src.models.mlp.mlp_model import PathSnapMLPClassifier
from src.models.rbf_svm.rbf_svm_model import PathSnapRBFSVMClassifier

warnings.filterwarnings("ignore")

MODELS_DIR = os.path.join(REPO_ROOT, "models")


def evaluate_hmm_on_probs(test_df, probs, model_label):
    work = test_df[["trip_id", "obs_idx", "candidate_arc_id", "true_arc_id"]].copy()
    work["pred_prob"] = np.asarray(probs, dtype=np.float32)

    cfg = HMMConfig(
        emission_weight=1.0,
        beta_m=5.0,
        beta_gap_scale=0.5,
        same_arc_bonus=0.25,
        reverse_penalty=12.0,
        disconnected_penalty=25.0,
        gap_threshold_s=5.0,
        break_gap_s=60.0,
        fill_intermediate_arcs=True,
    )
    decoder = HMMViterbiDecoder(cfg)
    raw_root = os.path.join(REPO_ROOT, "data", "raw")
    trip_ids = sorted(work["trip_id"].astype(str).unique())

    total_obs = 0
    weighted_hmm_hits = 0.0
    weighted_clf_hits = 0.0
    cont_list = []
    route_cont_list = []
    lcs_list = []
    decode_times = []

    for tid in trip_ids:
        trip = load_trip(raw_root, "student_test", tid)
        g = RoadNetworkGraph(trip["nodes"], trip["arcs"])
        cache = ShortestPathCache(g)
        t_df = work[work["trip_id"].astype(str) == tid]
        res = decoder.decode_member1_trip(
            trip=trip,
            trip_candidates_df=t_df,
            prob_col="pred_prob",
            graph=g,
            sp_cache=cache,
        )
        obs_true_map = t_df.drop_duplicates("obs_idx").set_index("obs_idx")["true_arc_id"].to_dict()
        ref_obs_arcs = [int(obs_true_map.get(oidx, -1)) for oidx in res.obs_indices]
        ref_route = trip["route"].tolist() if trip["route"] is not None else None
        ablation = compare_classifier_vs_hmm(
            graph=g,
            decode_result=res,
            reference_obs_arcs=ref_obs_arcs,
            reference_route_arcs=ref_route,
            sp_cache=cache,
        )
        n_obs = res.diagnostics["n_obs"]
        total_obs += n_obs
        weighted_clf_hits += ablation["classifier_only"]["obs_top1_accuracy"] * n_obs
        weighted_hmm_hits += ablation["classifier_plus_hmm"]["obs_top1_accuracy"] * n_obs
        cont_list.append(ablation["classifier_plus_hmm"]["obs_continuity"]["direct_continuity"])
        route_cont_list.append(ablation["classifier_plus_hmm"]["route_continuity"]["direct_continuity"])
        lcs_list.append(ablation["classifier_plus_hmm"]["sequence"]["lcs_f1"])
        decode_times.append(res.diagnostics["decode_time_s"])

    return {
        "pipeline": model_label,
        "weighted_clf_top1": weighted_clf_hits / max(1, total_obs),
        "weighted_hmm_top1": weighted_hmm_hits / max(1, total_obs),
        "mean_obs_continuity": float(np.mean(cont_list)),
        "mean_route_continuity": float(np.mean(route_cont_list)),
        "mean_lcs_f1": float(np.mean(lcs_list)),
        "total_decode_time_s": float(np.sum(decode_times)),
    }


def plot_all_models_comparison(comp_df, hmm_rows, out_path):
    ordered = comp_df.sort_values("Top-1 Accuracy", ascending=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Panel 1: Observation Top-1 Accuracy
    colors = ["#16a34a" if i == len(ordered) - 1 else "#2563eb" for i in range(len(ordered))]
    bars = axes[0, 0].barh(ordered["Model"], ordered["Top-1 Accuracy"] * 100.0, color=colors, alpha=0.9)
    axes[0, 0].set_xlim(75, 92)
    axes[0, 0].set_xlabel("Observation-Level Top-1 Accuracy (%)")
    axes[0, 0].set_title("Standalone Classifiers: Student-Test Top-1 Accuracy")
    axes[0, 0].grid(axis="x", alpha=0.3)
    for b in bars:
        w = b.get_width()
        axes[0, 0].text(w + 0.15, b.get_y() + b.get_height() / 2, f"{w:.2f}%", va="center", fontsize=8.5)

    # Panel 2: Log Loss (lower is better)
    ordered_ll = comp_df.sort_values("Log Loss", ascending=False)
    colors_ll = ["#16a34a" if v == ordered_ll["Log Loss"].min() else "#ea580c" for v in ordered_ll["Log Loss"]]
    bars_ll = axes[0, 1].barh(ordered_ll["Model"], ordered_ll["Log Loss"], color=colors_ll, alpha=0.9)
    axes[0, 1].set_xlabel("Binary Log Loss (Lower is Better)")
    axes[0, 1].set_title("Probability Calibration: Student-Test Log Loss")
    axes[0, 1].grid(axis="x", alpha=0.3)
    for b in bars_ll:
        w = b.get_width()
        axes[0, 1].text(w + 0.002, b.get_y() + b.get_height() / 2, f"{w:.4f}", va="center", fontsize=8.5)

    # Panel 3: Route Continuity & Arc Macro F1 (present for all 10 models)
    y_idx = np.arange(len(ordered))
    h = 0.36
    axes[1, 0].barh(y_idx + h / 2, ordered["Route Continuity"] * 100.0, height=h, label="Route Continuity (%)", color="#0d9488")
    axes[1, 0].barh(y_idx - h / 2, ordered["Arc Macro F1"] * 100.0, height=h, label="Arc Macro F1 (%)", color="#6366f1")
    axes[1, 0].set_yticks(y_idx)
    axes[1, 0].set_yticklabels(ordered["Model"])
    axes[1, 0].set_xlim(65, 100)
    axes[1, 0].set_xlabel("Percentage (%)")
    axes[1, 0].set_title("Route Continuity & Arc Macro F1")
    axes[1, 0].legend(loc="lower right", fontsize=8.5)
    axes[1, 0].grid(axis="x", alpha=0.3)

    # Panel 4: Standalone vs Classifier+HMM Top-1 Accuracy
    ax_hmm = axes[1, 1]
    if hmm_rows:
        labels = [r["pipeline"] for r in hmm_rows]
        clf_vals = [r["weighted_clf_top1"] * 100.0 for r in hmm_rows]
        hmm_vals = [r["weighted_hmm_top1"] * 100.0 for r in hmm_rows]
        x = np.arange(len(labels))
        w = 0.34
        b1 = ax_hmm.bar(x - w / 2, clf_vals, w, label="Classifier Only", color="#64748b")
        b2 = ax_hmm.bar(x + w / 2, hmm_vals, w, label="Classifier + HMM/Viterbi", color="#16a34a")
        ax_hmm.set_ylim(80, 93)
        ax_hmm.set_xticks(x)
        ax_hmm.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
        ax_hmm.set_ylabel("Weighted Student-Test Top-1 (%)")
        ax_hmm.set_title("Full Pipeline: Classifier vs. Classifier + HMM/Viterbi")
        ax_hmm.legend(loc="lower right", fontsize=8.5)
        ax_hmm.grid(axis="y", alpha=0.3)
        for b in list(b1) + list(b2):
            ht = b.get_height()
            ax_hmm.text(b.get_x() + b.get_width() / 2, ht + 0.15, f"{ht:.2f}%", ha="center", va="bottom", fontsize=8.5)

    plt.suptitle("PathSnap Unified Model Comparison (Student-Test Partition: 10 Trips, 19,160 Obs)", fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    print("Loading student_test dataset and shared artifacts...")
    _, test_df, feature_cols = load_data()
    scaler = joblib.load(os.path.join(MODELS_DIR, "feature_scaler.joblib"))

    X_raw = test_df[feature_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X_raw).astype(np.float32)
    y_test = test_df["label"].values
    meta_test = test_df[META_COLS]

    # Load existing CV records from Member 1 and Member 2 for reference
    cv_lookup = {}
    if os.path.exists(METRICS_CSV):
        m_df = pd.read_csv(METRICS_CSV)
        cv_rows = m_df[m_df["split"] == "cv_mean"]
        for _, r in cv_rows.iterrows():
            cv_lookup[str(r["model"])] = {
                "cv_top1": float(r["top1_acc"]),
                "cv_arc_f1": float(r["macro_f1"]),
                "cv_bin_f1": np.nan,
                "cv_log_loss": float(r["log_loss"]),
            }

    m1_json = os.path.join(REPO_ROOT, "outputs", "models_member1", "member1_model_comparison.json")
    if os.path.exists(m1_json):
        with open(m1_json, encoding="utf-8") as f:
            for item in json.load(f):
                key = "model1_logistic_regression" if "Logistic" in item["name"] else "model2_distance_weighted_knn"
                cv_lookup[key] = {
                    "cv_top1": float(item["cv"]["cv_top1_mean"]),
                    "cv_arc_f1": np.nan,
                    "cv_bin_f1": float(item["cv"]["cv_macro_f1_mean"]),
                    "cv_log_loss": float(item["cv"]["cv_log_loss_mean"]),
                }

    # Define all 7 saved standalone models in `models/`
    model_specs = [
        {
            "key": "model1_logistic_regression",
            "label": "Model 1: Logistic Regression",
            "family": "Linear and probabilistic",
            "config": "C=1.0, lbfgs, max_iter=300, StandardScaler",
            "predict_fn": lambda: joblib.load(os.path.join(MODELS_DIR, "model1_logistic_regression.joblib")).predict_proba(X_scaled)[:, 1],
        },
        {
            "key": "model2_distance_weighted_knn",
            "label": "Model 2: Distance-Weighted KNN",
            "family": "Instance-based",
            "config": "k=15, distance weights, kd_tree (32k ref)",
            "predict_fn": lambda: joblib.load(os.path.join(MODELS_DIR, "model2_distance_weighted_knn.joblib")).predict_proba(X_scaled)[:, 1],
        },
        {
            "key": "model3a_random_forest",
            "label": "Model 3a: Random Forest",
            "family": "Ensemble (Bagging)",
            "config": "n_estimators=200, class_weight=balanced",
            "predict_fn": lambda: joblib.load(os.path.join(MODELS_DIR, "model3a_random_forest.joblib")).predict_proba(X_raw)[:, 1],
        },
        {
            "key": "model3b_extra_trees",
            "label": "Model 3b: Extra Trees",
            "family": "Ensemble (Bagging)",
            "config": "n_estimators=200, class_weight=balanced",
            "predict_fn": lambda: joblib.load(os.path.join(MODELS_DIR, "model3b_extra_trees.joblib")).predict_proba(X_raw)[:, 1],
        },
        {
            "key": "model4_boosting_lightgbm_tuned",
            "label": "Model 4: LightGBM (Optuna Tuned)",
            "family": "Boosting",
            "config": "n_estimators=550, lr=0.0289, num_leaves=101",
            "predict_fn": lambda: bundle_predict_proba(
                load_bundle(os.path.join(MODELS_DIR, "model4_boosting_lightgbm_tuned.joblib")),
                test_df,
            ),
        },
        {
            "key": "model5_mlp_neural_network",
            "label": "Model 5: MLP Neural Network",
            "family": "Neural networks",
            "config": "128-64-32, BatchNorm, Dropout=0.15, AdamW",
            "predict_fn": lambda: PathSnapMLPClassifier.load(os.path.join(MODELS_DIR, "model5_mlp_neural_network.pt")).predict_proba(X_raw)[:, 1],
        },
        {
            "key": "model6_rbf_svm_nystroem",
            "label": "Model 6: RBF SVM (Nystrom)",
            "family": "Kernel-based",
            "config": "Nystrom(256), SGD hinge + Platt scaling",
            "predict_fn": lambda: PathSnapRBFSVMClassifier.load(os.path.join(MODELS_DIR, "model6_rbf_svm_nystroem.joblib")).predict_proba(X_raw)[:, 1],
        },
    ]

    rows = []
    probs_cache = {}
    print("\nEvaluating all 7 saved models on student_test (191,600 candidate rows / 19,160 obs)...")
    for spec in model_specs:
        # Warm load outside timing if needed? Better: measure model inference specifically after loading
        if spec["key"] == "model1_logistic_regression":
            m_obj = joblib.load(os.path.join(MODELS_DIR, "model1_logistic_regression.joblib"))
            t0 = time.perf_counter()
            probs = positive_proba(m_obj.predict_proba(X_scaled))
            inf_s = time.perf_counter() - t0
        elif spec["key"] == "model2_distance_weighted_knn":
            m_obj = joblib.load(os.path.join(MODELS_DIR, "model2_distance_weighted_knn.joblib"))
            t0 = time.perf_counter()
            probs = positive_proba(m_obj.predict_proba(X_scaled))
            inf_s = time.perf_counter() - t0
        elif spec["key"] == "model3a_random_forest":
            m_obj = joblib.load(os.path.join(MODELS_DIR, "model3a_random_forest.joblib"))
            t0 = time.perf_counter()
            probs = positive_proba(m_obj.predict_proba(X_raw))
            inf_s = time.perf_counter() - t0
        elif spec["key"] == "model3b_extra_trees":
            m_obj = joblib.load(os.path.join(MODELS_DIR, "model3b_extra_trees.joblib"))
            t0 = time.perf_counter()
            probs = positive_proba(m_obj.predict_proba(X_raw))
            inf_s = time.perf_counter() - t0
        elif spec["key"] == "model4_boosting_lightgbm_tuned":
            bundle = load_bundle(os.path.join(MODELS_DIR, "model4_boosting_lightgbm_tuned.joblib"))
            t0 = time.perf_counter()
            probs = bundle_predict_proba(bundle, test_df)
            inf_s = time.perf_counter() - t0
        elif spec["key"] == "model5_mlp_neural_network":
            m_obj = PathSnapMLPClassifier.load(os.path.join(MODELS_DIR, "model5_mlp_neural_network.pt"))
            t0 = time.perf_counter()
            probs = positive_proba(m_obj.predict_proba(X_raw))
            inf_s = time.perf_counter() - t0
        else:
            m_obj = PathSnapRBFSVMClassifier.load(os.path.join(MODELS_DIR, "model6_rbf_svm_nystroem.joblib"))
            t0 = time.perf_counter()
            probs = positive_proba(m_obj.predict_proba(X_raw))
            inf_s = time.perf_counter() - t0

        probs_cache[spec["key"]] = probs
        metrics = compute_metrics(meta_test, y_test, probs, inf_s)
        log_key = "boosting_lightgbm_tuned" if spec["key"] == "model4_boosting_lightgbm_tuned" else spec["key"]
        log_row(
            {"model": log_key, "split": "student_test", "seed": 42, "params": spec["config"], **metrics},
            METRICS_CSV,
        )
        preds_bin = (probs >= 0.5).astype(int)
        bin_acc = float(accuracy_score(y_test, preds_bin))
        bin_f1 = float(f1_score(y_test, preds_bin, average="macro", zero_division=0))

        cv_info = cv_lookup.get(spec["key"], {})
        rows.append({
            "Model": spec["label"],
            "Model Key": spec["key"],
            "Algorithm Family": spec["family"],
            "Key Configuration": spec["config"],
            "CV Top-1 Accuracy": cv_info.get("cv_top1", np.nan),
            "CV Binary Macro F1": cv_info.get("cv_bin_f1", np.nan),
            "CV Arc Macro F1": cv_info.get("cv_arc_f1", np.nan),
            "Top-1 Accuracy": metrics["top1_acc"],
            "Route Continuity": metrics["continuity"],
            "Arc Macro F1": metrics["macro_f1"],
            "Binary Macro F1": bin_f1,
            "Row Binary Accuracy": bin_acc,
            "Log Loss": metrics["log_loss"],
            "Inference Time (s)": inf_s,
            "Infer ms / 1k rows": metrics["infer_ms_per_1k_rows"],
        })
        print(
            f"  {spec['label']:<36} | Top-1={metrics['top1_acc']*100:.2f}% | "
            f"BinF1={bin_f1:.4f} | ArcF1={metrics['macro_f1']:.4f} | "
            f"LogLoss={metrics['log_loss']:.4f} | Cont={metrics['continuity']*100:.2f}% | "
            f"Inf={inf_s:.2f}s"
        )

    # Also include Member 2's untuned boosting baselines recorded on the exact same test_df
    if os.path.exists(METRICS_CSV):
        m_df = pd.read_csv(METRICS_CSV)
        for mkey, label, cfg_str in [
            ("boosting_hgb", "Model 4b: HistGradientBoosting", "max_iter=300, lr=0.1, max_leaf_nodes=63"),
            ("boosting_xgboost", "Model 4c: XGBoost", "n_estimators=300, lr=0.1, max_depth=8"),
            ("boosting_lightgbm", "Model 4d: LightGBM (Untuned)", "n_estimators=300, lr=0.1, num_leaves=63"),
        ]:
            ho = m_df[(m_df["model"] == mkey) & (m_df["split"] == "student_test")]
            cv = m_df[(m_df["model"] == mkey) & (m_df["split"] == "cv_mean")]
            if len(ho) > 0:
                r = ho.iloc[-1]
                cv_r = cv.iloc[-1] if len(cv) > 0 else None
                inf_s_est = float(r["infer_ms_per_1k_rows"]) * len(y_test) / 1000.0 / 1000.0
                rows.append({
                    "Model": label,
                    "Model Key": mkey,
                    "Algorithm Family": "Boosting",
                    "Key Configuration": cfg_str,
                    "CV Top-1 Accuracy": float(cv_r["top1_acc"]) if cv_r is not None else np.nan,
                    "CV Binary Macro F1": np.nan,
                    "CV Arc Macro F1": float(cv_r["macro_f1"]) if cv_r is not None else np.nan,
                    "Top-1 Accuracy": float(r["top1_acc"]),
                    "Route Continuity": float(r["continuity"]),
                    "Arc Macro F1": float(r["macro_f1"]),
                    "Binary Macro F1": np.nan,
                    "Row Binary Accuracy": np.nan,
                    "Log Loss": float(r["log_loss"]),
                    "Inference Time (s)": inf_s_est,
                    "Infer ms / 1k rows": float(r["infer_ms_per_1k_rows"]),
                })

    comp_df = pd.DataFrame(rows).sort_values(["Top-1 Accuracy", "Log Loss"], ascending=[False, True]).reset_index(drop=True)

    # Evaluate HMM/Viterbi full pipeline on Tuned LightGBM vs MLP
    print("\nRunning HMM/Viterbi comparison on top saved probabilistic models...")
    hmm_rows = [
        evaluate_hmm_on_probs(test_df, probs_cache["model4_boosting_lightgbm_tuned"], "LightGBM Tuned + HMM"),
        evaluate_hmm_on_probs(test_df, probs_cache["model5_mlp_neural_network"], "MLP (Model 5) + HMM"),
    ]
    hmm_df = pd.DataFrame(hmm_rows).sort_values("weighted_hmm_top1", ascending=False).reset_index(drop=True)
    for r in hmm_rows:
        print(
            f"  {r['pipeline']:<24} | Clf Top-1={r['weighted_clf_top1']*100:.2f}% -> "
            f"HMM Top-1={r['weighted_hmm_top1']*100:.2f}% | Obs Cont={r['mean_obs_continuity']*100:.2f}% | "
            f"Route Cont={r['mean_route_continuity']*100:.2f}% | LCS-F1={r['mean_lcs_f1']*100:.2f}%"
        )

    # Save unified CSV and Markdown tables
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "plots"), exist_ok=True)

    csv_out = os.path.join(RESULTS_DIR, "section_3_1_all_models_comparison.csv")
    md_out = os.path.join(RESULTS_DIR, "section_3_1_all_models_comparison.md")
    hmm_csv_out = os.path.join(RESULTS_DIR, "hmm_pipeline_comparison.csv")
    comp_df.to_csv(csv_out, index=False)
    hmm_df.to_csv(hmm_csv_out, index=False)

    # Format concise Markdown summary table (Deliverables.pdf Sec 3.1 style)
    best_standalone = comp_df.iloc[0]
    best_calibrated = comp_df.sort_values("Log Loss", ascending=True).iloc[0]
    best_hmm = hmm_df.iloc[0]

    md_lines = [
        "# Unified Model Comparison & Winner Summary (All 6 Families)",
        "",
        "## Most Accurate Models",
        f"- **Highest Standalone Top-1 Accuracy:** `{best_standalone['Model']}` — **{best_standalone['Top-1 Accuracy']*100:.2f}%** Top-1 Accuracy, **{best_standalone['Log Loss']:.4f}** Log Loss, **{best_standalone['Route Continuity']*100:.2f}%** Route Continuity",
        f"- **Best Calibrated Probabilities (Lowest Log Loss):** `{best_calibrated['Model']}` — **{best_calibrated['Log Loss']:.4f}** Log Loss, **{best_calibrated['Top-1 Accuracy']*100:.2f}%** Top-1 Accuracy",
        f"- **Most Accurate End-to-End Pipeline (Classifier + HMM/Viterbi):** `{best_hmm['pipeline']}` — **{best_hmm['weighted_hmm_top1']*100:.2f}%** Weighted Top-1 Accuracy, **{best_hmm['mean_obs_continuity']*100:.2f}%** Obs Continuity, **{best_hmm['mean_route_continuity']*100:.2f}%** Route Continuity, **{best_hmm['mean_lcs_f1']*100:.2f}%** Route LCS F1",
        "",
        "## Full Comparison Table (`student_test`: 10 trips / 19,160 GPS observations / 191,600 candidate rows)",
        "",
        "| Rank | Model | Family | CV Top-1 | Student-Test Top-1 | Binary Macro F1 | Arc Macro F1 | Route Continuity | Log Loss | Infer Time |",
        "| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for idx, r in comp_df.iterrows():
        cv_s = f"{r['CV Top-1 Accuracy']*100:.2f}%" if pd.notna(r["CV Top-1 Accuracy"]) else "N/A (full-fit)"
        bf1_s = f"{r['Binary Macro F1']:.4f}" if pd.notna(r["Binary Macro F1"]) else "N/A (logged)"
        md_lines.append(
            f"| {idx + 1} | {r['Model']} | {r['Algorithm Family']} | {cv_s} | "
            f"**{r['Top-1 Accuracy']*100:.2f}%** | {bf1_s} | {r['Arc Macro F1']:.4f} | "
            f"{r['Route Continuity']*100:.2f}% | {r['Log Loss']:.4f} | {r['Inference Time (s)']:.2f}s |"
        )

    md_lines += [
        "",
        "## Full Pipeline Comparison (Classifier + HMM/Viterbi on `student_test`)",
        "",
        "| Pipeline | Classifier-Only Top-1 | Classifier + HMM Top-1 | Obs Continuity | Full Route Continuity | Route LCS F1 | Decode Time |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for _, r in hmm_df.iterrows():
        md_lines.append(
            f"| {r['pipeline']} | {r['weighted_clf_top1']*100:.2f}% | **{r['weighted_hmm_top1']*100:.2f}%** | "
            f"{r['mean_obs_continuity']*100:.2f}% | {r['mean_route_continuity']*100:.2f}% | "
            f"{r['mean_lcs_f1']*100:.2f}% | {r['total_decode_time_s']:.2f}s |"
        )

    with open(md_out, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    plot_path = os.path.join(RESULTS_DIR, "plots", "all_models_student_test_comparison.png")
    plot_all_models_comparison(comp_df, hmm_rows, plot_path)

    print(f"\nSaved comparison CSV: {csv_out}")
    print(f"Saved comparison MD : {md_out}")
    print(f"Saved plot          : {plot_path}")


if __name__ == "__main__":
    main()
