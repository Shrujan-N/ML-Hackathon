import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
)


def safe_joblib_dump(obj, out_path, compress=3):
    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        joblib.dump(obj, f, compress=compress)
    os.replace(tmp_path, out_path)


def compute_obs_metrics(meta_df, probs, cand_k=10):
    n_obs = len(probs) // cand_k
    probs_2d = probs.reshape(n_obs, cand_k)
    best_pos = np.arange(n_obs) * cand_k + probs_2d.argmax(axis=1)

    obs_df = meta_df.iloc[best_pos][
        ["trip_id", "obs_idx", "timestamp", "candidate_arc_id", "from_node", "to_node", "true_arc_id"]
    ].copy()
    obs_df["pred_prob"] = probs[best_pos]
    obs_df["pred_arc_id"] = obs_df["candidate_arc_id"]
    obs_df["correct"] = (obs_df["pred_arc_id"] == obs_df["true_arc_id"]).astype(int)
    obs_df = obs_df.sort_values(["trip_id", "obs_idx"]).reset_index(drop=True)

    top1_acc = float(obs_df["correct"].mean())

    # Route continuity: consecutive observations in a trip share the same arc or connect to_node == from_node
    trip_ids = obs_df["trip_id"].values
    arcs_arr = obs_df["pred_arc_id"].values
    from_arr = obs_df["from_node"].values
    to_arr = obs_df["to_node"].values
    same_trip = trip_ids[:-1] == trip_ids[1:]
    same_arc = arcs_arr[:-1] == arcs_arr[1:]
    connected = to_arr[:-1] == from_arr[1:]
    cont_hits = int((same_trip & (same_arc | connected)).sum())
    cont_total = int(same_trip.sum())

    continuity = float(cont_hits / max(1, cont_total))
    out_cols = ["trip_id", "obs_idx", "timestamp", "true_arc_id", "pred_arc_id", "pred_prob", "correct"]
    return top1_acc, continuity, obs_df[out_cols]


def evaluate_predictions(y_true, probs, meta_df):
    preds = (probs >= 0.5).astype(int)
    bin_acc = float(accuracy_score(y_true, preds))
    prec = float(precision_score(y_true, preds, zero_division=0))
    rec = float(recall_score(y_true, preds, zero_division=0))
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    ll = float(log_loss(y_true, np.clip(probs, 1e-7, 1.0 - 1e-7)))
    cm = confusion_matrix(y_true, preds)
    top1_acc, continuity, obs_df = compute_obs_metrics(meta_df, probs)
    return {
        "binary_accuracy": bin_acc,
        "precision": prec,
        "recall": rec,
        "macro_f1": macro_f1,
        "log_loss": ll,
        "top1_accuracy": top1_acc,
        "route_continuity": continuity,
        "confusion_matrix": cm.tolist(),
        "obs_predictions": obs_df,
    }


def sample_trip_stratified_obs(df, obs_per_trip=100, seed=42):
    rng = np.random.default_rng(seed)
    chosen_groups = []
    for tid, grp in df.groupby("trip_id", sort=False):
        unique_obs = grp["obs_idx"].unique()
        if len(unique_obs) > obs_per_trip:
            picked = rng.choice(unique_obs, size=obs_per_trip, replace=False)
            chosen_groups.append(grp[grp["obs_idx"].isin(picked)])
        else:
            chosen_groups.append(grp)
    return pd.concat(chosen_groups, ignore_index=True)


def run_group_cv(df, feature_cols, make_model_fn, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    X_raw = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    groups = df["trip_id"].values

    fold_f1 = []
    fold_top1 = []
    fold_ll = []
    fold_cont = []

    for fold_idx, (tr_idx, val_idx) in enumerate(gkf.split(X_raw, y, groups), start=1):
        fold_scaler = StandardScaler()
        X_tr = fold_scaler.fit_transform(X_raw[tr_idx]).astype(np.float32)
        X_val = fold_scaler.transform(X_raw[val_idx]).astype(np.float32)

        clf = make_model_fn()
        clf.fit(X_tr, y[tr_idx])
        val_probs = clf.predict_proba(X_val)[:, 1]

        res = evaluate_predictions(y[val_idx], val_probs, df.iloc[val_idx])
        fold_f1.append(res["macro_f1"])
        fold_top1.append(res["top1_accuracy"])
        fold_ll.append(res["log_loss"])
        fold_cont.append(res["route_continuity"])
        print(
            f"  Fold {fold_idx}/{n_splits}: Macro F1={res['macro_f1']:.4f}, "
            f"Top-1={res['top1_accuracy']*100:.2f}%, LogLoss={res['log_loss']:.4f}"
        )

    return {
        "cv_macro_f1_mean": float(np.mean(fold_f1)),
        "cv_macro_f1_std": float(np.std(fold_f1)),
        "cv_top1_mean": float(np.mean(fold_top1)),
        "cv_top1_std": float(np.std(fold_top1)),
        "cv_log_loss_mean": float(np.mean(fold_ll)),
        "cv_continuity_mean": float(np.mean(fold_cont)),
    }


def plot_member1_results(results_list, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Confusion matrices for Model 1 and Model 2
    for idx, res in enumerate(results_list):
        cm = np.array(res["test"]["confusion_matrix"])
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(f"{res['name']}\nStudent-Test Confusion Matrix")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["0 (False)", "1 (True)"])
        ax.set_yticklabels(["0 (False)", "1 (True)"])
        thresh = cm.max() / 2.0
        for r in range(2):
            for c in range(2):
                color = "white" if cm[r, c] > thresh else "black"
                ax.text(c, r, f"{cm[r, c]:,}", ha="center", va="center", color=color, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Bar chart comparing CV Macro F1, Test Top-1, Test Macro F1, and Route Continuity
    ax_bar = axes[2]
    metrics = ["CV Macro F1", "Test Top-1 Acc", "Test Macro F1", "Route Continuity"]
    x = np.arange(len(metrics))
    width = 0.34
    m1 = results_list[0]
    m2 = results_list[1]
    vals1 = [
        m1["cv"]["cv_macro_f1_mean"],
        m1["test"]["top1_accuracy"],
        m1["test"]["macro_f1"],
        m1["test"]["route_continuity"],
    ]
    vals2 = [
        m2["cv"]["cv_macro_f1_mean"],
        m2["test"]["top1_accuracy"],
        m2["test"]["macro_f1"],
        m2["test"]["route_continuity"],
    ]
    b1 = ax_bar.bar(x - width / 2, vals1, width, label="Model 1: Logistic Regression", color="#2563eb")
    b2 = ax_bar.bar(x + width / 2, vals2, width, label="Model 2: Weighted KNN", color="#16a34a")
    ax_bar.set_ylim(0.0, 1.08)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, rotation=15, ha="right")
    ax_bar.set_ylabel("Score")
    ax_bar.set_title("Member 1 Models Performance Comparison")
    ax_bar.legend(loc="lower right", fontsize=8)
    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def run_member1_models(repo_root):
    proc_dir = os.path.join(repo_root, "data", "processed")
    art_dir = os.path.join(repo_root, "artifacts")
    out_dir = os.path.join(repo_root, "outputs", "models_member1")
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    with open(os.path.join(art_dir, "feature_schema.json"), "r", encoding="utf-8") as f:
        schema = json.load(f)
    feature_cols = schema["feature_columns"]
    scaler = joblib.load(os.path.join(art_dir, "feature_scaler.joblib"))

    print("Loading processed candidate datasets...")
    train_df = joblib.load(os.path.join(proc_dir, "train_candidates.pkl"))
    test_df = joblib.load(os.path.join(proc_dir, "student_test_candidates.pkl"))

    X_train_full = scaler.transform(train_df[feature_cols].values).astype(np.float32)
    y_train_full = train_df["label"].values
    X_test_full = scaler.transform(test_df[feature_cols].values).astype(np.float32)
    y_test_full = test_df["label"].values

    # Build trip-stratified reference subset for KNN (40 obs * 10 cands = 400 rows/trip -> 32,000 rows)
    knn_train_df = sample_trip_stratified_obs(train_df, obs_per_trip=40, seed=42)
    X_train_knn = scaler.transform(knn_train_df[feature_cols].values).astype(np.float32)
    y_train_knn = knn_train_df["label"].values

    model_specs = [
        {
            "name": "Model 1: Logistic Regression",
            "family": "Linear and probabilistic",
            "key_config": "C=1.0, solver=lbfgs, max_iter=300, StandardScaler",
            "cv_df": train_df,
            "X_fit": X_train_full,
            "y_fit": y_train_full,
            "fit_df": train_df,
            "make_fn": lambda: LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=42),
            "artifact_name": "model1_logistic_regression.joblib",
            "pred_csv": "model1_logreg_test_predictions.csv",
        },
        {
            "name": "Model 2: Distance-Weighted KNN",
            "family": "Instance-based",
            "key_config": "k=15, weights=distance, kd_tree, 32k trip-stratified ref",
            "cv_df": knn_train_df,
            "X_fit": X_train_knn,
            "y_fit": y_train_knn,
            "fit_df": knn_train_df,
            "make_fn": lambda: KNeighborsClassifier(
                n_neighbors=15, weights="distance", metric="euclidean", algorithm="kd_tree", leaf_size=50, n_jobs=-1
            ),
            "artifact_name": "model2_distance_weighted_knn.joblib",
            "pred_csv": "model2_knn_test_predictions.csv",
        },
    ]

    all_results = []
    comparison_rows = []

    for spec in model_specs:
        print(f"\n=== Running 5-fold GroupKFold CV for {spec['name']} ===")
        cv_res = run_group_cv(spec["cv_df"], feature_cols, spec["make_fn"], n_splits=5)

        print(f"Fitting final {spec['name']} on training data ({len(spec['X_fit'])} rows)...")
        clf = spec["make_fn"]()
        t0 = time.time()
        clf.fit(spec["X_fit"], spec["y_fit"])
        train_time_s = float(time.time() - t0)

        # Measure training fit performance on the training sample for overfitting diagnostics
        if len(spec["X_fit"]) > 200000:
            diag_df = sample_trip_stratified_obs(spec["fit_df"], obs_per_trip=200, seed=7)
            X_diag = scaler.transform(diag_df[feature_cols].values).astype(np.float32)
            y_diag = diag_df["label"].values
            train_probs = clf.predict_proba(X_diag)[:, 1]
            train_eval = evaluate_predictions(y_diag, train_probs, diag_df)
        else:
            train_probs = clf.predict_proba(spec["X_fit"])[:, 1]
            train_eval = evaluate_predictions(spec["y_fit"], train_probs, spec["fit_df"])

        t1 = time.time()
        test_probs = clf.predict_proba(X_test_full)[:, 1]
        inf_time_s = float(time.time() - t1)
        test_eval = evaluate_predictions(y_test_full, test_probs, test_df)

        # Save model artifact and observation-level predictions
        safe_joblib_dump(clf, os.path.join(art_dir, spec["artifact_name"]), compress=3)
        test_eval["obs_predictions"].to_csv(os.path.join(out_dir, spec["pred_csv"]), index=False)

        print(
            f"  [Train Diag] Macro F1={train_eval['macro_f1']:.4f}, Top-1={train_eval['top1_accuracy']*100:.2f}%\n"
            f"  [5-Fold CV]  Macro F1={cv_res['cv_macro_f1_mean']:.4f} +/- {cv_res['cv_macro_f1_std']:.4f}, "
            f"Top-1={cv_res['cv_top1_mean']*100:.2f}%\n"
            f"  [Test]       Macro F1={test_eval['macro_f1']:.4f}, Top-1={test_eval['top1_accuracy']*100:.2f}%, "
            f"Continuity={test_eval['route_continuity']*100:.2f}%, LogLoss={test_eval['log_loss']:.4f}, "
            f"TrainTime={train_time_s:.2f}s, InfTime={inf_time_s:.2f}s"
        )

        # Strip DataFrame before JSON serialization
        train_summary = {k: v for k, v in train_eval.items() if k != "obs_predictions"}
        test_summary = {k: v for k, v in test_eval.items() if k != "obs_predictions"}

        all_results.append({
            "name": spec["name"],
            "family": spec["family"],
            "key_config": spec["key_config"],
            "artifact_path": f"artifacts/{spec['artifact_name']}",
            "input_preprocessing": "artifacts/feature_scaler.joblib",
            "train_time_sec": round(train_time_s, 3),
            "inference_time_sec": round(inf_time_s, 3),
            "train_diag": train_summary,
            "cv": cv_res,
            "test": test_summary,
        })

        comparison_rows.append({
            "Model": spec["name"],
            "Algorithm Family": spec["family"],
            "Key Configuration": spec["key_config"],
            "CV Macro F1-score": round(cv_res["cv_macro_f1_mean"], 4),
            "Top-1 Accuracy": round(test_eval["top1_accuracy"], 4),
            "Route continuity": round(test_eval["route_continuity"], 4),
            "Student-Test Macro F1-score": round(test_eval["macro_f1"], 4),
            "Log Loss": round(test_eval["log_loss"], 4),
            "Inference Time": f"{inf_time_s:.2f}s",
            "Training Time": f"{train_time_s:.2f}s",
            "Student-Test Precision": round(test_eval["precision"], 4),
            "Student-Test Recall": round(test_eval["recall"], 4),
            "Student-Test Binary Accuracy": round(test_eval["binary_accuracy"], 4),
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_df.to_csv(os.path.join(out_dir, "member1_model_comparison.csv"), index=False)
    with open(os.path.join(out_dir, "member1_model_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    plot_member1_results(all_results, os.path.join(plots_dir, "member1_models_evaluation.png"))
    print("\n=== Member 1 Model Comparison Table ===")
    print(comp_df.to_string(index=False))


if __name__ == "__main__":
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run_member1_models(repo_dir)
