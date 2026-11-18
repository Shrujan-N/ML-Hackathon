"""Compile consolidated Deliverable 2 Technical Report (Markdown + PDF) from verified repo outputs."""
import json
import os
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image as RLImage,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from PIL import Image as PILImage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(rel_path):
    with open(os.path.join(REPO_ROOT, rel_path), "r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(rel_path):
    return pd.read_csv(os.path.join(REPO_ROOT, rel_path), dtype={"trip_id": str})


def fit_image(rel_path, max_width=6.7 * inch, max_height=4.3 * inch):
    full_path = os.path.join(REPO_ROOT, rel_path)
    with PILImage.open(full_path) as im:
        w, h = im.size
    aspect = h / float(w)
    draw_w = max_width
    draw_h = draw_w * aspect
    if draw_h > max_height:
        draw_h = max_height
        draw_w = draw_h / aspect
    return RLImage(full_path, width=draw_w, height=draw_h)


def make_table(data, col_widths, body_style):
    wrapped = []
    for r_idx, row in enumerate(data):
        w_row = []
        for cell in row:
            st = ParagraphStyle(
                "tbl_hdr" if r_idx == 0 else "tbl_cell",
                parent=body_style,
                fontSize=7.5 if r_idx > 0 else 8.0,
                leading=9.5 if r_idx > 0 else 10.0,
                textColor=colors.white if r_idx == 0 else colors.HexColor("#111827"),
                fontName="Helvetica-Bold" if r_idx == 0 else "Helvetica",
            )
            w_row.append(Paragraph(str(cell), st))
        wrapped.append(w_row)

    tbl = Table(wrapped, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


def build_report():
    inv_df = load_csv("outputs/eda/tables/dataset_inventory.csv")
    dq_df = load_csv("outputs/eda/tables/data_quality_report.csv")
    comp_df = load_csv("outputs/evaluation/section_3_1_all_models_comparison.csv")
    hmm_lgbm_df = load_csv("outputs/evaluation/hmm_ablation_lgbm_student_test.csv")
    hmm_pipe_df = load_csv("outputs/evaluation/hmm_pipeline_comparison.csv")
    api_df = load_csv("outputs/evaluation/api_10trip_evaluation.csv")

    schema = load_json("artifacts/feature_schema.json")
    best_params = load_json("outputs/tuning/best_params.json")
    m1_json = load_json("outputs/models_member1/member1_model_comparison.json")
    rf_json = load_json("artifacts/rf_results.json")
    mlp_json = load_json("outputs/member3/logs/mlp_student_test_metrics.json")
    svm_json = load_json("outputs/member3/logs/rbf_svm_student_test_metrics.json")
    final_json = load_json("artifacts/final_metrics.json")

    m1_lr = m1_json[0]
    m1_knn = m1_json[1]
    rf_row = rf_json[0]
    et_row = rf_json[1]

    # Lookup unified rows
    by_key = {r["Model Key"]: r for _, r in comp_df.iterrows()}
    hgb_r = by_key["boosting_hgb"]
    xgb_r = by_key["boosting_xgboost"]
    lgb_u = by_key["boosting_lightgbm"]
    lgb_t = by_key["model4_boosting_lightgbm_tuned"]
    mlp_r = by_key["model5_mlp_neural_network"]
    svm_r = by_key["model6_rbf_svm_nystroem"]
    rf_r = by_key["model3a_random_forest"]
    et_r = by_key["model3b_extra_trees"]
    lr_r = by_key["model1_logistic_regression"]
    knn_r = by_key["model2_distance_weighted_knn"]

    # Build Markdown version first
    md_sections = r"""# PathSnap: Distributed Map-Matching & Route Reconstruction
**PES University — Department of Computer Science and Engineering (RR Campus)**  
**Course:** UE24CS352A: Machine Learning | **Team Size:** 4 Members  
**GitHub Repository:** https://github.com/Shrujan-N/ML-Hackathon  

---

## 2.2 Dataset Understanding and Exploratory Data Analysis

### 1. Partition Scope & Leakage Prevention
All data exploration, cleaning, candidate generation, and model development were restricted to the **80 `train` trips** (`208,417` GPS observations) and **10 `student_test` trips** (`19,160` GPS observations). The **10 `evaluator_test` trips** were kept untouched and never inspected during preprocessing, feature selection, or hyperparameter tuning.

| Partition | Trips | Files | Total GPS Points | Mean / Min / Max GPS | Total Nodes | Total Arcs | Route Arcs | Median Offset | 95th Pct Offset |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 80 | 320 | 208,417 | 2,605.2 / 361 / 19,313 | 1,933,485 | 3,876,904 | 31,336 | 5.67 m | 141.23 m |
| `student_test` | 10 | 40 | 19,160 | 1,916.0 / 262 / 3,958 | 90,689 | 185,041 | 3,477 | 5.25 m | 16.80 m |

### 2. Data Quality & Missing-Value Analysis
- **Missing & Timestamp Integrity (`train` and `student_test`):** `0` NaN/missing coordinates and `0` non-monotonic timestamps (`dt <= 0`) across all 90 trips.
- **Sampling Frequency & Dropouts:** `96.50%` of `train` steps and `98.62%` of `student_test` steps are exact 1.0 s intervals (`1 Hz`). However, `train` contains `2,937` steps with `dt > 2 s` (`1,216` over `5 s`, max gap `153,064.0 s`), and `student_test` contains `138` steps over `2 s` (`44` over `5 s`, max gap `210.0 s` in trip `00000065`).
- **Graph & Reference Route Quirks:** Every road network has `0` degree-0 isolated nodes, yet `79/80` `train` graphs and `10/10` `student_test` graphs consist of multiple disconnected components. Consecutive `.route` entries also contain `13` disconnected arc transitions in `train` and `1` in `student_test`.
- **High-Offset Anomalies:** Two `train` trips (`00000033` with median offset `121.66 m`, and `00000054` with median offset `139.22 m`) plus an unmapped loop in `00000002` contain long off-route GPS segments more than 60 m away from `.route`.

### 3. Preprocessing Pipeline (`src/data/data_loader.py` & `src/features/feature_engineering.py`)
1. **Local Meter Projection:** Since raw `.track` and `.nodes` store `(longitude, latitude)` in angular degrees, ordinary Euclidean distance in degrees distorts East-West distances by `cos(lat)`. We project each trip onto a local tangent plane centered on its mean coordinates: `x_m = (lon - lon0) * 111320 * cos(radians(lat0))`, `y_m = (lat - lat0) * 110540`.
2. **Track-to-Route Monotonic DP Alignment:** Because `.track` has 5x to 30x more rows than `.route` (e.g., `2,503` GPS points vs. `179` route arcs in `student_test/00000000`), we align each GPS point to its traversed reference `arc_id` via monotonic dynamic programming over point-to-segment distances with a gentle directional penalty when moving (`speed > 1.5 m/s`).
3. **cKDTree Candidate Generation (`K = 10`):** For every GPS point, a `scipy.spatial.cKDTree` over segment midpoints retrieves `100` nearby arcs, which are refined by exact perpendicular point-to-segment distance to keep the top `10` candidates. `student_test` retains 100% of observations with a **Candidate Miss Rate of `0.511%`** (`99.489%` recall). On `train` only, points > 60 m from `.route` were excluded (`208,417 -> 186,093` obs), lowering active `train` Candidate Miss Rate from `7.856%` to **`1.350%`**.
4. **Leakage-Free Scaling:** `StandardScaler` (`artifacts/feature_scaler.joblib`) was fit strictly on `train`.

### 4. Engineered Features (16 Features)
- **Proximity & Margin:** `dist_to_arc_m`, `dist_min_diff_m`, `dist_ratio_to_min`, `cand_rank` (strongest negative correlation $r = -0.44$ with `label`).
- **Segment Geometry:** `proj_fraction`, `dist_to_from_node_m`, `dist_to_to_node_m`, `arc_length_m`.
- **Kinematic Directionality:** `heading_diff_rad` ($r = -0.30$), `heading_cos` ($r = +0.29$, resolves opposite-direction lanes on two-way streets), `heading_sin`, `gps_speed_mps`, `dt_prev_s` (using central differences and frozen heading when displacement <= 0.5 m).
- **Temporal & Graph Continuity:** `prev_gps_dist_to_arc_m`, `next_gps_dist_to_arc_m`, `is_connected_from_prev_top1` ($r = +0.31$).

---

## 3. Model Experimentation and Comparison

We trained and evaluated models across **all 6 algorithm families** (`Linear/Probabilistic`, `Instance-based`, `Ensemble`, `Boosting`, `Neural networks`, and `Kernel-based`).

### 3.1 Overall Model Comparison (`student_test`: 10 trips / 19,160 observations / 191,600 candidate rows)

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy | Route Continuity | Student-Test Macro F1 (Arc / Binary) | Log Loss | Inference Time |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Logistic Regression** | Linear & probabilistic | `C=1.0, lbfgs, max_iter=300, StandardScaler` | 0.8975 (Bin) | 85.33% | 93.08% | 0.7362 / 0.9075 | 0.0941 | 0.03s |
| **Model 2: Distance-Weighted KNN** | Instance-based | `k=15, distance, kd_tree, 32k trip ref` | 0.8922 (Bin) | 83.92% | 91.70% | 0.7067 / 0.9021 | 0.2225 | 12.12s |
| **Model 3a: Random Forest** | Ensemble (Bagging) | `n_estimators=200, class_weight=balanced` | N/A (10-trip val: 0.6780 Bin) | 85.59% | 92.34% | 0.7561 / 0.8988 | 0.1676 | 0.52s |
| **Model 3b: Extra Trees** | Ensemble (Bagging) | `n_estimators=200, class_weight=balanced` | N/A (10-trip val: 0.6878 Bin) | 85.35% | 92.18% | 0.7510 / 0.9006 | 0.1823 | 0.51s |
| **Model 4b: HistGradientBoosting** | Boosting | `max_iter=300, lr=0.1, max_leaf_nodes=63` | 0.7708 (Arc) | **86.21%** | 93.68% | **0.7643** / N/A | 0.0701 | 0.75s |
| **Model 4c: XGBoost** | Boosting | `n_estimators=300, lr=0.1, max_depth=8` | 0.7723 (Arc) | 86.08% | 93.77% | 0.7606 / N/A | 0.0704 | 0.17s |
| **Model 4d: LightGBM (Untuned)** | Boosting | `n_estimators=300, lr=0.1, num_leaves=63` | 0.7712 (Arc) | 86.07% | 93.68% | 0.7618 / N/A | 0.0703 | 0.51s |
| **Model 4: LightGBM (Optuna Tuned)** | Boosting | `n_estimators=550, lr=0.0289, leaves=101` | 0.8519 (Top-1, 3-fold) | 85.97% | 93.66% | 0.7614 / **0.9197** | **0.0697** | 1.23s |
| **Model 5: MLP Neural Network** | Neural networks | `128-64-32, BatchNorm, Dropout=0.15, AdamW` | N/A (15% trip val: 0.9241 Bin) | 85.70% | **93.81%** | 0.7524 / 0.9186 | 0.0719 | 0.10s |
| **Model 6: RBF SVM (Nyström)** | Kernel-based | `Nystrom(256), SGD hinge + Platt` | N/A (full-fit) | 84.93% | 93.19% | 0.7334 / 0.9038 | 0.0935 | 1.15s |

*(Note on Macro F1 columns: Member 2's shared harness computes `Arc Macro F1` over predicted vs. true `arc_id` classes, while candidate-row classification reports `Binary Macro F1` over `label in {0, 1}`. Both are reported above for full transparency. `N/A` indicates models evaluated via a single group-aware validation split or full-train fit due to compute time, such as 1,236s per MLP run.)*

### 3.2 Compulsory Analysis for Each Model

#### Model 1: Logistic Regression (`src/models/logistic_regression/`)
- **1. Suitability:** Candidate features exhibit strong monotonic trends (`cand_rank` $r=-0.44$, `is_connected_from_prev_top1` $r=+0.31$, `heading_cos` $r=+0.29$), making linear logistic regression a fast, well-calibrated baseline for HMM emissions.
- **2. Strengths & Limitations:** Extremely fast inference (`0.03s` for 191,600 rows) and low Log Loss (`0.0941`), but cannot capture non-linear feature interactions (e.g., ignoring noisy `heading_cos` when `gps_speed_mps < 1.0`).
- **5. Hyperparameters:** `C=1.0`, `solver='lbfgs'`, `max_iter=300`, `random_state=42`, with `StandardScaler`.
- **6. Optimisation Objective:** L2-regularized binary cross-entropy minimized across `1,860,930` training rows.
- **7 & 8. Student-Test Results:** Top-1 Accuracy = `85.33%`, Binary Macro F1 = `0.9075`, Arc Macro F1 = `0.7362`, Log Loss = `0.0941`, Continuity = `93.08%`.
- **9. Overfitting / Underfitting:** Zero overfitting (`train` diagnostic Macro F1 `0.9047`, 5-fold CV Macro F1 `0.8975 ± 0.0169`, `student_test` `0.9075`), mild underfitting compared to boosting/MLP due to linear capacity.

#### Model 2: Distance-Weighted KNN (`src/models/knn/`)
- **1. Suitability:** Compares test candidates directly against local geometric/kinematic neighborhoods (urban parallel streets vs. highway segments).
- **2. Strengths & Limitations:** Zero parametric assumptions and `0.07s` fit time, but slow inference (`12.12s` on `student_test`), sensitive to noisy coordinates in 16D unweighted Euclidean space, and poorly calibrated probability extremes (`Log Loss = 0.2225`).
- **5. Hyperparameters:** `n_neighbors=15`, `weights='distance'`, `metric='euclidean'`, `algorithm='kd_tree'`, `leaf_size=50`, fit on a trip-stratified `32,000`-row reference set (`40` obs/trip across all 80 train trips).
- **6. Optimisation Objective:** Non-parametric instance storage; inverse-distance weighted class voting at inference.
- **7 & 8. Student-Test Results:** Top-1 Accuracy = `83.92%`, Binary Macro F1 = `0.9021`, Arc Macro F1 = `0.7067`, Log Loss = `0.2225`, Continuity = `91.70%`.
- **9. Overfitting / Underfitting:** Local memorization on training points (`Train Diag Binary Macro F1 = 1.0000`, `Top-1 = 98.06%`) drops to `0.8922` 5-fold CV Macro F1 and `83.92%` test Top-1 due to `weights='distance'` assigning infinite self-weight on exact training vectors.

#### Model 3: Random Forest & Extra Trees (`src/models/random_forest/`)
- **1. Suitability:** Tree ensembles handle mixed feature scales (meters, radians, binary connectivity) without scaling and model non-linear intersection rules natively.
- **2. Strengths & Limitations:** Solid Top-1 accuracy (`85.59%` RF / `85.35%` ET) and fast sub-second inference (`0.52s`), but `class_weight='balanced'` shifts posterior probabilities upward, inflating Log Loss (`0.1676` RF, `0.1823` ET) relative to uncalibrated boosting/MLP.
- **5. Hyperparameters:** `n_estimators=200`, `class_weight='balanced'`, `max_depth=None`, `random_state=42`.
- **6. Optimisation Objective:** Gini impurity splitting averaged across 200 bootstrap/random-threshold decision trees.
- **7 & 8. Student-Test Results:** Random Forest achieved Top-1 = `85.59%`, Binary Macro F1 = `0.8988`, Arc Macro F1 = `0.7561`, Log Loss = `0.1676`, Continuity = `92.34%`. Extra Trees achieved Top-1 = `85.35%`, Binary Macro F1 = `0.9006`, Arc Macro F1 = `0.7510`, Log Loss = `0.1823`, Continuity = `92.18%`.
- **9. Overfitting / Underfitting:** Unbounded depth (`max_depth=None`) + 50k sample training in `train_rf.py` produced noticeable gap between sample validation (`57.23%` Top-1 on the 1-trip split) and full `student_test` (`85.59%`), plus higher Log Loss from unpruned leaves and balanced class weighting.

#### Model 4: Gradient Boosting — LightGBM, XGBoost & HistGradientBoosting (`src/models/boosting/`)
- **1. Suitability:** Sequential gradient-boosted decision trees learn multiplicative interactions (`dist_to_arc_m * heading_cos * gps_speed_mps`) and output sharp, well-calibrated log-odds probabilities ideal for HMM emission weighting.
- **2. Strengths & Limitations:** Highest Top-1 Accuracy (`86.21%` HGB, `86.08%` XGBoost, `86.07%` LightGBM), lowest Log Loss (`0.0697` Tuned LightGBM), and sub-second inference (`0.17s`–`1.23s`). Limitation is sensitivity to learning-rate/tree-complexity tuning to avoid leaf overfitting.
- **5. Hyperparameters:** Defaults: LightGBM (`n_estimators=300, lr=0.1, num_leaves=63, min_child_samples=50`), XGBoost (`n_estimators=300, lr=0.1, max_depth=8`), HGB (`max_iter=300, lr=0.1, max_leaf_nodes=63`). Tuned LightGBM (`n_estimators=550, lr=0.0289, num_leaves=101, min_child_samples=39`).
- **6. Optimisation Objective:** Binary logistic loss (`binary:logistic` / `logloss`) minimized via gradient boosting.
- **7 & 8. Student-Test Results:** HGB reached Top-1 = `86.21%`, Arc F1 = `0.7643`, Log Loss = `0.0701`, Continuity = `93.68%`; Tuned LightGBM reached Top-1 = `85.97%`, Binary F1 = `0.9197`, Arc F1 = `0.7614`, Log Loss = `0.0697`, Continuity = `93.66%`.
- **9. Overfitting / Underfitting:** Minimal overfitting across 5-fold GroupKFold CV: CV Top-1 (`84.30%`–`84.41%`) tracks `student_test` (`86.07%`–`86.21%`) closely, with fold std `±2.95%` reflecting trip-level GPS noise variation across unseen cities.

#### Model 5: PyTorch MLP Neural Network (`src/models/mlp/`)
- **1. Suitability:** Smooth continuous non-linear MLP with BatchNorm and Dropout learns joint geometric-kinematic representations and outputs calibrated sigmoid probabilities.
- **2. Strengths & Limitations:** Strongest Route Continuity (`93.81%`), second-best Log Loss (`0.0719`), and fast GPU/CPU inference (`0.10s`), at the cost of long training time (`1,235.9 s` over 30 epochs on CUDA).
- **5. Hyperparameters:** `hidden_dims=(128, 64, 32)`, `BatchNorm1d`, `ReLU`, `dropout=0.15`, `AdamW(lr=1e-3, weight_decay=1e-4)`, `batch_size=2048`, `max_epochs=30`, `patience=5`, `StandardScaler`.
- **6. Optimisation Objective:** Unweighted `BCEWithLogitsLoss` (chosen intentionally over class-weighted BCE to preserve true posterior calibration for HMM decoding), scheduled with `ReduceLROnPlateau`.
- **7 & 8. Student-Test Results:** Top-1 Accuracy = `85.70%`, Binary Macro F1 = `0.9186`, Arc Macro F1 = `0.7524`, Log Loss = `0.0719`, Continuity = `93.81%`, Precision = `0.8669`, Recall = `0.8397`.
- **9. Overfitting / Underfitting:** Well-regularized convergence: best validation epoch was Epoch 28 (`val_loss = 0.0708`, `val_macro_f1 = 0.9241`), closely matching `student_test` (`log_loss = 0.0719`, `macro_f1 = 0.9186`) without divergence.

#### Model 6: Kernel-Based RBF SVM (`src/models/rbf_svm/`)
- **1. Suitability:** Maps standardized features into a non-linear Gaussian kernel space to separate parallel/curved road geometry.
- **2. Strengths & Limitations:** Exact SVC scales quadratically ($O(N^2)$) and cannot run on `1.86M` rows; our `Nystroem(256)` + chunked SGD hinge + Platt calibration scales linearly (`125.7s` train, `1.15s` test inference) and beats linear LogReg in Log Loss (`0.0935` vs. `0.0941`), but 256 Nyström components slightly underfit complex intersections compared to 550 boosted trees (`84.93%` vs. `85.97%`).
- **5. Hyperparameters:** `nystroem_components=256`, `nystroem_fit_samples=50,000`, `C=1.0`, `sgd_epochs=6`, `chunk_size=50,000`, `calibration_fraction=0.10`.
- **6. Optimisation Objective:** L2-regularized hinge loss (`SGDClassifier(loss='hinge')`) over 256 RBF Nyström features, followed by Platt logistic calibration on a 10% holdout split.
- **7 & 8. Student-Test Results:** Top-1 Accuracy = `84.93%`, Binary Macro F1 = `0.9038`, Arc Macro F1 = `0.7334`, Log Loss = `0.0935`, Continuity = `93.19%`.
- **9. Overfitting / Underfitting:** Mild underfitting due to finite 256-component kernel rank; `student_test` Log Loss (`0.0935`) stays close to Logistic Regression without high-variance overfitting.

---

## 4. Bayesian Hyperparameter Optimisation (Optuna TPE on LightGBM)

1. **Search Space & Distributions (`src/tuning/bayes_opt.py`):**
   - `n_estimators`: int `[100, 600]` (step 50)
   - `learning_rate`: log-uniform `[0.02, 0.30]`
   - `num_leaves`: log-int `[15, 255]`
   - `min_child_samples`: log-int `[10, 200]`
   - `subsample`: uniform `[0.50, 1.00]`
   - `colsample_bytree`: uniform `[0.50, 1.00]`
   - `reg_alpha`: log-uniform `[1e-3, 10.0]`
   - `reg_lambda`: log-uniform `[1e-3, 10.0]`
2. **Optimisation Objective & Partitioning:** Maximize 3-fold `GroupKFold` CV Top-1 Accuracy on a deterministic 50% training trip subset (`40` train trips, `seed=42`). `student_test` was never touched during tuning.
3. **Trials & Pruning:** **30 total trials** using `TPESampler` (`8` startup trials) and `MedianPruner`: **26 completed**, **4 pruned** (`trials 8, 9, 14, 28`).
4. **Best Parameters (Trial 20, CV Top-1 = `0.8519`):** `n_estimators=550`, `learning_rate=0.0289`, `num_leaves=101`, `min_child_samples=39`, `subsample=0.8604`, `colsample_bytree=0.8964`, `reg_alpha=0.0074`, `reg_lambda=0.0044`.
5. **Untuned vs. Tuned LightGBM Comparison (Full 80-Trip Fit -> `student_test`):**

| Configuration | CV Top-1 (Subset/5-Fold) | Student-Test Top-1 | Student-Test Arc F1 | Student-Test Log Loss | Route Continuity | Fit Time | Infer Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Untuned LightGBM** | 0.8430 (5-fold) | **86.07%** | **0.7618** | 0.0703 | **93.68%** | **10.45s** | **0.51s** |
| **Optuna-Tuned LightGBM** | **0.8519** (3-fold sub) | 85.97% | 0.7614 | **0.0697** | 93.66% | 22.96s | 1.23s |

**Did Tuned Outperform Untuned?** Tuning improved probability calibration (`Log Loss` improved from `0.0703` to **`0.0697`** — lowest of any model) and reached `0.8519` on the 3-fold tuning subset, while untuned LightGBM retained a negligible `+0.10%` edge in standalone Top-1 (`86.07%` vs `85.97%`) and 2x faster training/inference. Because calibrated log-odds directly feed the HMM emission layer, the tuned LightGBM bundle was exported to `artifacts/final_model.joblib`.

---

## 5. HMM and Viterbi Route Reconstruction

1. **Hidden States:** Each observation `t` has hidden states corresponding to its top-`K=10` candidate road arcs `candidate_arc_id` from Member 1's candidate generator (`src/hmm/viterbi.py`).
2. **Emission Probabilities:** Derived in log-space from classifier posterior probabilities: `log P(z_t | e_i) = w_em * log(max(p_t(e_i), 1e-6))` with `w_em = 1.0`, falling back to Gaussian distance likelihood (`sigma_z = 10.0 m`) if classifier probabilities are missing.
3. **Transition Probabilities:** Following Newson & Krumm (2009), shortest directed network distance `d_route(e_i, e_j)` between projections on consecutive candidate arcs is compared against great-circle GPS distance `d_gps(z_{t-1}, z_t)`: `log P(e_j | e_i) = -|d_route - d_gps| / beta_eff`, with `beta = 5.0 m`, same-arc stability bonus `+0.25`, and U-turn reverse penalty `-12.0`.
4. **Missing Candidates & GPS Gaps:** Missing candidate sets trigger `carry_forward` from the previous valid step. For GPS gaps (`dt > 5.0 s`), `beta_eff` relaxes by `0.5` per second and Dijkstra's cutoff expands dynamically up to `5,000 m`. Extreme outages (`dt > 60 s` or jump `> 2,500 m`) split the Viterbi chain to avoid stale drift.
5. **Disconnected Road Segments:** Rather than crashing on `-inf` transitions caused by map breaks or one-way errors, unreachable pairs receive a finite penalty (`-25.0`), keeping Viterbi decoding robust even across the `1` disconnected `.route` transition in `student_test`.
6. **Experimental Ablation (All 10 `student_test` Trips):**

| Pipeline | Classifier-Only Top-1 | Classifier + HMM Top-1 | Obs Continuity | Full Route Continuity | Route LCS F1 | Decode Time |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Tuned LightGBM + HMM** | 85.97% | **88.95%** (+2.98%) | **96.82%** | **99.52%** | **88.64%** | 5.20s total |
| **MLP (Model 5) + HMM** | 85.70% | 88.32% (+2.62%) | 96.76% | 99.46% | 88.04% | 11.12s total |

- **Key Improvements:** Every single trip except `00000091` improved substantially with HMM decoding: e.g., `00000007` jumped from `92.84%` to **`99.78%`** (`+6.94%`, `100%` continuity), `00000065` (which has a `210 s` GPS gap) jumped from `86.20%` to **`91.41%`** (`+5.21%`), and `00000077` rose from `70.70%` to **`74.92%`** (`+4.22%`, LCS F1 `29.75% -> 53.92%`).
- **Failure Case (`00000091`):** Top-1 decreased slightly (`85.81% -> 83.30%`, `-2.52%`) while Route Continuity improved (`90.26% -> 94.96%`) and LCS F1 surged (`68.98% -> 87.62%`), because the HMM smoothed an ambiguous parallel-lane fork onto a connected path where our monotonic DP label assigned the adjacent service segment.

---

## 6. Geographical Visualisation Using Folium

Interactive multi-layer HTML maps (`src/visualization/folium_map.py` and `src/data/eda_analysis.py`) are committed in `outputs/eda/maps/` and `outputs/maps/`:
- `outputs/eda/maps/eda_map_train_00000001.html` & `eda_map_student_test_00000000.html`
- `outputs/maps/student_test_00000007_lgbm_hmm.html` & `student_test_00000065_lgbm_hmm.html`
- `outputs/maps/student_test_00000000_mlp_hmm.html`, `00000007_mlp_hmm.html`, `00000065_mlp_hmm.html`, `00000076_mlp_hmm.html`

Each map plots **Raw GPS Observations** (blue markers), **Relevant Road Network** (gray lines), **Reference Route** (green polyline), **Classifier-Only Route** (dashed orange), and **Predicted HMM Route** (solid red), with automatic `fit_bounds`, custom HTML legend, and `folium.LayerControl`.

---

## 7. API, UI and Deployment

- **Laptop 1 (`api/main.py`):** FastAPI server on `0.0.0.0:8000` serving `GET /`, `POST /predict`, `POST /evaluate` (server-side evaluation where `true_arc_id` never returns to the UI), and `GET /logs` (ring buffer + stdout CMD logs).
- **Laptop 2 (`ui/app.py`):** Streamlit Evaluator UI displaying all 10 trips, single-trip prediction & Folium map visualization, and a 10-trip LAN stress-test dashboard.
- **Live 10-Trip Verification (`outputs/evaluation/api_10trip_evaluation.csv`):** All 10 `student_test` trips passed `200 OK` across both `/predict` and `/evaluate`.

---

## 8. Academic Integrity & AI Assistance Disclosure

All code, training scripts, hyperparameter trials, HMM decoding ablations, Folium maps, and API logs in this report were independently executed by our 4-member team on the permitted `train` (80 trips) and `student_test` (10 trips) partitions. No `evaluator_test` labels or cross-team artifacts were accessed. AI assistance was used for code structuring, boilerplate refactoring, and formatting assistance, while all experiments, metric verification, and technical explanations were validated directly against repository logs by the team.
"""

    md_path = os.path.join(REPO_ROOT, "PathSnap_Technical_Report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_sections)

    # Compile PDF via ReportLab
    os.makedirs(os.path.join(REPO_ROOT, "report"), exist_ok=True)
    pdf_paths = [
        os.path.join(REPO_ROOT, "PathSnap_Technical_Report.pdf"),
        os.path.join(REPO_ROOT, "report", "PathSnap_Technical_Report.pdf"),
    ]

    styles = getSampleStyleSheet()
    title_st = ParagraphStyle("TitleCustom", parent=styles["Title"], fontSize=18, leading=22, textColor=colors.HexColor("#1e3a8a"), spaceAfter=6)
    sub_st = ParagraphStyle("SubCustom", parent=styles["Normal"], fontSize=10, leading=13, textColor=colors.HexColor("#475569"), spaceAfter=12)
    h1_st = ParagraphStyle("H1Custom", parent=styles["Heading1"], fontSize=13.5, leading=17, textColor=colors.HexColor("#1e3a8a"), spaceBefore=12, spaceAfter=6)
    h2_st = ParagraphStyle("H2Custom", parent=styles["Heading2"], fontSize=11, leading=14, textColor=colors.HexColor("#0f172a"), spaceBefore=8, spaceAfter=4)
    h3_st = ParagraphStyle("H3Custom", parent=styles["Heading3"], fontSize=9.8, leading=12.5, textColor=colors.HexColor("#1e40af"), spaceBefore=6, spaceAfter=3)
    body_st = ParagraphStyle("BodyCustom", parent=styles["Normal"], fontSize=9.2, leading=12.5, spaceAfter=5, textColor=colors.HexColor("#1f2937"))
    cap_st = ParagraphStyle("CapCustom", parent=styles["Italic"], fontSize=8.2, leading=10.5, alignment=1, textColor=colors.HexColor("#475569"), spaceAfter=8)

    story = []

    def add_p(txt, style=body_st):
        story.append(Paragraph(txt, style))

    def add_fig(rel_path, caption, max_h=3.6 * inch):
        story.append(Spacer(1, 4))
        story.append(fit_image(rel_path, max_width=6.7 * inch, max_height=max_h))
        story.append(Paragraph(caption, cap_st))

    # Header
    add_p("PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION", title_st)
    add_p("<b>PES University | Department of CSE (RR Campus) | UE24CS352A: Machine Learning</b><br/>"
          "Consolidated Team Technical Report | Public GitHub: https://github.com/Shrujan-N/ML-Hackathon", sub_st)

    # Section 2.2
    add_p("2.2 Dataset Understanding and Exploratory Data Analysis", h1_st)
    add_p("All exploration, preprocessing, candidate generation, and model development were strictly limited to the "
          "<b>80 train trips</b> (208,417 GPS points) and <b>10 student_test trips</b> (19,160 GPS points). "
          "The 10 evaluator_test trips remained completely untouched to prevent any data leakage.")

    inv_tbl_data = [
        ["Partition", "Trips", "Files", "Total GPS", "Mean / Min / Max GPS", "Nodes", "Directed Arcs", "Route Arcs", "Med / p95 Offset"]
    ]
    for _, r in inv_df.iterrows():
        inv_tbl_data.append([
            r["partition"], int(r["n_trips"]), int(r["total_files"]), f"{int(r['total_gps_points']):,}",
            f"{r['mean_gps_per_trip']} / {int(r['min_gps_per_trip'])} / {int(r['max_gps_per_trip'])}",
            f"{int(r['total_nodes']):,}", f"{int(r['total_arcs']):,}", f"{int(r['total_route_arcs']):,}",
            f"{r['median_noise_dist_m']}m / {r['p95_noise_dist_m']}m"
        ])
    story.append(make_table(inv_tbl_data, [0.8*inch, 0.45*inch, 0.45*inch, 0.72*inch, 1.05*inch, 0.72*inch, 0.8*inch, 0.65*inch, 1.06*inch], body_st))
    story.append(Spacer(1, 6))

    dq_tbl_data = [["Partition", "NaN/Missing", "Non-Mono dt", "% Exact 1s", "Gaps >2s / >5s", "Max Gap (s)", "Iso Nodes", "Multi-Comp Trips", "Disc. .route", "Offset >30m"]]
    for _, r in dq_df.iterrows():
        dq_tbl_data.append([
            r["partition"], int(r["missing_or_nan_values"]), int(r["non_monotonic_timestamps"]),
            f"{r['pct_steps_exactly_1s']}%", f"{int(r['steps_gt_2s'])} / {int(r['steps_gt_5s'])}",
            f"{r['max_single_gap_s']}", int(r["total_isolated_nodes"]), int(r["trips_with_multi_components"]),
            int(r["disconnected_route_transitions"]), int(r["trips_with_median_offset_gt_30m"])
        ])
    story.append(make_table(dq_tbl_data, [0.78*inch, 0.62*inch, 0.65*inch, 0.62*inch, 0.78*inch, 0.68*inch, 0.55*inch, 0.72*inch, 0.65*inch, 0.65*inch], body_st))
    story.append(Spacer(1, 6))

    add_p("1. How did we preprocess the data?", h2_st)
    add_p("<b>Coordinate Projection:</b> Raw .track and .nodes store (longitude, latitude) in angular degrees. Because 1 degree of longitude "
          "shrinks by cos(latitude), Euclidean degree distance is physically invalid. We project every trip to local tangent-plane meters "
          "centered on its mean coordinate: x_m = (lon - lon0) * 111320 * cos(lat0), y_m = (lat - lat0) * 110540.<br/>"
          "<b>Track-to-Route Monotonic DP Alignment:</b> .track contains 5x to 30x more rows than .route (e.g., student_test/00000000 has 2,503 GPS rows vs. 179 route arcs). "
          "We aligned each GPS point to its reference arc_id via chunked monotonic dynamic programming with perpendicular distance plus a directional penalty when moving.<br/>"
          "<b>cKDTree Candidate Generation & Cleaning:</b> Using scipy.spatial.cKDTree over segment midpoints (M_pre=100 refined to top K=10), "
          "student_test retained 100% of observations with <b>0.511% Candidate Miss Rate</b> (99.49% recall). In train only, off-route outlier points >60m from .route "
          "(mostly parking/detour drift in trips 00000033 and 00000054) were filtered (208,417 -> 186,093 obs), cutting train miss rate from 7.856% to <b>1.350%</b>. "
          "StandardScaler (artifacts/feature_scaler.joblib) was fit strictly on train.", body_st)

    add_p("2. Observed Patterns & 16 Engineered Features", h2_st)
    add_p("<b>Patterns Observed:</b> Zero missing/NaN values and strictly monotonic timestamps across all 90 trips. "
          "96.50% (train) and 98.62% (student_test) of steps occur at 1 Hz, with intermittent dropouts up to 153,064 s (train) and 210 s (student_test). "
          "79/80 train graphs have multiple disconnected components and 13 reference .route steps are non-adjacent, showing that supplied routes require fallback handling.<br/>"
          "<b>Extracted Features (16):</b> (i) Spatial proximity: dist_to_arc_m, dist_min_diff_m, dist_ratio_to_min, cand_rank (r=-0.44); "
          "(ii) Geometry: proj_fraction, dist_to_from_node_m, dist_to_to_node_m, arc_length_m; "
          "(iii) Kinematics: heading_diff_rad (r=-0.30), heading_cos (r=+0.29, separates forward vs. opposite two-way lanes), heading_sin, gps_speed_mps, dt_prev_s; "
          "(iv) Temporal & Topological: prev_gps_dist_to_arc_m, next_gps_dist_to_arc_m, is_connected_from_prev_top1 (r=+0.31).", body_st)

    add_fig("outputs/eda/plots/01_trip_length_and_duration.png", "Figure 2.1: Trip observation count and spatial span distributions (train n=80 vs student_test n=10).", max_h=2.5*inch)
    add_fig("outputs/eda/plots/02_sampling_intervals_and_gaps.png", "Figure 2.2: 1 Hz sampling interval peak and GPS signal dropout tail (>2s).", max_h=2.5*inch)
    add_fig("outputs/eda/plots/03_road_network_and_gps_noise.png", "Figure 2.3: Road network graph scale and CDF of GPS noise distance to reference route.", max_h=2.5*inch)
    add_fig("outputs/eda/plots/04_sample_trips_overlay.png", "Figure 2.4: Road network, reference route, and raw GPS overlay (note unmapped detour in train/00000002).", max_h=3.4*inch)
    add_fig("outputs/eda/plots/05_feature_distributions_and_correlations.png", "Figure 2.5: Feature separation (label=1 vs label=0) and Pearson correlation with target.", max_h=3.3*inch)

    story.append(PageBreak())

    # Section 3.1
    add_p("3. Model Experimentation and Comparison", h1_st)
    add_p("3.1 Overall Model Comparison (student_test: 10 trips, 19,160 obs, 191,600 candidate rows)", h2_st)
    add_p("We trained and compared models across <b>all 6 algorithm families</b>. "
          "Both Arc Macro F1 (Member 2 shared harness over arc_id classes) and Binary Macro F1 (row-level candidate classification) are reported transparently.", body_st)

    s31_tbl = [["Model", "Family", "Key Config", "CV Macro F1", "Top-1 Acc", "Route Cont.", "Student-Test Macro F1", "Log Loss", "Infer Time"]]
    for _, r in comp_df.iterrows():
        cv_f1 = "N/A (full-fit)"
        if pd.notna(r["CV Binary Macro F1"]):
            cv_f1 = f"{r['CV Binary Macro F1']:.4f} (Bin)"
        elif pd.notna(r["CV Arc Macro F1"]):
            cv_f1 = f"{r['CV Arc Macro F1']:.4f} (Arc)"
        elif r["Model Key"] == "model4_boosting_lightgbm_tuned":
            cv_f1 = "0.8519 (Top1, 3f)"
        bf1 = f"{r['Binary Macro F1']:.4f}" if pd.notna(r["Binary Macro F1"]) else "-"
        st_f1 = f"{r['Arc Macro F1']:.4f} / {bf1}"
        s31_tbl.append([
            r["Model"], r["Algorithm Family"], r["Key Configuration"], cv_f1,
            f"{r['Top-1 Accuracy']*100:.2f}%", f"{r['Route Continuity']*100:.2f}%",
            st_f1, f"{r['Log Loss']:.4f}", f"{r['Inference Time (s)']:.2f}s"
        ])
    story.append(make_table(s31_tbl, [1.05*inch, 0.72*inch, 1.25*inch, 0.72*inch, 0.55*inch, 0.58*inch, 0.75*inch, 0.52*inch, 0.56*inch], body_st))
    story.append(Spacer(1, 8))

    # Section 3.2
    add_p("3.2 Compulsory Analysis for Every Model", h2_st)

    model_blocks = [
        ("Model 1: Logistic Regression (Linear & Probabilistic)",
         "Suitability & Strengths/Limitations: Strong linear correlations (cand_rank r=-0.44, is_connected r=+0.31, heading_cos r=+0.29) allow Logistic Regression to produce calibrated probabilities (Log Loss 0.0941) in just 0.03s inference. Limitation: linear boundary misses non-linear stationary-heading interactions.",
         "Config & Objective: C=1.0, solver=lbfgs, max_iter=300, StandardScaler fit on train/fold. Minimized L2-regularized binary cross-entropy across 1,860,930 rows (12.78s fit).",
         f"Experimental Results & Overfitting Check: student_test Top-1={lr_r['Top-1 Accuracy']*100:.2f}%, Arc F1={lr_r['Arc Macro F1']:.4f}, Bin F1={lr_r['Binary Macro F1']:.4f}, LogLoss={lr_r['Log Loss']:.4f}, Cont={lr_r['Route Continuity']*100:.2f}%. No overfitting (Train Diag Bin F1=0.9047, 5-fold CV Bin F1=0.8975±0.0169, Test=0.9075); mild linear underfitting."),
        ("Model 2: Distance-Weighted KNN (Instance-Based)",
         "Suitability & Strengths/Limitations: Matches queries directly against local road-regime neighborhoods (urban parallel roads vs highways) with 0.07s fit time. Limitations: 12.12s inference time (~360x slower than LogReg) and poorly calibrated extreme weights (Log Loss 0.2225) in unweighted 16D Euclidean space.",
         "Config & Objective: k=15, weights='distance', metric='euclidean', kd_tree (leaf_size=50) on a trip-stratified 32,000-row reference set (40 obs/trip across 80 train trips). Non-parametric inverse-distance voting.",
         f"Experimental Results & Overfitting Check: student_test Top-1={knn_r['Top-1 Accuracy']*100:.2f}%, Arc F1={knn_r['Arc Macro F1']:.4f}, Bin F1={knn_r['Binary Macro F1']:.4f}, LogLoss={knn_r['Log Loss']:.4f}, Cont={knn_r['Route Continuity']*100:.2f}%. Memorizes training points (Train Diag Bin F1=1.0000, Top-1=98.06%) vs 5-fold CV Bin F1=0.8922, showing local variance."),
        ("Model 3: Random Forest & Extra Trees (Ensemble)",
         "Suitability & Strengths/Limitations: Bagged decision trees handle unscaled physical features and non-linear junction rules cleanly with 0.51-0.52s inference. Limitation: class_weight='balanced' inflates positive posterior probabilities, raising Log Loss (0.1676-0.1823).",
         "Config & Objective: n_estimators=200, class_weight='balanced', max_depth=None, random_state=42. Minimized Gini impurity across bootstrap/randomized splits.",
         f"Experimental Results & Overfitting Check: RF achieved Top-1={rf_r['Top-1 Accuracy']*100:.2f}%, Arc F1={rf_r['Arc Macro F1']:.4f}, Bin F1={rf_r['Binary Macro F1']:.4f}, LogLoss={rf_r['Log Loss']:.4f}, Cont={rf_r['Route Continuity']*100:.2f}%. ET achieved Top-1={et_r['Top-1 Accuracy']*100:.2f}%, LogLoss={et_r['Log Loss']:.4f}, Cont={et_r['Route Continuity']*100:.2f}%. Unbounded depth on 50k sample rows caused validation variance (val Top-1 57.2% vs test 85.6%)."),
        ("Model 4: Gradient Boosting — HGB, XGBoost & LightGBM (Boosting)",
         "Suitability & Strengths/Limitations: Sequential error correction captures multiplicative interactions (distance x heading x speed). Achieves top standalone Top-1 (86.21% HGB) and lowest Log Loss (0.0697 Tuned LightGBM) with fast tree inference.",
         "Config & Objective: Untuned LightGBM (300 trees, lr=0.1, leaves=63), XGBoost (300 trees, depth=8), HGB (300 iters, max_leaf=63), Optuna-Tuned LightGBM (550 trees, lr=0.0289, leaves=101). Binary logistic loss.",
         f"Experimental Results & Overfitting Check: HGB Top-1={hgb_r['Top-1 Accuracy']*100:.2f}% (LogLoss {hgb_r['Log Loss']:.4f}), XGB Top-1={xgb_r['Top-1 Accuracy']*100:.2f}%, Tuned LGB Top-1={lgb_t['Top-1 Accuracy']*100:.2f}% (Bin F1={lgb_t['Binary Macro F1']:.4f}, LogLoss={lgb_t['Log Loss']:.4f}). 5-fold CV Top-1 (84.30-84.41%) matches student_test closely, showing strong generalization."),
        ("Model 5: PyTorch MLP Neural Network (Neural Networks)",
         "Suitability & Strengths/Limitations: Smooth deep decision boundaries with BatchNorm and Dropout yield highest Route Continuity (93.81%) and second-best Log Loss (0.0719) in 0.10s inference, though training took 1,235.9s over 30 CUDA epochs.",
         "Config & Objective: 128-64-32 Linear+BatchNorm+ReLU+Dropout(0.15), AdamW(lr=1e-3, wd=1e-4), batch=2048, ReduceLROnPlateau, early stopping (patience=5). Unweighted BCEWithLogitsLoss preserves true posterior probabilities.",
         f"Experimental Results & Overfitting Check: student_test Top-1={mlp_r['Top-1 Accuracy']*100:.2f}%, Arc F1={mlp_r['Arc Macro F1']:.4f}, Bin F1={mlp_r['Binary Macro F1']:.4f}, LogLoss={mlp_r['Log Loss']:.4f}, Cont={mlp_r['Route Continuity']*100:.2f}%. Best epoch=28 (val_loss=0.0708, val_F1=0.9241) closely tracks test (0.0719, 0.9186)."),
        ("Model 6: Kernel-Based RBF SVM (Nystrom + SGD Hinge + Platt)",
         "Suitability & Strengths/Limitations: Exact SVC is O(N^2) and infeasible for 1.86M rows; Nystrom(256) + chunked SGD hinge + Platt calibration scales linearly (125.7s train, 1.15s infer) and matches LogReg calibration (Log Loss 0.0935), though 256 basis anchors limit complex junction resolution.",
         "Config & Objective: Nystrom(components=256, fit_samples=50k), SGDClassifier(loss='hinge', C=1.0, epochs=6), 10% holdout Platt sigmoid calibration.",
         f"Experimental Results & Overfitting Check: student_test Top-1={svm_r['Top-1 Accuracy']*100:.2f}%, Arc F1={svm_r['Arc Macro F1']:.4f}, Bin F1={svm_r['Binary Macro F1']:.4f}, LogLoss={svm_r['Log Loss']:.4f}, Cont={svm_r['Route Continuity']*100:.2f}%. Mild underfitting from fixed 256-dim kernel approximation; no high-variance overfitting."),
    ]
    for title_m, p1, p2, p3 in model_blocks:
        add_p(title_m, h3_st)
        add_p(p1 + "<br/>" + p2 + "<br/>" + p3, body_st)

    # Section 3.3
    add_p("3.3 Model Visualisations", h2_st)
    add_fig("outputs/evaluation/plots/all_models_student_test_comparison.png", "Figure 3.1: Unified comparison across all 10 model variants and HMM pipelines on student_test.", max_h=3.6*inch)
    add_fig("outputs/evaluation/plots/model_comparison_overview.png", "Figure 3.2: 5-fold GroupKFold CV comparison across boosting backends (mean ± std across folds).", max_h=3.2*inch)
    add_fig("outputs/evaluation/plots/cv_fold_boxplot_top1.png", "Figure 3.3: Fold-to-fold Top-1 variability across GroupKFold splits.", max_h=2.5*inch)
    add_fig("outputs/evaluation/plots/all_models_confusion_matrices.png", "Figure 3.4: Actual student_test candidate confusion matrices across all 6 model families.", max_h=3.6*inch)
    add_fig("outputs/member3/plots/model5_mlp_loss_curves.png", "Figure 3.5: Real epoch-wise training and validation curves for Model 5 (MLP). No artificial curves were created for non-epoch models.", max_h=2.5*inch)

    story.append(PageBreak())

    # Section 4
    add_p("4. Bayesian Hyperparameter Optimisation (Optuna TPE on LightGBM)", h1_st)
    add_p("<b>Search Configuration:</b> Optimised 8 LightGBM parameters (n_estimators [100-600], log learning_rate [0.02-0.3], "
          "log num_leaves [15-255], log min_child_samples [10-200], subsample [0.5-1.0], colsample_bytree [0.5-1.0], "
          "log reg_alpha [1e-3-10], log reg_lambda [1e-3-10]).<br/>"
          "<b>Objective & Trials:</b> Maximized 3-fold GroupKFold CV Top-1 accuracy on a 50% training trip subset (40 train trips; student_test untouched). "
          "Out of 30 Optuna TPE trials with MedianPruner, 26 completed and 4 were pruned.<br/>"
          f"<b>Best Trial (Trial 20, CV Top-1 = {best_params['best_value']:.4f}):</b> n_estimators=550, lr=0.0289, num_leaves=101, "
          "min_child_samples=39, subsample=0.8604, colsample_bytree=0.8964, reg_alpha=0.0074, reg_lambda=0.0044.", body_st)

    t4_tbl = [
        ["Model", "CV Score", "Student-Test Top-1", "Student-Test Arc F1", "Log Loss", "Fit Time", "Infer Time"],
        ["Untuned LightGBM", "0.8430 (5-fold Top-1)", "86.07%", "0.7618", "0.0703", "10.45s", "0.51s"],
        ["Optuna-Tuned LightGBM", "0.8519 (3-fold sub)", "85.97%", "0.7614", "0.0697", "22.96s", "1.23s"],
    ]
    story.append(make_table(t4_tbl, [1.5*inch, 1.3*inch, 0.9*inch, 0.9*inch, 0.7*inch, 0.7*inch, 0.7*inch], body_st))
    story.append(Spacer(1, 4))
    add_p("Tuning achieved the best probability calibration across the entire benchmark (<b>Log Loss = 0.0697</b> vs 0.0703), while untuned LightGBM kept a +0.10% edge in standalone Top-1. Because calibrated probabilities drive HMM emissions, the tuned LightGBM bundle was exported as artifacts/final_model.joblib.", body_st)
    add_fig("outputs/tuning/optimization_history.png", "Figure 4.1: Optuna TPE optimization history across 26 completed trials.", max_h=2.4*inch)
    add_fig("outputs/tuning/param_importance.png", "Figure 4.2: fANOVA hyperparameter importance (colsample_bytree and learning_rate dominate).", max_h=2.4*inch)
    add_fig("outputs/evaluation/boosting_feature_importance.png", "Figure 4.3: Feature importance of the final tuned LightGBM model.", max_h=2.6*inch)

    story.append(PageBreak())

    # Section 5
    add_p("5. HMM and Viterbi Route Reconstruction", h1_st)
    add_p("<b>1. Hidden States:</b> Top-K=10 candidate road arcs per GPS observation.<br/>"
          "<b>2. Emission Probabilities:</b> Log classifier posterior w_em * log(clip(p, 1e-6, 1.0)) with Gaussian distance fallback (sigma_z=10m).<br/>"
          "<b>3. Transition Probabilities:</b> Newson & Krumm (2009) exponential discrepancy -|d_route - d_gps| / beta_eff (beta=5.0m), same_arc_bonus=+0.25, reverse_penalty=-12.0.<br/>"
          "<b>4. Missing Candidates & GPS Gaps:</b> Carry-forward on missing candidates; dynamic beta relaxation (+0.5/s for dt>5s) and expanding Dijkstra cutoff up to 5000m; sequence break restart above dt=60s or jump=2500m.<br/>"
          "<b>5. Disconnected Segments:</b> Finite disconnected_penalty=-25.0 prevents -inf collapse across map breaks.", body_st)

    hmm_tbl = [["Trip ID", "Obs", "Gaps", "Clf-Only Top-1", "LGBM+HMM Top-1", "Delta", "Clf Cont.", "HMM Obs Cont.", "HMM Route Cont.", "LCS F1 (Clf->HMM)"]]
    for _, r in hmm_lgbm_df.iterrows():
        tid_str = str(r["trip_id"]).split(".")[0].zfill(8)
        hmm_tbl.append([
            tid_str, int(r["n_obs"]), int(r["n_gps_gaps"]),
            f"{r['clf_obs_top1']*100:.2f}%", f"{r['hmm_obs_top1']*100:.2f}%", f"{r['delta_obs_top1']*100:+.2f}%",
            f"{r['clf_direct_cont']*100:.2f}%", f"{r['hmm_direct_cont']*100:.2f}%",
            f"{r['hmm_route_direct_cont']*100:.2f}%", f"{r['clf_lcs_f1']*100:.1f}%->{r['hmm_lcs_f1']*100:.1f}%"
        ])
    story.append(make_table(hmm_tbl, [0.68*inch, 0.40*inch, 0.42*inch, 0.70*inch, 0.78*inch, 0.52*inch, 0.65*inch, 0.70*inch, 0.75*inch, 1.10*inch], body_st))
    story.append(Spacer(1, 6))
    add_p("<b>6. Experimental Conclusion:</b> HMM/Viterbi raised Tuned LightGBM Top-1 Accuracy from <b>85.97% to 88.95% (+2.98%)</b>, "
          "observation continuity to <b>96.82%</b>, full route continuity to <b>99.52%</b>, and Route LCS F1 to <b>88.64%</b> (outperforming MLP+HMM at 88.32%). "
          "On trip 00000007 Top-1 rose +6.94% (to 99.78%), and on trip 00000065 (with a 210s GPS gap) Top-1 rose +5.21%. "
          "Only trip 00000091 dropped in pointwise Top-1 (-2.52%) because HMM chose a cleaner continuous parallel arc where our DP ground-truth label stayed on an adjacent ramp (LCS F1 still jumped 69.0% -> 87.6%).", body_st)

    # Section 6
    add_p("6. Geographical Visualisation Using Folium", h1_st)
    add_p("Interactive multi-layer HTML maps generated via src/visualization/folium_map.py and src/data/eda_analysis.py are committed under "
          "outputs/eda/maps/ and outputs/maps/ (e.g., student_test_00000007_lgbm_hmm.html, student_test_00000065_lgbm_hmm.html). "
          "Each map includes distinguishable colours (blue GPS dots, gray local road network, green reference route, dashed orange classifier-only route, solid red HMM route), "
          "auto-zoom bounds, HTML legend, and toggleable folium.LayerControl.", body_st)
    add_fig("outputs/maps/section_6_folium_comparison_preview.png", "Figure 6.1: Exported geographical route comparison matching interactive Folium map layers (00000007 and 00000065).", max_h=3.2*inch)

    story.append(PageBreak())

    # Section 7
    add_p("7. API, UI and Deployment", h1_st)
    add_p("<b>Two-Laptop Distributed Setup:</b> Laptop 1 runs uvicorn api.main:app --host 0.0.0.0 --port 8000 (exposing GET /, POST /predict, POST /evaluate, GET /logs), "
          "loading artifacts/final_model.joblib (Tuned LightGBM) with server-side ground-truth evaluation. "
          "Laptop 2 runs streamlit run ui/app.py, displays the 10 evaluation trips, sends candidate batches over LAN, and renders Folium maps and accuracy metrics.<br/>"
          "All 10 student_test trips were verified end-to-end against FastAPI TestClient across /predict and /evaluate.", body_st)
    add_fig("outputs/evaluation/plots/section_7_api_cmd_response.png", "Figure 7.1: Laptop 1 CMD FastAPI startup and live [PREDICT] / [EVAL] logs across all 10 trips.", max_h=2.9*inch)
    add_fig("outputs/evaluation/plots/section_7_ui_dashboard_preview.png", "Figure 7.2: Laptop 2 Evaluator UI 10-trip summary and server-side evaluation metrics.", max_h=2.8*inch)

    # Section 8
    add_p("8. Academic Integrity and AI Content Disclosure", h1_st)
    add_p("All code, preprocessing, experiments, Bayesian optimisation trials, HMM ablations, Folium maps, and API tests were independently run by our team on permitted train and student_test data. "
          "Evaluator testing data remained untouched. AI tools were used for code structuring and report formatting assistance; all numerical tables, plots, and conclusions are directly verifiable from repository logs.", body_st)

    for p_out in pdf_paths:
        doc = SimpleDocTemplate(
            p_out,
            pagesize=A4,
            leftMargin=0.65 * inch,
            rightMargin=0.65 * inch,
            topMargin=0.6 * inch,
            bottomMargin=0.6 * inch,
            title="PathSnap Consolidated Technical Report",
        )
        doc.build(story)
        print("Built PDF:", p_out)


if __name__ == "__main__":
    build_report()
