# PathSnap: Distributed Map-Matching & Route Reconstruction
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
