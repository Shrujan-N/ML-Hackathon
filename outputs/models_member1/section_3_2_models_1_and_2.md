# Section 3.2: Member 1 Models (Models 1 & 2)

## Consolidated Metrics Summary (Member 1)

| Model | Algorithm Family | Key Configuration | CV Macro F1-score | Top-1 Accuracy (`student_test`) | Route Continuity (`student_test`) | Student-Test Macro F1-score | Log Loss (`student_test`) | Inference Time (`student_test`) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model 1: Logistic Regression** | Linear and probabilistic | `C=1.0, solver=lbfgs, max_iter=300, StandardScaler` | 0.8975 (+/- 0.0169) | 85.33% | 93.08% | 0.9075 | 0.0934 | 0.04s (0.035s) |
| **Model 2: Distance-Weighted KNN** | Instance-based | `k=15, weights=distance, kd_tree, 32k trip-stratified ref` | 0.8922 (+/- 0.0105) | 83.92% | 91.70% | 0.9021 | 0.1445 | 11.58s |

*(Both models use `StandardScaler` fit strictly on `train` via `artifacts/feature_scaler.joblib` and 5-fold `GroupKFold` grouped by `trip_id` across the 80 `train` trips. Evaluation on `student_test` covers all 10 trips / 19,160 observations / 191,600 candidate rows.)*

---

## Model 1: Logistic Regression

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Each GPS observation has 10 candidate arcs characterized by strong monotonic geometric and kinematic signals: `cand_rank` ($r = -0.44$ correlation with label), `heading_diff_rad` ($r = -0.30$), `heading_cos` ($r = +0.29$), `is_connected_from_prev_top1` ($r = +0.31$), and `dist_to_arc_m` ($r = -0.17$).
   - Because downstream HMM/Viterbi decoding requires well-calibrated candidate emission probabilities $P(y=1 \mid x)$, Logistic Regression is an ideal linear probabilistic baseline that outputs calibrated sigmoid probabilities directly and executes inference in milliseconds (`0.035 s` for 191,600 rows).

2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Fast inference (`0.035 s` across all 19,160 test observations), very well-calibrated probabilities (lowest Log Loss of `0.0934`), minimal variance across unseen trips (`train` Macro F1 `0.9047` vs. `student_test` Macro F1 `0.9075`), and tiny deployment footprint (`713` bytes).
   - **Limitations:** Decision boundary is strictly linear in the standardized 16-feature space, so it cannot learn non-linear interactions such as "if `gps_speed_mps < 1.0`, ignore `heading_cos` because GPS bearing is noisy while stationary, and rely instead on `prev_gps_dist_to_arc_m` and `is_connected_from_prev_top1`." This limits its standalone Top-1 Accuracy to `85.33%` without non-linear models or HMM sequential smoothing.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - Regularization strength `C=1.0` (L2 penalty), optimization algorithm `solver='lbfgs'`, `max_iter=300`, `random_state=42`, with all 16 input features standardized using `StandardScaler` fit strictly on `train` (or per CV training fold).
6. **What loss function or optimisation objective was used?**
   - L2-regularized binary cross-entropy (negative log-likelihood / logistic loss) minimized over all `1,860,930` training candidate rows using L-BFGS (`TrainTime = 12.78 s`).

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - Across all 10 `student_test` trips (`191,600` rows / `19,160` observations), Model 1 achieved Binary Accuracy = `96.69%`, Precision = `0.8344`, Recall = `0.8323`, and Route Continuity = `93.08%`.
   - Confusion matrix (`student_test`): `TN = 169,389`, `FP = 3,149`, `FN = 3,196`, `TP = 15,866`.
8. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   - **Observation-level Top-1 Accuracy:** `85.33%` (compared to `54.21%` for the pure geometric closest-arc baseline).
   - **Macro F1-score:** `0.9075`.
   - **Log Loss:** `0.0934`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - **No overfitting, mild underfitting due to linear capacity.** Evidence: 5-fold Group CV Macro F1 is `0.8975 +/- 0.0169` (Top-1 `83.31%`, Log Loss `0.0995`), training diagnostic Macro F1 is `0.9047` (Top-1 `85.04%`), and `student_test` Macro F1 is `0.9075` (Top-1 `85.33%`). Because training, CV, and test scores are nearly identical, the model generalizes reliably across unseen trips while leaving headroom for non-linear tree/neural models.

---

## Model 2: Distance-Weighted KNN

### A. Model Understanding
1. **Why did you consider this algorithm suitable for the PathSnap problem?**
   - Map-matching candidates naturally form local geometric regimes (e.g., dense urban parallel streets vs. rural single-highway corridors vs. roundabout/junction turns). A non-parametric distance-weighted nearest-neighbor classifier predicts candidate validity by comparing standardized distance, rank, heading alignment, and local topology directly against similar historical road situations.

2. **What are the major strengths and limitations of this algorithm for this task?**
   - **Strengths:** Captures arbitrary non-linear local decision boundaries in feature space with near-zero fit time (`0.07 s` to build the KD-tree on the 32,000-row trip-stratified reference set) and competitive Macro F1 (`0.9021` on `student_test`).
   - **Limitations:** Inference is computationally expensive (`11.58 s` on `student_test`, ~330x slower than Logistic Regression) and memory scales with the reference dataset size, making full-1.86M-row KNN impractical for real-time FastAPI inference. Furthermore, inverse-distance weighting produces less calibrated probabilities (higher Log Loss of `0.1445`) when Euclidean distances in 16D standardized space suffer from irrelevant/noisy dimensions such as `dt_prev_s` or `heading_sin`.

### B. Training and Configuration
5. **What were the important hyperparameters and their initial values?**
   - `n_neighbors=15`, `weights='distance'` (inverse Euclidean distance weighting), `metric='euclidean'`, `algorithm='kd_tree'`, `leaf_size=50`, `n_jobs=-1`.
   - Because exact KD-tree querying against `1.86M` training rows is slow at API inference time, we built a balanced **trip-stratified reference set of 32,000 candidate rows** (`40` random observations $\times$ `10` candidates per trip across all 80 `train` trips, `random_state=42`).
6. **What loss function or optimisation objective was used?**
   - KNN is a lazy, non-parametric learner with no parametric training loss; it stores standardized training vectors in a KD-tree and outputs inverse-distance-weighted class votes over the 15 nearest neighbors at query time.

### C. Experimental Results
7. **What was the model's performance on the student-testing dataset?**
   - On `student_test` (`191,600` rows / `19,160` observations), Model 2 achieved Binary Accuracy = `96.53%`, Precision = `0.8336`, Recall = `0.8135`, and Route Continuity = `91.70%`.
   - Confusion matrix (`student_test`): `TN = 169,442`, `FP = 3,096`, `FN = 3,556`, `TP = 15,506`.
8. **What were the observation-level Top-1 Accuracy, Macro F1-score and Log Loss on the student-testing dataset?**
   - **Observation-level Top-1 Accuracy:** `83.92%`.
   - **Macro F1-score:** `0.9021`.
   - **Log Loss:** `0.1445`.
9. **Did you observe overfitting or underfitting? What evidence supports your conclusion?**
   - **Local overfitting / memorization on training points, with slightly worse generalization than Logistic Regression.** Evidence: With `weights='distance'`, training-set diagnostic performance reaches `1.0000` Macro F1 and `98.06%` Top-1 Accuracy (because exact self-neighbors have zero distance and infinite weight), whereas 5-fold Group CV drops to `0.8922 +/- 0.0105` Macro F1 (`82.34%` Top-1, Log Loss `0.1533`) and `student_test` reaches `0.9021` Macro F1 (`83.92%` Top-1, Log Loss `0.1445`). Equal Euclidean feature weighting in 16D allows lower-signal features (like `dt_prev_s` and `gps_speed_mps`) to dilute nearest-neighbor geometry compared to Logistic Regression's learned weights.
