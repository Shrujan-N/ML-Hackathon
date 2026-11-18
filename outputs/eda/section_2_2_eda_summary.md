# Section 2.2: Dataset Understanding and Exploratory Data Analysis (Member 1)

## 1. Dataset Partitioning & Inventory
All exploration and preprocessing statistics were computed separately on the permitted **Training** (`80` trips, `320` files) and **Student Testing** (`10` trips, `40` files) partitions. The `10` Evaluator Testing trips were kept completely untouched.

| Partition | Trips | Files | Total GPS Points | Mean / Min / Max Points | Total Road Nodes | Total Road Arcs | Reference Route Arcs | Median GPS-to-Route Error (m) | 95th Pct Error (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 80 | 320 | 208,417 | 2,605.2 / 361 / 19,313 | 1,933,485 | 3,876,904 | 31,336 | 5.67 | 141.23 |
| `student_test` | 10 | 40 | 19,160 | 1,916.0 / 262 / 3,958 | 90,689 | 185,041 | 3,477 | 5.25 | 16.80 |

## 2. How the Data Was Preprocessed
1. **Coordinate Projection:** Raw `.track` and `.nodes` coordinates are given in angular `(longitude, latitude)` degrees. Because `1` degree of longitude shrinks by `cos(latitude)`, all spatial computations project coordinates into local tangent-plane meters centered on each trip's mean `(ref_lon, ref_lat)` via `project_to_local_meters()` in `src/data/data_loader.py`.
2. **Track-to-Route Alignment:** Each `.route` file contains the chronological sequence of traversed `arc_id`s (`31,336` arcs total in `train`, `3,477` in `student_test`), which is much shorter than `.track` (`208,417` and `19,160` observations) because multiple 1 Hz GPS points occur per road segment. We used monotonic dynamic programming (`align_track_to_route()`) combining perpendicular distance and moving-heading alignment to assign each GPS observation its ground-truth `true_arc_id`.
3. **Training Outlier Cleaning & Leakage-Free Scaling:** Stationary off-road points (`dist > 60 m` and `speed < 1.0 m/s`) were filtered in `train` only (`208,417 -> 207,564` observations), while `100%` of `student_test` (`19,160` observations) was retained unmodified. `StandardScaler` (`artifacts/feature_scaler.joblib`) was fit strictly on `train` features.

## 3. Data Quality Findings & Observed Patterns
- **Missing Values & Ordering:** `0` missing/NaN coordinates and `0` non-monotonic timestamps across all `90` trips (`outputs/eda/tables/data_quality_report.csv`).
- **Sampling Frequency & Dropouts:** `96.50%` of `train` intervals and `98.62%` of `student_test` intervals are exactly `1.0 s` (`1 Hz`). However, `train` has `2,937` intervals `> 2 s` and `1,216` intervals `> 5 s` (max gap `153,064.0 s`), and `student_test` has `138` intervals `> 2 s` (`44` `> 5 s`, max `210.0 s`).
- **Road Network Structure:** `0` isolated nodes exist, yet `79/80` `train` trips and `10/10` `student_test` trips contain multiple disconnected graph components. Furthermore, `.route` itself contains `13` disconnected arc transitions in `train` and `1` in `student_test`, confirming that Viterbi decoding must support gap recovery.
- **High-Offset Anomaly Trips:** Two `train` trips (`00000033` with `121.66 m` median offset, `00000054` with `139.22 m` median offset) have extended off-road stationary parking/drift intervals, inflating `train` 95th-percentile GPS noise (`141.23 m`) and raw candidate miss rate (`7.856%`). Excluding those 2 trips, the remaining `78` `train` trips (`183,172` obs) have a `1.221%` candidate miss rate, and `student_test` has a `0.511%` candidate miss rate.

## 4. Candidate Generation & Engineered Features
Using `cKDTree` over road segment midpoints (`m_pre = 100` refined by exact point-to-segment distance to `K = 10` candidates per GPS observation), geometric nearest-arc matching achieves only `49.473%` Top-1 accuracy on `train` and `54.207%` on `student_test`, motivating ML classification over `16` features (`artifacts/feature_schema.json`):

- **Proximity & Relative Geometry:** `dist_to_arc_m`, `dist_min_diff_m`, `dist_ratio_to_min`, `cand_rank`, `proj_fraction`, `dist_to_from_node_m`, `dist_to_to_node_m`, `arc_length_m`.
- **Direction & Kinematics (Separates parallel/opposite lanes):** `heading_diff_rad`, `heading_cos`, `heading_sin`, `gps_speed_mps`, `dt_prev_s`.
- **Local Temporal & Graph Context:** `prev_gps_dist_to_arc_m`, `next_gps_dist_to_arc_m`, `is_connected_from_prev_top1`.

## 5. Quick Guide for Teammates (Loading Preprocessed Data)
```python
import json, joblib

with open("artifacts/feature_schema.json") as f:
    schema = json.load(f)

feature_cols = schema["feature_columns"]
train_df = joblib.load("data/processed/train_candidates.pkl")          # 2,075,640 rows
test_df = joblib.load("data/processed/student_test_candidates.pkl")    #   191,600 rows
scaler = joblib.load("artifacts/feature_scaler.joblib")                # fit on train only
```
- **Raw features** are stored in the DataFrames (ideal for Random Forest / Boosting). For linear, KNN, SVM, or MLP models, transform with `scaler.transform(df[feature_cols].values)` (or fit inside each CV fold).
- **Group-Aware CV Key:** Group by `trip_id` (`80` groups in `train_df`), and rank/predict per `(trip_id, obs_idx)`.
