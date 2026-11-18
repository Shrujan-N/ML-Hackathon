# Section 2.2: Dataset Understanding and Exploratory Data Analysis

## 1. Dataset Partitions and File Structure

All exploration and preprocessing were performed strictly on the permitted development partitions: **80 `train` trips** and **10 `student_test` trips** (`evaluator_test` was kept completely untouched to prevent data leakage).

Each trip directory contains four headerless whitespace-separated files:
- `.track`: `(N, 3)` GPS observations (`longitude`, `latitude`, `timestamp` in seconds).
- `.nodes`: `(V, 2)` road-network vertices (`longitude`, `latitude`, 0-based `node_id`).
- `.arcs`: `(E, 2)` directed road segments (`from_node_id`, `to_node_id`, 0-based `arc_id`).
- `.route`: `(M,)` chronological sequence of ground-truth traversed `arc_id`s.

| Partition | Trips | Files | Total GPS Points | Mean / Min / Max GPS per Trip | Total Nodes | Total Directed Arcs | Total Route Arcs | Median GPS-to-Route Offset (m) | 95th Pct Offset (m) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `train` | 80 | 320 | 208,417 | 2,605.2 / 361 / 19,313 | 1,933,485 | 3,876,904 | 31,336 | 5.67 m | 141.23 m |
| `student_test` | 10 | 40 | 19,160 | 1,916.0 / 262 / 3,958 | 90,689 | 185,041 | 3,477 | 5.25 m | 16.80 m |

A key structural finding is that **`.track` observations do not map 1-to-1 with `.route` rows**. Because GPS is logged at roughly 1 Hz while a vehicle spends multiple seconds traversing each road segment, `.track` typically has 5x to 30x more rows than `.route` (for example, `student_test/00000000` has 2,503 GPS rows vs. 179 reference route arcs; `train/00000001` has 1,315 GPS rows vs. 525 route arcs).

---

## 2. Data Quality & Missing-Value Analysis

| Quality Metric | `train` (80 trips) | `student_test` (10 trips) |
| :--- | :---: | :---: |
| Missing or NaN coordinates/timestamps | 0 | 0 |
| Non-monotonic timestamp steps (`dt <= 0`) | 0 | 0 |
| Steps at exact 1 Hz (`dt == 1.0s`) | 96.50% | 98.62% |
| Sampling gaps `dt > 2.0s` / `dt > 5.0s` | 2,937 / 1,216 | 138 / 44 |
| Maximum single GPS gap (`dt`) | 153,064.0 s | 210.0 s |
| Isolated road-network nodes (degree 0) | 0 | 0 |
| Trips with $>1$ undirected graph components | 79 / 80 | 10 / 10 |
| Disconnected consecutive transitions in `.route` | 13 | 1 |
| High-offset anomaly trips (median $> 30\text{m}$) | 2 (`00000033`, `00000054`) | 0 |

### Observed Patterns and Anomalies
1. **Nominal 1 Hz sampling with intermittent dropouts:** While 96.50% (`train`) and 98.62% (`student_test`) of steps are spaced by 1.0 s, long GPS dropouts occur (up to 153,064 s in `train` and 210 s in `student_test`). Naive Euclidean step velocity or fixed-step HMM jumps break during these gaps unless `dt_prev` is explicitly tracked.
2. **Off-route drift / unmapped detour segments:** Two `train` trips—`00000033` (5,844 points, median offset 121.66 m) and `00000054` (19,313 points, median offset 139.22 m)—contain large blocks of GPS points located 120–210 m away from any segment in `.route`. Similarly, `train/00000002` contains a short off-route loop outside the reference sequence (`outputs/eda/plots/04_sample_trips_overlay.png`).
3. **Reference route disconnections and multi-component graphs:** Almost every trip graph contains multiple disconnected components (79/80 in `train`, 10/10 in `student_test`), and `.route` itself contains 13 non-adjacent arc transitions in `train` and 1 in `student_test`. This confirms that supplied `.route` files are imperfect reference labels and that downstream HMM/Viterbi decoding must support disconnected-segment fallbacks.

---

## 3. Data Preprocessing Steps (`src/data/data_loader.py` & `src/features/feature_engineering.py`)

1. **Coordinate Projection to Local Meters:** Since `.track` and `.nodes` store `(lon, lat)` in angular degrees (verified across European/global city bounding boxes), ordinary Euclidean distance in degrees is physically invalid because $1^\circ$ of longitude shrinks by $\cos(\text{lat})$. Each trip is projected to a local tangent plane in meters centered at the trip's mean coordinates:
   - $x_m = (\text{lon} - \text{lon}_0) \cdot 111320 \cos(\text{lat}_0)$, $\quad y_m = (\text{lat} - \text{lat}_0) \cdot 110540$.
2. **Track-to-Route Monotonic Alignment:** To create observation-level supervised ground-truth labels (`true_arc_id`), we align `.track` to `.route` using monotonic dynamic programming over point-to-segment perpendicular distances with a gentle heading penalty when moving (`speed > 1.5 m/s`) and bounded forward jumps (`max_jump = 40`).
3. **Candidate Generation via Spatial Indexing (`cKDTree`):** Road networks reach up to 155,003 nodes and 307,392 arcs (`train/00000013`). A `scipy.spatial.cKDTree` over segment midpoints queries the $M_{\text{pre}} = 100$ nearest segments per GPS point, refines them using exact point-to-segment distance, and retains the **top $K = 10$ candidates**.
   - `student_test` Candidate Miss Rate (100% of points retained): **0.511%** (99.489% recall; geometric top-1 accuracy = **54.207%**).
   - `train` Candidate Miss Rate (raw): **7.856%** (driven by off-route anomalies in `00000033` and `00000054`).
4. **Training Outlier Filtering & Leakage-Free Scaling:** Points $>60\text{ m}$ away from the reference `.route` were excluded from `train` only (`208,417 -> 186,093` observations, lowering active `train` Candidate Miss Rate to **1.350%** and raising geometric top-1 to **54.853%**). `student_test` (`19,160` observations, `191,600` candidate rows) was left **100% unfiltered**. `StandardScaler` (`artifacts/feature_scaler.joblib`) was fit strictly on `train`.

---

## 4. Extracted Features and Rationale (16 Features)

| Feature | Type / Units | Rationale |
| :--- | :--- | :--- |
| `dist_to_arc_m` | Spatial (m) | Core emission proximity between noisy GPS point and candidate road segment. |
| `dist_min_diff_m` | Relative (m) | Margin (`d - min_d`) over the closest candidate at timestamp $t$; captures local ambiguity at intersections. |
| `dist_ratio_to_min` | Relative | Scale-free distance ratio `(d + 0.5) / (min_d + 0.5)` robust across urban/highway noise levels. |
| `cand_rank` | Ranking (0–9) | Distance rank among the 10 candidates at observation $t$ ($r = -0.44$ correlation with label). |
| `proj_fraction` | Geometric [0, 1] | Clamped projection position along the arc; values pinned at `0.0` or `1.0` often indicate a nearby junction rather than on-segment travel. |
| `dist_to_from_node_m` / `dist_to_to_node_m` | Geometric (m) | Distance to segment endpoints; distinguishes mid-arc observations from node-entry/exit points. |
| `arc_length_m` | Geometric (m) | Physical length of road segment; very short micro-arcs near junctions frequently create false proximity matches. |
| `heading_diff_rad` / `heading_cos` / `heading_sin` | Kinematic | Angle alignment between central-difference GPS movement vector and directed arc `(u -> v)`. Crucial for separating opposite lanes on two-way roads (`heading_cos` peaks at `+1` for true arcs vs. both `+1` and `-1` for false candidates). |
| `gps_speed_mps` / `dt_prev_s` | Kinematic | Instantaneous speed and sampling gap; alerts downstream models when GPS heading is noisy (stationary) or when large time gaps occur. |
| `prev_gps_dist_to_arc_m` / `next_gps_dist_to_arc_m` | Temporal (m) | Distance from $p_{t-1}$ and $p_{t+1}$ to this candidate arc; suppresses single-point multipath spikes. |
| `is_connected_from_prev_top1` | Topological {0, 1} | Checks whether the candidate arc matches or connects (`from_node == prev_to_node`) from the top-3 closest candidates at $t-1$ ($r = +0.31$ positive correlation with label). |

---

## 5. Generated Phase 1 Visualisations & Artifacts
- **Tables:** `outputs/eda/tables/dataset_inventory.csv`, `data_quality_report.csv`, `trip_level_stats.csv`
- **Plots:** `outputs/eda/plots/01_trip_length_and_duration.png` through `05_feature_distributions_and_correlations.png`
- **Interactive Folium Maps:** `outputs/eda/maps/eda_map_train_00000001.html` and `eda_map_student_test_00000000.html` (with layer controls for road network, reference route, raw GPS points, and GPS gaps $> 2\text{ s}$ / $> 5\text{ s}$)
- **Processed Datasets & Schema:** `data/processed/train_candidates.pkl`, `student_test_candidates.pkl`, `student_test_candidates.csv`, `artifacts/feature_scaler.joblib`, `artifacts/feature_schema.json`
