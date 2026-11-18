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
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data.data_loader import (
    get_trip_ids,
    load_trip,
    project_to_local_meters,
    compute_gps_kinematics,
    point_to_segment_dist,
    align_track_to_route,
)

FEATURE_COLUMNS = [
    "dist_to_arc_m",
    "dist_min_diff_m",
    "dist_ratio_to_min",
    "cand_rank",
    "proj_fraction",
    "dist_to_from_node_m",
    "dist_to_to_node_m",
    "arc_length_m",
    "heading_diff_rad",
    "heading_cos",
    "heading_sin",
    "gps_speed_mps",
    "dt_prev_s",
    "prev_gps_dist_to_arc_m",
    "next_gps_dist_to_arc_m",
    "is_connected_from_prev_top1",
]

METADATA_COLUMNS = [
    "trip_id",
    "obs_idx",
    "timestamp",
    "candidate_arc_id",
    "from_node",
    "to_node",
    "true_arc_id",
]


def generate_candidates_and_features_for_trip(trip, k=10, m_pre=100, filter_offroad_stationary=False):
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]
    trip_id = trip["trip_id"]

    ref_lon = float(track[:, 0].mean())
    ref_lat = float(track[:, 1].mean())
    tx, ty = project_to_local_meters(track[:, 0], track[:, 1], ref_lon, ref_lat)
    nx, ny = project_to_local_meters(nodes[:, 0], nodes[:, 1], ref_lon, ref_lat)

    timestamps = track[:, 2]
    dt_prev, speed_mps, gps_heading = compute_gps_kinematics(tx, ty, timestamps)

    true_arc_ids, _, matched_route_dist = align_track_to_route(tx, ty, timestamps, nx, ny, arcs, route)

    u = arcs[:, 0]
    v = arcs[:, 1]
    ax = nx[u]
    ay = ny[u]
    bx = nx[v]
    by = ny[v]
    arc_dx = bx - ax
    arc_dy = by - ay
    arc_len = np.hypot(arc_dx, arc_dy)
    arc_angle = np.arctan2(arc_dy, arc_dx)
    mx = 0.5 * (ax + bx)
    my = 0.5 * (ay + by)

    # Spatial index on arc midpoints for fast pre-filtering on large city graphs
    tree = cKDTree(np.column_stack([mx, my]))
    n_pre = min(m_pre, len(arcs))
    _, pre_cand = tree.query(np.column_stack([tx, ty]), k=n_pre)
    if pre_cand.ndim == 1:
        pre_cand = pre_cand[:, None]

    pre_dists, pre_proj = point_to_segment_dist(
        tx[:, None],
        ty[:, None],
        ax[pre_cand],
        ay[pre_cand],
        bx[pre_cand],
        by[pre_cand],
    )

    k_actual = min(k, n_pre)
    order = np.argsort(pre_dists, axis=1)[:, :k_actual]
    topk_arcs = np.take_along_axis(pre_cand, order, axis=1)
    topk_dists = np.take_along_axis(pre_dists, order, axis=1)
    topk_proj = np.take_along_axis(pre_proj, order, axis=1)

    n_obs = len(track)
    raw_hits_k = int((topk_arcs == true_arc_ids[:, None]).any(axis=1).sum())
    raw_hits_1 = int((topk_arcs[:, 0] == true_arc_ids).sum())

    # Optionally remove extreme off-route outlier points (>60m from reference route) from training only
    keep_mask = np.ones(n_obs, dtype=bool)
    if filter_offroad_stationary:
        keep_mask = matched_route_dist <= 60.0
        if not keep_mask.any():
            keep_mask[:] = True

    active_hits_k = int((topk_arcs[keep_mask] == true_arc_ids[keep_mask, None]).any(axis=1).sum())
    active_hits_1 = int((topk_arcs[keep_mask, 0] == true_arc_ids[keep_mask]).sum())
    active_obs = int(keep_mask.sum())

    # Previous and next GPS points (clamped at trajectory ends)
    idx_all = np.arange(n_obs)
    prev_idx = np.clip(idx_all - 1, 0, n_obs - 1)
    next_idx = np.clip(idx_all + 1, 0, n_obs - 1)

    prev_dists, _ = point_to_segment_dist(
        tx[prev_idx, None],
        ty[prev_idx, None],
        ax[topk_arcs],
        ay[topk_arcs],
        bx[topk_arcs],
        by[topk_arcs],
    )
    next_dists, _ = point_to_segment_dist(
        tx[next_idx, None],
        ty[next_idx, None],
        ax[topk_arcs],
        ay[topk_arcs],
        bx[topk_arcs],
        by[topk_arcs],
    )

    # Check if each candidate connects from any of the top-3 closest arcs at t-1
    is_conn = np.zeros((n_obs, k_actual), dtype=np.float32)
    cand_from = u[topk_arcs]
    cand_to = v[topk_arcs]
    for i in range(1, n_obs):
        prev_top3_arcs = topk_arcs[i - 1, :3]
        prev_top3_to = v[prev_top3_arcs]
        same_arc = np.isin(topk_arcs[i], prev_top3_arcs)
        node_conn = np.isin(cand_from[i], prev_top3_to)
        is_conn[i] = np.where(same_arc | node_conn, 1.0, 0.0)

    # Filter observations (for train only if enabled) after temporal neighbor features are built
    sel_obs = np.where(keep_mask)[0]
    m_sel = len(sel_obs)

    k_arcs = topk_arcs[sel_obs]
    k_dists = topk_dists[sel_obs]
    k_proj = topk_proj[sel_obs]
    k_prev_d = prev_dists[sel_obs]
    k_next_d = next_dists[sel_obs]
    k_conn = is_conn[sel_obs]

    min_d = k_dists[:, [0]]
    dist_min_diff = k_dists - min_d
    dist_ratio = (k_dists + 0.5) / (min_d + 0.5)
    cand_rank = np.broadcast_to(np.arange(k_actual, dtype=np.float32), (m_sel, k_actual))

    d_from_node = np.hypot(tx[sel_obs, None] - ax[k_arcs], ty[sel_obs, None] - ay[k_arcs])
    d_to_node = np.hypot(tx[sel_obs, None] - bx[k_arcs], ty[sel_obs, None] - by[k_arcs])
    k_arc_len = arc_len[k_arcs]

    ang_diff = np.arctan2(
        np.sin(gps_heading[sel_obs, None] - arc_angle[k_arcs]),
        np.cos(gps_heading[sel_obs, None] - arc_angle[k_arcs]),
    )
    heading_diff_rad = np.abs(ang_diff)
    heading_cos = np.cos(ang_diff)
    heading_sin = np.sin(ang_diff)

    k_speed = np.broadcast_to(speed_mps[sel_obs, None], (m_sel, k_actual))
    k_dt = np.broadcast_to(dt_prev[sel_obs, None], (m_sel, k_actual))
    k_true_arc = np.broadcast_to(true_arc_ids[sel_obs, None], (m_sel, k_actual))
    labels = (k_arcs == k_true_arc).astype(np.int8)

    df = pd.DataFrame({
        "trip_id": np.repeat(trip_id, m_sel * k_actual),
        "obs_idx": np.repeat(sel_obs.astype(np.int32), k_actual),
        "timestamp": np.repeat(timestamps[sel_obs].astype(np.float32), k_actual),
        "candidate_arc_id": k_arcs.ravel().astype(np.int32),
        "from_node": u[k_arcs].ravel().astype(np.int32),
        "to_node": v[k_arcs].ravel().astype(np.int32),
        "true_arc_id": k_true_arc.ravel().astype(np.int32),
        "dist_to_arc_m": k_dists.ravel().astype(np.float32),
        "dist_min_diff_m": dist_min_diff.ravel().astype(np.float32),
        "dist_ratio_to_min": dist_ratio.ravel().astype(np.float32),
        "cand_rank": cand_rank.ravel().astype(np.float32),
        "proj_fraction": k_proj.ravel().astype(np.float32),
        "dist_to_from_node_m": d_from_node.ravel().astype(np.float32),
        "dist_to_to_node_m": d_to_node.ravel().astype(np.float32),
        "arc_length_m": k_arc_len.ravel().astype(np.float32),
        "heading_diff_rad": heading_diff_rad.ravel().astype(np.float32),
        "heading_cos": heading_cos.ravel().astype(np.float32),
        "heading_sin": heading_sin.ravel().astype(np.float32),
        "gps_speed_mps": k_speed.ravel().astype(np.float32),
        "dt_prev_s": k_dt.ravel().astype(np.float32),
        "prev_gps_dist_to_arc_m": k_prev_d.ravel().astype(np.float32),
        "next_gps_dist_to_arc_m": k_next_d.ravel().astype(np.float32),
        "is_connected_from_prev_top1": k_conn.ravel().astype(np.float32),
        "label": labels.ravel(),
    })

    stats = {
        "n_obs_raw": n_obs,
        "n_obs_kept": active_obs,
        "raw_hits_k": raw_hits_k,
        "raw_hits_1": raw_hits_1,
        "active_hits_k": active_hits_k,
        "active_hits_1": active_hits_1,
    }
    return df, stats


def plot_feature_diagnostics(train_df, out_path):
    sample_df = train_df if len(train_df) <= 300000 else train_df.sample(n=300000, random_state=42)
    pos = sample_df[sample_df["label"] == 1]
    neg = sample_df[sample_df["label"] == 0]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    axes[0, 0].hist(np.clip(pos["dist_to_arc_m"], 0, 50), bins=35, alpha=0.65, label="True Arc (label=1)", color="#16a34a", density=True)
    axes[0, 0].hist(np.clip(neg["dist_to_arc_m"], 0, 50), bins=35, alpha=0.55, label="False Candidate (label=0)", color="#dc2626", density=True)
    axes[0, 0].set_title("Perpendicular Distance to Arc (Clipped at 50m)")
    axes[0, 0].set_xlabel("dist_to_arc_m")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()

    axes[0, 1].hist(pos["heading_cos"], bins=30, alpha=0.65, label="True Arc (label=1)", color="#16a34a", density=True)
    axes[0, 1].hist(neg["heading_cos"], bins=30, alpha=0.55, label="False Candidate (label=0)", color="#dc2626", density=True)
    axes[0, 1].set_title("Cosine Heading Alignment (GPS vs. Directed Arc)")
    axes[0, 1].set_xlabel("heading_cos")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].legend()

    axes[1, 0].hist(np.clip(pos["dist_min_diff_m"], 0, 30), bins=30, alpha=0.65, label="True Arc (label=1)", color="#16a34a", density=True)
    axes[1, 0].hist(np.clip(neg["dist_min_diff_m"], 0, 30), bins=30, alpha=0.55, label="False Candidate (label=0)", color="#dc2626", density=True)
    axes[1, 0].set_title("Excess Distance Over Closest Candidate")
    axes[1, 0].set_xlabel("dist_min_diff_m (m)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].legend()

    corrs = sample_df[FEATURE_COLUMNS].corrwith(sample_df["label"]).sort_values()
    colors = ["#dc2626" if c < 0 else "#16a34a" for c in corrs.values]
    axes[1, 1].barh(corrs.index, corrs.values, color=colors, alpha=0.85)
    axes[1, 1].axvline(0.0, color="#333333", linewidth=0.8)
    axes[1, 1].set_title("Pearson Correlation of Engineered Features with Label")
    axes[1, 1].set_xlabel("Correlation with Target (label)")

    plt.suptitle("PathSnap Feature Diagnostics on Training Candidate Set", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def build_feature_datasets(repo_root, k=10, m_pre=100):
    t_start = time.time()
    raw_root = os.path.join(repo_root, "data", "raw")
    proc_dir = os.path.join(repo_root, "data", "processed")
    art_dir = os.path.join(repo_root, "artifacts")
    plots_dir = os.path.join(repo_root, "outputs", "eda", "plots")
    for d in [proc_dir, art_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    summary = {}
    dfs = {}

    for part in ["train", "student_test"]:
        trip_ids = get_trip_ids(raw_root, part)
        filter_flag = (part == "train")
        part_dfs = []
        totals = {
            "n_obs_raw": 0,
            "n_obs_kept": 0,
            "raw_hits_k": 0,
            "raw_hits_1": 0,
            "active_hits_k": 0,
            "active_hits_1": 0,
        }

        print(f"Generating candidates & features for {part} ({len(trip_ids)} trips)...")
        for tid in trip_ids:
            trip = load_trip(raw_root, part, tid)
            df_t, st = generate_candidates_and_features_for_trip(
                trip,
                k=k,
                m_pre=m_pre,
                filter_offroad_stationary=filter_flag,
            )
            part_dfs.append(df_t)
            for key in totals:
                totals[key] += st[key]

        df_part = pd.concat(part_dfs, ignore_index=True)
        dfs[part] = df_part

        raw_miss_rate = 1.0 - totals["raw_hits_k"] / max(1, totals["n_obs_raw"])
        active_miss_rate = 1.0 - totals["active_hits_k"] / max(1, totals["n_obs_kept"])
        geom_top1_raw = totals["raw_hits_1"] / max(1, totals["n_obs_raw"])
        geom_top1_active = totals["active_hits_1"] / max(1, totals["n_obs_kept"])

        summary[part] = {
            "n_trips": len(trip_ids),
            "n_obs_raw": totals["n_obs_raw"],
            "n_obs_kept": totals["n_obs_kept"],
            "n_candidate_rows": len(df_part),
            "positive_rows": int(df_part["label"].sum()),
            "raw_candidate_miss_rate_pct": round(100.0 * raw_miss_rate, 3),
            "active_candidate_miss_rate_pct": round(100.0 * active_miss_rate, 3),
            "geom_top1_acc_raw_pct": round(100.0 * geom_top1_raw, 3),
            "geom_top1_acc_active_pct": round(100.0 * geom_top1_active, 3),
        }
        print(f"  {part}: rows={len(df_part)}, miss_rate(raw)={summary[part]['raw_candidate_miss_rate_pct']}%, miss_rate(kept)={summary[part]['active_candidate_miss_rate_pct']}%, geom_top1={summary[part]['geom_top1_acc_active_pct']}%")

    train_df = dfs["train"]
    test_df = dfs["student_test"]

    # Check for any NaN or Inf values before scaling
    assert not np.isnan(train_df[FEATURE_COLUMNS].values).any(), "NaNs found in train features"
    assert not np.isinf(train_df[FEATURE_COLUMNS].values).any(), "Infs found in train features"
    assert not np.isnan(test_df[FEATURE_COLUMNS].values).any(), "NaNs found in student_test features"
    assert not np.isinf(test_df[FEATURE_COLUMNS].values).any(), "Infs found in student_test features"

    # Fit StandardScaler strictly on train data to prevent data leakage
    scaler = StandardScaler()
    scaler.fit(train_df[FEATURE_COLUMNS].values)

    scaler_path = os.path.join(art_dir, "feature_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    schema = {
        "candidate_k": k,
        "pre_candidates_m": m_pre,
        "metadata_columns": METADATA_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": "label",
        "scaler_path": "artifacts/feature_scaler.joblib",
        "scaler_fit_partition": "train_only",
        "train_filter": "removed off-route outlier points (>60m from reference route, e.g. parking/detour drift in 00000033 and 00000054)",
        "student_test_filter": "none (100% of observations retained)",
        "stats": summary,
    }
    schema_path = os.path.join(art_dir, "feature_schema.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    print("Saving processed candidate datasets...")
    joblib.dump(train_df, os.path.join(proc_dir, "train_candidates.pkl"), compress=3)
    joblib.dump(test_df, os.path.join(proc_dir, "student_test_candidates.pkl"), compress=3)
    test_df.to_csv(os.path.join(proc_dir, "student_test_candidates.csv"), index=False)
    train_df.head(50000).to_csv(os.path.join(proc_dir, "train_candidates_sample.csv"), index=False)
    train_df.to_csv(os.path.join(proc_dir, "train_candidates.csv.gz"), index=False, compression="gzip")

    plot_feature_diagnostics(train_df, os.path.join(plots_dir, "05_feature_distributions_and_correlations.png"))

    print(f"Feature engineering completed in {time.time() - t_start:.2f}s.")
    return summary


if __name__ == "__main__":
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    build_feature_datasets(repo_dir)
