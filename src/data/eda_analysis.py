import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import folium

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_loader import (
    get_trip_ids,
    load_trip,
    project_to_local_meters,
    align_track_to_route,
)


def compute_graph_stats(n_nodes, arcs):
    if n_nodes == 0 or len(arcs) == 0:
        return n_nodes, 0
    u = arcs[:, 0]
    v = arcs[:, 1]
    deg = np.bincount(u, minlength=n_nodes) + np.bincount(v, minlength=n_nodes)
    isolated_nodes = int((deg == 0).sum())

    ones = np.ones(len(arcs), dtype=np.int8)
    adj = csr_matrix((ones, (u, v)), shape=(n_nodes, n_nodes))
    n_comp, _ = connected_components(adj, directed=False, return_labels=True)
    return isolated_nodes, int(n_comp)


def build_folium_map(trip, out_html_path):
    track = trip["track"]
    nodes = trip["nodes"]
    arcs = trip["arcs"]
    route = trip["route"]
    n_route_arcs = len(route) if route is not None else 0

    ref_lon = float(track[:, 0].mean())
    ref_lat = float(track[:, 1].mean())
    tx, ty = project_to_local_meters(track[:, 0], track[:, 1], ref_lon, ref_lat)
    nx, ny = project_to_local_meters(nodes[:, 0], nodes[:, 1], ref_lon, ref_lat)

    # Keep road arcs whose midpoint falls within 150m of the trajectory bounding box
    xmin, xmax = tx.min() - 150.0, tx.max() + 150.0
    ymin, ymax = ty.min() - 150.0, ty.max() + 150.0
    u = arcs[:, 0]
    v = arcs[:, 1]
    mx = 0.5 * (nx[u] + nx[v])
    my = 0.5 * (ny[u] + ny[v])
    near_mask = (mx >= xmin) & (mx <= xmax) & (my >= ymin) & (my <= ymax)
    near_arcs = arcs[near_mask]

    if len(near_arcs) > 4000:
        step = max(1, len(near_arcs) // 4000)
        near_arcs = near_arcs[::step]

    m = folium.Map(location=[ref_lat, ref_lon], zoom_start=14, tiles="OpenStreetMap")

    fg_network = folium.FeatureGroup(name="Nearby Road Network (.arcs)")
    road_lines = [
        [[float(nodes[a, 1]), float(nodes[a, 0])], [float(nodes[b, 1]), float(nodes[b, 0])]]
        for a, b in near_arcs
    ]
    if road_lines:
        folium.PolyLine(
            locations=road_lines,
            color="#888888",
            weight=1.5,
            opacity=0.45,
        ).add_to(fg_network)
    fg_network.add_to(m)

    fg_route = folium.FeatureGroup(name="Reference Route (.route)")
    if route is not None and len(route) > 0:
        route_lines = [
            [
                [float(nodes[arcs[aid, 0], 1]), float(nodes[arcs[aid, 0], 0])],
                [float(nodes[arcs[aid, 1], 1]), float(nodes[arcs[aid, 1], 0])],
            ]
            for aid in route
        ]
        folium.PolyLine(
            locations=route_lines,
            color="#1b9e77",
            weight=5,
            opacity=0.85,
            tooltip=f"Reference Route ({n_route_arcs} arcs)",
        ).add_to(fg_route)
    fg_route.add_to(m)

    fg_gps = folium.FeatureGroup(name="Raw GPS Observations (.track)")
    gps_coords = [[float(row[1]), float(row[0])] for row in track]
    folium.PolyLine(
        locations=gps_coords,
        color="#2b6cb0",
        weight=2,
        opacity=0.65,
        dash_array="5, 5",
    ).add_to(fg_gps)

    pt_step = max(1, len(track) // 600)
    for i in range(0, len(track), pt_step):
        folium.CircleMarker(
            location=[float(track[i, 1]), float(track[i, 0])],
            radius=3,
            color="#1d4ed8",
            fill=True,
            fill_opacity=0.8,
            popup=f"obs {i} | t={track[i, 2]:.1f}s",
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    fg_gaps = folium.FeatureGroup(name="GPS Gaps (dt > 2s) & Start/End")
    if len(track) > 1:
        dts = np.diff(track[:, 2])
        gap_idxs = np.where(dts > 2.0)[0]
        for idx in gap_idxs[:200]:
            color = "#dc2626" if dts[idx] > 5.0 else "#ea580c"
            folium.CircleMarker(
                location=[float(track[idx + 1, 1]), float(track[idx + 1, 0])],
                radius=6,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.9,
                popup=f"GPS gap at obs {idx}->{idx+1}: dt={dts[idx]:.1f}s",
            ).add_to(fg_gaps)

    folium.Marker(
        location=[float(track[0, 1]), float(track[0, 0])],
        popup="Start",
        icon=folium.Icon(color="green", icon="play", prefix="fa"),
    ).add_to(fg_gaps)
    folium.Marker(
        location=[float(track[-1, 1]), float(track[-1, 0])],
        popup="End",
        icon=folium.Icon(color="red", icon="stop", prefix="fa"),
    ).add_to(fg_gaps)
    fg_gaps.add_to(m)

    sw = [float(track[:, 1].min()), float(track[:, 0].min())]
    ne = [float(track[:, 1].max()), float(track[:, 0].max())]
    m.fit_bounds([sw, ne])
    folium.LayerControl(collapsed=False).add_to(m)

    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background-color: white; border: 2px solid #555; border-radius: 6px;
                padding: 10px 14px; font-family: Arial, sans-serif; font-size: 12px;">
      <b>PathSnap EDA - {trip['partition']} / {trip['trip_id']}</b><br>
      <span style="color:#1d4ed8;">&#9679;</span> Raw GPS Track ({len(track)} pts)<br>
      <span style="color:#1b9e77;">&#9644;</span> Reference Route ({n_route_arcs} arcs)<br>
      <span style="color:#888888;">&#9644;</span> Road Network (.arcs)<br>
      <span style="color:#ea580c;">&#9679;</span> Gap &gt; 2s &nbsp;
      <span style="color:#dc2626;">&#9679;</span> Gap &gt; 5s
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.save(out_html_path)


def plot_sample_grid(sample_trips, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.flatten()

    for ax, trip in zip(axes, sample_trips):
        track = trip["track"]
        nodes = trip["nodes"]
        arcs = trip["arcs"]
        route = trip["route"]

        ref_lon = float(track[:, 0].mean())
        ref_lat = float(track[:, 1].mean())
        tx, ty = project_to_local_meters(track[:, 0], track[:, 1], ref_lon, ref_lat)
        nx, ny = project_to_local_meters(nodes[:, 0], nodes[:, 1], ref_lon, ref_lat)

        pad = 150.0
        xmin, xmax = tx.min() - pad, tx.max() + pad
        ymin, ymax = ty.min() - pad, ty.max() + pad
        u, v = arcs[:, 0], arcs[:, 1]
        mx = 0.5 * (nx[u] + nx[v])
        my = 0.5 * (ny[u] + ny[v])
        mask = (mx >= xmin) & (mx <= xmax) & (my >= ymin) & (my <= ymax)
        near = arcs[mask]
        if len(near) > 4000:
            near = near[:: max(1, len(near) // 4000)]

        for a, b in near:
            ax.plot([nx[a], nx[b]], [ny[a], ny[b]], color="#cccccc", linewidth=0.6, zorder=1)

        if route is not None and len(route) > 0:
            for idx, aid in enumerate(route):
                a, b = arcs[aid]
                lbl = "Reference Route" if idx == 0 else None
                ax.plot([nx[a], nx[b]], [ny[a], ny[b]], color="#16a34a", linewidth=2.0, zorder=3, label=lbl)

        ax.scatter(tx, ty, s=3.5, color="#2563eb", alpha=0.45, zorder=2, label="Raw GPS (.track)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.margins(0.05)
        n_rt = len(route) if route is not None else 0
        ax.set_title(f"{trip['partition']} / {trip['trip_id']} ({len(track)} pts, {n_rt} route arcs)")
        ax.set_xlabel("Local Easting (m)")
        ax.set_ylabel("Local Northing (m)")
        ax.legend(loc="upper right", fontsize=8)

    plt.suptitle("Sample GPS Trajectories vs. Road Network and Reference Routes", fontsize=13, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def run_eda(repo_root):
    raw_root = os.path.join(repo_root, "data", "raw")
    tables_dir = os.path.join(repo_root, "outputs", "eda", "tables")
    plots_dir = os.path.join(repo_root, "outputs", "eda", "plots")
    maps_dir = os.path.join(repo_root, "outputs", "eda", "maps")
    for d in [tables_dir, plots_dir, maps_dir]:
        os.makedirs(d, exist_ok=True)

    rows = []
    all_dts = {"train": [], "student_test": []}
    all_noise_dists = {"train": [], "student_test": []}

    for part in ["train", "student_test"]:
        trip_ids = get_trip_ids(raw_root, part)
        print(f"Scanning {part} ({len(trip_ids)} trips)...")
        for tid in trip_ids:
            trip = load_trip(raw_root, part, tid)
            track = trip["track"]
            nodes = trip["nodes"]
            arcs = trip["arcs"]
            route = trip["route"]

            missing_vals = int(np.isnan(track).sum() + np.isnan(nodes).sum())
            ref_lon = float(track[:, 0].mean())
            ref_lat = float(track[:, 1].mean())
            tx, ty = project_to_local_meters(track[:, 0], track[:, 1], ref_lon, ref_lat)
            nx, ny = project_to_local_meters(nodes[:, 0], nodes[:, 1], ref_lon, ref_lat)

            bbox_w_km = float((tx.max() - tx.min()) / 1000.0)
            bbox_h_km = float((ty.max() - ty.min()) / 1000.0)
            duration_s = float(track[-1, 2] - track[0, 2]) if len(track) > 1 else 0.0

            dts = np.diff(track[:, 2]) if len(track) > 1 else np.array([])
            non_mono = int((dts <= 0).sum()) if len(dts) > 0 else 0
            pct_1s = float(100.0 * (np.abs(dts - 1.0) <= 1e-4).mean()) if len(dts) > 0 else 100.0
            gaps_gt_5s = int((dts > 5.0).sum()) if len(dts) > 0 else 0
            max_gap = float(dts.max()) if len(dts) > 0 else 0.0
            if len(dts) > 0:
                all_dts[part].append(dts)

            iso_nodes, n_comp = compute_graph_stats(len(nodes), arcs)

            disc_route = 0
            if route is not None and len(route) > 1:
                disc_route = int(sum(1 for i in range(len(route) - 1) if arcs[route[i], 1] != arcs[route[i + 1], 0]))

            _, _, matched_dist = align_track_to_route(tx, ty, track[:, 2], nx, ny, arcs, route)
            all_noise_dists[part].append(matched_dist)

            rows.append({
                "partition": part,
                "trip_id": tid,
                "n_gps_points": len(track),
                "duration_sec": round(duration_s, 1),
                "n_nodes": len(nodes),
                "n_arcs": len(arcs),
                "n_route_arcs": len(route) if route is not None else 0,
                "mean_lon": round(ref_lon, 5),
                "mean_lat": round(ref_lat, 5),
                "bbox_width_km": round(bbox_w_km, 3),
                "bbox_height_km": round(bbox_h_km, 3),
                "missing_values": missing_vals,
                "non_monotonic_steps": non_mono,
                "pct_exact_1s_dt": round(pct_1s, 2),
                "n_gaps_gt_5s": gaps_gt_5s,
                "max_dt_gap_s": round(max_gap, 1),
                "isolated_nodes": iso_nodes,
                "n_connected_components": n_comp,
                "disconnected_route_transitions": disc_route,
                "median_gps_to_route_dist_m": round(float(np.median(matched_dist)), 2),
                "p95_gps_to_route_dist_m": round(float(np.percentile(matched_dist, 95)), 2),
            })

    df_trips = pd.DataFrame(rows)
    df_trips.to_csv(os.path.join(tables_dir, "trip_level_stats.csv"), index=False)

    inv_rows = []
    for part in ["train", "student_test"]:
        sub = df_trips[df_trips["partition"] == part]
        inv_rows.append({
            "partition": part,
            "n_trips": len(sub),
            "total_files": len(sub) * 4,
            "total_gps_points": int(sub["n_gps_points"].sum()),
            "mean_gps_per_trip": round(float(sub["n_gps_points"].mean()), 1),
            "min_gps_per_trip": int(sub["n_gps_points"].min()),
            "max_gps_per_trip": int(sub["n_gps_points"].max()),
            "total_nodes": int(sub["n_nodes"].sum()),
            "total_arcs": int(sub["n_arcs"].sum()),
            "total_route_arcs": int(sub["n_route_arcs"].sum()),
            "median_noise_dist_m": round(float(np.median(np.concatenate(all_noise_dists[part]))), 2),
            "p95_noise_dist_m": round(float(np.percentile(np.concatenate(all_noise_dists[part]), 95)), 2),
        })
    df_inv = pd.DataFrame(inv_rows)
    df_inv.to_csv(os.path.join(tables_dir, "dataset_inventory.csv"), index=False)

    dq_rows = []
    for part in ["train", "student_test"]:
        sub = df_trips[df_trips["partition"] == part]
        dts_all = np.concatenate(all_dts[part])
        dq_rows.append({
            "partition": part,
            "missing_or_nan_values": int(sub["missing_values"].sum()),
            "non_monotonic_timestamps": int(sub["non_monotonic_steps"].sum()),
            "pct_steps_exactly_1s": round(float(100.0 * (np.abs(dts_all - 1.0) <= 1e-4).mean()), 2),
            "steps_gt_2s": int((dts_all > 2.0).sum()),
            "steps_gt_5s": int((dts_all > 5.0).sum()),
            "max_single_gap_s": round(float(dts_all.max()), 1),
            "total_isolated_nodes": int(sub["isolated_nodes"].sum()),
            "trips_with_multi_components": int((sub["n_connected_components"] > 1).sum()),
            "disconnected_route_transitions": int(sub["disconnected_route_transitions"].sum()),
            "trips_with_median_offset_gt_30m": int((sub["median_gps_to_route_dist_m"] > 30.0).sum()),
        })
    df_dq = pd.DataFrame(dq_rows)
    df_dq.to_csv(os.path.join(tables_dir, "data_quality_report.csv"), index=False)

    # Plot 1: Trip length and duration
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for part, col in [("train", "#2563eb"), ("student_test", "#16a34a")]:
        sub = df_trips[df_trips["partition"] == part]
        axes[0].hist(sub["n_gps_points"], bins=15, alpha=0.55, label=f"{part} (n={len(sub)})", color=col, density=True)
        axes[1].hist(np.hypot(sub["bbox_width_km"], sub["bbox_height_km"]), bins=15, alpha=0.55, label=part, color=col, density=True)
    axes[0].set_title("GPS Observations per Trip")
    axes[0].set_xlabel("Number of GPS Points")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].set_title("Trajectory Spatial Span (Diagonal km)")
    axes[1].set_xlabel("Bounding Box Diagonal (km)")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "01_trip_length_and_duration.png"), dpi=180)
    plt.close(fig)

    # Plot 2: Sampling intervals and gaps
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    tr_dts = np.concatenate(all_dts["train"])
    st_dts = np.concatenate(all_dts["student_test"])
    bins_dt = np.arange(0.5, 11.5, 1.0)
    axes[0].hist(np.clip(tr_dts, 0, 10.5), bins=bins_dt, alpha=0.65, label="train", color="#2563eb", density=True)
    axes[0].hist(np.clip(st_dts, 0, 10.5), bins=bins_dt, alpha=0.55, label="student_test", color="#16a34a", density=True)
    axes[0].set_title("Sampling Interval Distribution (Clipped at 10s)")
    axes[0].set_xlabel("dt between consecutive GPS points (s)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    gap_bins = [2, 5, 10, 30, 60, 10000]
    gap_labels = ["2-5s", "5-10s", "10-30s", "30-60s", ">60s"]
    tr_gaps = tr_dts[tr_dts > 2.0]
    counts, _ = np.histogram(tr_gaps, bins=gap_bins)
    axes[1].bar(gap_labels, counts, color="#ea580c", alpha=0.85)
    axes[1].set_title(f"Irregular GPS Gaps in Train (Total dt > 2s: {len(tr_gaps)})")
    axes[1].set_xlabel("Gap Duration Bucket")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_sampling_intervals_and_gaps.png"), dpi=180)
    plt.close(fig)

    # Plot 3: Road network scale and GPS noise CDF
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for part, col in [("train", "#2563eb"), ("student_test", "#16a34a")]:
        sub = df_trips[df_trips["partition"] == part]
        axes[0].scatter(sub["n_nodes"] / 1000.0, sub["n_arcs"] / 1000.0, alpha=0.7, label=part, color=col)
    axes[0].set_title("Road Network Size per Trip")
    axes[0].set_xlabel("Nodes (thousands)")
    axes[0].set_ylabel("Directed Arcs (thousands)")
    axes[0].legend()

    for part, col in [("train", "#2563eb"), ("student_test", "#16a34a")]:
        d_arr = np.sort(np.concatenate(all_noise_dists[part]))
        y_cdf = np.linspace(0, 1, len(d_arr))
        axes[1].plot(np.clip(d_arr, 0, 50), y_cdf, label=part, color=col, linewidth=2)
    axes[1].axvline(15.0, color="#dc2626", linestyle="--", linewidth=1.2, label="15m reference line")
    axes[1].set_title("CDF of GPS Perpendicular Error to Reference Route")
    axes[1].set_xlabel("Distance to Matched Reference Arc (m, clipped at 50m)")
    axes[1].set_ylabel("Cumulative Fraction of Points")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_road_network_and_gps_noise.png"), dpi=180)
    plt.close(fig)

    # Plot 4: 2x2 sample trip overlay
    train_ids = get_trip_ids(raw_root, "train")
    test_ids = get_trip_ids(raw_root, "student_test")
    sample_trips = [
        load_trip(raw_root, "train", train_ids[0]),
        load_trip(raw_root, "train", train_ids[1]),
        load_trip(raw_root, "student_test", test_ids[0]),
        load_trip(raw_root, "student_test", test_ids[1]),
    ]
    plot_sample_grid(sample_trips, os.path.join(plots_dir, "04_sample_trips_overlay.png"))

    build_folium_map(sample_trips[0], os.path.join(maps_dir, f"eda_map_train_{sample_trips[0]['trip_id']}.html"))
    build_folium_map(sample_trips[2], os.path.join(maps_dir, f"eda_map_student_test_{sample_trips[2]['trip_id']}.html"))

    print("\n=== Dataset Inventory ===")
    print(df_inv.to_string(index=False))
    print("\n=== Data Quality Report ===")
    print(df_dq.to_string(index=False))
    anomalies = df_trips[df_trips["median_gps_to_route_dist_m"] > 30.0]
    if len(anomalies) > 0:
        print("\n=== High-Offset Anomaly Trips (median GPS-to-route > 30m) ===")
        print(anomalies[["partition", "trip_id", "n_gps_points", "median_gps_to_route_dist_m", "p95_gps_to_route_dist_m"]].to_string(index=False))


if __name__ == "__main__":
    repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    run_eda(repo_dir)
