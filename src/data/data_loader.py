import os
import numpy as np


def get_trip_ids(data_root, partition):
    part_dir = os.path.join(data_root, partition)
    trips = [
        d for d in sorted(os.listdir(part_dir))
        if os.path.isdir(os.path.join(part_dir, d))
    ]
    return trips


def load_trip(data_root, partition, trip_id):
    base_path = os.path.join(data_root, partition, trip_id, trip_id)

    track = np.loadtxt(base_path + ".track", dtype=np.float64)
    if track.ndim == 1:
        track = track.reshape(1, -1)

    nodes = np.loadtxt(base_path + ".nodes", dtype=np.float64)
    if nodes.ndim == 1:
        nodes = nodes.reshape(1, -1)

    arcs = np.loadtxt(base_path + ".arcs", dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)

    route_path = base_path + ".route"
    route = None
    if os.path.exists(route_path):
        route = np.loadtxt(route_path, dtype=np.int64)
        if route.ndim == 0:
            route = route.reshape(1)

    return {
        "trip_id": trip_id,
        "partition": partition,
        "track": track,
        "nodes": nodes,
        "arcs": arcs,
        "route": route,
    }


def project_to_local_meters(lon, lat, ref_lon, ref_lat):
    # Degree coordinates cannot be used directly with Euclidean distance because
    # 1 degree of longitude shrinks by cos(latitude) as we move away from the equator.
    lon_scale = 111320.0 * np.cos(np.radians(ref_lat))
    lat_scale = 110540.0
    x_m = (lon - ref_lon) * lon_scale
    y_m = (lat - ref_lat) * lat_scale
    return x_m, y_m


def compute_gps_kinematics(track_x, track_y, timestamps):
    n = len(track_x)
    dt_prev = np.ones(n, dtype=np.float64)
    if n > 1:
        dt_raw = np.diff(timestamps)
        dt_prev[1:] = np.where(dt_raw > 0.0, dt_raw, 1.0)

    # Use central difference for interior points so heading is smoother against 1 Hz GPS jitter
    dx = np.zeros(n, dtype=np.float64)
    dy = np.zeros(n, dtype=np.float64)
    dt_vec = np.ones(n, dtype=np.float64)

    if n >= 3:
        dx[1:-1] = track_x[2:] - track_x[:-2]
        dy[1:-1] = track_y[2:] - track_y[:-2]
        dt_vec[1:-1] = np.maximum(timestamps[2:] - timestamps[:-2], 1e-3)
        dx[0] = track_x[1] - track_x[0]
        dy[0] = track_y[1] - track_y[0]
        dt_vec[0] = dt_prev[1]
        dx[-1] = track_x[-1] - track_x[-2]
        dy[-1] = track_y[-1] - track_y[-2]
        dt_vec[-1] = dt_prev[-1]
    elif n == 2:
        dx[:] = track_x[1] - track_x[0]
        dy[:] = track_y[1] - track_y[0]
        dt_vec[:] = dt_prev[1]

    step_dist = np.hypot(dx, dy)
    speed_mps = step_dist / dt_vec
    gps_heading = np.arctan2(dy, dx)

    # If the car is nearly stationary, keep the last valid heading
    last_angle = 0.0
    for i in range(n):
        if step_dist[i] > 0.5:
            last_angle = gps_heading[i]
        else:
            gps_heading[i] = last_angle

    return dt_prev, speed_mps, gps_heading


def point_to_segment_dist(px, py, ax, ay, bx, by):
    dx = bx - ax
    dy = by - ay
    seg_len2 = dx * dx + dy * dy
    safe_len2 = np.where(seg_len2 > 1e-12, seg_len2, 1.0)
    proj = ((px - ax) * dx + (py - ay) * dy) / safe_len2
    proj_clamped = np.clip(proj, 0.0, 1.0)
    cx = ax + proj_clamped * dx
    cy = ay + proj_clamped * dy
    dist = np.hypot(px - cx, py - cy)
    return dist, proj_clamped


def align_track_to_route(track_x, track_y, timestamps, nodes_x, nodes_y, arcs, route, max_jump=40):
    n = len(track_x)
    m = len(route) if route is not None else 0
    if m == 0:
        return np.full(n, -1, dtype=np.int64), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.float64)
    if m == 1:
        u, v = arcs[route[0]]
        dist, _ = point_to_segment_dist(track_x, track_y, nodes_x[u], nodes_y[u], nodes_x[v], nodes_y[v])
        return np.full(n, route[0], dtype=np.int64), np.zeros(n, dtype=np.int64), dist

    u_nodes = arcs[route, 0]
    v_nodes = arcs[route, 1]
    ax = nodes_x[u_nodes]
    ay = nodes_y[u_nodes]
    bx = nodes_x[v_nodes]
    by = nodes_y[v_nodes]
    arc_angles = np.arctan2(by - ay, bx - ax)

    _, speed_mps, gps_heading = compute_gps_kinematics(track_x, track_y, timestamps)
    moving_mask = (speed_mps > 1.5).astype(np.float32)

    # Compute emission matrix in chunks to keep memory usage small on long trips
    dist_mat = np.empty((n, m), dtype=np.float32)
    cost_mat = np.empty((n, m), dtype=np.float32)
    chunk_size = 2000
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        d_chunk, _ = point_to_segment_dist(
            track_x[start:end, None],
            track_y[start:end, None],
            ax[None, :],
            ay[None, :],
            bx[None, :],
            by[None, :],
        )
        dist_mat[start:end] = d_chunk

        # Add a gentle directional penalty when moving so opposite/perpendicular turns align cleanly
        ang_diff = gps_heading[start:end, None] - arc_angles[None, :]
        heading_pen = 2.0 * (1.0 - np.cos(ang_diff)) * moving_mask[start:end, None]
        cost_mat[start:end] = d_chunk + heading_pen

    # Monotonic DP along the chronological route sequence
    dp = np.full((n, m), np.inf, dtype=np.float64)
    back = np.zeros((n, m), dtype=np.int32)
    dp[0] = cost_mat[0]

    jump_limit = min(max_jump, m)
    j_indices = np.arange(m, dtype=np.int32)

    for i in range(1, n):
        prev = dp[i - 1]
        best_prev = prev.copy()
        best_j = j_indices.copy()
        for k in range(1, jump_limit):
            cand = prev[:-k] + 0.5 * k
            mask = cand < best_prev[k:]
            best_prev[k:][mask] = cand[mask]
            best_j[k:][mask] = j_indices[:-k][mask]
        dp[i] = best_prev + cost_mat[i]
        back[i] = best_j

    if not np.isfinite(dp[-1]).any():
        route_pos = np.argmin(dist_mat, axis=1).astype(np.int64)
    else:
        route_pos = np.zeros(n, dtype=np.int64)
        route_pos[-1] = int(np.argmin(dp[-1]))
        for i in range(n - 1, 0, -1):
            route_pos[i - 1] = back[i, route_pos[i]]

    true_arc_ids = route[route_pos]
    matched_dist = dist_mat[np.arange(n), route_pos].astype(np.float64)
    return true_arc_ids, route_pos, matched_dist
