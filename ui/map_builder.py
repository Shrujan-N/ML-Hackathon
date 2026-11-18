"""
map_builder.py — Folium interactive map builder for PathSnap UI.

Renders:
- GPS track observations (CircleMarkers)
- Predicted route (Polylines connecting arc from_node to to_node)
- Reference ground truth route (if available)
"""

import os
import folium
import numpy as np
import pandas as pd


def load_raw_nodes(raw_trip_dir: str):
    """
    Load node coordinates from <trip_id>.nodes file.
    Format: each line is 'lon lat'
    Returns dict: {node_id: (lat, lon)}
    """
    for fname in os.listdir(raw_trip_dir):
        if fname.endswith(".nodes"):
            path = os.path.join(raw_trip_dir, fname)
            coords = np.loadtxt(path, dtype=np.float64)
            if coords.ndim == 1:
                coords = coords.reshape(1, -1)
            # nodes file has lon, lat
            return {i: (coords[i, 1], coords[i, 0]) for i in range(len(coords))}
    return {}


def load_raw_track(raw_trip_dir: str):
    """
    Load raw GPS track from <trip_id>.track file.
    Format: 'lon lat timestamp'
    Returns array of (lat, lon)
    """
    for fname in os.listdir(raw_trip_dir):
        if fname.endswith(".track"):
            path = os.path.join(raw_trip_dir, fname)
            data = np.loadtxt(path, dtype=np.float64)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            # lat is col 1, lon is col 0
            return list(zip(data[:, 1], data[:, 0]))
    return []


def build_folium_map(
    trip_id: str,
    predictions_df: pd.DataFrame,
    repo_root: str,
    partition: str = "student_test"
) -> folium.Map:
    """
    Builds an interactive Folium map visualization.
    predictions_df must contain: from_node, to_node, predicted_arc_id, confidence
    """
    # Find trip folder (e.g. "00000000" or int string)
    tid_int = int(trip_id) if str(trip_id).isdigit() else trip_id
    tid_str = f"{tid_int:08d}" if isinstance(tid_int, int) else str(trip_id)

    raw_dir = os.path.join(repo_root, "data", "raw", partition, tid_str)
    if not os.path.exists(raw_dir):
        # try without leading zeroes
        raw_dir = os.path.join(repo_root, "data", "raw", partition, str(trip_id))

    node_dict = {}
    gps_coords = []
    if os.path.exists(raw_dir):
        node_dict = load_raw_nodes(raw_dir)
        gps_coords = load_raw_track(raw_dir)

    # Center map on average GPS point or node
    if len(gps_coords) > 0:
        center_lat = np.mean([p[0] for p in gps_coords])
        center_lon = np.mean([p[1] for p in gps_coords])
    elif len(node_dict) > 0:
        first_node = next(iter(node_dict.values()))
        center_lat, center_lon = first_node[0], first_node[1]
    else:
        center_lat, center_lon = 45.0, 7.0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="CartoDB positron")

    # Add GPS track points
    gps_fg = folium.FeatureGroup(name="GPS Track (Raw Observations)")
    for i, (lat, lon) in enumerate(gps_coords):
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color="#2563eb",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=0.7,
            popup=f"Obs {i}: ({lat:.6f}, {lon:.6f})"
        ).add_to(gps_fg)
    gps_fg.add_to(m)

    # Add predicted route segments
    pred_fg = folium.FeatureGroup(name="Predicted Route (Model 3 RF)")
    seen_arcs = set()
    for _, row in predictions_df.iterrows():
        arc_id = row.get("predicted_arc_id", None)
        u = row.get("from_node", None)
        v = row.get("to_node", None)
        conf = row.get("confidence", 1.0)
        
        if u in node_dict and v in node_dict:
            u_pt = node_dict[u]
            v_pt = node_dict[v]
            arc_key = (u, v)
            if arc_key not in seen_arcs:
                seen_arcs.add(arc_key)
                folium.PolyLine(
                    locations=[u_pt, v_pt],
                    color="#16a34a",
                    weight=4,
                    opacity=0.85,
                    popup=f"Arc {arc_id} | Conf: {conf:.2f} | Nodes: {u}->{v}"
                ).add_to(pred_fg)

    pred_fg.add_to(m)
    folium.LayerControl().add_to(m)
    return m
