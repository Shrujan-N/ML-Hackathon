"""
map_builder.py — Folium interactive map builder for PathSnap UI.

Renders:
- Base map layers (Google Streets, OpenStreetMap, CartoDB Positron)
- Nearby road network graph (<trip_id>.arcs) so streets remain visible even offline
- Ground truth reference route (<trip_id>.route)
- GPS track observations (<trip_id>.track)
- Predicted route arcs (from FastAPI backend response)
- Start/End badges and a floating map legend
"""

import os
import folium
import numpy as np
import pandas as pd


def _find_trip_file(raw_trip_dir: str, ext: str):
    """Return path to the first file in raw_trip_dir ending with ext, or None."""
    if not raw_trip_dir or not os.path.isdir(raw_trip_dir):
        return None
    for fname in sorted(os.listdir(raw_trip_dir)):
        if fname.endswith(ext):
            return os.path.join(raw_trip_dir, fname)
    return None


def load_raw_nodes(raw_trip_dir: str):
    """
    Load node coordinates from <trip_id>.nodes file.
    Format: each line is 'lon lat'
    Returns dict: {node_id: (lat, lon)}
    """
    path = _find_trip_file(raw_trip_dir, ".nodes")
    if path is None:
        return {}
    coords = np.loadtxt(path, dtype=np.float64)
    if coords.size == 0:
        return {}
    if coords.ndim == 1:
        coords = coords.reshape(1, -1)
    # nodes file has (lon, lat)
    return {i: (float(coords[i, 1]), float(coords[i, 0])) for i in range(len(coords))}


def load_raw_track(raw_trip_dir: str):
    """
    Load raw GPS track from <trip_id>.track file.
    Format: 'lon lat timestamp'
    Returns list of (lat, lon) tuples.
    """
    path = _find_trip_file(raw_trip_dir, ".track")
    if path is None:
        return []
    data = np.loadtxt(path, dtype=np.float64)
    if data.size == 0:
        return []
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # lat is col 1, lon is col 0
    return [(float(row[1]), float(row[0])) for row in data]


def load_raw_arcs(raw_trip_dir: str) -> np.ndarray:
    """
    Load directed road graph arcs from <trip_id>.arcs file.
    Format: each line is 'from_node to_node'
    Returns (E, 2) int64 array.
    """
    path = _find_trip_file(raw_trip_dir, ".arcs")
    if path is None:
        return np.empty((0, 2), dtype=np.int64)
    arcs = np.loadtxt(path, dtype=np.int64)
    if arcs.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if arcs.ndim == 1:
        arcs = arcs.reshape(1, -1)
    return arcs


def load_raw_route(raw_trip_dir: str) -> np.ndarray:
    """
    Load ground truth arc index sequence from <trip_id>.route file.
    Returns 1D int64 array of arc IDs.
    """
    path = _find_trip_file(raw_trip_dir, ".route")
    if path is None:
        return np.empty((0,), dtype=np.int64)
    route = np.loadtxt(path, dtype=np.int64)
    if route.size == 0:
        return np.empty((0,), dtype=np.int64)
    if route.ndim == 0:
        route = route.reshape(1)
    return route


def _resolve_raw_dir(repo_root: str, partition: str, trip_id: str) -> str:
    """Resolve raw trip folder matching either zero-padded (00000052) or raw ID."""
    tid_str = f"{int(trip_id):08d}" if str(trip_id).isdigit() else str(trip_id)
    candidate = os.path.join(repo_root, "data", "raw", partition, tid_str)
    if os.path.isdir(candidate):
        return candidate
    alt = os.path.join(repo_root, "data", "raw", partition, str(trip_id))
    return alt if os.path.isdir(alt) else candidate


def build_folium_map(
    trip_id: str,
    predictions_df: pd.DataFrame = None,
    repo_root: str = ".",
    partition: str = "student_test",
) -> folium.Map:
    """
    Build an interactive Folium map visualization for a trip.
    predictions_df (optional) contains: from_node, to_node, predicted_arc_id, confidence.
    """
    raw_dir = _resolve_raw_dir(repo_root, partition, str(trip_id))

    node_dict = {}
    gps_coords = []
    arcs = np.empty((0, 2), dtype=np.int64)
    route = np.empty((0,), dtype=np.int64)

    if os.path.isdir(raw_dir):
        node_dict = load_raw_nodes(raw_dir)
        gps_coords = load_raw_track(raw_dir)
        arcs = load_raw_arcs(raw_dir)
        route = load_raw_route(raw_dir)

    # Determine bounding box and map center from GPS points or predicted/route nodes
    bound_points = list(gps_coords)
    if not bound_points and predictions_df is not None and not predictions_df.empty:
        for _, row in predictions_df.iterrows():
            u, v = row.get("from_node"), row.get("to_node")
            if u in node_dict:
                bound_points.append(node_dict[u])
            if v in node_dict:
                bound_points.append(node_dict[v])

    if bound_points:
        lats = [p[0] for p in bound_points]
        lons = [p[1] for p in bound_points]
        center_lat = float(np.mean(lats))
        center_lon = float(np.mean(lons))
        min_lat, max_lat = float(np.min(lats)), float(np.max(lats))
        min_lon, max_lon = float(np.min(lons)), float(np.max(lons))
    elif node_dict:
        pts = list(node_dict.values())
        center_lat, center_lon = pts[0][0], pts[0][1]
        min_lat = max_lat = center_lat
        min_lon = max_lon = center_lon
    else:
        center_lat, center_lon = 56.81, 35.86
        min_lat = max_lat = center_lat
        min_lon = max_lon = center_lon

    # Initialize map without hardcoding CartoDB only; add reliable multi-tile layers
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=None,
        control_scale=True,
    )

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}",
        attr="Google Maps",
        name="Google Streets",
        overlay=False,
        control=True,
        show=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="OpenStreetMap",
        name="OpenStreetMap",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        attr="&copy; OpenStreetMap contributors &copy; CARTO",
        name="CartoDB Positron",
        overlay=False,
        control=True,
        show=False,
    ).add_to(m)

    # Light background canvas + invalidateSize fix so tiles/vector layers render cleanly in Streamlit tabs
    map_var = m.get_name()
    render_fix = f"""
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            background-color: #f1f5f9;
        }}
        .leaflet-container {{
            background: #f1f5f9 !important;
        }}
    </style>
    <script>
        (function() {{
            function refreshLeafletMap() {{
                var mapObj = window["{map_var}"];
                if (mapObj && typeof mapObj.invalidateSize === "function") {{
                    mapObj.invalidateSize(true);
                }}
            }}
            setTimeout(refreshLeafletMap, 120);
            setTimeout(refreshLeafletMap, 500);
            setTimeout(refreshLeafletMap, 1200);
            window.addEventListener("resize", refreshLeafletMap);
            document.addEventListener("visibilitychange", refreshLeafletMap);
        }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(render_fix))

    # 1. Nearby Road Network (.arcs) within ~350m padding of the trip bounding box
    if len(arcs) > 0 and len(node_dict) > 0 and bound_points:
        cos_lat = max(0.1, float(np.cos(np.radians(center_lat))))
        pad_lat = 350.0 / 110540.0
        pad_lon = 350.0 / (111320.0 * cos_lat)
        lat_lo, lat_hi = min_lat - pad_lat, max_lat + pad_lat
        lon_lo, lon_hi = min_lon - pad_lon, max_lon + pad_lon

        road_lines = []
        for u_raw, v_raw in arcs:
            u = int(u_raw)
            v = int(v_raw)
            if u in node_dict and v in node_dict:
                u_pt = node_dict[u]
                v_pt = node_dict[v]
                mid_lat = 0.5 * (u_pt[0] + v_pt[0])
                mid_lon = 0.5 * (u_pt[1] + v_pt[1])
                if lat_lo <= mid_lat <= lat_hi and lon_lo <= mid_lon <= lon_hi:
                    road_lines.append([u_pt, v_pt])

        if len(road_lines) > 4500:
            step = max(1, len(road_lines) // 4500)
            road_lines = road_lines[::step]

        if road_lines:
            net_fg = folium.FeatureGroup(name="Road Network (.arcs)")
            folium.PolyLine(
                locations=road_lines,
                color="#94a3b8",
                weight=2.0,
                opacity=0.75,
            ).add_to(net_fg)
            net_fg.add_to(m)

    # 2. Ground Truth Reference Route (.route)
    if len(route) > 0 and len(arcs) > 0 and len(node_dict) > 0:
        route_lines = []
        n_arcs = len(arcs)
        for aid_raw in route:
            aid = int(aid_raw)
            if 0 <= aid < n_arcs:
                u = int(arcs[aid, 0])
                v = int(arcs[aid, 1])
                if u in node_dict and v in node_dict:
                    route_lines.append([node_dict[u], node_dict[v]])
        if route_lines:
            ref_fg = folium.FeatureGroup(name="Reference Route (.route)")
            folium.PolyLine(
                locations=route_lines,
                color="#16a34a",
                weight=6,
                opacity=0.65,
                tooltip=f"Reference Route ({len(route_lines)} arcs)",
            ).add_to(ref_fg)
            ref_fg.add_to(m)

    # 3. Raw GPS Track (.track)
    if gps_coords:
        gps_fg = folium.FeatureGroup(name="GPS Track (Raw Observations)")
        folium.PolyLine(
            locations=gps_coords,
            color="#2563eb",
            weight=2,
            opacity=0.6,
            dash_array="5, 5",
        ).add_to(gps_fg)

        pt_step = max(1, len(gps_coords) // 600)
        for i in range(0, len(gps_coords), pt_step):
            lat, lon = gps_coords[i]
            folium.CircleMarker(
                location=[lat, lon],
                radius=3.2,
                color="#1d4ed8",
                fill=True,
                fill_color="#3b82f6",
                fill_opacity=0.85,
                popup=f"Obs {i}: ({lat:.6f}, {lon:.6f})",
            ).add_to(gps_fg)
        gps_fg.add_to(m)

    # 4. Predicted Route Segments
    if predictions_df is not None and not predictions_df.empty and len(node_dict) > 0:
        pred_fg = folium.FeatureGroup(name="Predicted Route")
        seen_arcs = set()
        for _, row in predictions_df.iterrows():
            arc_id = row.get("predicted_arc_id", None)
            u_raw = row.get("from_node", None)
            v_raw = row.get("to_node", None)
            if pd.isna(u_raw) or pd.isna(v_raw):
                continue
            u = int(u_raw)
            v = int(v_raw)
            conf = float(row.get("confidence", 1.0))
            is_corr = row.get("is_correct", None)

            if u in node_dict and v in node_dict:
                arc_key = (u, v)
                if arc_key not in seen_arcs:
                    seen_arcs.add(arc_key)
                    match_txt = ""
                    if is_corr is not None:
                        match_txt = f" | Match: {'Correct' if bool(is_corr) else 'Mismatch'}"
                    folium.PolyLine(
                        locations=[node_dict[u], node_dict[v]],
                        color="#dc2626",
                        weight=3.5,
                        opacity=0.95,
                        popup=f"Arc {arc_id} | Conf: {conf:.2f} | Nodes: {u}->{v}{match_txt}",
                    ).add_to(pred_fg)
        pred_fg.add_to(m)

    # 5. Start & End Markers (pure CSS/HTML badges — work 100% offline)
    if gps_coords:
        markers_fg = folium.FeatureGroup(name="Start / End")
        folium.Marker(
            location=gps_coords[0],
            tooltip="Trip Start",
            icon=folium.DivIcon(
                html='<div style="background:#15803d;color:#fff;font-size:10px;font-weight:700;'
                     'padding:2px 6px;border-radius:4px;border:1px solid #fff;'
                     'box-shadow:0 1px 4px rgba(0,0,0,0.4);white-space:nowrap;">START</div>'
            ),
        ).add_to(markers_fg)
        folium.Marker(
            location=gps_coords[-1],
            tooltip="Trip End",
            icon=folium.DivIcon(
                html='<div style="background:#b91c1c;color:#fff;font-size:10px;font-weight:700;'
                     'padding:2px 6px;border-radius:4px;border:1px solid #fff;'
                     'box-shadow:0 1px 4px rgba(0,0,0,0.4);white-space:nowrap;">END</div>'
            ),
        ).add_to(markers_fg)
        markers_fg.add_to(m)

    if bound_points and (max_lat > min_lat or max_lon > min_lon):
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(25, 25))

    folium.LayerControl(collapsed=True).add_to(m)

    # 6. Floating Legend
    has_preds = predictions_df is not None and not predictions_df.empty
    pred_legend = (
        '<div><span style="display:inline-block;width:14px;height:4px;background:#dc2626;'
        'vertical-align:middle;margin-right:6px;"></span>Predicted Route</div>'
        if has_preds
        else ""
    )
    legend_html = f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 16px;
        z-index: 9999;
        background: rgba(255, 255, 255, 0.94);
        color: #0f172a;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 8px 12px;
        font-size: 12px;
        line-height: 1.55;
        font-family: system-ui, -apple-system, sans-serif;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
    ">
        <div style="font-weight:700;margin-bottom:4px;">PathSnap Map Legend</div>
        <div><span style="display:inline-block;width:14px;height:3px;background:#94a3b8;vertical-align:middle;margin-right:6px;"></span>Road Network (.arcs)</div>
        <div><span style="display:inline-block;width:14px;height:5px;background:#16a34a;vertical-align:middle;margin-right:6px;"></span>Reference Route (.route)</div>
        {pred_legend}
        <div><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#3b82f6;border:1px solid #1d4ed8;vertical-align:middle;margin-left:3px;margin-right:9px;"></span>GPS Track (.track)</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m
