"""Mandatory interactive Folium geographical visualisations for PathSnap (Member 3).

Meets Deliverables.pdf Section 6 requirements:
- Visualises original GPS observations, relevant road-network geometry,
  reference route (where available), optional classifier-only route, and predicted route.
- Distinguishable colours and markers per layer.
- Clear legend and optional metrics summary card.
- Automatic geographical extent and zoom fitting (`fit_bounds`).
- Interactive layer controls (`folium.LayerControl`).
- Exports standalone interactive HTML files and HTML strings (for Member 4's UI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import folium
import numpy as np
from branca.element import Element

from src.hmm.road_graph import RoadNetworkGraph


def _build_arc_polylines(
    graph: RoadNetworkGraph,
    arc_ids: Sequence[int],
) -> List[List[Tuple[float, float]]]:
    """Merge consecutive connected arcs into continuous lat/lon polylines for crisp rendering."""
    polylines: List[List[Tuple[float, float]]] = []
    current_line: List[Tuple[float, float]] = []
    prev_to_node: Optional[int] = None

    for raw_id in arc_ids:
        arc_id = int(raw_id)
        if arc_id < 0 or arc_id >= graph.num_arcs:
            continue
        u = int(graph.from_nodes[arc_id])
        v = int(graph.to_nodes[arc_id])
        pt_u = (float(graph.nodes_lonlat[u, 1]), float(graph.nodes_lonlat[u, 0]))
        pt_v = (float(graph.nodes_lonlat[v, 1]), float(graph.nodes_lonlat[v, 0]))

        if current_line and prev_to_node == u:
            current_line.append(pt_v)
        else:
            if current_line:
                polylines.append(current_line)
            current_line = [pt_u, pt_v]
        prev_to_node = v

    if current_line:
        polylines.append(current_line)
    return polylines


def _legend_html(
    trip_id: str,
    has_road_network: bool,
    has_reference: bool,
    has_classifier_only: bool,
    has_predicted: bool,
) -> str:
    items: List[str] = []
    items.append(
        '<div><span style="display:inline-block;width:11px;height:11px;border-radius:50%;'
        'background:#2563eb;border:1px solid #1d4ed8;margin-right:7px;vertical-align:middle;"></span>'
        "Raw GPS Observations</div>"
    )
    if has_road_network:
        items.append(
            '<div><span style="display:inline-block;width:16px;height:3px;'
            'background:#64748b;margin-right:7px;vertical-align:middle;"></span>'
            "Relevant Road Network</div>"
        )
    if has_reference:
        items.append(
            '<div><span style="display:inline-block;width:16px;height:4px;'
            'background:#16a34a;margin-right:7px;vertical-align:middle;"></span>'
            "Reference Route (Ground Truth)</div>"
        )
    if has_classifier_only:
        items.append(
            '<div><span style="display:inline-block;width:16px;height:0;border-top:3px dashed #f97316;'
            'margin-right:7px;vertical-align:middle;"></span>'
            "Classifier-Only Predicted Route</div>"
        )
    if has_predicted:
        items.append(
            '<div><span style="display:inline-block;width:16px;height:4px;'
            'background:#dc2626;margin-right:7px;vertical-align:middle;"></span>'
            "Predicted Route (Classifier + HMM)</div>"
        )

    body = "".join(items)
    return f"""
    <div style="
        position: fixed;
        bottom: 24px;
        left: 24px;
        z-index: 9999;
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        line-height: 1.7;
        color: #0f172a;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15);
    ">
        <div style="font-weight: 700; margin-bottom: 4px; font-size: 13px;">
            PathSnap Map — Trip {trip_id}
        </div>
        {body}
    </div>
    """


def _metrics_card_html(metrics_summary: Dict[str, Any]) -> str:
    rows = "".join(
        f"<tr><td style='padding:2px 8px 2px 0;color:#475569;'>{k}</td>"
        f"<td style='padding:2px 0;font-weight:600;text-align:right;'>{v}</td></tr>"
        for k, v in metrics_summary.items()
    )
    return f"""
    <div style="
        position: fixed;
        bottom: 24px;
        right: 24px;
        z-index: 9999;
        background-color: rgba(255, 255, 255, 0.95);
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 10px 14px;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 12px;
        color: #0f172a;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.15);
    ">
        <div style="font-weight: 700; margin-bottom: 4px; font-size: 13px;">Route Metrics</div>
        <table style="border-collapse:collapse;">{rows}</table>
    </div>
    """


def create_trip_map(
    graph: RoadNetworkGraph,
    track_lonlat: np.ndarray,
    trip_id: str = "unknown",
    timestamps: Optional[Sequence[float]] = None,
    reference_route_arcs: Optional[Sequence[int]] = None,
    predicted_route_arcs: Optional[Sequence[int]] = None,
    classifier_only_arcs: Optional[Sequence[int]] = None,
    metrics_summary: Optional[Dict[str, Any]] = None,
    road_margin_deg: float = 0.004,
    max_road_arcs: int = 4000,
    max_gps_markers: int = 1200,
    tiles: str = "OpenStreetMap",
) -> folium.Map:
    """Build an interactive Folium map comparing GPS observations, road network, and routes.

    Note: `reference_route_arcs` is strictly optional; pass `None` in evaluator UI contexts
    where ground-truth labels must remain hidden server-side.
    """
    pts = np.asarray(track_lonlat, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2 or len(pts) == 0:
        raise ValueError("track_lonlat must be a non-empty array of shape (T, 2) [lon, lat]")

    lons = pts[:, 0]
    lats = pts[:, 1]
    min_lon, max_lon = float(np.nanmin(lons)), float(np.nanmax(lons))
    min_lat, max_lat = float(np.nanmin(lats)), float(np.nanmax(lats))
    center_lat = 0.5 * (min_lat + max_lat)
    center_lon = 0.5 * (min_lon + max_lon)

    fmap = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles=tiles,
        prefer_canvas=True,
        control_scale=True,
    )

    # Layer 1: Relevant road network cropped to trip bounding box + margin
    local_arc_ids = graph.arcs_in_bbox(
        min_lon - road_margin_deg,
        min_lat - road_margin_deg,
        max_lon + road_margin_deg,
        max_lat + road_margin_deg,
    )
    has_road_network = len(local_arc_ids) > 0
    if has_road_network:
        if len(local_arc_ids) > max_road_arcs:
            step = max(1, len(local_arc_ids) // max_road_arcs)
            display_arcs = local_arc_ids[::step]
        else:
            display_arcs = local_arc_ids

        road_group = folium.FeatureGroup(name="Relevant Road Network", show=True)
        # Render efficiently using GeoJSON MultiLineString (10x lighter HTML than thousands of PolyLines)
        coords = [
            [
                [float(graph.nodes_lonlat[int(graph.from_nodes[aid]), 0]),
                 float(graph.nodes_lonlat[int(graph.from_nodes[aid]), 1])],
                [float(graph.nodes_lonlat[int(graph.to_nodes[aid]), 0]),
                 float(graph.nodes_lonlat[int(graph.to_nodes[aid]), 1])],
            ]
            for aid in display_arcs
        ]
        road_geojson = {
            "type": "Feature",
            "properties": {"name": "Road Network"},
            "geometry": {"type": "MultiLineString", "coordinates": coords},
        }
        folium.GeoJson(
            road_geojson,
            style_function=lambda _: {
                "color": "#64748b",
                "weight": 1.6,
                "opacity": 0.45,
            },
        ).add_to(road_group)
        road_group.add_to(fmap)

    # Layer 2: Reference route (Ground truth — only when provided)
    has_reference = reference_route_arcs is not None and len(reference_route_arcs) > 0
    if has_reference:
        ref_group = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
        for line in _build_arc_polylines(graph, reference_route_arcs):
            folium.PolyLine(
                locations=line,
                color="#16a34a",
                weight=6.5,
                opacity=0.80,
                tooltip=f"Reference Route (Trip {trip_id})",
            ).add_to(ref_group)
        ref_group.add_to(fmap)

    # Layer 3: Classifier-only predicted route (optional ablation layer)
    has_clf_only = classifier_only_arcs is not None and len(classifier_only_arcs) > 0
    if has_clf_only:
        clf_group = folium.FeatureGroup(name="Classifier-Only Predicted Route", show=False)
        for line in _build_arc_polylines(graph, classifier_only_arcs):
            folium.PolyLine(
                locations=line,
                color="#f97316",
                weight=3.8,
                opacity=0.85,
                dash_array="8 6",
                tooltip=f"Classifier-Only Route (Trip {trip_id})",
            ).add_to(clf_group)
        clf_group.add_to(fmap)

    # Layer 4: Predicted route (Classifier + HMM/Viterbi)
    has_predicted = predicted_route_arcs is not None and len(predicted_route_arcs) > 0
    if has_predicted:
        pred_group = folium.FeatureGroup(name="Predicted Route (Classifier + HMM)", show=True)
        pred_lines = _build_arc_polylines(graph, predicted_route_arcs)
        for line in pred_lines:
            folium.PolyLine(
                locations=line,
                color="#dc2626",
                weight=4.0,
                opacity=0.90,
                tooltip=f"Predicted Route — HMM/Viterbi (Trip {trip_id})",
            ).add_to(pred_group)
        if pred_lines:
            folium.CircleMarker(
                location=pred_lines[0][0],
                radius=7,
                color="#991b1b",
                fill=True,
                fill_color="#ef4444",
                fill_opacity=1.0,
                tooltip="Predicted Route Start",
            ).add_to(pred_group)
            folium.CircleMarker(
                location=pred_lines[-1][-1],
                radius=7,
                color="#1e293b",
                fill=True,
                fill_color="#dc2626",
                fill_opacity=1.0,
                tooltip="Predicted Route End",
            ).add_to(pred_group)
        pred_group.add_to(fmap)

    # Layer 5: Raw GPS observations
    gps_group = folium.FeatureGroup(name="Original GPS Observations", show=True)
    gps_latlon = [(float(lat), float(lon)) for lon, lat in zip(lons, lats) if not (np.isnan(lon) or np.isnan(lat))]
    if gps_latlon:
        # Subtle connecting line for temporal progression
        folium.PolyLine(
            locations=gps_latlon,
            color="#3b82f6",
            weight=1.5,
            opacity=0.35,
            dash_array="3 4",
        ).add_to(gps_group)

        stride = max(1, len(gps_latlon) // max_gps_markers)
        for idx in range(0, len(gps_latlon), stride):
            lat_i, lon_i = gps_latlon[idx]
            ts_str = f", t={timestamps[idx]:.1f}s" if (timestamps is not None and idx < len(timestamps)) else ""
            folium.CircleMarker(
                location=(lat_i, lon_i),
                radius=2.8,
                color="#1d4ed8",
                weight=1,
                fill=True,
                fill_color="#2563eb",
                fill_opacity=0.85,
                tooltip=f"GPS obs #{idx}{ts_str} ({lat_i:.5f}, {lon_i:.5f})",
            ).add_to(gps_group)

        # Start & End markers for GPS trajectory
        folium.Marker(
            location=gps_latlon[0],
            tooltip=f"GPS Start (Trip {trip_id})",
            icon=folium.Icon(color="blue", icon="play", prefix="fa"),
        ).add_to(gps_group)
        folium.Marker(
            location=gps_latlon[-1],
            tooltip=f"GPS End (Trip {trip_id})",
            icon=folium.Icon(color="darkblue", icon="stop", prefix="fa"),
        ).add_to(gps_group)
    gps_group.add_to(fmap)

    # Fit view bounds to trajectory
    fmap.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]], padding=(24, 24))

    # Layer control + legend + optional metrics summary
    folium.LayerControl(collapsed=False).add_to(fmap)
    fmap.get_root().html.add_child(
        Element(
            _legend_html(
                trip_id=trip_id,
                has_road_network=has_road_network,
                has_reference=has_reference,
                has_classifier_only=has_clf_only,
                has_predicted=has_predicted,
            )
        )
    )
    if metrics_summary:
        fmap.get_root().html.add_child(Element(_metrics_card_html(metrics_summary)))

    return fmap


def save_trip_map(fmap: folium.Map, output_path: str | Path) -> Path:
    """Save a Folium map to an interactive standalone HTML file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fmap.save(str(p))
    return p


def render_map_html(fmap: folium.Map) -> str:
    """Render Folium map to an HTML string (useful for embedding in Member 4's UI)."""
    return fmap.get_root().render()
