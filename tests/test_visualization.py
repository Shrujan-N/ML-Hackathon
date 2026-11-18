"""Unit & real-trip smoke tests for Member 3's Folium visualisation (`src/visualization/`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.hmm import RoadNetworkGraph
from src.visualization import create_trip_map, render_map_html, save_trip_map


def test_folium_map_all_layers_and_hidden_reference(tmp_path: Path):
    nodes = np.array(
        [[77.590, 12.970], [77.592, 12.971], [77.594, 12.972], [77.592, 12.973]],
        dtype=np.float64,
    )
    arcs = np.array([[0, 1], [1, 2], [0, 3]], dtype=np.int64)
    g = RoadNetworkGraph(nodes, arcs)
    track = np.array([[77.5902, 12.9701], [77.5918, 12.9710], [77.5938, 12.9719]], dtype=np.float64)

    # Full comparison map (dev / report mode with reference route)
    fmap = create_trip_map(
        graph=g,
        track_lonlat=track,
        trip_id="test_01",
        timestamps=[1.0, 2.0, 3.0],
        reference_route_arcs=[0, 1],
        classifier_only_arcs=[0, 2],
        predicted_route_arcs=[0, 1],
        metrics_summary={"Top-1 Acc": "96.2%", "Continuity": "100.0%"},
    )
    out_file = save_trip_map(fmap, tmp_path / "test_trip.html")
    assert out_file.exists()
    html = out_file.read_text(encoding="utf-8")
    assert "Original GPS Observations" in html
    assert "Relevant Road Network" in html
    assert "Reference Route (Ground Truth)" in html
    assert "Classifier-Only Predicted Route" in html
    assert "Predicted Route (Classifier + HMM)" in html
    assert "Route Metrics" in html

    # Evaluator UI mode: reference_route_arcs=None must NOT render reference route layer
    fmap_ui = create_trip_map(
        graph=g,
        track_lonlat=track,
        trip_id="eval_hidden",
        reference_route_arcs=None,
        predicted_route_arcs=[0, 1],
    )
    html_ui = render_map_html(fmap_ui)
    assert "Reference Route (Ground Truth)" not in html_ui
    assert "Predicted Route (Classifier + HMM)" in html_ui


def test_folium_map_on_real_student_test_trip(tmp_path: Path):
    trip_dir = Path("student_test/00000007")
    if not trip_dir.exists():
        pytest.skip("student_test/00000007 not present")

    g = RoadNetworkGraph.from_raw_files(
        trip_dir / "00000007.nodes",
        trip_dir / "00000007.arcs",
    )
    track_data = np.loadtxt(trip_dir / "00000007.track", dtype=np.float64)
    route = [int(l.strip()) for l in (trip_dir / "00000007.route").open()]

    fmap = create_trip_map(
        graph=g,
        track_lonlat=track_data[:, :2],
        timestamps=track_data[:, 2],
        trip_id="00000007",
        reference_route_arcs=route,
        predicted_route_arcs=None,
    )
    out = save_trip_map(fmap, tmp_path / "student_test_00000007_template.html")
    assert out.exists()
    assert out.stat().st_size > 10_000
