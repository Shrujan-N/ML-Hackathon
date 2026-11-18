"""Directed road-network graph and shortest-path transition cache (Member 3).

Provides:
- Geodesic Haversine distance and local tangent-plane projection onto directed road arcs
- Memory-efficient CSR directed adjacency graph supporting 150k+ nodes / 300k+ arcs
- Bounded Dijkstra shortest-path queries with transition caching and path reconstruction
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in meters between two WGS84 (lon, lat) points."""
    rlon1 = math.radians(lon1)
    rlat1 = math.radians(lat1)
    rlon2 = math.radians(lon2)
    rlat2 = math.radians(lat2)
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = (
        math.sin(dlat * 0.5) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon * 0.5) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(max(0.0, a))))


def haversine_vec_m(
    lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray
) -> np.ndarray:
    """Vectorized Haversine distance in meters."""
    rlon1 = np.radians(lon1)
    rlat1 = np.radians(lat1)
    rlon2 = np.radians(lon2)
    rlat2 = np.radians(lat2)
    dlon = rlon2 - rlon1
    dlat = rlat2 - rlat1
    a = (
        np.sin(dlat * 0.5) ** 2
        + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon * 0.5) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.clip(np.sqrt(np.clip(a, 0.0, 1.0)), 0.0, 1.0))


@dataclass
class ProjectionResult:
    """Result of projecting a GPS point onto a directed road arc."""

    perp_dist_m: float
    frac_along: float  # Clamped to [0.0, 1.0] from from_node -> to_node
    proj_lon: float
    proj_lat: float


@dataclass
class CacheStats:
    """Diagnostic counters for shortest-path transition queries."""

    same_arc_hits: int = 0
    direct_adj_hits: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    dijkstra_runs: int = 0
    unreachable_count: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            "same_arc_hits": self.same_arc_hits,
            "direct_adj_hits": self.direct_adj_hits,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "dijkstra_runs": self.dijkstra_runs,
            "unreachable_count": self.unreachable_count,
        }


class RoadNetworkGraph:
    """Directed road-network graph backed by compact CSR arrays.

    Accepts parsed `nodes_lonlat` and `arcs` arrays (e.g. from Member 1's data
    module) so Member 3 does not duplicate Member 1's parsing/EDA pipeline.
    """

    def __init__(
        self,
        nodes_lonlat: np.ndarray,
        arcs: np.ndarray,
    ) -> None:
        nodes_arr = np.asarray(nodes_lonlat, dtype=np.float64)
        arcs_arr = np.asarray(arcs, dtype=np.int64)

        if nodes_arr.ndim != 2 or nodes_arr.shape[1] != 2:
            raise ValueError(f"nodes_lonlat must have shape (N, 2), got {nodes_arr.shape}")
        if arcs_arr.ndim != 2 or arcs_arr.shape[1] != 2:
            raise ValueError(f"arcs must have shape (M, 2), got {arcs_arr.shape}")

        self.nodes_lonlat = nodes_arr  # shape (N, 2): [lon, lat]
        self.arcs = arcs_arr  # shape (M, 2): [from_node, to_node]
        self.num_nodes = int(nodes_arr.shape[0])
        self.num_arcs = int(arcs_arr.shape[0])

        if self.num_arcs > 0:
            if arcs_arr.min() < 0 or arcs_arr.max() >= self.num_nodes:
                raise ValueError("Arc node indices out of bounds for nodes_lonlat")

        # Precompute directed arc endpoints & geodesic lengths (meters)
        if self.num_arcs > 0:
            self.from_nodes = arcs_arr[:, 0]
            self.to_nodes = arcs_arr[:, 1]
            p0 = nodes_arr[self.from_nodes]
            p1 = nodes_arr[self.to_nodes]
            self.arc_lengths_m = haversine_vec_m(p0[:, 0], p0[:, 1], p1[:, 0], p1[:, 1])
        else:
            self.from_nodes = np.empty(0, dtype=np.int64)
            self.to_nodes = np.empty(0, dtype=np.int64)
            self.arc_lengths_m = np.empty(0, dtype=np.float64)

        # Build CSR outgoing adjacency for fast bounded Dijkstra
        self._build_csr()

    def _build_csr(self) -> None:
        counts = np.bincount(self.from_nodes, minlength=self.num_nodes)
        indptr = np.zeros(self.num_nodes + 1, dtype=np.int64)
        np.cumsum(counts, out=indptr[1:])

        order = np.argsort(self.from_nodes, kind="mergesort")
        self.csr_indptr = indptr
        self.csr_to_nodes = self.to_nodes[order]
        self.csr_arc_ids = order.astype(np.int64)
        self.csr_weights_m = self.arc_lengths_m[order]

    @classmethod
    def from_raw_files(
        cls,
        nodes_path: str | Path,
        arcs_path: str | Path,
    ) -> "RoadNetworkGraph":
        """Lightweight loader for standalone testing/visualization when raw files are passed directly."""
        nodes = np.loadtxt(nodes_path, dtype=np.float64)
        if nodes.ndim == 1 and nodes.size == 2:
            nodes = nodes.reshape(1, 2)
        arcs = np.loadtxt(arcs_path, dtype=np.int64)
        if arcs.ndim == 1 and arcs.size == 2:
            arcs = arcs.reshape(1, 2)
        return cls(nodes_lonlat=nodes, arcs=arcs)

    def arc_endpoints(self, arc_id: int) -> Tuple[int, int]:
        return int(self.from_nodes[arc_id]), int(self.to_nodes[arc_id])

    def arc_coords_latlon(self, arc_id: int) -> List[Tuple[float, float]]:
        """Return [(lat_from, lon_from), (lat_to, lon_to)] for Folium plotting."""
        u, v = int(self.from_nodes[arc_id]), int(self.to_nodes[arc_id])
        return [
            (float(self.nodes_lonlat[u, 1]), float(self.nodes_lonlat[u, 0])),
            (float(self.nodes_lonlat[v, 1]), float(self.nodes_lonlat[v, 0])),
        ]

    def is_directly_connected(self, arc_a: int, arc_b: int) -> bool:
        """True if arc_a == arc_b or to_node(arc_a) == from_node(arc_b)."""
        if arc_a < 0 or arc_b < 0 or arc_a >= self.num_arcs or arc_b >= self.num_arcs:
            return False
        return (arc_a == arc_b) or (int(self.to_nodes[arc_a]) == int(self.from_nodes[arc_b]))

    def is_reverse_arc(self, arc_a: int, arc_b: int) -> bool:
        """True if arc_b is the exact opposite directed edge of arc_a."""
        if arc_a == arc_b:
            return False
        return (
            int(self.from_nodes[arc_a]) == int(self.to_nodes[arc_b])
            and int(self.to_nodes[arc_a]) == int(self.from_nodes[arc_b])
        )

    def project_point_onto_arc(self, lon: float, lat: float, arc_id: int) -> ProjectionResult:
        """Project (lon, lat) onto directed arc `arc_id` in a local tangent plane (meters)."""
        u = int(self.from_nodes[arc_id])
        v = int(self.to_nodes[arc_id])
        lon_u, lat_u = float(self.nodes_lonlat[u, 0]), float(self.nodes_lonlat[u, 1])
        lon_v, lat_v = float(self.nodes_lonlat[v, 0]), float(self.nodes_lonlat[v, 1])

        mean_lat_rad = math.radians(0.5 * (lat_u + lat_v))
        m_per_deg_lat = 111_132.954
        m_per_deg_lon = 111_319.491 * max(1e-6, math.cos(mean_lat_rad))

        ax, ay = 0.0, 0.0
        bx = (lon_v - lon_u) * m_per_deg_lon
        by = (lat_v - lat_u) * m_per_deg_lat
        px = (lon - lon_u) * m_per_deg_lon
        py = (lat - lat_u) * m_per_deg_lat

        seg_len_sq = bx * bx + by * by
        if seg_len_sq <= 1e-12:
            frac = 0.0
            proj_x, proj_y = ax, ay
        else:
            frac = max(0.0, min(1.0, (px * bx + py * by) / seg_len_sq))
            proj_x = frac * bx
            proj_y = frac * by

        perp_dist = math.hypot(px - proj_x, py - proj_y)
        proj_lon = lon_u + frac * (lon_v - lon_u)
        proj_lat = lat_u + frac * (lat_v - lat_u)
        return ProjectionResult(
            perp_dist_m=perp_dist,
            frac_along=frac,
            proj_lon=proj_lon,
            proj_lat=proj_lat,
        )

    def arcs_in_bbox(
        self,
        min_lon: float,
        min_lat: float,
        max_lon: float,
        max_lat: float,
    ) -> np.ndarray:
        """Return arc_ids whose either endpoint lies within the given lon/lat bounding box."""
        if self.num_arcs == 0:
            return np.empty(0, dtype=np.int64)
        node_in = (
            (self.nodes_lonlat[:, 0] >= min_lon)
            & (self.nodes_lonlat[:, 0] <= max_lon)
            & (self.nodes_lonlat[:, 1] >= min_lat)
            & (self.nodes_lonlat[:, 1] <= max_lat)
        )
        mask = node_in[self.from_nodes] | node_in[self.to_nodes]
        return np.flatnonzero(mask)


class ShortestPathCache:
    """Bounded directed shortest-path cache between road-network nodes/arcs.

    Key design points:
    1. Same-arc and direct-adjacency transitions are resolved in O(1) without Dijkstra.
    2. Multi-hop queries run bounded single-source Dijkstra with a distance cutoff
       and cache all settled nodes from that source at the queried cutoff so multiple
       candidate targets from the same source arc reuse a single search!
    3. Unreachable pairs record the cutoff tested; if a later GPS gap requires a wider
       cutoff, the search cleanly expands instead of returning a stale miss.
    """

    def __init__(self, graph: RoadNetworkGraph, max_cached_sources: int = 2048) -> None:
        self.graph = graph
        self.max_cached_sources = max_cached_sources
        # src_node -> (cutoff_m, dist_dict, parent_dict)
        # parent_dict[node] = (prev_node, arc_id)
        self._source_cache: Dict[
            int, Tuple[float, Dict[int, float], Dict[int, Tuple[int, int]]]
        ] = {}
        self.stats = CacheStats()

    def clear(self) -> None:
        self._source_cache.clear()
        self.stats = CacheStats()

    def _run_bounded_dijkstra(
        self, src_node: int, cutoff_m: float
    ) -> Tuple[Dict[int, float], Dict[int, Tuple[int, int]]]:
        """Run bounded Dijkstra from `src_node` up to `cutoff_m`."""
        self.stats.dijkstra_runs += 1
        g = self.graph
        indptr = g.csr_indptr
        to_nodes = g.csr_to_nodes
        arc_ids = g.csr_arc_ids
        weights = g.csr_weights_m

        dist: Dict[int, float] = {src_node: 0.0}
        parent: Dict[int, Tuple[int, int]] = {}
        heap: List[Tuple[float, int]] = [(0.0, src_node)]

        while heap:
            d_u, u = heapq.heappop(heap)
            if d_u > cutoff_m:
                break
            if d_u > dist.get(u, math.inf):
                continue
            start = int(indptr[u])
            end = int(indptr[u + 1])
            for idx in range(start, end):
                v = int(to_nodes[idx])
                nd = d_u + float(weights[idx])
                if nd <= cutoff_m and nd < dist.get(v, math.inf):
                    dist[v] = nd
                    parent[v] = (u, int(arc_ids[idx]))
                    heapq.heappush(heap, (nd, v))

        if len(self._source_cache) >= self.max_cached_sources:
            # Evict oldest inserted key (FIFO) to bound memory on huge graphs
            oldest_key = next(iter(self._source_cache))
            del self._source_cache[oldest_key]
        self._source_cache[src_node] = (cutoff_m, dist, parent)
        return dist, parent

    def shortest_node_path(
        self,
        src_node: int,
        dst_node: int,
        cutoff_m: float,
    ) -> Tuple[Optional[float], Optional[List[int]]]:
        """Return (distance_m, list_of_arc_ids) from `src_node` to `dst_node` within `cutoff_m`.

        Returns (None, None) if unreachable within `cutoff_m`.
        """
        if src_node == dst_node:
            return 0.0, []

        cached = self._source_cache.get(src_node)
        if cached is not None:
            cached_cutoff, dist_map, parent_map = cached
            if dst_node in dist_map:
                self.stats.cache_hits += 1
                return dist_map[dst_node], self._reconstruct_arcs(src_node, dst_node, parent_map)
            if cached_cutoff >= cutoff_m:
                self.stats.cache_hits += 1
                self.stats.unreachable_count += 1
                return None, None

        self.stats.cache_misses += 1
        dist_map, parent_map = self._run_bounded_dijkstra(src_node, cutoff_m)
        if dst_node not in dist_map:
            self.stats.unreachable_count += 1
            return None, None
        return dist_map[dst_node], self._reconstruct_arcs(src_node, dst_node, parent_map)

    @staticmethod
    def _reconstruct_arcs(
        src_node: int,
        dst_node: int,
        parent_map: Dict[int, Tuple[int, int]],
    ) -> List[int]:
        arcs: List[int] = []
        cur = dst_node
        while cur != src_node:
            prev_node, arc_id = parent_map[cur]
            arcs.append(arc_id)
            cur = prev_node
        arcs.reverse()
        return arcs

    def arc_to_arc_distance(
        self,
        arc_a: int,
        frac_a: float,
        arc_b: int,
        frac_b: float,
        cutoff_m: float,
    ) -> Tuple[Optional[float], bool, Optional[List[int]]]:
        """Compute directed route distance between projected points on `arc_a` and `arc_b`.

        Returns:
            (route_dist_m, is_backward_on_same_arc, intermediate_arcs)
            - `route_dist_m` is None if `arc_b` is unreachable from `arc_a` within `cutoff_m`.
            - `is_backward_on_same_arc` is True if `arc_a == arc_b` and `frac_b < frac_a`.
            - `intermediate_arcs` is the list of directed arcs strictly between `arc_a` and `arc_b`.
        """
        g = self.graph
        len_a = float(g.arc_lengths_m[arc_a])
        len_b = float(g.arc_lengths_m[arc_b])

        # Case 1: Same arc
        if arc_a == arc_b:
            self.stats.same_arc_hits += 1
            delta = (frac_b - frac_a) * len_a
            if delta >= -1e-6:
                return max(0.0, delta), False, []
            return abs(delta), True, []

        rem_a = max(0.0, (1.0 - frac_a) * len_a)
        into_b = max(0.0, frac_b * len_b)
        end_a = int(g.to_nodes[arc_a])
        start_b = int(g.from_nodes[arc_b])

        # Case 2: Direct directed adjacency (0 intermediate hops)
        if end_a == start_b:
            self.stats.direct_adj_hits += 1
            return rem_a + into_b, False, []

        # Case 3: Multi-hop bounded shortest path from to_node(arc_a) to from_node(arc_b)
        node_cutoff = max(0.0, cutoff_m - rem_a - into_b)
        if node_cutoff <= 0.0:
            self.stats.unreachable_count += 1
            return None, False, None

        sp_dist, inter_arcs = self.shortest_node_path(end_a, start_b, node_cutoff)
        if sp_dist is None:
            return None, False, None
        return rem_a + sp_dist + into_b, False, inter_arcs
