"""Configuration for PathSnap HMM & Viterbi route reconstruction (Member 3)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class HMMConfig:
    """Hyperparameters and fallback policies for the HMM/Viterbi map-matcher.

    All parameter tuning MUST be performed using only the 80 training trips
    and 10 student-testing trips (never evaluator-testing data).
    """

    # Emission model parameters
    emission_weight: float = 1.0
    prob_eps: float = 1e-6
    use_distance_fallback: bool = True
    sigma_z_m: float = 10.0  # Gaussian GPS noise std (meters) when classifier probs unavailable

    # Transition model parameters (Newson & Krumm 2009 exponential discrepancy model)
    beta_m: float = 5.0  # Scale parameter (meters) for |d_route - d_gps|
    beta_gap_scale: float = 0.5  # Relaxation factor per second above 1 s sampling interval
    same_arc_bonus: float = 0.25  # Mild log-space stability preference for staying on same arc
    reverse_penalty: float = 12.0  # Log-space penalty for backward motion along the same arc
    disconnected_penalty: float = 25.0  # Finite log-penalty for unreachable candidate pairs

    # GPS gap and sequence-break handling
    gap_threshold_s: float = 5.0  # Delta-t above which a step is flagged as a GPS gap
    break_gap_s: float = 60.0  # Delta-t above which Viterbi restarts to prevent stale drift
    break_distance_m: float = 2500.0  # Great-circle jump above which Viterbi restarts

    # Bounded shortest-path search settings
    max_speed_mps: float = 45.0  # ~162 km/h generous speed cap for dynamic cutoff
    min_search_radius_m: float = 250.0
    max_search_radius_m: float = 5000.0
    search_dist_multiplier: float = 2.5
    search_dist_buffer_m: float = 150.0

    # Missing-candidate and output reconstruction policies
    missing_candidate_policy: str = "carry_forward"  # "carry_forward" or "none"
    fill_intermediate_arcs: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "HMMConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
