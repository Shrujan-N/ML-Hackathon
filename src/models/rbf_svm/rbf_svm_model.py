"""Model 6 (Buffer): Kernel-based RBF SVM classifier (Member 3).

Provides:
- Exact `SVC(kernel='rbf', probability=True)` when `N <= exact_svc_max_samples`
- Memory-safe scalable RBF Kernel SVM (`StandardScaler` -> `Nystroem(kernel='rbf')`
  -> chunked L2 hinge-loss linear SVM `SGDClassifier(loss='hinge')` -> Platt sigmoid
  calibration on an internal held-out split) when `N > exact_svc_max_samples`, so it
  trains on 100% of Member 1's 1.86M candidate rows within laptop RAM limits.
- Sklearn-compatible `fit(X, y)` and `predict_proba(X)` interface for Member 2's CV harness.
- Does NOT fabricate epoch-wise training curves (respecting Deliverables.pdf Sec 3.3).
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class RBFSVMConfig:
    """Configuration for PathSnap Model 6 (Kernel-based RBF SVM)."""

    mode: str = "auto"  # "auto", "exact", or "nystroem"
    C: float = 1.0
    gamma: str | float = "scale"
    class_weight: Optional[str] = None  # None keeps calibrated posteriors; "balanced" reweights hinge loss
    nystroem_components: int = 256
    nystroem_gamma: Optional[float] = None  # None -> 1 / n_features after scaling
    nystroem_fit_samples: int = 50_000  # Anchor sample size for Nystroem basis estimation
    exact_svc_max_samples: int = 25_000
    chunk_size: int = 50_000
    sgd_epochs: int = 6
    calibration_fraction: float = 0.10
    scale_features: bool = True
    random_state: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PathSnapRBFSVMClassifier:
    """Kernel-based RBF SVM candidate classifier with Platt-calibrated probabilities."""

    def __init__(self, config: Optional[RBFSVMConfig] = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise ValueError("Pass either `config` or keyword arguments, not both.")
        self.config = config if config is not None else RBFSVMConfig(**kwargs)
        self.scaler_: Optional[StandardScaler] = None
        self.exact_svc_: Optional[SVC] = None
        self.nystroem_: Optional[Nystroem] = None
        self.sgd_svm_: Optional[SGDClassifier] = None
        self.platt_calibrator_: Optional[LogisticRegression] = None
        self.resolved_mode_: Optional[str] = None
        self.classes_ = np.array([0, 1], dtype=np.int64)
        self.train_time_s_: float = 0.0
        self.n_train_samples_: int = 0

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return self.config.to_dict()

    def set_params(self, **params: Any) -> "PathSnapRBFSVMClassifier":
        cfg_dict = self.config.to_dict()
        cfg_dict.update(params)
        self.config = RBFSVMConfig(**cfg_dict)
        return self

    def _transform_scaled(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        if self.scaler_ is not None:
            return self.scaler_.transform(X_arr).astype(np.float32)
        return X_arr

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PathSnapRBFSVMClassifier":
        """Fit RBF SVM (exact SVC for small N, memory-safe chunked Nystroem+Hinge SVM for large N)."""
        t0 = time.perf_counter()
        cfg = self.config
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
        n_samples, n_features = X_arr.shape
        self.n_train_samples_ = int(n_samples)

        if cfg.mode == "auto":
            mode = "exact" if n_samples <= cfg.exact_svc_max_samples else "nystroem"
        elif cfg.mode in {"exact", "nystroem"}:
            mode = cfg.mode
        else:
            raise ValueError(f"Unsupported mode '{cfg.mode}'. Use 'auto', 'exact', or 'nystroem'.")

        self.resolved_mode_ = mode
        self.exact_svc_ = None
        self.nystroem_ = None
        self.sgd_svm_ = None
        self.platt_calibrator_ = None

        if mode == "exact":
            if cfg.scale_features:
                self.scaler_ = StandardScaler()
                X_tr = self.scaler_.fit_transform(X_arr)
            else:
                self.scaler_ = None
                X_tr = X_arr
            self.exact_svc_ = SVC(
                kernel="rbf",
                C=cfg.C,
                gamma=cfg.gamma,
                class_weight=cfg.class_weight,
                probability=True,
                random_state=cfg.random_state,
            )
            self.exact_svc_.fit(X_tr, y_arr)
            self.train_time_s_ = time.perf_counter() - t0
            return self

        # Scalable Nystroem RBF Kernel SVM + Platt sigmoid calibration on internal split
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=cfg.calibration_fraction, random_state=cfg.random_state
        )
        tr_idx, cal_idx = next(splitter.split(X_arr, y_arr))

        if cfg.scale_features:
            self.scaler_ = StandardScaler()
            self.scaler_.fit(X_arr[tr_idx])
        else:
            self.scaler_ = None

        # Fit Nystroem RBF kernel basis on stratified anchor subset of training fold
        rng = np.random.RandomState(cfg.random_state)
        n_tr = len(tr_idx)
        n_anchor = min(cfg.nystroem_fit_samples, n_tr)
        anchor_sel = rng.choice(tr_idx, size=n_anchor, replace=False)
        X_anchor = self._transform_scaled(X_arr[anchor_sel])

        ny_gamma = cfg.nystroem_gamma if cfg.nystroem_gamma is not None else (1.0 / max(1, n_features))
        n_comp = min(cfg.nystroem_components, max(10, n_anchor - 1))
        self.nystroem_ = Nystroem(
            kernel="rbf",
            gamma=ny_gamma,
            n_components=n_comp,
            random_state=cfg.random_state,
        )
        self.nystroem_.fit(X_anchor)

        # Alpha in SGDClassifier corresponds to 1 / (C * N) for L2 hinge-loss SVM
        alpha = 1.0 / max(1.0, cfg.C * float(n_tr))
        self.sgd_svm_ = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=alpha,
            learning_rate="optimal",
            random_state=cfg.random_state,
        )

        # Optional class weights for chunked partial_fit
        if cfg.class_weight == "balanced":
            n_pos = max(1, int(np.sum(y_arr[tr_idx] == 1)))
            n_neg = max(1, int(np.sum(y_arr[tr_idx] == 0)))
            w0 = n_tr / (2.0 * n_neg)
            w1 = n_tr / (2.0 * n_pos)
        else:
            w0, w1 = 1.0, 1.0

        # Train hinge-loss SVM over 100% of training rows in memory-safe chunks
        for _ in range(cfg.sgd_epochs):
            perm = rng.permutation(tr_idx)
            for start in range(0, n_tr, cfg.chunk_size):
                b_idx = perm[start : start + cfg.chunk_size]
                Xb = self.nystroem_.transform(self._transform_scaled(X_arr[b_idx])).astype(np.float32)
                yb = y_arr[b_idx]
                if cfg.class_weight == "balanced":
                    sw = np.where(yb == 1, w1, w0).astype(np.float32)
                    self.sgd_svm_.partial_fit(Xb, yb, classes=self.classes_, sample_weight=sw)
                else:
                    self.sgd_svm_.partial_fit(Xb, yb, classes=self.classes_)

        # Platt sigmoid calibration on held-out calibration split
        cal_scores = self._decision_scores_chunked(X_arr[cal_idx])
        self.platt_calibrator_ = LogisticRegression(C=1e6, solver="lbfgs")
        self.platt_calibrator_.fit(cal_scores.reshape(-1, 1), y_arr[cal_idx])

        self.train_time_s_ = time.perf_counter() - t0
        return self

    def _decision_scores_chunked(self, X: np.ndarray) -> np.ndarray:
        assert self.nystroem_ is not None and self.sgd_svm_ is not None
        n = len(X)
        scores = np.empty(n, dtype=np.float64)
        for start in range(0, n, self.config.chunk_size):
            end = min(n, start + self.config.chunk_size)
            Xb = self.nystroem_.transform(self._transform_scaled(X[start:end])).astype(np.float32)
            scores[start:end] = self.sgd_svm_.decision_function(Xb)
        return scores

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.resolved_mode_ is None:
            raise RuntimeError("PathSnapRBFSVMClassifier must be fitted before calling predict_proba.")
        X_arr = np.asarray(X, dtype=np.float32)
        if self.resolved_mode_ == "exact":
            assert self.exact_svc_ is not None
            probs = self.exact_svc_.predict_proba(self._transform_scaled(X_arr))
            return np.clip(probs, 1e-7, 1.0 - 1e-7)

        assert self.platt_calibrator_ is not None
        scores = self._decision_scores_chunked(X_arr)
        probs = self.platt_calibrator_.predict_proba(scores.reshape(-1, 1))
        return np.clip(probs, 1e-7, 1.0 - 1e-7)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(np.int64)

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "model_name": "RBF SVM",
            "algorithm_family": "Kernel-based",
            "resolved_mode": self.resolved_mode_,
            "n_train_samples": self.n_train_samples_,
            "config": self.config.to_dict(),
            "train_time_s": self.train_time_s_,
        }

    def save(self, path: str | Path) -> Path:
        if self.resolved_mode_ is None:
            raise RuntimeError("Cannot save an unfitted PathSnapRBFSVMClassifier.")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": self.config.to_dict(),
                "resolved_mode": self.resolved_mode_,
                "n_train_samples": self.n_train_samples_,
                "scaler": self.scaler_,
                "exact_svc": self.exact_svc_,
                "nystroem": self.nystroem_,
                "sgd_svm": self.sgd_svm_,
                "platt_calibrator": self.platt_calibrator_,
                "train_time_s": self.train_time_s_,
            },
            p,
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "PathSnapRBFSVMClassifier":
        payload = joblib.load(Path(path))
        obj = cls(config=RBFSVMConfig(**payload["config"]))
        obj.resolved_mode_ = payload["resolved_mode"]
        obj.n_train_samples_ = int(payload.get("n_train_samples", 0))
        obj.scaler_ = payload["scaler"]
        obj.exact_svc_ = payload["exact_svc"]
        obj.nystroem_ = payload["nystroem"]
        obj.sgd_svm_ = payload["sgd_svm"]
        obj.platt_calibrator_ = payload["platt_calibrator"]
        obj.train_time_s_ = float(payload.get("train_time_s", 0.0))
        return obj
