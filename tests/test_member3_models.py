"""Unit tests for Member 3's Model 5 (MLP) and Model 6 (RBF SVM) wrappers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification

from src.models.mlp import MLPConfig, PathSnapMLPClassifier, plot_mlp_training_curves
from src.models.rbf_svm import PathSnapRBFSVMClassifier, RBFSVMConfig


def _synthetic_candidate_data(n_samples: int = 800, n_features: int = 12):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=2,
        weights=[0.8, 0.2],
        random_state=42,
    )
    groups = np.repeat(np.arange(20), n_samples // 20)
    return X.astype(np.float32), y.astype(np.int64), groups


def test_mlp_classifier_fit_curves_save_load(tmp_path: Path):
    X, y, groups = _synthetic_candidate_data()
    cfg = MLPConfig(
        hidden_dims=(32, 16),
        dropout=0.1,
        max_epochs=6,
        patience=4,
        batch_size=64,
        random_state=42,
    )
    clf = PathSnapMLPClassifier(cfg)
    clf.fit(X[:640], y[:640], groups=groups[:640])

    probs = clf.predict_proba(X[640:])
    assert probs.shape == (160, 2)
    assert np.all((probs > 0.0) & (probs < 1.0))
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    # Real epoch-wise training & validation history recorded
    hist = clf.get_training_history()
    assert 1 <= len(hist) <= 6
    assert {"epoch", "train_loss", "val_loss", "train_macro_f1", "val_macro_f1"} <= set(hist[0].keys())

    # Plot curve PNG
    curve_path = plot_mlp_training_curves(hist, tmp_path / "mlp_curves.png")
    assert curve_path.exists() and curve_path.stat().st_size > 5_000

    # Save & load parity
    model_path = clf.save(tmp_path / "mlp_model.pt")
    loaded = PathSnapMLPClassifier.load(model_path, device=str(clf.device_))
    probs_loaded = loaded.predict_proba(X[640:])
    assert np.allclose(probs, probs_loaded, atol=1e-5)


def test_rbf_svm_exact_and_nystroem_modes(tmp_path: Path):
    X, y, _ = _synthetic_candidate_data(n_samples=500, n_features=10)

    # Exact RBF SVM mode
    exact_clf = PathSnapRBFSVMClassifier(RBFSVMConfig(mode="exact", C=1.0, random_state=42))
    exact_clf.fit(X[:400], y[:400])
    assert exact_clf.resolved_mode_ == "exact"
    p_exact = exact_clf.predict_proba(X[400:])
    assert p_exact.shape == (100, 2)
    assert np.allclose(p_exact.sum(axis=1), 1.0, atol=1e-5)

    # Nystroem scalable RBF kernel mode
    ny_clf = PathSnapRBFSVMClassifier(
        RBFSVMConfig(mode="nystroem", nystroem_components=100, C=1.0, random_state=42)
    )
    ny_clf.fit(X[:400], y[:400])
    assert ny_clf.resolved_mode_ == "nystroem"
    p_ny = ny_clf.predict_proba(X[400:])
    assert p_ny.shape == (100, 2)
    assert np.allclose(p_ny.sum(axis=1), 1.0, atol=1e-5)

    # Save & load parity
    saved = ny_clf.save(tmp_path / "rbf_svm.joblib")
    reloaded = PathSnapRBFSVMClassifier.load(saved)
    assert np.allclose(p_ny, reloaded.predict_proba(X[400:]), atol=1e-6)
    meta = reloaded.get_metadata()
    assert meta["algorithm_family"] == "Kernel-based"
