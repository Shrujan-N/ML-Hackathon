import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNNCandidateModel:
    needs_scaling = True

    def __init__(self, n_neighbors=15, max_ref_rows=32000, seed=42):
        self.name = "model2_distance_weighted_knn"
        self.max_ref_rows = max_ref_rows
        self.seed = seed
        self.clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            metric="euclidean",
            algorithm="kd_tree",
            leaf_size=50,
            n_jobs=-1,
        )

    def fit(self, X, y):
        if self.max_ref_rows and len(X) > self.max_ref_rows:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(len(X), size=self.max_ref_rows, replace=False)
            self.clf.fit(X[idx], y[idx])
        else:
            self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def make_model(**kwargs):
    return KNNCandidateModel(**kwargs)
