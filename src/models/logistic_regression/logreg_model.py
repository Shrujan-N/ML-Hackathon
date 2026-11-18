from sklearn.linear_model import LogisticRegression


class LogRegCandidateModel:
    needs_scaling = True

    def __init__(self, C=1.0, max_iter=300, seed=42):
        self.name = "model1_logistic_regression"
        self.clf = LogisticRegression(C=C, solver="lbfgs", max_iter=max_iter, random_state=seed)

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


def make_model(**kwargs):
    return LogRegCandidateModel(**kwargs)
