import numpy as np
from sklearn.base import BaseEstimator
from xgboost._typing import ArrayLike

DEFAULT_THRESHOLD = .5


class Classifier(BaseEstimator):
    def __init__(self, clf) -> None:
        self.clf = clf
        self.threshold = DEFAULT_THRESHOLD

    def fit(self, X, y, **kwargs) -> 'Classifier':
        self.clf.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X: ArrayLike):
        return self.clf.predict_proba()[:, 1]

    def predict(self, X, threshold=None):
        if not threshold:
            threshold = self.threshold
        return np.where(self.predict_proba(X) > threshold, 1, 0)
