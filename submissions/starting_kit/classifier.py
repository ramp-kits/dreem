from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier


class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = RandomForestClassifier(n_estimators=10)

    def fit(self, X, y):
        return self.clf.fit(X, y)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

