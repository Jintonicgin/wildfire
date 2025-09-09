import numpy as np

class EnsembleRegressor:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = [model.predict(X) for model in self.models]
        return np.mean(np.column_stack(preds), axis=1)
    
class EnsembleClassifier:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models])
        return np.apply_along_axis(lambda arr: np.bincount(arr).argmax(), axis=1, arr=preds)