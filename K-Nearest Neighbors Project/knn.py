import numpy as np
from collections import Counter

# Length of line segment between two points
def euclidean_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

class KNN:
    def __init__(self, k=3):
        self.k = k
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        predictions = [self._predict(x) for x in X]
    
    def _predict(self, x):
        # computing distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # getting closest k value
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority decider
        majority = Counter(k_nearest_labels).most_common()
        return majority[0][0]
    