import numpy as np


class DBSCAN:
    def __init__(self, epsilon=0.1, min_samples=5):
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.labels_ = None

    def fit(self, X):
        self.labels_ = np.zeros(X.shape[0])
        cluster = 1

        for i in range(len(X)):
            if self.labels_[i] != 0:
                continue

            neighbors, neighbors_idx = self.find_neighbors(i, X)

            if len(neighbors) >= self.min_samples - 1:
                self.labels_[i] = cluster
                self.check_neighbors(i, neighbors_idx, X, cluster)
                self.clusters.append(cluster)
                cluster += 1

    def check_neighbors(self, center_idx, neighbors_idx, X, cluster):
        for i in neighbors_idx:
            if self.labels_[i] != 0:
                continue
            
            self.labels_[i] = cluster
            neighbors, n_idx = self.find_neighbors(i, X)

            if len(neighbors) >= self.min_samples - 1:
                self.check_neighbors(i, n_idx, X, cluster)
    
    def find_neighbors(self, center_idx, X):
        neighbors = []
        neighbors_idx = []

        for i, sample in enumerate(X):
            if i == center_idx:
                continue
            
            if np.linalg.norm(X[i] - X[center_idx]) <= self.epsilon:
                neighbors.append(sample)
                neighbors_idx.append(i)

        return np.array(neighbors), np.array(neighbors_idx)
