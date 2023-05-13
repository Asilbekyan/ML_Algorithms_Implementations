import numpy as np
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


class SpectralClustering:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.labels_ = None
        self.L = None

    def fit_predict(self, X):
        self.labels_ = np.zeros(X.shape[0])
        n = X.shape[0]

        W = self.init_weights(X)
        D = np.diag([np.sqrt(i) for i in W.sum(axis=0)])
        self.L = D - W
        
        eigenvalues, eigenvectors = np.linalg.eig(self.L)
        H = eigenvectors[:self.n_clusters].T

        kmeans = KMeans(n_clusters=self.n_clusters).fit(X)
        self.labels_ = kmeans.labels_

        return self.labels_

    def init_weights(self, X):
        mean = X.mean()
        n = X.shape[0]
        
        W = np.zeros((n, n))
        var = np.median(X)

        for i in range(n):
            for j in range(n):
                if i == j:
                    W[i][j] = 0
                else:
                    W[i][j] = np.exp(np.linalg.norm(X[i] - X[j]) ** 2 / (2 * var**2))
            
        return W



if __name__ == '__main__':
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    model = SpectralClustering(n_clusters=4)
    y_pred = model.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.show()
