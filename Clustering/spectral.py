import numpy as np
import scipy
import matplotlib.pyplot as plt 

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


class SpectralClustering:
    def __init__(self, n_clusters, sigma=1):
        self.n_clusters = n_clusters
        self.sigma = sigma
        self.labels_ = None
        self.L = None

    def fit_predict(self, X):
        self.labels_ = np.zeros(X.shape[0])
        n = X.shape[0]

        W = self.init_weights(X)
        D = np.diag(W.sum(axis=0))
        self.L = D - W
        
        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(self.L, k=self.n_clusters+1, which='SM')
        H = eigenvectors
        kmeans = KMeans(n_clusters=self.n_clusters).fit(H)
        self.labels_ = kmeans.labels_

        return self.labels_

    def init_weights(self, X):
        mean = X.mean()
        n = X.shape[0]
        
        W = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                d = np.linalg.norm(X[i] - X[j])
                W[i][j] = d
                W[j][i] = d
        
        W = np.exp(-W**2 / (2 * self.sigma**2))

        return W



if __name__ == '__main__':
    X, y_true = make_blobs(n_samples=500, centers=4, cluster_std=0.60, random_state=42)

    model = SpectralClustering(n_clusters=4)
    
    y_pred = model.fit_predict(X)
    
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
    plt.show()
