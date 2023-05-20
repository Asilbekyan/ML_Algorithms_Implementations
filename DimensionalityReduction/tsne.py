import numpy as np

class t_SNE:
    def __init__(self, n_components=2, epochs=20, learning_rate=0.01, sigma=30, alpha=0.5):
        self.n_components = n_components
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.alpha = alpha
        self.prev_y = None

    def fit_transform(self, X):
        n = X.shape[0]
        P = np.zeros((n, n))

        # Calculating conditional probs using high-dimensional Euclidean distances
        for i in range(n):
            sum_ = 0
            for j in range(n):
                P[i][j] += np.exp(-np.linalg.norm(X[i] - X[j]**2) / (2 * self.sigma ** 2))
                
                if i != j:
                    sum_ += np.exp(-np.linalg.norm(X[i] - X[j]**2) / (2 * self.sigma**2))
            P[i] /= sum_

        # Provide symmetry
        for i in range(n):
            for j in range(i + 1):
                P[j][i] = P[i][j] = (P[i][j] + P[j][i]) / 2

        # Initializing low-dimensional data and saving initial as previous for further using
        y = np.random.normal(0, 1e-4, size=(n, self.n_components))
        self.prev_y = y.copy()

        # Starting iterations
        for epoch in range(self.epochs):
            Q = self.compute_dist(y)
            grad = self.compute_grad(P, Q, y)
            print('Epoch: ', epoch, 'Loss: ', self.cost_function(P, Q))
            y = self.update_values(y, grad)
        
        return y

    def compute_dist(self, y):
        # Computes low-dimensional affinities
        n = y.shape[0]
        Q = np.zeros((n, n))

        for i in range(n):
            sum_ = 0
            for j in range(n):
                Q[i][j] += np.exp(-np.linalg.norm(X[i] - X[j])**2)

                if i != j:
                    sum_ += np.exp(-np.linalg.norm(X[i] - X[j])**2)
            Q[i] /= sum_
        
        return Q

    def compute_grad(self, P, Q, y):
        # Computes gradients
        grad = np.zeros((y.shape[0], self.n_components))

        for i in range(len(y)):
            for j in range(len(y)): 
                grad[i] += 4 * (P[i][j] - Q[i][j]) * (y[i] - y[j]) * (1 + np.linalg.norm(y[i] - y[j])**2)

        return grad

    def cost_function(self, P, Q):
        n = P.shape[0]
        cost = 0

        for i in range(n):
            for j in range(n):
                cost += P[i][j] * np.log(P[i][j] / Q[i][j])

        return cost

    def update_values(self, y, grad):
        # Updates current values given gradient and previous values
        current_y = y.copy()
        
        for i in range(len(y)):
            y[i] = y[i] + self.learning_rate * grad[i] + self.alpha * (y[i] - self.prev_y[i])
        
        self.prev_y = current_y

        return y
        

if __name__ == '__main__':
    X = np.random.rand(100, 25)

    model = t_SNE(learning_rate=0.1, alpha=0.01)
    y = model.fit_transform(X)

    print(y.shape)








