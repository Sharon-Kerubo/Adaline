import numpy as np

class Adaline(object):
    def __init__(self, learning_rate=0.01, n_iterations = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
    
    def fit(self, X, y):
        self.weights = np.random.random((1 + X.shape[1]))
        self.cost = []

        for _ in range(self.n_iterations):
            errors= 0
            for _ in zip(X,y):
                errors = y - self.predict(X)
                self.weights[1:] += (self.learning_rate * X.T.dot(errors))
                self.weights[0] += self.learning_rate * errors.sum()
                cost = (errors**2).sum()/2.0
                self.cost.append(cost)

    def net_input(self, X):
        output = np.dot(X, self.weights[1:])+ self.weights[0]
        return output

    def predict(self, X):
        return self.sigmoid(self.net_input(X))

    def sigmoid(self, X):
        return np.exp(-np.logaddexp(0, -X))