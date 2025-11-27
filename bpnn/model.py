import numpy as np

class BPRegressor:
    def __init__(self, layers, lr=0.005, epochs=10, verbose=True):
        self.layers = layers
        self.lr = lr
        self.epochs = epochs
        self.verbose = verbose
        self.weights = []
        self.biases = []
        self.losses = []

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1-x)

    def _init_weights(self, input_dim):
        self.layers[0] = input_dim
        for i in range(len(self.layers)-1):
            w = np.random.randn(self.layers[i], self.layers[i+1]) * np.sqrt(2.0 / self.layers[i])
            b = np.zeros((1, self.layers[i+1]))
            self.weights.append(w)
            self.biases.append(b)

    def _forward(self, X):
        activations = [X]
        z_vals = []
        for i in range(len(self.weights)):
            z = activations[-1] @ self.weights[i] + self.biases[i]
            z_vals.append(z)
            # 最后一层为线性输出
            activations.append(self.sigmoid(z) if i < len(self.weights) - 1 else z)
        return activations, z_vals

    def _backward(self, activations, y):
        m = y.shape[0]
        delta = activations[-1] - y.reshape(-1, 1)
        grad_w, grad_b = [], []

        for i in range(len(self.weights)-1, -1, -1):
            dw = activations[i].T @ delta / m
            db = np.sum(delta, axis=0, keepdims=True) / m
            
            grad_w.insert(0, dw)
            grad_b.insert(0, db)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.sigmoid_derivative(activations[i])

        return grad_w, grad_b

    def fit(self, X, y):
        if not self.weights:
            self._init_weights(X.shape[1])

        for epoch in range(self.epochs):
            acts, z_vals = self._forward(X)
            loss = np.mean((acts[-1] - y.reshape(-1,1))**2)
            self.losses.append(loss)

            gw, gb = self._backward(acts, y)
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * gw[i]
                self.biases[i] -= self.lr * gb[i]

            if self.verbose and epoch % max(1, self.epochs // 10) == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Loss = {loss:.6f}")
        return self

    def predict(self, X):
        return self._forward(X)[0][-1].flatten()
