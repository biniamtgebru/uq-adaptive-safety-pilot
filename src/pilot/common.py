"""Shared tools: small MLP ensemble (pure-NumPy inference) and split-conformal helpers."""
import numpy as np
from sklearn.neural_network import MLPRegressor


class Ensemble:
    def __init__(self, M=5, hidden=(32, 32), seed=0):
        self.M, self.hidden, self.seed = M, hidden, seed

    def fit(self, X, Y):
        self.xm, self.xs = X.mean(0), X.std(0) + 1e-9
        self.ys = Y.std(0) + 1e-9
        Xn, Yn = (X - self.xm) / self.xs, Y / self.ys
        rng = np.random.default_rng(self.seed)
        self.nets = []
        for m in range(self.M):
            idx = rng.integers(0, len(X), len(X))   # bootstrap
            net = MLPRegressor(hidden_layer_sizes=self.hidden, activation='tanh', solver='adam',
                               max_iter=120, early_stopping=True,
                               random_state=self.seed + m, learning_rate_init=3e-3)
            net.fit(Xn[idx], Yn[idx])
            self.nets.append(net)
        self.W = [np.stack([n.coefs_[l] for n in self.nets]) for l in range(3)]
        self.B = [np.stack([n.intercepts_[l] for n in self.nets]) for l in range(3)]

    def predict_all(self, x):           # x (4,) -> (M,2)
        z = (x - self.xm) / self.xs
        h = np.tanh(np.einsum('j,mjk->mk', z, self.W[0]) + self.B[0])
        h = np.tanh(np.einsum('mj,mjk->mk', h, self.W[1]) + self.B[1])
        return (np.einsum('mj,mjk->mk', h, self.W[2]) + self.B[2]) * self.ys

    def predict_batch(self, X):         # X (n,4) -> mean (n,2), std (n)
        Z = (X - self.xm) / self.xs
        outs = []
        for m in range(self.M):
            h = np.tanh(Z @ self.W[0][m] + self.B[0][m])
            h = np.tanh(h @ self.W[1][m] + self.B[1][m])
            outs.append((h @ self.W[2][m] + self.B[2][m]) * self.ys)
        outs = np.stack(outs)
        return outs.mean(0), np.linalg.norm(outs.std(0), axis=1)



# ---------------- conformal helpers ----------------
def quantile_level(n, a):
    return min(1.0, np.ceil((n + 1) * (1 - a)) / n)


def conformal_q(scores, a):
    return np.quantile(scores, quantile_level(len(scores), a), method='higher')


