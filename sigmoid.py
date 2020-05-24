import numpy as np
from sklearn.datasets import load_iris


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy(X, y, w):
    y_hat = sigmoid(np.dot(X, w))
    return -np.sum(y * np.log(y_hat))


def logistic_sigmoid(X, y, w_init, lr, tol=1e-4, max_iter=10000):
    w = [w_init]
    N = X.shape[0]
    it = 0
    check_w_after = 20
    while it < max_iter:
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[i]
            yi = y[i]
            zi = sigmoid(np.dot(xi, w[-1]))
            w_new = w[-1] + lr * (yi - zi) * xi
            it += 1
            if it % check_w_after == 0:
                if abs(cross_entropy(X, y, w_new) - cross_entropy(X, y, w[-check_w_after])) < tol:
                    return w, it
            w.append(w_new)
    return w, max_iter


data = load_iris()
X = data.data
y = data.target
X_train, y_train = [], []
for i in range(len(y)):
    if y[i] == 1:
        y_train.append(y[i])
        X_train.append(X[i])
    elif y[i] == 0:
        y_train.append(0)
        X_train.append(X[i])
X = np.array(X_train)
y = np.array(y_train)

if __name__ == "__main__":
    w, it = logistic_sigmoid(X, y, w_init=np.random.randn(4), lr=0.05, tol=1e-4)
    print(w[-1], it)
    print(sigmoid(np.dot(X, w[-1])))
