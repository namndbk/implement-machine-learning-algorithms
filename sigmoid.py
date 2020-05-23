import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def cross_entropy(X, y, w):
    y_hat = sigmoid(np.dot(X, w))
    s = 0.0
    for i in range(X.shape[0]):
        s -= y[i] * np.log(y_hat[i])
    return s


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


if __name__ == "__main__":
    np.random.seed(2)
    X = np.array(
        [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75,
         5.00, 5.50])
    X_new = []
    for i in range(X.shape[0]):
        X_new.append([X[i], 1.0])
    X = np.array(X_new)
    y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
    w, it = logistic_sigmoid(X, y, w_init=np.random.randn(2), lr=0.05, tol=1e-4)
    print(w[-1])
    print(sigmoid(np.dot(X, w[-1])))