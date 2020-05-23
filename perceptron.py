import numpy as np


def sgn(x, w):
    return np.sign(np.dot(x, w))


def has_converged(X, y, w):
    return np.array_equal(sgn(X, w), y)


def perceptron(X, y, w_init, alpha=1.0):
    w = [w_init]
    N = X.shape[0]
    mis_points = []
    while True:
        mix_id = np.random.permutation(N)
        for i in range(N):
            xi = X[i]
            yi = y[i]
            if sgn(xi, w[-1]) != yi:
                mis_points.append(mix_id[i])
                w_new = w[-1] + alpha * yi * xi
                w.append(w_new)
        if has_converged(X, y, w[-1]):
            break
    return w, mis_points


np.random.seed(2)

means = [[2, 2], [4, 2]]
cov = [[.3, .2], [.2, .3]]
N = 10
X0 = np.random.multivariate_normal(means[0], cov, N).T
X1 = np.random.multivariate_normal(means[1], cov, N).T

X = np.concatenate((X0, X1), axis=1)
y = np.concatenate((np.ones((1, N)), -1 * np.ones((1, N))), axis=1)
# Xbar
X = np.concatenate((np.ones((1, 2 * N)), X), axis=0)

if __name__ == "__main__":
    X_new = []
    y_new = []
    for i in range(X.shape[1]):
        a = np.zeros(3)
        a[0] = X[0][i]
        a[1] = X[1][i]
        a[2] = X[2][i]
        X_new.append(a)
        y_new.append(y[0][i])
    X_new = np.array(X_new)
    y_new = np.array(y_new)
    w, mis_points = perceptron(X_new, y_new, [0, 0, 0])
    print(sgn(X_new, w[-1]))
    print(y_new)
    print(has_converged(X_new, y_new, w[-1]))
    print(w[-1])
