from sklearn.model_selection import train_test_split


import numpy as np

class Kmeans(object):

    def __init__(self, k):
        self.k = k

    def init_centers(self, X, k):
        return X[np.random.choice(X.shape[0], k, replace=False)]


    def assign_labels(self, X, centers):
        y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            d = X[i] - centers
            d = np.linalg.norm(d, axis=1)
            y[i] = np.argmin(d)
        return y


    def update_centers(self, X, y, k):
        centers = np.zeros((k, X.shape[1]))
        for i in range(k):
            X_i = X[y == i, :]
            centers[i] = np.mean(X_i, axis=0)
        return centers


    def fit(self, X, k):
        self.centers = self.init_centers(X, k)
        y = []
        while True:
            y_old = y
            y = self.assign_labels(X, self.centers)
            if np.array_equal(y, y_old):
                break
            self.centers = self.update_centers(X, y, k)
    

    def get_centers(self):
        return self.centers



