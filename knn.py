import numpy as np


from scipy.spatial.distance import cdist
from scipy.stats import itemfreq


class KNN(object):

    _metric = ["euclidean", "cosine"]

    def __init__(self, K, metric="euclidean"):
        self.K = K
        self.metric = metric


    def l2_distance(self, X_test):
        return cdist(X_test, self.X, "euclidean")


    def cosine_similarity(self, X_test):
        return cdist(X_test, self.X, "cosine")
    

    def distance(self, X_test):
        return cdist(X_test, self.X, metric=self.metric)


    def fit(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = np.unique(y)


    def predict(self, X_test):
        X_test = np.array(X_test)
        if self.metric not in self._metric:
            self.metric = "euclidean"
        dist = self.distance(X_test)
        dist = np.argsort(dist, axis=1)
        k_nearest = dist[:, :self.K]
        labels = self.y[k_nearest]
        result = []
        for label in labels:
            label, count = np.unique(label, return_counts=True)
            result.append(label[np.argmax(count)])
        return np.array(result)

def optimizer(X_train, y_train, X_test, y_test):
    current_k = None
    current_acc = 0.0
    for k in range(3, 16, 2):
        model = KNN(k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > current_acc:
            current_acc = acc
            current_k = k
    print("Best k: %d, acc: %.2f %%" % (current_k, (current_acc*100)))


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print(X_train.shape, X_test.shape)
    # model = KNN(3)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # print(accuracy_score(y_test, y_pred))
    optimizer(X_train, y_train, X_test, y_test)
    