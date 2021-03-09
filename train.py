from adaline import Adaline
import numpy as np
from sklearn.preprocessing import StandardScaler


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == np.around(y_pred)) / len(y_true)
    return accuracy
with open('pimadiabetes.data.txt') as f:
    data = np.array(list(list(map(lambda x: float(x), line.strip().split(' '))) for line in f.readlines()))
    X = data[:, 0:-1]
    y = data[:, -1]
    y -= 1
D_train = int(len(data)* 0.7)
X_train = X[:D_train]
y_train = y[:D_train]
X_test = X[D_train:]
y_test = y[D_train:]

sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

adl = Adaline(0.01, 1000)
adl.fit(X_train, y_train)
predictions = adl.predict(X_test)
print(accuracy(y_test,predictions))