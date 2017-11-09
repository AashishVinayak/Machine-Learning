'''
Support Vector Machine code from scratch
this code doesn't work
but demonstrates an implementation of SVM on a single feature of IRIS dataset
'''


# the dependencies
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def tag_svm(lab, size):
    t = np.zeros([100, 1], dtype=float)
    a = []
    b = []
    for i in range(size):
        if lab[i, 0] == 'Iris-setosa':
            a.append(i)
            t[i, 0] = 1
        elif lab[i, 0] == 'Iris-versicolor':
            b.append(i)
            t[i, 0] = -1
    return t


def svm_c(x, y):
    errors = []
    theta = np.zeros([1, 1], dtype=float)
    for j in range(1500):
        error = 0
        for i in range(100):
            if (y[i]*np.dot(x[i], theta)) < 1:
                theta = theta + 0.0001*(x[i] * y[i] - (2 * (1/(i+1)) * theta))
                error = 1
            else:
                theta = theta + 0.0001*(-2 * (1/i) * theta)
        errors.append(error)
    return theta


# loading the iris dataset
file  = pd.read_csv('iris.csv')
data = pd.DataFrame(file)

# input matrix
X = np.array(data)[0:100, 0:1]                                          # X : 100 X 4
X = X.reshape(100, 1)
# label vector
y = np.array(data)[0:100, 4:5]                                           # y : 100 X 1
y = y.reshape(100, 1)

m = np.size(y[:, 0])

# plotting the dataset
Tag = tag_svm(y, m)
tag = np.array(Tag).reshape(m, 1)
weight = svm_c(X, tag)
print(weight)

plt.plot(X, y, '.b')
plt.plot(X, np.dot(X, weight), '.r')
plt.show()