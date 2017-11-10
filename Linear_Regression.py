'''
linear regression model using gradient descent
L2 regularisation
Dataset used: assignment 1 of coursera ml course
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# loading the dataset
file = pd.read_csv("ex1.csv")
data = pd.DataFrame(file)                       # 96 X 2

# training set preparation
X_train = np.array(data)[:, 0:1]
y_train = np.array(data)[:, 1]
m_train = y_train.size
y_train = y_train.reshape(m_train, 1)


# initialising
theta = np.zeros([2,1], dtype = float)
X_train_input = np.append(np.ones([m_train, 1], dtype=float), X_train, axis=1)
alpha = 0.01                                   # learning rate


# gradient descent
for i in range(6500):
    h = X_train_input.dot(theta)
    error = h - y_train
    theta = theta - (alpha/m_train) * np.sum(np.dot(X_train_input.T, error)) + (3.0/m_train)*np.sum(np.square(theta[1:, 0]))

# predictions
predictions = np.array(np.dot(X_train_input, theta))

print("the weights are:")
print(theta)

plt.plot(X_train[:, 0], y_train, 'xr')
plt.plot(X_train, np.dot(X_train_input, theta))
plt.show()