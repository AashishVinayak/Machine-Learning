file = pd.read_csv("data1.txt")
data = pd.DataFrame(file)


X = np.array(data)[:, 0:2]
y = np.array(data)[:, 2]
m = np.size(y)

y = np.reshape(y, [99, 1])

pos = []
neg = []

for i in range(m):
    if y[i, 0] == 1:
        pos.append(i)
    else:
        neg.append(i)

plt.plot(X[pos, 0], X[pos, 1], '|')
plt.plot(X[neg, 0], X[neg, 1], 'o')


# initialising

theta = np.random.randint(-40, 40, [3, 1])
x = np.append(np.ones([m, 1], dtype=float), X, axis=1)


# gradient descent
for i in range(3500):
    h = sigmoid(np.dot(x, theta))
    err = y-h
    theta = theta + (0.012/m) * np.sum(np.dot(x.T, err))


predict_ah = np.zeros([m,1], dtype=float)
h = sigmoid(x.dot(theta))
for i in range(m):
    if h[i, 0] >= 0.5:
        predict_ah[i, 0] = 1.0

