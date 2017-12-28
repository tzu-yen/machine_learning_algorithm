import numpy as np
import matplotlib.pyplot as plt


X, Y = [], []
for line in open("./data/data_poly.csv"):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x**2])
    Y.append(float(y))

X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:,1], Y)
plt.show()

w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()

r2 = 1 - np.dot(d1, d1) / np.dot(d2, d2)
print r2