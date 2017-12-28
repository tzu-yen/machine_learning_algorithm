import numpy as np 
import matplotlib.pyplot as plt 

X = []
Y = []

for line in open('./data/data_1d.csv'):
	x, y = line.split(',')
	X.append(x)
	Y.append(y)

X = np.array(X, dtype=float)
Y = np.array(Y, dtype=float)

plt.scatter(X, Y)
plt.show()

denominator = np.dot(X, X) - X.mean() * X.sum()
a = (np.dot(X, Y) - Y.mean() * X.sum()) / denominator
b = (Y.mean() * np.dot(X, X) - X.mean() * np.dot(X, Y)) / denominator

Yhat = a * X + b

plt.scatter(X, Y)
plt.plot(X, Yhat)
plt.show()

d1 = Y - Yhat
d2 = Y - Y.mean()
r2 = 1 - d1.dot(d1) / d2.dot(d2)

print r2