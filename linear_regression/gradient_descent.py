import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3

X = np.zeros((N, D))
X[:, 0] = 1
X[:5, 1] = 1
X[5:, 2] = 1

Y = np.array([0] * 5 + [1] * 5)

#Closed form solution won't work. LinAlgError: Singular matrix
#w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

#Gradient Descent
costs = []

w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

for t in xrange(1000):
	Yhat = X.dot(w)
	delta = Yhat - Y
	w = w - learning_rate * X.T.dot(delta)
	mse = delta.dot(delta) / N
	print mse
	costs.append(mse)

plt.plot(costs)
plt.show()

plt.plot(Yhat, label='prediciton')
plt.plot(Y, label='target')
plt.legend()
plt.show()

print w
