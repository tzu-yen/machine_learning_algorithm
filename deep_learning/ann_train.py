import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from process import get_data

def y2indicator(y, K):
	N = len(y)
	ind = np.zeros((N, K))
	ind[np.arange(N), y] = 1
	return ind

X, Y = get_data()
X, Y = shuffle(X, Y)
Y = Y.astype(np.int32)

N = len(Y)
M = 5
D = X.shape[1]
K = len(set(Y))

Xtrain = X[:-100]
Ytrain = Y[:-100]
Ytrain_ind = y2indicator(Ytrain, K)

Xtest = X[-100:]
Ytest = Y[-100:]
Ytest_ind = y2indicator(Ytest, K)

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(a):
	exp_A = np.exp(a)
	return exp_A / exp_A.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
	Z = np.tanh(X.dot(W1) + b1)
	return softmax(Z.dot(W2) + b2), Z

def predict(P_Y_given_X):
	return np.argmax(P_Y_given_X, axis=1)

def classification_rate(Y, P):
	return np.mean(Y==P)

def cross_entropy(T, pY):
	return -np.sum(T*np.log(pY))

train_costs = []
test_costs = []
learning_rate = 0.001

for epoch in xrange(10000):
	pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
	pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)

	ctrain = cross_entropy(Ytrain_ind, pYtrain)
	ctest = cross_entropy(Ytest_ind, pYtest)
	train_costs.append(ctrain)
	test_costs.append(ctest)

	W2 -= learning_rate * Ztrain.T.dot(pYtrain - Ytrain_ind)
	b2 -= learning_rate * (pYtrain - Ytrain_ind).sum(axis=0)
	dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain**2)
	W1 -= learning_rate * Xtrain.T.dot(dZ)
	b1 -= learning_rate * dZ.sum(axis=0)
	if epoch % 1000 == 0:
		print epoch, ctrain, ctest
print "Final Train Classification Rate: ", classification_rate(Ytrain, predict(pYtrain))
print "Final Test Classification Rate: ", classification_rate(Ytest, predict(pYtest))

legend1, = plt.plot(train_costs, label='train cost')
legend2, = plt.plot(test_costs, label='test cost')
plt.legend([legend1, legend2])
plt.show()