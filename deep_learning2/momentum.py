import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b1, derivative_b2


def main():
	# 1.batch SGD
	# 2. batch SGD with momentum
	# 3. batch SGD with Nesterov

	max_iter = 20                
	print_period = 10

	X, Y = get_normalized_data()
	lr = 0.00004
	reg = 0.01

	Xtrain, Ytrain = X[:-1000], Y[:-1000]
	Xtest, Ytest = X[-1000:], Y[-1000:]
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	N, D = X.shape
	batch_sz = 500
	n_batches = N / batch_sz

	M = 300    # number of hidden neurons
	K = 10     # number of output classes

	W1 = np.random.randn(D, M) / np.sqrt(D+M)
	b1 = np.zeros(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.zeros(K)

	#1. batch SGD
	LL_batch = []
	CR_batch = []

	for i in xrange(max_iter):
		for j in xrange(n_batches):
			Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz,]
			Ybatch = Ytrain_ind[j*batch_sz:(j+1)*batch_sz,]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			W2 -= lr * (derivative_w2(Z, Ybatch, pYbatch) + reg * W2)
			b2 -= lr * (derivative_b2(Ybatch, pYbatch) + reg * b2)
			W1 -= lr * (derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1)
			b1 -= lr * (derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1)

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_batch.append(ll)
				print "Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll)

				err = error_rate(pY, Ytest)
				CR_batch.append(err)
				print "Error rate:", err

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print "Final error rate:", error_rate(pY, Ytest)

	#2. batch with momentum
	W1 = np.random.randn(D, M) / np.sqrt(D+M)
	b1 = np.zeros(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.zeros(K)
	LL_momentum = []
	CR_momentum = []
	mu = 0.9
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0

	for i in xrange(max_iter):
		for j in xrange(n_batches):
			Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz,]
			Ybatch = Ytrain_ind[j*batch_sz:(j+1)*batch_sz,]

			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			dW2 = mu * dW2 - lr * (derivative_w2(Z, Ybatch, pYbatch) + reg * W2)
			W2 += dW2
			db2 = mu * db2 - lr * (derivative_b2(Ybatch, pYbatch) + reg * b2)
			b2 += db2
			dW1 = mu * dW1 - lr * (derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg * W1)
			W1 += dW1
			db1 = mu * db1 - lr * (derivative_b1(Z, Ybatch, pYbatch, W2) + reg * b1)
			b1 += db1

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_momentum.append(ll)
				print "Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll)

				err = error_rate(pY, Ytest)
				CR_momentum.append(err)
				print "Error rate:", err

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print "Final error rate:", error_rate(pY, Ytest)


	#3. batch with Nesterov momentum
	W1 = np.random.randn(D, M) / np.sqrt(D+M)
	b1 = np.zeros(M)
	W2 = np.random.randn(M, K) / np.sqrt(M)
	b2 = np.zeros(K)
	LL_nesterov = []
	CR_nesterov = []
	mu = 0.9
	dW2 = 0
	db2 = 0
	dW1 = 0
	db1 = 0

	for i in xrange(max_iter):
		for j in xrange(n_batches):
			Xbatch = Xtrain[j*batch_sz:(j+1)*batch_sz,]
			Ybatch = Ytrain_ind[j*batch_sz:(j+1)*batch_sz,]
			pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

			dW2 = mu*mu*dW2 - (1+mu)*lr*(derivative_w2(Z, Ybatch, pYbatch)+reg*W2)
			W2 += dW2
			db2 = mu*mu*db2 - (1+mu)*lr*(derivative_b2(Ybatch, pYbatch)+reg*b2)
			b2 += db2
			dW1 = mu*mu*dW1 - (1+mu)*lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2)+reg*W1)
			W1 += dW1
			db1 = mu*mu*db1 - (1+mu)*lr*(derivative_b1(Z, Ybatch, pYbatch, W2)+reg*b1)
			b1 += db1

			if j % print_period == 0:
				pY, _ = forward(Xtest, W1, b1, W2, b2)
				ll = cost(pY, Ytest_ind)
				LL_nesterov.append(ll)
				print "Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll)

				err = error_rate(pY, Ytest)
				CR_nesterov.append(err)
				print "Error rate:", err

	pY, _ = forward(Xtest, W1, b1, W2, b2)
	print "Final error rate:", error_rate(pY, Ytest)

	plt.plot(LL_batch, label="batch")
	plt.plot(LL_momentum, label='momentum')
	plt.plot(LL_nesterov, label='nesterov')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()

			


