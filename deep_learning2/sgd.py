import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from datetime import datetime

from util import get_transformed_data, forward, error_rate, cost, derivative_W, derivative_b, y2indicator

def main():
	X, Y, _, _ = get_transformed_data()
	X = X[:, :300]

	mu = X.mean(axis=0)
	std = X.std(axis=0)
	np.place(std, std == 0, 1)
	X = (X - mu) / std

	Xtrain, Ytrain = X[:-1000], Y[:-1000]
	Xtest, Ytest = X[-1000:], Y[-1000:]

	N, D = Xtrain.shape
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	#Full
	W = np.random.randn(D, 10) / 28
	b = np.zeros(10)
	LL = []
	learning_rate = 0.0001
	reg = 0.01
	t0 = datetime.now()

	for i in xrange(200):
		pY = forward(Xtrain, W, b)

		W -= learning_rate * (derivative_W(pY, Ytrain_ind, Xtrain) + reg * W)
		b -= learning_rate * (derivative_b(pY, Ytrain_ind) + reg * b)

		pYtest = forward(Xtest, W, b)
		ll = cost(pYtest, Ytest_ind)
		LL.append(ll)

		if i % 10 == 0:
			err = error_rate(pYtest, Ytest)
			print "Cost at iter %d: %.6f" % (i, ll)
			print "Error rate:", err

	pY = forward(Xtest, W, b)
	print "Final error rate:", error_rate(pY, pYtest)
	print "Elapsed time for full GD:", datetime.now() - t0

	#SGD
	W = np.random.randn(D, 10) / 28
	b = np.zeros(10)
	LL_stochastic = []
	learning_rate = 0.0001
	reg = 0.01
	t0 = datetime.now()

	for i in xrange(1):    # one epoch
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for n in xrange(min(N, 500)):
			x = tmpX[n, :].reshape(1, D)
			y = tmpY[n, :].reshape(1, 10)
			p_y = forward(x, W, b)

			W -= learning_rate * (derivative_W(p_y, y, x) + reg * W)
			b -= learning_rate * (derivative_b(p_y, y) + reg * b)

			p_y_test = forward(Xtest, W, b)
			ll = cost(p_y_test, Ytest_ind)
			LL_stochastic.append(ll)

			if n % (N/2) == 0:
				err = error_rate(p_y_test, Ytest)
				print "Cost at iteration %d: %.6f" % (i, ll)
				print "Error rate:", err
	p_y = forward(Xtest, W, b)
	print "Final Error rate:", error_rate(p_y, Ytest)
	print "Elapsed time for SGD:", datetime.now() - t0

	#Batch

	W = np.random.randn(D, 10) / 28
	b = np.zeros(10)
	LL_batch = []
	learning_rate = 0.0001
	reg = 0.01
	batch_sz = 500
	n_batches = N / batch_sz
	t0 = datetime.now()

	for i in xrange(50):    
		tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
		for j in xrange(n_batches):
			x = tmpX[j*batch_sz:(j*batch_sz + batch_sz), :]
			y = tmpY[j*batch_sz:(j*batch_sz + batch_sz), :]
			p_y = forward(x, W, b)

			W -= learning_rate * (derivative_W(p_y, y, x) + reg * W)
			b -= learning_rate * (derivative_b(p_y, y) + reg * b)

			p_y_test = forward(Xtest, W, b)
			ll = cost(p_y_test, Ytest_ind)
			LL_batch.append(ll)

			if j % (n_batches/2) == 0:
				err = error_rate(p_y_test, Ytest)
				print "Cost at iteration %d: %.6f" % (i, ll)
				print "Error rate:", err
	p_y = forward(Xtest, W, b)
	print "Final Error rate:", error_rate(p_y, Ytest)
	print "Elapsed time for Batch GD:", datetime.now() - t0

	x1 = np.linspace(0, 1, len(LL))
	plt.plot(x1, LL, label='full')
	x2 = np.linspace(0, 1, len(LL_stochastic))
	plt.plot(x2, LL_stochastic, label='stochastic')
	x3 = np.linspace(0, 1, len(LL_batch))
	plt.plot(x3, LL_batch, label='batch')
	plt.legend()
	plt.show()

if __name__ == '__main__':
	main()