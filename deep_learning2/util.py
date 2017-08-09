import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def get_transformed_data():
	df = pd.read_csv('../large_files/train.csv')
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)

	Y = data[:, 0]
	X = data[:, 1:]
	mu = X.mean(axis=0)
	X = X - mu
	pca = PCA()
	Z = pca.fit_transform(X)
	return Z, Y, pca, mu

def get_normalized_data():
	df = pd.read_csv("../large_files/train.csv")
	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)
	Y = data[:, 0]
	X = data[:, 1:]
	mu = X.mean(axis=0)
	std = X.std(axis=0)
	np.place(std, std == 0, 1)    	# std = 1 if std == 0 else std
	X = (X - mu) / std
	return X, Y

def plot_cumulative_variance(pca):
	P = []
	for p in pca.explained_variance_ratio:
		if len(P) == 0:
			P.append(p)
		else:
			P.append(p + P[-1])
	plt.plot(P)
	plt.show()

def forward(X, W, b):
	a = X.dot(W) + b
	exp_a = np.exp(a)
	return exp_a / exp_a.sum(axis=1, keepdims=True)

def predict(pY):
	return np.argmax(pY, axis=1)

def error_rate(Y, T):
	prediction = predict(Y)
	return np.mean(prediction!=T)

def cost(Y, T):
	'''T needs to be indicator matrix'''
	return (-T*np.log(Y)).sum()

def cost2(Y, T):
	'''T does not have to be indicator'''
	N = len(T)
	return -np.log(Y[np.arange(N), T].sum())

def derivative_W(Y, T, X):
	return X.T.dot(Y - T)

def derivative_b(Y, T):
	return (Y - T).sum(axis=0)

def y2indicator(y):
	N, K = len(y), len(set(y))
	ind = np.zeros((N, K))
	ind[np.arange(N), y.astype(np.int32)] = 1
	return ind

def benchmark_full():
	X, Y = get_normalized_data()

	print "Performing Logistic Regression..."

	Xtrain = X[:-1000]
	Ytrain = Y[:-1000]
	Xtest = X[-1000:]
	Ytest = Y[-1000:]

	N, D = Xtrain.shape
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	W = np.random.randn(D, 10) / 28    	# 28 = np.sqrt(D+10)
	b = np.zeros(10)
	LL = []
	LLtest = []
	CRtest = []

	learning_rate = 0.00004
	reg = 0.01

	for i in xrange(500):
		pY = forward(Xtrain, W, b)
		ll = cost(pY, Ytrain_ind)
		LL.append(ll)

		pYtest = forward(Xtest, W, b)
		lltest = cost(pYtest, Ytest_ind)
		LLtest.append(lltest)

		err = error_rate(pYtest, Ytest)
		CRtest.append(err)

		W -= learning_rate * (derivative_W(pY, Ytrain_ind, Xtrain) + reg * W)
		b -= learning_rate * (derivative_b(pY, Ytrain_ind) + reg * b)
		if i % 10 == 0:
			print "Cost at iteration %d: %.6f" % (i, ll)
			print "Error rate:", err

	pY = forward(Xtest, W, b)
	print 'Final error rate:', error_rate(pY, Ytest)
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)
	plt.show()

def benchmark_pca():
	X, Y, _, _ = get_transformed_data()
	X = X[:, :300]

	mu = X.mean(axis=0)
	std = X.std(axis=0)
	X = (X - mu) / std

	Xtrain, Ytrain = X[:-1000], Y[:-1000]
	Xtest, Ytest = X[-1000:], Y[-1000:]

	N, D = Xtrain.shape
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	W = np.random.randn(D, 10) / 28
	b = np.zeros(10)

	LL = []
	LLtest = []
	CRtest = []

	learning_rate = 0.0001
	reg = 0.01

	for i in xrange(200):
		pY = forward(Xtrain, W, b)
		ll = cost(pY, Ytrain_ind)
		LL.append(ll)

		pYtest = forward(Xtest, W, b)
		lltest = cost(pYtest, Ytest_ind)
		LLtest.append(lltest)

		err = error_rate(pYtest, Ytest)
		CRtest.append(err)

		W -= learning_rate * (derivative_W(pY, Ytrain_ind, Xtrain) + reg * W)
		b -= learning_rate * (derivative_b(pY, Ytrain_ind) + reg * b)
		if i % 10 == 0:
			print "Cost at iteration %d: %.6f" % (i, ll)
			print "Error rate:", err

	pY = forward(Xtest, W, b)
	print 'Final error rate:', error_rate(pY, Ytest)
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)
	plt.show()



if __name__ == '__main__':
	#benchmark_full()
	benchmark_pca()