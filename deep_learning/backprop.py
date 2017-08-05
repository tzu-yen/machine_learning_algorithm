import numpy as np
import matplotlib.pyplot as plt

def forward(X, W1, b1, W2, b2):
	#output of hidden layer
	Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
	#softmax on output layer
	exp_A = np.exp(Z.dot(W2) + b2)
	Y = exp_A / exp_A.sum(axis=1, keepdims=True)
	return Y, Z

def classification_rate(Y, P):
	return np.mean(Y == P)

def cost(T, Y):
	#for gradient descent
	return np.sum(-T * np.log(Y))

def derivative_w2(Z, T, Y):
	# w2 = weights of output layer
	N, K = T.shape
	M = Z.shape[1]

	#slow - using for loops
	# ret1 = np.zeros((M, K))
	# for n in xrange(N):
	# 	for m in xrange(M):
	# 		for k in xrange(K):
	# 			ret1[m,k] += (T[n,k] - Y[n,k]) * Z[n,m]

	# ret2 = np.zeros((M, K))
	# for n in xrange(N):
	# 	for k in xrange(K):
	# 		ret2[:,k] += (T[n,k] - Y[n,k]) * Z[n,:]

	# ret3 = np.zeros((M, K))
	# for n in xrange(N):
	# 	ret3 += np.outer(Z[n], T[n] - Y[n])

	return Z.T.dot(Y - T)   #return MXK matrix

def derivative_b2(T, Y):
	return (Y - T).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
	N, D = X.shape
	M, K = W2.shape

	#slow
	# ret1 = np.zeros((D, M))
	# for n in xrange(N):
	# 	for k in xrange(K):
	# 		for m in xrange(M):
	# 			for d in xrange(D):
	# 				ret1[d,m] += (T[n,k] - Y[n,k]) * W2[m,k] * Z[n,m] * (1 - Z[n,m]) * X[n,d]
	
	dZ = (Y - T).dot(W2.T) * Z * (1 - Z)
	return X.T.dot(dZ)

def derivative_b1(T, Y, W2, Z):
	return ((Y - T).dot(W2.T) * Z * (1 - Z)).sum(axis=0)


def main():
	Nclass = 500
	X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
	X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
	X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
	X = np.vstack((X1, X2, X3))
	Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    
	N = len(Y)
	K = len(set(Y))
	#One-hot encoding for the targets
	# T = np.zeros((N, K))
	# for i in xrange(N):
	# 	T[i, Y[i]] = 1
	T = np.zeros((N, K))
	T[np.arange(N), Y.astype(np.int32)] = 1

	#plot the data
	plt.scatter(X[:, 0], X[:, 1], c=Y, s=100, alpha=0.5)
	plt.show()

	D = X.shape[1]
	# number of neurous in the hidden layer
	M = 3 

	#randomly initialize parameters
	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K)
	b2 = np.random.randn(K)

	learning_rate = 10e-7
	costs = []

	for epoch in xrange(100000):
		output, hidden = forward(X, W1, b1, W2, b2)
		if epoch % 100 == 0:
			c = cost(T, output)
			P = np.argmax(output, axis=1)
			r = classification_rate(Y, P)
			print "cost:", c, "classification_rate:", r
			costs.append(c)

		#gradient descent
		W2 -= learning_rate * derivative_w2(hidden, T, output)
		b2 -= learning_rate * derivative_b2(T, output)
		W1 -= learning_rate * derivative_w1(X, hidden, T, output, W2)
		b1 -= learning_rate * derivative_b1(T, output, W2, hidden)
	
	plt.plot(costs)
	plt.show()



if __name__ == '__main__':
	main()