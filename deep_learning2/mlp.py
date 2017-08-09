import numpy as np

def forward(X, W1, b1, W2, b2):
	#sigmoid
	#Z = 1 / (1 + np.exp(- X.dot(W1) - b1))

	#Rectifier linear unit
	Z = X.dot(W1) + b1
	Z[Z<0] = 0

	A = Z.dot(W2) + b2
	exp_A = np.exp(A)
	Y = exp_A / exp_A.sum(axis=1, keepdims=True)
	return Y, Z

def derivative_w2(Z, T, Y):
	return Z.T.dot(Y - T)

def derivative_b2(T, Y):
	return (Y - T).sum(axis=0)

def derivative_w1(X, Z, T, Y, W2):
	# sigmoid
	#return X.T.dot((Y - T).dot(W2.T) * Z * (1 - Z))
	# relu
	return X.T.dot((Y - T).dot(W2.T) * (Z > 0))

def derivative_b1(Z, T, Y, W2):
	#sigmoid
	#return ((Y - T).dot(W2.T) * Z * (1 - Z)).sum(axis=0)
	#relu
	return ((Y - T).dot(W2.T) * (Z > 0)).sum(axis=0)