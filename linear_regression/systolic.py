import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_excel('./data/mlr02.xls')

X = df.as_matrix()

plt.scatter(X[:, 1], X[:, 0])
plt.show()

plt.scatter(X[:, 2], X[:, 0])
plt.show()

df['ones'] = 1
Y = df['X1']
X = df[['X2', 'X3', 'ones']]

X2_only = df[['X2', 'ones']]
X3_only = df[['X3', 'ones']]

def get_r2(X, Y):
	w = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
	Yhat = X.dot(w)

	d1 = Y - Yhat
	d2 = Y - Y.mean()
	r2 = 1 - np.dot(d1, d1) / np.dot(d2, d2)
	return r2

print "r2 for x2 only", get_r2(X2_only, Y)
print "r2 for x3 only", get_r2(X3_only, Y)
print "r2 for x2 and x3", get_r2(X, Y)
