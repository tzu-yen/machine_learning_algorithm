import numpy as np
import matplotlib.pyplot as pyplot
from sklearn.utils import shuffle
from util import getBinaryData, sigmoid, sigmoid_cost, error_rate


class LogisticModel():
	def __init__(self):
		pass

	def fit(self, X, Y, learning_rate=10e-7, reg=0, epochs=120000, show_fig=False):
		X, Y = shuffle(X, Y)
		Xvalid, Yvalid = X[-1000:], Y[-1000:]	#validation set
		X, Y = X[:-1000], Y[:-1000]
		N, D = X.shape
		self.W = np.random.randn(D) / np.sqrt(D)
		self.b = 0

		costs = []
		best_validation_error = 1

		for i in xrange(epochs):
			pY = self.forward(X)

			self.W -= learning_rate*(X.T.dot(pY-Y)+reg*self.W)
			self.b -= learning_rate*((pY-Y).sum()+reg*self.b)

			if i%20 == 0:
				pYvalid = self.forward(Xvalid)
				c = sigmoid_cost(Yvalid, pYvalid)
				costs.append(c)
				e = error_rate(Yvalid, np.round(pYvalid))
				print 'i:', i, 'cost:', c, 'error:', e
				if e < best_validation_error:
					best_validation_error = e
		print best_validation_error

		if show_fig:
			plt.plot(costs)
			plt.show()

	def forward(self, X):
		return sigmod(X.dot(self.W)+self.b)

	def predict(self, X):
		pY = self.forward(X)
		return np.round(pY)

	def score(self, X, Y):
		prediction = self.predict(X)
		return 1 - error_rate(Y, prediction)

def main():
	X, Y = getBinaryData()
	X0 = X[Y==0, :]
	X1 = X[Y==1, :]
	X1 = np.repeat(X1, 9, axis=0)
	X = np.vstack([X0, X1])
	Y = np.array([0]*len(X0)+[1]*len(X1))

	model = LogisticModel()
	model.fit(X, Y, show_fig=True)
	model.score()

