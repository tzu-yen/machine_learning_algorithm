import numpy as np
import pandas as pd


def sigmoid_cost(T, Y):
	return -(T*np.log(T)+(1-T)*np.log(1-T)).sum()

def sigmoid(A):
	return 1.0 / (1.0+np.exp(-A)