
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[5]:

def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.as_matrix()
    #separate labels from data
    X = data[:, :-1]
    Y = data[:, -1]
    #normalize numerical columns
    X[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()
    X[:,2] = (X[:,2] - X[:,2].mean()) / X[:,2].std()
    #one-hot encoding for categorical column
    N, D = X.shape
    X2 = np.zeros((N, D+3))        #four different categorical values
    X2[:, 0:(D-1)] = X[:, 0:(D-1)] #most of columns are the same
    
    for n in xrange(N):
        t = int(X[n,D-1])          #four time intervals encoded as 0, 1, 2, 3
        X2[n, D-1+t] = 1
    
    Z = np.zeros((N, 4))
    Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
    # X2[:-4] = Z
    assert(np.abs(X2[:,-4:] - Z).sum() < 10e-10)
    
    return X2, Y


# In[6]:

def get_binary_data():
    X, Y = get_data()
    X2 = X[Y<=1]
    Y2 = Y[Y<=1]
    return X2, Y2


# In[7]:

X, Y = get_data()


# In[ ]:



