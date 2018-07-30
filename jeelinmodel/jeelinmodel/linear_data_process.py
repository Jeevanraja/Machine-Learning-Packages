import numpy as np
import matplotlib.pyplot as plt

def liner_get_data():
    N = 1000
    D = 1000
    X = np.random.randn(N,D)
    true_w = np.array([2, 10, 15] + [0]*(D+1-3))
    X = np.column_stack((np.array([[1]*N]).T, X))
    y = X.dot(true_w) + np.random.randn(N)*0.5
    X_test = X[:N//10]
    y_test = y[:N//10]
    Xtrain = X[N//10:]
    ytrain = y[N//10:]
    y = y[N//10:]
    X = Xtrain
    y = ytrain
    return X,y,X_test,y_test
        
