import numpy as np
import matplotlib.pyplot as plt

def logic_get_data():
    D=2
    K=3
    N= K * 1000
    X1=np.random.randn(N//K,D)+np.array([2,2])
    X2=np.random.randn(N//K,D)+np.array([0,-2])
    X3=np.random.randn(N//K,D)+np.array([-2,2])
    X=np.vstack((X1,X2,X3))
    X=np.column_stack((np.array([[1]*N]).T,X))
    y=np.array([0]*(N//K)+[1]*(N//K)+[2]*(N//K))
    Y=np.zeros((N,K))
    for i in range(N):
        Y[i,y[i]]=1 
    X_test = X[:N//10]
    Y_test = Y[:N//10]
    Xtrain = X[N//10:]
    ytrain = Y[N//10:]
    y = y[N//10:]
    y_test=y[:N//10]
    X = Xtrain
    Y = ytrain
    Y_test=y_test
    return X,Y,y,X_test,Y_test,y_test
        
