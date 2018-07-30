import numpy as np
import matplotlib.pyplot as plt

class LinearModel(object):
    """
    An L2-regularized linear model that uses SGD to minimize the in-sample error function.
    """

def ridgefit (X,y):
    """
    Trains a  Mutiple linear regression model using gradient descent 
    with L1+L2 regularization
    """
        # Step 0: Initialize the parameters
   
    N,D = X.shape
    w = np.random.randn(D)
    J =[]
    eta = 0.0001
    I = 1000
    l2 =10
    for t in range(1000):
        y_hat = X.dot(w)
        J.append((np.dot(y-y_hat,y-y_hat)+l2*(np.dot(w.T,w))))
        w -= eta*(X.T.dot(y_hat-y)+l2*w)      
    y_hat = X.dot(w)
    R2= 1-np.sum((y - y_hat)**2)/np.sum((y-np.mean(y))**2)
    ax= plt.figure(figsize=(15,12))
    ax= plt.plot(J)
    relerror =np.abs((y - y_hat) / y)
    mre=np.median(relerror).astype(np.float32)
    result =[(w),('coefficient of determination(r2_score',R2),('MRE',mre),('plot(J)',ax)]
    return (result)
        
