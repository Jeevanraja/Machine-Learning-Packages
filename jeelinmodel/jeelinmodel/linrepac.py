import numpy as np
import matplotlib.pyplot as plt


def linfit (X,y,eta=0.001,iteration=1000):
    """
    Trains a  Mutiple linear regression model using gradient descent 
    with L1+L2 regularization
    """
    N,D = X.shape
    w = np.random.randn(D)
    J =[]
    for t in range(iteration):
        y_hat = X.dot(w)
        J.append(np.dot(y-y_hat,y-y_hat))
        w -= eta*(X.T.dot(y_hat-y))     
    y_hat = X.dot(w)
    R2= 1-np.sum((y - y_hat)**2)/np.sum((y-np.mean(y))**2)
    ax= plt.figure(figsize=(15,12))
    plt.plot(J)
    relerror =np.abs((y - y_hat) / y)
    mre=np.median(relerror).astype(np.float32)
    result =[(w),('coefficient of determination(r2_score',R2),('MRE',mre),('plot(J)',ax)]
    return (result)
    
   
