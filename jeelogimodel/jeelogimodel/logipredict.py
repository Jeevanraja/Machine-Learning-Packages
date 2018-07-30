import numpy as np
import matplotlib.pyplot as plt
from jeelogimodel import logic_data_process
from jeelogimodel import logipac

       
def logipredict(fit,X_test,y_test):
    
    def softmax(H):
        eH=np.exp(H)
        return eH/eH.sum(axis=1,keepdims=True)
    
    def cross_entropy(Y_test,P_test):
        return -np.sum(Y_test*np.log(P_test))
    
    def classification_rate(y_test,P_test):
        return np.mean(y_test==P_test.argmax(axis=1))

    W = fit[0].T
    P_test = softmax(X_test.dot(W))
    result =[('Classification rate',classification_rate(y_test,P_test)),('P_test',P_test),]
    return (result)
 
