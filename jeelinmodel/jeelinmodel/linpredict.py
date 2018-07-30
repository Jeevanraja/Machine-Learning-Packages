import numpy as np
import matplotlib.pyplot as plt
from jeelinmodel import linear_data_process
from jeelinmodel import linrepac
        
def linpredict(fit,X_test,y_test):
    w=fit[0].T
    yhat_test=X_test.dot(w)
    R2=1-np.sum((y_test-yhat_test)**2)/np.sum((y_test-y_test.mean())**2)
    Rsquared="Rsquared is: "
    relerror =np.abs((y_test - yhat_test) / y_test)
    mre=np.median(relerror)
    MRE= "Median Relative Error: "
    act="Actual Values are: "
    pred="Predictions are: "
    result=([MRE, mre],[Rsquared, R2],[act,y_test],[pred,yhat_test])
    return (result)
    
