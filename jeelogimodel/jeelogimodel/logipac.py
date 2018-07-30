import numpy as np
import matplotlib.pyplot as plt

def sigmoid(h):
    return 1/ (1+ np.exp(-h))

def softmax(H):
    eH=np.exp(H)
    return eH/eH.sum(axis=1,keepdims=True)
    
def cross_entropy(Y,P):
    return -np.sum(Y*np.log(P))
    
def classification_rate(y,P):
    return np.mean(y==P.argmax(axis=1))

        
def logifit(X,Y,y):
    
    
    D = X.shape[1]
    K = Y.shape[1]
    W=np.random.randn(D,K)
    J = []
    eta = 1e-3
    epochs = int(1e3)
    for i in range(epochs):
        P=softmax(X.dot(W))
        J.append(cross_entropy(Y,P))
        W -= eta*X.T.dot(P-Y)
    P=softmax(X.dot(W))
    plt.figure(figsize=(12,9))
    plt.title('Cross Entropy Plot for Logistic Regression')
    plt.plot(J)
    roc_matrix=np.column_stack((P.argmax(axis=1),np.round(P.argmax(axis=1)),y))
    roc_matrix = roc_matrix[roc_matrix[:,0].argsort()[::-1]]
    tp = np.cumsum((roc_matrix[:,1] == 1) & (roc_matrix[:,2] == 1))/ np.sum(roc_matrix[:,2] == 1)
    fp = np.cumsum((roc_matrix[:,1] == 1) & (roc_matrix[:,2] == 0))/ np.sum(roc_matrix[:,2] == 0)
    tp = np.array([0] + tp.tolist()+[1])
    fp = np.array([0] + fp.tolist()+[1])
    plt.figure(figsize=(12,9))
    plt.step(fp,tp)
    plt.title('AUC Plot for Logistic Regression')
    auc= np.sum(np.array([u*v for u,v in zip(tp[1:],[j-i for i,j in zip(fp,fp[1:])])]))
    result =[(W),('Classification rate',classification_rate(y,P)),('AUC Score',auc)]
    return (result)



def binarylogifit(X,y):
    D = X.shape[1]
    w = np.random.randn(D+1)
    J =[]
    eta = 0.003
    epochs = 1000
    for i in range(epochs):
        p = sigmoid(X.dot(w))
        J.append(cross_entropy(y,p))
        w -= eta*X.T.dot(p - y)
    p = sigmoid(X.dot(w))
    plt.figure(figsize =(12,9))
    plt.plot(J)
    roc_matrix=np.column_stack((P.argmax(axis=1),np.round(P.argmax(axis=1)),y))
    roc_matrix = roc_matrix[roc_matrix[:,0].argsort()[::-1]]
    tp = np.cumsum((roc_matrix[:,1] == 1) & (roc_matrix[:,2] == 1))/ np.sum(roc_matrix[:,2] == 1)
    fp = np.cumsum((roc_matrix[:,1] == 1) & (roc_matrix[:,2] == 0))/ np.sum(roc_matrix[:,2] == 0)
    tp = np.array([0] + tp.tolist()+[1])
    fp = np.array([0] + fp.tolist()+[1])
    plt.figure(figsize=(12,9))
    plt.step(fp,tp)
    plt.title('AUC Plot for Logistic Regression')
    auc= np.sum(np.array([u*v for u,v in zip(tp[1:],[j-i for i,j in zip(fp,fp[1:])])]))
    result =[(W),('Classification rate',classification_rate(y,P)),('AUC Score',auc)]
    return (result)
