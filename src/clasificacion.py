import numpy as np

def prob2labels(y_prob):
    n,c=y_prob.shape
    labels=np.zeros(n,dtype=int)
    labels=np.argmax(y_prob,axis=1)
    return labels

def prob2labels2(y_prob):
    n,c=y_prob.shape
    labels=np.zeros(n,dtype=int)
    maximos=np.max(y_prob,axis=1)
    valores=np.array(range(0,c,1))
    for i in range(0,n):
        labels[i]=np.dot(np.equal(y_prob[i,:],np.array(maximos[i])).astype(int),valores)
    return labels


def accuracy(y,y_pred):
    accuracy=0.0
    for i in range(0,len(y)):
        if (y[i]==y_pred[i]):
            accuracy=accuracy+1
    accuracy=accuracy/len(y)
    return accuracy

def confusion(y,y_pred):
    classes=int(y.max()+1)
    matrix=np.zeros( (classes,classes))
    n,=y.shape
    for i in range(0,len(y)):
        matrix[y[i],y_pred[i]]=(matrix[y[i],y_pred[i]]+1)
    return matrix

def confusion2accuracy(confusion):
    c,c = confusion.shape
    accuracy = 0.0
    accuracy=np.sum(np.diag(confusion))/np.sum(confusion)
    return accuracy