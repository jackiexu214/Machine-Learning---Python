import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import scipy.io
import matplotlib.pyplot as plt
import pickle

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD
    
    first = X[:,0]
    second = X[:,1]
    first = first.reshape(150,1)
    second = second.reshape(150,1)
    means = []
    covmat = []
    num = 0
    for i in y:
        if i>num:
            num = i
    j = 1
    while j != (num.astype(int) + 1):
        i = 0
        total = 0
        totalcnt = 0
        while i != len(y):
            if y[i].astype(int) == j:
                total += first[i].astype(int) 
                totalcnt += 1
            i += 1
        means.append((total/totalcnt))
        j += 1
    j = 1
    while j != (num.astype(int) + 1):
        i = 0
        total = 0
        totalcnt = 0
        while i != len(y):
            if y[i].astype(int) == j:
                total += second[i].astype(int)  
                totalcnt += 1
            i += 1
        means.append(total/totalcnt)
        j += 1
    means = np.matrix(means)
    means = means.reshape(2,5)
    first = np.squeeze(np.asarray(first))
    second = np.squeeze(np.asarray(second))
    covmat = np.cov(first,second)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    first = X[:,0]
    second = X[:,1]
    first = first.reshape(150,1)
    second = second.reshape(150,1)
    means = []
    covmats = []
    num = 0
    for i in y:
        if i>num:
            num = i
    j = 1
    allfirst = []
    allsecond = []
    while j != (num.astype(int) + 1):
        i = 0
        total = 0
        totalcnt = 0
        ft = []
        while i != len(y):
            if y[i].astype(int) == j:
                total += first[i].astype(int)
                ft.append(first[i])
                totalcnt += 1
            i += 1
        means.append((total/totalcnt))
        j += 1
        ft = np.matrix(ft)
        allfirst.append(ft)
    j = 1
    while j != (num.astype(int) + 1):
        i = 0
        total = 0
        totalcnt = 0
        sd = []
        while i != len(y):
            if y[i].astype(int) == j:
                total += second[i].astype(int) 
                sd.append(second[i])
                totalcnt += 1
            i += 1
        means.append(total/totalcnt)
        j += 1
        sd = np.matrix(sd)
        allsecond.append(sd)
    means = np.matrix(means)
    means = means.reshape(2,5)
    i = 0
    while i < 5:
        a = np.squeeze(np.asarray(allfirst[i]))
        b = np.squeeze(np.asarray(allsecond[i]))
        covmats.append(np.cov(a,b))
        i += 1
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    pdf = []
    pi = np.pi
    deter = np.linalg.det(covmat)
    cnt = 0
    cnt2 = 0
    while cnt < len(Xtest):
        cnt2 = ytest[cnt].astype(int)
        cnt2 -= 1
        fpt = 1/np.sqrt(np.dot(((2*pi)**cnt2),deter))
        pt1 = (np.transpose(np.subtract(Xtest[cnt,:].reshape(2,1),means[:,cnt2])))
        pt2 = np.dot((np.linalg.inv(covmat)),(np.subtract(Xtest[cnt,:].reshape(2,1),means[:,cnt2])))
        pt3 = np.dot(pt1,pt2)
        pt4 = np.exp(np.dot((-1/2),pt3))
        pdf.append(np.dot(fpt,pt4))
        cnt += 1
    pdf = np.squeeze(np.asarray(pdf))
    highnum = 0
    curi = 0
    i = 0
    while i < 100:
        if pdf[i] > highnum:
            highnum = pdf[i]
            curi = i
        i += 1
    acc = 100 - (highnum/ytest[curi])*100
    acc = np.squeeze(np.asscalar(acc))
    return acc

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    
    # IMPLEMENT THIS METHOD
    pdf = []
    pi = np.pi
    deter = np.linalg.det(covmat)
    cnt = 0
    cnt2 = 0
    while cnt < len(Xtest):
        cnt2 = ytest[cnt].astype(int)
        cnt2 -= 1
        fpt = 1/np.sqrt(np.dot(((2*pi)**cnt2),deter))
        pt1 = (np.transpose(np.subtract(Xtest[cnt,:].reshape(2,1),means[:,cnt2])))
        pt2 = np.dot((np.linalg.inv(covmats[cnt2])),(np.subtract(Xtest[cnt,:].reshape(2,1),means[:,cnt2])))
        pt3 = np.dot(pt1,pt2)
        pt4 = np.exp(np.dot((-1/2),pt3))
        pdf.append(np.dot(fpt,pt4))
        cnt += 1
    pdf = np.squeeze(np.asarray(pdf))
    highnum = 0
    curi = 0
    i = 0
    while i < 100:
        if pdf[i] > highnum:
            highnum = pdf[i]
            curi = i
        i += 1
    acc = 100 - (highnum/ytest[curi])*100
    acc = np.squeeze(np.asscalar(acc))
    return acc

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1                                                                
    # IMPLEMENT THIS METHOD                                                   
    a = np.linalg.inv(np.dot(np.transpose(X),X))
    b = np.dot(np.transpose(X), y)
    w = np.dot(a,b)
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD                                                   
    a = np.dot(lambd,np.identity(X.shape[1]))
    b = (np.dot(np.transpose(X),X)/X.shape[0])
    c = np.linalg.inv(np.add(a,b))
    d = (np.dot(np.transpose(X), y)/X.shape[0])
    w = np.dot(c, d)
    return w


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # rmse
    
    # IMPLEMENT THIS METHOD
    N = Xtest.shape[0]
    a = np.transpose(np.subtract(ytest,np.dot(Xtest,w)))
    b = np.subtract(ytest,np.dot(Xtest,w))
    c = np.dot(a,b)
    d = np.sqrt(c)
    rmse = (d/N)
    rmse = np.squeeze(np.asscalar(rmse))
    return rmse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD                                             
    w = np.asmatrix(w)
    w = np.transpose(w)
    N = X.shape[0]
    a = np.multiply(2,N)
    b = np.transpose(np.subtract(y,np.dot(X,w)))
    c = np.subtract(y,np.dot(X,w))
    d = np.multiply(lambd,np.transpose(w))
    e = np.dot(d,w)
    f = np.multiply((1/2),e)
    g = (b/a)
    h = np.dot(g,c)
    error = np.add(h,f)
    a = np.dot(np.transpose(np.multiply(-1,y)),X)
    b = np.dot(np.transpose(w),np.dot(np.transpose(X),X))
    c = np.multiply(lambd,np.transpose(w))
    d = (a/X.shape[0])
    e = (b/X.shape[0])
    error_grad = d + e + c
    error = np.squeeze(np.asscalar(error))
    error_grad = np.squeeze(np.asarray(error_grad))
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xd - (N x (d+1))                                                         
    # IMPLEMENT THIS METHOD
    Xd = np.zeros((x.shape[0],pmax))
    xi = 1
    for xi in range (0,x.shape[0]):
        for p in range(0,pmax):
            Xd[xi,p] = np.power(x[xi],p)
    
    return Xd

# Main script

# Problem 1
# load the sample data                                                                 
X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')            

# LDA
means,covmat = ldaLearn(X,y)
ldaacc = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# Problem 2

X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')   
# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('RMSE without intercept '+str(mle))
print('RMSE with intercept '+str(mle_i))

# Problem 3
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    rmses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses3)
plt.show()

# Problem 4
k = 101
lambdas = np.linspace(0, 0.004, num=k)
i = 0
rmses4 = np.zeros((k,1))
opts = {'maxiter' : 100}    # Preferred value.                                                
w_init = np.zeros((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l_1 = np.zeros((X_i.shape[1],1))
    for j in range(len(w_l.x)):
        w_l_1[j] = w_l.x[j]
    rmses4[i] = testOLERegression(w_l_1,Xtest_i,ytest)
    i = i + 1
plt.plot(lambdas,rmses4)
plt.show()

# Problem 5
pmax = 7
lambda_opt = lambdas[np.argmin(rmses4)]
rmses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    rmses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    rmses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)
plt.plot(range(pmax),rmses5)
plt.legend(('No Regularization','Regularization'))
plt.show()