import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    z = np.float64(z)
    
    return 1 / (1 + np.exp(-z))
       

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""

    mat = loadmat('mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    
    
    #Your code here
    train_data = np.array([])
    train_label = np.array([])
    validation_data = np.array([])
    validation_label = np.array([])
    test_data = np.array([])
    test_label =  np.array([])
    train_label_total = np.array([])
   
    train0 = mat.get('train0')
    train1 = mat.get('train1')
    train2 = mat.get('train2')
    train3 = mat.get('train3')
    train4 = mat.get('train4')
    train5 = mat.get('train5')
    train6 = mat.get('train6')
    train7 = mat.get('train7')
    train8 = mat.get('train8')
    train9 = mat.get('train9')
    
    test0 = mat.get('test0')
    test1 = mat.get('test1')
    test2 = mat.get('test2')
    test3 = mat.get('test3')
    test4 = mat.get('test4')
    test5 = mat.get('test5')
    test6 = mat.get('test6')
    test7 = mat.get('test7')
    test8 = mat.get('test8')
    test9 = mat.get('test9')
    
        
    train_data_total = np.vstack((train0, train1, train2, train3, train4, train5, train6, train7, train8, train9))
    test_data = np.vstack((test0, test1, test2, test3, test4, test5, test6, test7, test8, test9))
    
    train_data_total = train_data_total/255
    test_data = test_data/255
    
    train_data_t = train_data_total.astype(float)
    test_data_t = test_data.astype(float)

    #First While Loop  
    train_data_t.shape
    train_data_length = train_data_t.shape[1]
    x = 0
    while x <= train_data_length:
        if 0 in train_data_t[x]:
            train_data_length = train_data_length - 1
            train_data_t[:, x] = []
    x = x + 1
    train_data_t.shape

    #Second While Loop
    test_data.shape
    test_data_length = test_data.shape[1]
    y = 0
    while y <= test_data_length:
        if 0 in test_data[y]:
            test_data_length = test_data_length - 1
            test_data[:, y] = []
    y = y + 1
    test_data.shape
   
    #First For Loop for labels
    for z in range(0, train_data_t.shape[0]):
        num = 0
        if num == 0:
            train0.shape[0]
        elif num == 1:
            train1.shape[0]
        elif num == 2:
            train2.shape[0]
        elif num == 3:
            train3.shape[0]
        elif num == 4:
            train4.shape[0]
        elif num == 5:
            train5.shape[0]
        elif num == 6:
            train6.shape[0]
        elif num == 7:
            train7.shape[0]
        elif num == 8:
            train8.shape[0]
        elif num == 9:
            train9.shape[0]

        train_label_total = np.array((train_label_total, num))
    
    #Second For Loop for labels
    z = 1
    for z in range(1, test_data.shape[0], 1):
        num = 0
        if num == 0:
            test0.shape[0]
        elif num == 1:
            test1.shape[0]
        elif num == 2:
            test2.shape[0]
        elif num == 3:
            test3.shape[0]
        elif num == 4:
            test4.shape[0]
        elif num == 5:
            test5.shape[0]
        elif num == 6:
            test6.shape[0]
        elif num == 7:
            test7.shape[0]
        elif num == 8:
            test8.shape[0]
        elif num == 9:
            test9.shape[0]
        test_label = np.array((test_label, num))
    
    s = 'Size of train labels'
    s 
    
    train_label_total.shape
    
    train_data_totals = range(train_data_t.shape[0])
    indx = np.random.permutation(train_data_totals)
    
    train_data = train_data_t[indx[0:50000],:]
    train_label = train_data_t[indx[0:50000],:]
    
    validation_data = train_data_t[indx[50000:],:]
    validation_label = train_data_t[indx[50000:],:]
    
    s1 = 'train_data and validation_data'
    s1
    train_data.shape
    validation_data.shape
    train_data = train_data/255
    validation_data = validation_data/255
    s2 = 'labels'
    s2
    train_label.shape
    validation_label.shape
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
    #
    grad_w1 = np.zeros((50, 11),dtype=np.float)
    grad_w2 = np.zeros((10, 51),dtype=np.float)
    
    training_label_total = np.zeros((training_label.shape[0], 10))
    for g in range (0, training_label.shape[0]):
        for a in range (0, 10):
            if a in training_label[g]:
                training_label_total[g, a] = a
            else:
                training_label_total[g, a] = 0
                
    training_label = training_label_total
    training_label
    
    vecX = np.array([])
    vecY = np.array([])
    
    zj_total = np.array([])
    yk_total = np.array([])
    delta_k = np.array([])
    
    aj = np.dot(w1,training_data.shape[0])
    vecX = np.append(vecX.shape, sigmoid(aj))
    zj_total = np.concatenate((zj_total, vecX), axis = 0)

    bk = np.dot(vecX,w2.shape[0])
    vecY = np.append(vecY,sigmoid(bk))
    tk = training_label[:training_data.shape[0],:n_class]
    yk = vecY[:n_class]
    delta_k_scale = yk - tk
    delta_k = np.append(delta_k, np.array([delta_k_scale]))
    
    t_w2 = np.zeros((10, 51),dtype=np.float)
    t_w2 = np.dot(delta_k.shape[0],vecX)
    grad_w2 = t_w2
        
    t_w1 = np.zeros((50, 11), dtype=np.float)
    scale = np.dot(delta_k.shape[0],w2)
    scale *= np.dot(1-vecX,vecX)
    
    r_total = training_data[:training_data.shape[0], :n_input]
    t_w1 = np.dot(scale.shape[0],r_total)
            
    grad_w1 = t_w1
    yk_total = np.concatenate((yk_total, vecY),axis = 0)
    vecX = np.array([])
    vecY = np.array([])
    
    ynk = yk_total[:training_data.shape[0]]
    tnk = training_label[:training_data.shape[0], :training_label[:training_data.shape[0]:].shape[1]]
    err1 = np.dot(yk_total,tnk.shape[0])
    err2 = np.dot(err1,np.log(ynk))
    err3 = err2*(-1)
    err4 = np.dot((1-tnk.shape[0]),np.log(1-ynk))
    total_err = err3 + err4
    
    tmp_obj_val = 0
    tmp_obj_val = np.dot(w1.shape,w1.shape)
    tmp_obj_val += np.dot(w2.shape,w2.shape)
   
    N = training_data.shape[0]
    tmp_obj_val *= (lambdaval / (2*N)) + total_err
    obj_val = tmp_obj_val

    grad_w2 = np.dot(w2,lambdaval)
    grad_w2 = (1/N)*grad_w2
    
    grad_w1 = np.dot(w1,lambdaval)
    grad_w1 *= (1/N)            
                
    grad_w1.shape
    grad_w2.shape
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    return (obj_val,obj_grad)
    




def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.array([])
    #Your code here
    
    prob = np.zeros((data.shape[0], 1), dtype=np.float)
    for x in range (0, data.shape[0]):
        data = np.array([data, np.matrix([1])])
    
    for y in range (0, data.shape[0]):
        q = sigmoid(data[y:].shape[0]*w1.conj().transpose())
        q = np.array([np.matrix([1]),q])
        sig = np.dot(w2,q.shape[0])
        prob[y] = np.amax(sigmoid(sig))
        
    labels = prob
    return labels
 
    



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;
                   
# set the number of nodes in output unit
n_class = 10;                   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 0;


args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)

#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))


#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

print('\n Test set Accuracy:'  + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

#saving parameters to pickle

file_name = "params.pickle"
pickle.dump(n_input,open(file_name,'wb'))
pickle.dump(n_hidden,open(file_name,'wb'))
pickle.dump(w1,open(file_name,'wb'))
pickle.dump(w2,open(file_name,'wb'))
pickle.dump(lambdaval,open(file_name,'wb'))