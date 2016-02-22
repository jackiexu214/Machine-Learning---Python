import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
from sklearn import svm

def preprocess():
    """ 
     Input:
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
     - feature selection
    """
    
    mat = loadmat('mnist_all.mat'); #loads the MAT object as a Dictionary
    
    n_feature = mat.get("train1").shape[1];
    n_sample = 0;
    for i in range(10):
        n_sample = n_sample + mat.get("train"+str(i)).shape[0];
    n_validation = 1000;
    n_train = n_sample - 10*n_validation;
    
    # Construct validation data
    validation_data = np.zeros((10*n_validation,n_feature));
    for i in range(10):
        validation_data[i*n_validation:(i+1)*n_validation,:] = mat.get("train"+str(i))[0:n_validation,:];
        
    # Construct validation label
    validation_label = np.ones((10*n_validation,1));
    for i in range(10):
        validation_label[i*n_validation:(i+1)*n_validation,:] = i*np.ones((n_validation,1));
    
    # Construct training data and label
    train_data = np.zeros((n_train,n_feature));
    train_label = np.zeros((n_train,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("train"+str(i)).shape[0];
        train_data[temp:temp+size_i-n_validation,:] = mat.get("train"+str(i))[n_validation:size_i,:];
        train_label[temp:temp+size_i-n_validation,:] = i*np.ones((size_i-n_validation,1));
        temp = temp+size_i-n_validation;
        
    # Construct test data and label
    n_test = 0;
    for i in range(10):
        n_test = n_test + mat.get("test"+str(i)).shape[0];
    test_data = np.zeros((n_test,n_feature));
    test_label = np.zeros((n_test,1));
    temp = 0;
    for i in range(10):
        size_i = mat.get("test"+str(i)).shape[0];
        test_data[temp:temp+size_i,:] = mat.get("test"+str(i));
        test_label[temp:temp+size_i,:] = i*np.ones((size_i,1));
        temp = temp + size_i;
    
    # Delete features which don't provide any useful information for classifiers
    sigma = np.std(train_data, axis = 0);
    index = np.array([]);
    for i in range(n_feature):
        if(sigma[i] > 0.001):
            index = np.append(index, [i]);
    train_data = train_data[:,index.astype(int)];
    validation_data = validation_data[:,index.astype(int)];
    test_data = test_data[:,index.astype(int)];

    # Scale data to 0 and 1
    train_data = train_data/255.0;
    validation_data = validation_data/255.0;
    test_data = test_data/255.0;
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z));
    
def blrObjFunction(params, *args):
    """
    blrObjFunction computes 2-class Logistic Regression error function and
    its gradient.

    Input:
        initialWeights: the weight vector of size (D + 1) x 1 
        train_data: the data matrix of size N x D
        labeli: the label vector of size N x 1 where each entry can be either 0 or 1
                representing the label of corresponding feature vector

    Output: 
        error: the scalar value of error function of 2-class logistic regression
        error_grad: the vector of size (D+1) x 1 representing the gradient of
                    error function
    """
    initialWeights = params
    train_data, labeli = args
    n_data = train_data.shape[0];
    n_feature = train_data.shape[1];
    error = 0;
    error_grad = np.zeros((n_feature+1,1));
    
    ##################
    # YOUR CODE HERE #
    ##################
    initialWeights = initialWeights.reshape(716,1)
    x = np.concatenate((np.ones((n_data,1)),train_data),axis = 1)
    y = sigmoid(np.dot(x,initialWeights))
    a = np.log(1 - y)
    b = (1 - labeli)
    c = np.log(y)
    d = np.multiply(labeli,c)
    e = np.multiply(b,a)
    f = np.add(d,e)
    g = np.sum(f)
    error = np.multiply(-1,g)
    a = (y - labeli)
    error_grad = np.dot(np.transpose(a),x)
    error_grad = np.squeeze(np.asarray(error_grad))
    return error, error_grad

def blrPredict(W, data):
    """
     blrObjFunction predicts the label of data given the data and parameter W 
     of Logistic Regression
     
     Input:
         W: the matrix of weight of size (D + 1) x 10. Each column is the weight 
         vector of a Logistic Regression classifier.
         X: the data matrix of size N x D
         
     Output: 
         label: vector of size N x 1 representing the predicted label of 
         corresponding feature vector given in data matrix

    """
    label = np.zeros((data.shape[0],1));
    
    ##################
    # YOUR CODE HERE #
    ##################
    x = np.concatenate((np.ones((data.shape[0],1)),data),axis = 1)
    y = sigmoid(np.dot(x,W))
    label = np.argmax(y,axis = 1)
    label = label.reshape(label.shape[0],1)
    return label


"""
Script for Logistic Regression
"""
train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();
# number of classes
n_class = 10;

# number of training samples
n_train = train_data.shape[0];

# number of features
n_feature = train_data.shape[1];

T = np.zeros((n_train, n_class));
for i in range(n_class):
    T[:,i] = (train_label == i).astype(int).ravel();
    
# Logistic Regression with Gradient Descent
W = np.zeros((n_feature+1, n_class));
initialWeights = np.zeros((n_feature+1,1));
opts = {'maxiter' : 50};
for i in range(n_class):
    labeli = T[:,i].reshape(n_train,1);
    args = (train_data, labeli);
    nn_params = minimize(blrObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
    W[:,i] = nn_params.x.reshape((n_feature+1,));

# Find the accuracy on Training Dataset
predicted_label = blrPredict(W, train_data);
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

# Find the accuracy on Validation Dataset
predicted_label = blrPredict(W, validation_data);
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

# Find the accuracy on Testing Dataset
predicted_label = blrPredict(W, test_data);
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

pickle_file = open('params.pickle', 'wb')
p = pickle.Pickler(pickle_file)
p.dump(W)
pickle_file.close()

"""
Script for Support Vector Machine
"""

print('\n\n--------------SVM-------------------\n\n')
##################
# YOUR CODE HERE #
##################

train_label = train_label.ravel()
validation_label = validation_label.ravel()
test_label = test_label.ravel()

#Linear Kernel
print('\n Linear Kernel')
linear = svm.SVC(kernel = 'linear')
linear.fit(train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = linear.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = linear.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = linear.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

#Radial Basis Function when Gamma = 1

print('\n Radial Basis Function, Gamma = 1')
radial = svm.SVC(kernel = 'rbf', gamma = 1)
radial.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

#Radial Basis Function: Normal

print('\n Radial Basis Function, Gamma = default')
radialdefault = svm.SVC(kernel = 'rbf', gamma = 0.0)
radialdefault.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radialdefault.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radialdefault.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radialdefault.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

#Radial Basis Function: C = 1,10,20,30,40,50,60,70,80,90,100

print('\n Radial Basis Function, C = 1,10,20,30,40,50,60,70,80,90,100')
print('\n C = 1')
radial_c1 = svm.SVC(kernel = 'rbf', C = 1, gamma = 0.0)
radial_c1.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c1.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c1.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c1.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 10')
radial_c10 = svm.SVC(kernel = 'rbf', C = 10, gamma = 0.0)
radial_c10.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c10.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c10.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c10.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 20')
radial_c20 = svm.SVC(kernel = 'rbf', C = 20, gamma = 0.0)
radial_c20.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c20.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c20.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c20.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 30')
radial_c30 = svm.SVC(kernel = 'rbf', C = 30, gamma = 0.0)
radial_c30.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c30.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c30.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c30.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 40')
radial_c40 = svm.SVC(kernel = 'rbf', C = 40, gamma = 0.0)
radial_c40.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c40.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c40.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c40.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 50')
radial_c50 = svm.SVC(kernel = 'rbf', C = 50, gamma = 0.0)
radial_c50.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c50.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c50.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c50.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 60')
radial_c60 = svm.SVC(kernel = 'rbf', C = 60, gamma = 0.0)
radial_c60.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c60.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c60.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c60.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 70')
radial_c70 = svm.SVC(kernel = 'rbf', C = 70, gamma = 0.0)
radial_c70.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c70.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c70.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c70.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 80')
radial_c80 = svm.SVC(kernel = 'rbf', C = 80, gamma = 0.0)
radial_c80.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c80.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c80.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c80.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 90')
radial_c90 = svm.SVC(kernel = 'rbf', C = 90, gamma = 0.0)
radial_c90.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c90.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c90.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c90.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')

print('\n C = 100')
radial_c100 = svm.SVC(kernel = 'rbf', C = 100, gamma = 0.0)
radial_c100.fit (train_data,train_label)

#Finding Training Dataset Accuracy
predicted_label = radial_c100.predict(train_data)
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

#Finding Validation Dataset Accuracy
predicted_label = radial_c100.predict(validation_data)
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')

#Finding Testing Dataset Accuracy
predicted_label = radial_c100.predict(test_data)
print('\n Testing set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')