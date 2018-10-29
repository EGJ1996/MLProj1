# -*- coding: utf-8 -*-
"""required implementations for project 1."""

import numpy as np
from proj1_helpers import *
from myhelpers import *

def least_squares_GD(y, tx, initial_w, gamma):
    """Gradient descent algorithm."""
    
    tol = 1e-5
    max_iters = 1000
    #initializing parameters
    
    if initial_w is None:
        initial_w = np.random.rand(tx.shape,1)
        
    w_old = initial_w
    n_iter = 0
    err = 1
    
    # looping to max_iters or to the reached tolerance
    while n_iter < max_iters and err > tol:
        n_iter += 1
        grad = compute_gradient(y,tx,w_old)
   
        # update optimal set of parameters
        w_new = w_old - gamma*grad
        err = np.linalg.norm(w_new - w_old)
        w_old = w_new
        
    return w_new, compute_mse(y,tx,w_new)

'''
We will try to find a good model that uses gradient descent by iterating over different values of gammas and different values of
k for the kfold cross validation
'''
def best_GD(y,tx,initial_w):
    gammas = [0.1,0.01,0.001]
    kfolds = [1,5,10]
    deg = [2,3,4,5] # Higher degrees may lead to overfitting
    allAcc = []
    bestFold, bestDeg, bestG, bestAcc = [0,0,0,0]
    
    for k_fold in kfolds:
        
        for g in gammas:
            
            for d in deg:
            
                ind = build_k_indices(y.shape[0], k_fold, seed)
                X_test,y_test,X_train,y_train = cross_validation(y,tx,k_fold//2,ind)
                X_test = build_poly(X_test,d)
                X_test = np.c_[np.ones(X_test.shape[0]),X_test]
                X_train = build_poly(X_train,d)
                X_train = np.c_[np.ones(X_train.shape[0]),X_train]
                w_opt = least_squares_GD(y_train, X_train, None, g)[0]
                y_pred = predict(w_opt, X_test)
                acc = (1 - (abs(y_pred - y_test).sum() / y_test.shape[0]))
                allAcc.append(acc)
                if acc > bestAcc:
                    bestAcc = acc
                    bestFold = k_fold
                    bestDeg = d
                    bestG = g
    
    return bestFold, bestDeg, bestG, bestAcc, np.mean(allAcc)
                

    
def least_squares_SGD(y, tx, initial_w,gamma):
    """Stochastic gradient descent algorithm."""
   
    max_iters = 1000
    tol = 1e-5
    w_old = initial_w
    n_iter = 0
    err = 1
    
    # looping to max_iters or to the reached tolerance
    while n_iter < max_iters and err > tol:
        n_iter += 1
        
        # Use the standard minibatch_size = 1
        for miniy, minitx in batch_iter(y,tx,1):
            g = compute_gradient(miniy, minitx,w_old)
            
        w_new = w_old - gamma*g
        err = np.linalg.norm(w_new-w_old)
        w_old = w_new
    
    return w_new, compute_mse(y,tx,w_new)


'''
We will try to find a good model that uses stochatic gradient descent by iterating over different values of gammas, different values of
k for the kfold cross validation and different values of deg for building a polynomial basis
'''
def best_least_squares_SGD(y,tx,initial_w):
    gammas = [0.1,0.01,0.001]
    kfolds = [1,5,10]
    deg = [2,3,4,5]
    allAcc = []
    bestFold, bestDeg, bestG, bestAcc = [0,0,0,0]
    
    for k_fold in kfolds:
        
        for g in gammas:
            
            for d in deg:
            
                ind = build_k_indices(y.shape[0], k_fold, seed)
                X_test,y_test,X_train,y_train = cross_validation(y,tx,k_fold//2,ind)
                X_test = build_poly(X_test,d)
                X_test = np.c_[np.ones(X_test.shape[0]),X_test]
                X_train = build_poly(X_train,d)
                X_train = np.c_[np.ones(X_train.shape[0]),X_train]
                w_opt = least_squares_SGD(y_train, X_train, None, g)[0]
                y_pred = predict(w_opt, X_test)
                acc = (1 - (abs(y_pred - y_test).sum() / y_test.shape[0]))
                allAcc.append(acc)
                if acc > bestAcc:
                    bestAcc = acc
                    bestDeg = d
                    bestG = g
                    bestFold = k_fold
    
    return bestFold, bestDeg, bestG, bestAcc, np.mean(allAcc)
                

def least_squares(y, tx):
    """calculate the least squares solution by solving the normal equations."""
    
    transp=tx.T
    w = np.linalg.solve(transp@tx, transp@y)
    
    return w, compute_mse(y,tx,w)


    
def ridge_regression(y, tx, lambda_):
    """implement ridge regression by solving normal equations."""
    transp = tx.T
    m = transp@tx + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    w = np.linalg.solve(m, transp@y)
    
    return w,compute_mse(y,tx,w)

'''
In order to find a good model for ridge regression, we apply the same method as for the cases of GD and SGD, by iterating over the values
of k for the kfold cross validation, over the degrees for the polynomial basis and over the lambdas.
'''
def best_ridge(y,tx):
    kfolds = [2,5,10]
    deg = [2,3,4,5,10,15] # Higher degrees may lead to overfitting
    lambdas = [0.001,0.01,0.1,0.5,0.7]
    allAcc = []
    bestFold, bestDeg, bestL, bestAcc = [0,0,0,0]
    
    for k_fold in kfolds:
        
        for l in lambdas:
            
            for d in deg:
            
                ind = build_k_indices(y.shape[0], k_fold, 123344)
                X_test,y_test,X_train,y_train = cross_validation(y,tx,k_fold//2,ind)
                X_test = build_poly(X_test,d)
                X_test = np.c_[np.ones(X_test.shape[0]),X_test]
                X_train = build_poly(X_train,d)
                X_train = np.c_[np.ones(X_train.shape[0]),X_train]
                w_opt = ridge_regression(y_train, X_train,l)[0]
                y_pred = predict(w_opt, X_test)
                acc = (1 - (abs(y_pred - y_test).sum() / y_test.shape[0]))
                allAcc.append(acc)
                if acc > bestAcc:
                    bestFold = k_fold
                    bestDeg = d
                    bestL = l
                    bestAcc = acc
    
    return bestFold, bestDeg, bestL, bestAcc, np.mean(allAcc)

'''
Since ridge regression is the one that performs better, I will try to explore it a bit further by applying ridge regression with cross validation in the whole dataset, such that the data are grouped by their jet numbers. This approach proved quite efficient for the logistic regression and it may prove the same for the ridge regression.
'''

def best_ridge_jetno(y,tx):
    kfolds = [2,5,10]
    deg = [2,3,4,5] # Higher degrees may lead to overfitting
    lambdas = [0.01,0.1,0.5]
    preds = []
    allAcc = []
    bestFold, bestDeg, bestL, bestAcc = [0,0,0,0]
   
    for k_fold in kfolds:

        for l in lambdas:

            for d in deg:
                ind = build_k_indices(y.shape[0], k_fold, 123344)

                X_test,y_test,X_train,y_train = cross_validation(y,tx,k_fold//2,ind)

                jet_xtrain, jet_ytrain = group_by_jet_no(X_train,y_train)
                jet_xtest, jet_ytest = group_by_jet_no(X_test,y_test)
                for i in range(3):
                    Xi_train = jet_xtrain[i]
                    Xi_train = build_poly(Xi_train,d)
                    Xi_train = np.c_[np.ones(Xi_train.shape[0]),Xi_train]
                    yi_train = jet_ytrain[i]

                    Xi_test = jet_xtest[i]
                    Xi_test = build_poly(Xi_test,d)
                    Xi_test = np.c_[np.ones(Xi_test.shape[0]),Xi_test]
                    yi_test = jet_ytest[i]

                    w_opt = ridge_regression(yi_train, Xi_train,l)[0]

                    y_pred = predict(w_opt, Xi_test)
                    preds.append(y_pred == yi_test)
                
                acc = 0
                
                for i in range(len(preds)):
                    acc+=np.sum(preds[i])

                acc = acc / y_test.shape[0]
                #acc = (1 - (abs(y_pred - y_test).sum() / y_test.shape[0]))
                allAcc.append(acc)
                if acc > bestAcc:
                    bestFold = k_fold
                    bestDeg = d
                    bestL = l
                    bestAcc = acc

    return bestFold, bestDeg, bestL, bestAcc, np.mean(allAcc)



def group_features_by_jet(x):
    return {  
        0: x[:, 22] == 0,
        1: x[:, 22] == 1,
        2: np.logical_or(x[:, 22] == 2, x[:, 22] == 3)  
    }


def cross_validation_with_jet_no(y, x, k_indices, k, lambdas, degrees):
    cols_to_delete = [[15,18,20,25,28,22,29,8],[15,18,20,25,28,22,23],[15,18,20,25,28,4]]
    test_indice = k_indices[k]
    train_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)].reshape(-1)

    Y_test = y[test_indice]
    Y_train = y[train_indice]
    X_test = x[test_indice]
    X_train = x[train_indice]

    # Split data into three subset by jet_no
    jets_train,y_train = group_by_jet_no(X_train,Y_train)
    train_indexes = get_group_indexes(X_train)
    jets_test,y_test = group_by_jet_no(X_test,Y_test)
    test_indexes = get_group_indexes(X_test)
    
    predicted_y_train = np.zeros(len(Y_train))
    predicted_y_test = np.zeros(len(Y_test))

    for index in range(len(jets_train)):
        x_train = jets_train[index]
        x_test = jets_test[index]
               
        x_train = clean_data(x_train)
        x_test = clean_data(x_test)
        
        x_train = build_poly(x_train, degrees[index])
        x_test = build_poly(x_test, degrees[index])
        
        x_train = np.c_[np.ones(x_train.shape[0]),x_train]
        x_test = np.c_[np.ones(x_test.shape[0]),x_test]

        w, loss = ridge_regression(y_train[index], x_train, lambdas[index])
        
        predicted_y_train[train_indexes[index]] = np.reshape(predict(w, x_train),-1)
        predicted_y_test[test_indexes[index]] = np.reshape(predict(w, x_test),-1)

    acc_train = compute_accuracy(predicted_y_train, Y_train)
    acc_test = compute_accuracy(predicted_y_test, Y_test)

    return acc_train, acc_test


def sigmoid(z):

    return 1 / (1 + np.exp(-z))

def log_cost(y, tx, w, lambda_ = 0):

    epsilon = 1e-10

    h= sigmoid(tx.dot(w));
    # epsilon is added in order to avoid log(0) or log(-inf)
    
    cost = (-y) * np.log(h + epsilon) - (1-y) * np.log(1-h + epsilon)
    cost = cost.mean() + np.dot(w.T,w)*lambda_ / (2 * y.shape[0])
    return np.squeeze(cost)


def learning_by_gradient_descent(y, tx, w, gamma,lambda_ = 0):
    
    h = sigmoid(tx.dot(w)) - y

    grad = tx.T.dot(h) / y.shape[0]
    
    grad = grad + w * lambda_ / y.shape[0]
    
    w = w - gamma * grad
    return log_cost(y,tx,w,lambda_), w


def logistic_regression(y, tx,initial_w, gamma, lambda_ = 0):
    
    if initial_w is None:
        initial_w = np.random.rand(tx.shape[1],1)
    
    ws = [initial_w]

    losses = []

    w = initial_w
    
    eps = 1e-7
    
    maxIter = 1000

    niter = 0

    while True or niter > maxIter:
        loss, w = learning_by_gradient_descent(y,tx,w,gamma,lambda_)
        
        ws.append(w)
        
        losses.append(loss)

        #print("loss ",loss)
      
        
        if len(losses) >= 2:
            if losses[-2] - losses[-1] < eps:
                break
        
        niter = niter + 1
        
    
    return losses, ws

def test_reg_logistic_regresion(X_train, y_train, X_test, y_test, gamma, lambda_ = 0):
    
    #In this function, we test the accuracy of our logistic model

    initial_w = np.random.randn(X_train.shape[1],1)
    
    losses, ws = logistic_regression(y_train, X_train, initial_w, gamma, lambda_)

    y_pred = predict(ws[-1], X_test)

    acc = (1 - (abs(y_pred - y_test).sum() / y_test.shape[0]))

    return acc


def predict(weights, data):
    
    y_pred = np.dot(data, weights)
    
    
    y_pred[np.where(y_pred <= 0)] = -1
    
    y_pred[np.where(y_pred > 0)] = 1

    return y_pred

