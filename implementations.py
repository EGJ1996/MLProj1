# -*- coding: utf-8 -*-
"""required implementations for project 1."""

import numpy as np
from proj1_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    
    #initializing parameters
    w_old = initial_w
    n_iter = 0
    
    #looping to max_iters
    while n_iter < max_iters:
        n_iter += 1
        grad = compute_gradient(y,tx,w_old)
   
        # update optimal set of parameters
        w_new = w_old - gamma*grad
        w_old = w_new
        
        print("Gradient Descent({bi}/{ti}):  w5={w0}, w17={w1}".format(
            bi=n_iter, ti=max_iters - 1,  w0=w_new[5], w1=w_new[17]))
        
    
    return w_new, compute_mse(y,tx,w_new)


    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
   
    w_old = initial_w
    
    for n_iter in range(max_iters):
        
        # Use the standard minibatch_size = 1
        for miniy, minitx in batch_iter(y,tx,1):
            g = compute_gradient(miniy, minitx,w_old)
            
        w_new = w_old - gamma*g
        w_old = w_new
        
        print("Stoch Gradient Descent({bi}/{ti}):  w5={w0}, w17={w1}".format(
             bi=n_iter, ti=max_iters - 1,  w0=w_new[5], w1=w_new[17]))
        
    return w_new, compute_mse(y,tx,w_new)




def least_squares(y, tx):
    """calculate the least squares solution by solving the normal equations."""
    
    transp=tx.T
    w = np.linalg.solve(transp@tx, transp@y)
    
    return w, compute_mse(y,tx,w)


    
#def ridge_regression(y, tx, lambda_):
 #    """implement ridge regression by solving normal equations."""
    
  #  m = tx.T@tx + 2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
   # w = np.linalg.solve(m, tx.T@y)
    
    #return w, compute_mse(y,tx,w)


    
#def logistic_regression(y, tx, initial_w, max_iters, gamma):
    
#def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    