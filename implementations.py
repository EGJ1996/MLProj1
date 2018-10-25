# -*- coding: utf-8 -*-
"""required implementations for project 1."""

import numpy as np
from proj1_helpers import *
from myhelpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma, tol=1e-4):
    """Gradient descent algorithm."""
    
    #initializing parameters
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


    
def least_squares_SGD(y, tx, initial_w, max_iters, gamma, tol=1e-4):
    """Stochastic gradient descent algorithm."""
   
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
    
    return w, compute_mse(y,tx,w)


    
def logistic_regression(y, tx, initial_w, max_iters, gamma, tol=1e-8):
    """ Logistic regression using the GRADIENT DESCENT method
        the stopping criterium is the abs of the difference of two successive losses"""
    
    # Initializing parameters
    w = initial_w
    err = 1
    niter = 0
    losses = []
    
    while niter < max_iters and err > tol:
        niter += 1
        
        # Compute loss, gradient
        loss = compute_logloss(y, tx, w)
        grad = compute_loggrad(y, tx, w)
        
        # Update
        w = w - gamma*grad
        
        # Check criterium
        losses.append(loss)
        if len(losses) > 1:
            err = np.abs(losses[-1] - losses[-2])            
            
        # Print output
        if niter % 50 == 0:
            print("Current iteration={i}, loss={l}, norm_grad = {g}".format(i=niter, l=loss, g=grad.T@grad))
        
    return w, loss
    
    
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, tol = 1e-7):
    """ Regularized logistic regression using the GRADIENT DESCENT method
        the stopping criterium is the abs of the difference of two successive losses"""
    
    # Initializing parameters
    w = initial_w
    err = 1
    niter = 0
    losses = []
    
    while niter < max_iters and err > tol:
        niter += 1
        
        # Compute loss, gradient
        loss = compute_logloss(y, tx, w) + 0.5*lambda_*(w.T@w)
        grad = compute_loggrad(y, tx, w) + lambda_*w
        
        # Update
        w = w - gamma*grad
        
        # Check criterium
        losses.append(loss)
        if len(losses) > 1:
            err = np.abs(losses[-1] - losses[-2])
            
        # Print output
        if niter % 50 == 0:
            print("Current iteration={i}, loss={l}, norm_w = {n}".format(i=niter, l=loss, n=w.T@w))
        
    return w, loss
    