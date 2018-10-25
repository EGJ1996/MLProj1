# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np

def compute_mse(y, tx, w):
    """Calculate the mse."""

    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)


 
def compute_gradient(y, tx, w):
    """Compute the gradient."""
    
    e = y - tx.dot(w)
    
    grad = -tx.T.dot(e) / len(e)
    return grad



def standardize(x):
    x = (x - np.mean(x,axis=0)) / np.std(x, axis=0)
    return x


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
            
            
def get_best_rmse(lam, deg, losses):
    """Get the best (lam, deg) from the result of grid search."""
 
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], deg[min_row], lam[min_col]



def get_best_accuracy(lam, deg, accuracies):
    """Get the best (lam, deg) from the result of grid search."""
    
    max_row, max_col = np.unravel_index(np.argmax(accuracies), accuracies.shape)
    return accuracies[max_row, max_col], deg[max_row], lam[max_col]


"""
Helper functions for the logistic regression
"""

def sigmoid_scalar(t):
    """Apply sigmoid function on t."""
    
    if t >= 0:
        return 1.0 / (1.0 + np.exp(-t))
    else:
        return np.exp(t) / (1 + np.exp(t))
    
sigmoid = np.vectorize(sigmoid_scalar)



def compute_logloss(y, tx, w):
    
    one = np.ones(len(y))
    first = (np.log(1 + np.exp(tx@w))) @ one

    second = y @ (tx@w)
    
    return first-second



def compute_loggrad(y, tx, w):
    """returns the gradient of logloss."""
    
    sig = sigmoid(tx@w)
    return tx.T @ (sig - y)



def compute_hessian(y, tx, w):
    """
    Returns the hessian of the logloss function.
    (with this implementation the computational cost in terms of time and memory 
    can be very high, since we build a NxN matrix
    """
    
    sig = np.squeeze(sigmoid(tx@w))
    
    S = np.diag(sig - sig*sig)
    SX = S@tx
    
    return tx.T @ SX



def logistic_newton(y, tx, initial_w, max_iters, gamma, tol=1e-8):
    """ Logistic regression using the NEWTON method.
        The stopping criterium is the abs of the difference of two successive losses"""
    
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
        hess = compute_hessian(y, tx, w)
      
        # Update with Newton
        v = np.linalg.solve(hess,grad)
        w = w - gamma*v
        
        # Check criterium
        losses.append(loss)
        if len(losses) > 1:
            err = np.abs(losses[-1] - losses[-2])            
            
        # Print output
        if niter % 50 == 0:
            print("Current iteration={i}, loss={l}, norm_grad = {g}".format(i=niter, l=loss, g=grad.T@grad))
        
    return w, loss



def reg_logistic_newton(y, tx, lambda_, initial_w, max_iters, gamma, tol = 1e-8):
    """ Regularized logistic regression using the NEWTON method.
        The stopping criterium is the abs of the difference of two successive losses"""
    
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
        hess = compute_hessian(y, tx, w) + np.diag(lambda_*np.ones(len(w)))
      
        # Update with Newton
        v = np.linalg.solve(hess,grad)
        w = w - gamma*v
        
        # Check criterium
        losses.append(loss)
        if len(losses) > 1:
            err = np.abs(losses[-1] - losses[-2])            
            
        # Print output
        if niter % 50 == 0:
            print("Current iteration={i}, loss={l}, norm_grad = {g}".format(i=niter, l=loss, g=grad.T@grad))
        
    return w, loss