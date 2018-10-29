# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from datetime import datetime
from myhelpers import *
from implementations import *


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones((len(y),1))
    yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def build_poly(data,degree):
    
    X = np.ones(data.shape[0])
    for j in range(data.shape[1]):
        X = np.c_[X,data[:,j]]
        for d in range(2,degree+1):
            X = np.c_[X,pow(data[:,j],d)]
    
    return X 


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_submission(y1, id1, y2, id2, y3, id3, y4, id4, total_ids, name_file):
    """
    Reconstruct the prediction on the whole dataset starting from the partial ones obtained with the different models
    the name_file should be in the string format 'name.csv' 
    """
    
    ypred = np.zeros(len(total_ids))
    
    ypred[id1] = y1
    ypred[id2] = y2
    ypred[id3] = y3
    ypred[id4] = y4
    
    create_csv_submission(total_ids, ypred, name_file)
    
 
def build_k_indices(y_sz, k_fold, seed):
    """build k indices for k-fold."""
    
    
    num_row = y_sz
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k, k_indices):
    """Returns the matrices and vectors Test and Train used in a k-fold cross validation"""
    
    # get k'th subgroup in test, others in train.

    X_test = x[k_indices[k]]
    y_test = y[k_indices[k]]
    X_train = np.delete(x,k_indices[k],0)
    y_train = np.delete(y,k_indices[k])
    
    y_test = y_test.reshape((y_test.shape[0],1))
    y_train = y_train.reshape((y_train.shape[0],1))
    #X_test = np.c_[X_test, np.ones(X_test.shape[0])]
    #X_train = np.c_[X_train,np.ones(X_train.shape[0])]
    return X_test, y_test, X_train, y_train



    
def build_poly(data,degree):
                                  
    X = np.ones(data.shape[0])
    for j in range(data.shape[1]):
        X = np.c_[X,data[:,j]]
        for d in range(2,degree+1):
            X = np.c_[X,pow(data[:,j],d)]
    
    return X  



def cross_validation_jet(jet_list,y_jet,gamma,lambda_,deg):
    
    accuracy = []
    maxAccuracy = 0
    for i in range(len(jet_list)):
        X = jet_list[i]
        ind = build_k_indices(y_jet[i].shape[0],4,1232342)
        X_test,y_test,X_train,y_train = cross_validation(y_jet[i],X,2,ind)
        X_train = build_poly(X_train,deg)
        X_train = np.c_[np.ones(X_train.shape[0]),X_train]
        X_test = build_poly(X_test,deg)
        X_test = np.c_[np.ones(X_test.shape[0]),X_test]
        w_opt = logistic_regression(y_train,X_train,None,gamma,lambda_)[1][-1]
        y_pred = predict(w_opt,X_test)
        tempAcc = (1 - (abs(y_pred - y_test).sum() / y_test.shape[0]))
        accuracy.append(tempAcc)
        if tempAcc > maxAccuracy:
                maxAccuracy  = tempAcc
    return maxAccuracy,np.mean(accuracy)

'''    
def process_data(X,y,cols_to_delete):
    for i in range(4):
        X = np.c_[X,np.zeros(X.shape[0])]
        
    newCols = []
    for k in range(4):
        col = X[:,30+k]
        col[np.where(X[:,22] ==k)] = 1
        X[:,30+k] = col
        
    X = np.delete(X,cols_to_delete,1)
    X = clean_data(X)
    return X
'''
def gridSearchLogisticReg(X,y):
    
    gammas = [0.1,0.01,0.001]
    lambdas = [0.1,0.01,0.001]
    k_val = [2,3,4]
    degrees = list(range(1,10))
    bestkF, bestG, bestL, bestD = [-1, -1, -1, -1]
    bestAcc = 0
    allAcc = []
    for k_folds in k_val: # loop over the number of folds
                    
            for g in gammas:
                
                for l in lambdas:
                    
                    for d in degrees:
                        
                        k_indices = build_k_indices(y.shape[0], k_folds,12343)
                        
                        x_test, y_test, x_train, y_train = cross_validation(y,X,(k_folds)//2,k_indices)
                        x_train = build_poly(x_train, d)
                        
                        x_train = np.c_[np.ones(x_train.shape[0]),x_train]
                        x_test= build_poly(x_test, d)
                        x_test = np.c_[np.ones(x_test.shape[0]),x_test]
                        
                        w_opt = logistic_regression(y_train, x_train,None, g, l)[1][-1]
                        y_pred = predict(w_opt,x_test)
                        tempAcc = (1 - (abs(y_pred - y_test).sum() / y_test.shape[0]))
                                                   
                        allAcc.append(tempAcc)
                        
                        #print('tempAcc', tempAcc)

                        if tempAcc > bestAcc:
                            bestkF = k_folds
                            bestG = g
                            bestL = l
                            bestD = d
                            bestAcc = tempAcc
    
    return bestkF, bestG, bestL, bestD, bestAcc, np.mean(allAcc)
            
   

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})
      



def group_by_jet_no(data,yb):
                                  
    jet_lists = []
    y_jet = []
    
    for i in range(4):
        jet_lists.append(data[np.where(data[:,22] == i)])
        y_jet.append(np.where(data[:,22] == i))
    
    
    cols_del = [[15,18,20,25,28,22,29,8],[15,18,20,25,28,22,23],[15,18,20,25,28,4]]

    for i,col in enumerate(cols_del):

        jet_lists[i] = np.delete(jet_lists[i],col,1)

    jet_lists[3] = np.delete(jet_lists[3],cols_del[2],1)

    for i in range(len(y_jet)):
        y_jet[i] = yb[y_jet[i]]


    jet_lists[2] = np.concatenate((jet_lists[2],jet_lists[3]))
    jet_lists = jet_lists[:3]


    y_jet[2] = np.concatenate((y_jet[2],y_jet[3]))
    y_jet = y_jet[:3]

    return jet_lists,y_jet

def get_group_indexes(X):
    jet_indexes = []
    for i in range(2):
        jet_indexes.append(X[:,22] == i)
    
    jet_indexes.append(np.logical_or(X[:,22] == 2,X[:,22] == 3))
    
    return jet_indexes
    
    #jet_list.append(data[[np.where(data[:,22] == 2), data[np.where(data[:,22] == 3)]]])                                 
    
'''After grouping the data based on the jet_no, we noticed the following:
For the first group, the one whose jet_no is 0, we can delete the columns [15,18,20,25,28,22,29,8]. The columns [15,18,20,25,28] correspond
to the phi variables, which as we have explained in the main file of the project, are uniformly distributed and do not affect our model.
The columns, 22 and 29 correspond to the variables PRI_jet_num and PRI_jet_all_pt, which are 0s for the first group. The 8th column correspond to the column DER_pt_tot, which for all the particles with jet_num = 0, is almost always equal to DER_pt_h, or in the cases when
their values differ, the difference is less than 10^-3.

For the 2nd, the particles with jet_no = 1, we can again delete the columns [15,18,20,25,28], which again correspond to the phi variables. 
For this group we can also delete the column 22 that corresponds to the variable PRI_jet_num, which is always 2, and the column 23 that corresponds to the variable PRI_jet_leading_pt which almost always has values equal to Pri_jet_all_pt, and in the cases when their values differ, the difference is less than 10^-3. Hence for the 2nd group, we can delete the following columns [15,18,20,25,28,22,23].

For the 3rd group, the particles with jet_no = 2, we can again delete the columns [15,18,20,25,28] which correspond to the phi variables. We can also delete the columns 22 and 23 using the same reasoning as for the 2nd group. We can also delete the 4th column that corresponds to the variable DER_deltaeta_jet_jet, which for all the particles that belong to this group, is equal to the difference between the values in the column 24 that correspond to the variable PRI_jet_leading_eta and the values in the column 27 that correspond to the variable PRI_jet_subleading_eta. Hence, for the 3rd group we can delete the following columns [15,18,20,25,28,22,23,4].

For the 4th group, the particles with jet_no = 3, we can delete the columns [15,18,20,25,28,4], following the same reasoning as for the 3rd group.

Since the number of particles that belong to the 3rd or the 4th group is almost equal to the number of particles that belong to the 1st group and the number of particles that belong to the 2nd group, we decided to have the particles with jet_num = 3 and jet_num = 4 as a single group.
'''
def clean_data(input_data):
    
    cols_to_delete = [15, 18, 20, 25, 28, 22]
    
    for i in range(input_data.shape[1]):
        if (input_data[:,i] == -999).all():
            if i not in cols_to_delete:
                cols_to_delete.append(i)
     
    
    
    input_data[np.where(input_data == -999)] = 0
    
    means_by_columns = np.mean(input_data, axis=0)


    for i in range(input_data.shape[1]):
        input_data[:,np.where(input_data[:,i]==-999)] = means_by_columns[i]
      

    for i in range(input_data.shape[1]):
        if (input_data[:,i].mean() == 0 or input_data[:,i].std() == 0).all():
            cols_to_delete.append(i)
     
    input_data = np.delete(input_data,cols_to_delete,1)
        
    input_data = standardize(input_data)
    return input_data
                              
def compute_accuracy(yp, yr):
    count = 0
    for idx, value in enumerate(yr):
        if value == yp[idx]:
            count += 1
    return count / len(yr)

