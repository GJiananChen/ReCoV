"""
This scripts perform weighted sampling
"""
from typing import Dict, List, Tuple
import numpy as np

def get_sample_probs(weights:np.array, tau:float) -> np.array:
    '''
    Gets sampling probability based on weights
    '''
    exp_i = np.exp(weights/tau)
    return exp_i/np.sum(exp_i)

def sample_folds(n_folds:int, weights:np.array, tau:float) -> Tuple[List,Dict]:
    '''
    Given weights of the sample, performs weighted sampling into n_folds
    taking temperature tau into account
    '''
    n_samples = len(weights)
    #Assuming 'best' is the best fold and 'worst' is the worst fold and 'rest' are in between folds
    fold_id = {"best":[],"rest":[],"worst":[]}
    p_i = get_sample_probs(weights,tau)
    if tau<0.0001:
        #tau = 0 case, deterministic
        ordering = np.argsort(weights)[::-1]
    elif tau>10000:
        #infinite tau case
        ordering = np.random.permutation(len(weights))
    else:
        ordering = np.random.choice(n_samples,size=n_samples,replace=False,p=p_i)
    fold_id["best"] = ordering[:int(n_samples/n_folds)]
    fold_id["rest"] = ordering[int(n_samples/n_folds):(n_folds-1)*int(n_samples/n_folds)]
    fold_id["worst"] = ordering[(n_folds-1)*int(n_samples/n_folds):]
    # Convert into train_test splits
    rest_ids = fold_id["rest"]
    #split into n-2 folds with two reserved for best and worst
    rest_ids = [rest_ids[i*int(np.ceil(len(rest_ids)/(n_folds-2))):(i+1)*int(np.ceil(len(rest_ids)/(n_folds-2)))] for i in range(n_folds-2)]
    folds = [fold_id["best"]] + rest_ids + [fold_id["worst"]]
    fold_splits = []
    for i in range(n_folds):
        temp = folds.copy()
        test_ids = np.random.permutation(temp.pop(i))
        train_ids = np.random.permutation(np.concatenate(temp))
        fold_splits.append((train_ids,test_ids))
    return fold_splits, fold_id