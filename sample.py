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

# def weighted_random_sampling(probs:np.array,n_samples) -> Tuple[np.array,np.array]:
#     # https://link.springer.com/referenceworkentry/10.1007/978-0-387-30162-4_478#:~:text=In%20weighted%20random%20sampling%20(WRS,determined%20by%20its%20relative%20weight.
#     rand_u = np.random.rand(len(probs))
#     k = np.array([x**(1/probs[i]) for i,x in enumerate(rand_u)])
#     idx = np.argsort(k)[::-1]
#     return idx[:int(n_samples)], idx[int(n_samples):]
#     # ordering = np.random.choice(len(probs),size=n_samples,replace=False,p=probs)
#     # return ordering

# def sample_folds(n_folds:int, weights:np.array, tau:float) -> Tuple[List,Dict]:
#     '''
#     Given weights of the sample, performs weighted sampling into n_folds
#     taking temperature tau into account
#     '''
#     n_samples = len(weights)/n_folds
#     #Assuming 'best' is the best fold and 'worst' is the worst fold and 'rest' are in between folds
#     fold_id = {"best":[],"rest":[],"worst":[]}
#     # Step 1: Get the Probabilites for dividing into fold 0
#     # p_i = get_sample_probs(weights,tau)
#     # Step 2: Find samples to go into fold 0
#     fold_idx,rest_idx = weighted_random_sampling(p_i,n_samples)
#     fold_id["best"] = fold_idx
#     #Step 3: Find samples to go into fold rest
#     weights_temp = weights[rest_idx]
#     # p_i = get_sample_probs(weights_temp, tau)
#     fold_idx, rest_idx = weighted_random_sampling(p_i,(n_folds-2)*n_samples)
#     fold_id["rest"] = fold_idx
#     fold_id["worst"] = rest_idx
#     #Convert into train_test splits
#     rest_ids = np.random.permutation(fold_id["rest"])
#     rest_ids = [rest_ids[i*int(np.ceil(len(rest_ids)/(n_folds-2))):(i+1)*int(np.ceil(len(rest_ids)/(n_folds-2)))] for i in range(n_folds-2)]
#     folds = [fold_id["best"]] + rest_ids + [fold_id["worst"]]
#     fold_splits = []
#     for i in range(n_folds):
#         temp = folds.copy()
#         test_ids = np.random.permutation(temp.pop(i))
#         train_ids = np.random.permutation(np.concatenate(temp))
#         fold_splits.append((train_ids,test_ids))
#     return fold_splits, fold_id

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
    # rest_ids = np.random.permutation(fold_id["rest"])
    rest_ids = fold_id["rest"]
    rest_ids = [rest_ids[i*int(np.ceil(len(rest_ids)/(n_folds-2))):(i+1)*int(np.ceil(len(rest_ids)/(n_folds-2)))] for i in range(n_folds-2)]
    folds = [fold_id["best"]] + rest_ids + [fold_id["worst"]]
    fold_splits = []
    for i in range(n_folds):
        temp = folds.copy()
        test_ids = np.random.permutation(temp.pop(i))
        train_ids = np.random.permutation(np.concatenate(temp))
        fold_splits.append((train_ids,test_ids))
    return fold_splits, fold_id