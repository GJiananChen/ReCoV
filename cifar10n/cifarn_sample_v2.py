'''
Observations:
1. just auc leads to slow convergence
2. like leads to one shot convergence, but is dependednt on the model
3. Model regularization leads to worsening performance of NEP
4. Tau implies confidence of sampling, high tau leads to more like uniform dist and vice versa
5. Lower Tau leads to faster convergence but also leads to rigidness of folds
6. High regularization and low Tau leads to subsequent decrease in performace
'''
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import warnings
warnings.filterwarnings("ignore")
import random
import h5py

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from sklearn.ensemble import RandomForestClassifier

from sample import sample_folds

file_loc = Path(__file__).resolve().parent.parent

def train_one_run(fold_splits, filter_noise=False, noisy_idx=[]):
    accs = []
    test_ids = []
    pred_probs = []
    test_labels = []
    for fold, (train, test) in enumerate(fold_splits):
        if filter_noise and len(noisy_idx)>0:
            #select 80% indice to drop randomly
            drop_idx = np.random.permutation(noisy_idx)[:int(NOISY_DROP*len(noisy_idx))]
            train = list(set(train) - set(drop_idx))
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train].to_list(), y.iloc[test].to_list()
        # rf = RandomForestClassifier(n_jobs=-1)
        rf = LogisticRegression()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        probs = rf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        # auc = roc_auc_score(y_test, y_pred)
        # print(f'Fold: {fold}, ACC={acc:.3f}, AUC={auc:.3f}')
        accs.append(acc)
        test_ids.append(test)
        pred_probs.append(probs)
        test_labels.append(y_test)
    return accs, test_ids, pred_probs, test_labels

def auc_weight_function(auc_val):
    #0.5 is random hence easily achievable
    weight = np.exp(3.5*(auc_val - 0.7))
    #the val of weight will be 1 we debias it with exp**(-0.3)
    weight = weight - 0.7
    return weight

def rank_weights(aucs, test_ids, pred_probs, test_labels, memory)->np.array:
    '''
    Gives weighting to all the samples in the dataset
    High weight implies more probability of being selected in the top fold
    '''
    n_folds = len(aucs)
    number_ids = [len(i) for i in test_ids]
    test_ids_all = np.concatenate(test_ids)
    pred_probs_all = np.concatenate(pred_probs)
    test_labels_all = np.concatenate(test_labels)
    # aucsweights = np.linspace(0,1,n_folds)
    # weights_auc = aucsweights[np.argsort(aucs)]
    # weights_auc = np.max(aucs)/aucs
    # weights_auc = np.max(aucs) - aucs
    # weights_auc = 1 - (weights_auc - weights_auc.min())/(weights_auc.max()-weights_auc.min())
    # weights_auc = np.concatenate([[weights_auc[i]]*number_ids[i] for i in range(n_folds)])
    # weights_auc = 2*(np.array(aucs) - 0.5)
    weights_auc = aucs
    # weights_auc = auc_weight_function(np.array(aucs))
    weights_auc = np.concatenate([[weights_auc[i]]*number_ids[i] for i in range(n_folds)])
    idx_temp = np.stack((np.arange(len(test_labels_all)),test_labels_all))
    temp =  pred_probs_all[idx_temp[0,:],idx_temp[1,:]]
    weights_like = 1 - (temp.max() - temp)
    # weights = 1*weights_auc + weights_like
    weights = weights_like
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.3*weights + 0.7*memory
    # print(weights[gt])
    return memory

def plot_weights(x,noise_labels):
    labels = np.zeros_like(x)
    labels[noise_labels] = 1
    data_idx = np.arange(len(x))
    sorting = np.argsort(x)
    labels = labels[sorting]
    x = x[sorting]
    x_clean = x[np.where(labels==0)[0]]
    x_noise = x[np.where(labels==1)[0]]
    fig = plt.figure()
    plt.subplot(1,2,1)
    sns.histplot(x)
    plt.subplot(1,2,2)
    plt.scatter(data_idx[np.where(labels==0)[0]],x_clean,s=1)
    plt.scatter(data_idx[np.where(labels==1)[0]],x_noise,s=1)
    plt.legend(["clean","noise"])
    plt.savefig(str(file_loc / f"results/cifar/cifar_n_sample_{NOISE_TYPE}_{TAU}_v2.png"))

def closest_value(input_list, input_value):
    difference = lambda input_list: abs(input_list - input_value)

    res = min(input_list, key=difference)

    return res

def pbinom(q,size,prob=0.5):
    """
    Calculates the cumulative of the binomial distribution
    """
    result=binom.cdf(k=q,n=size,p=prob,loc=0)
    return result

def qbinom(q, size, prob=0.5):
    """
    Calculates the quantile of the binomial distribution
    """
    result=binom.ppf(q=q,n=size,p=prob,loc=0)
    return result

def smallest_index(lst):
    return lst.index(min(lst))

def threshold_indice(lst, threshold=0.5):
    return [x[0] for x in enumerate(lst) if x[1] < threshold]

if __name__ == '__main__':
    N_RUNS = 20
    N_FOLDS = 5
    TAU = 0.1
    NOISE_TYPE = 'worst' # ['clean', 'random1', 'random2', 'random3', 'aggre', 'worst']
    RANDOM_STATE = 1
    SUBSET_LENGTH = 50000
    MEMORY_NOISE_THRES = 0.5
    #Dropping bottom 5% of the dataset
    NOISY_DROP = 0.8

    if not (file_loc / "results/cifar").is_dir():
        os.mkdir(file_loc / "results/cifar")

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    cifar10n_pt = str(file_loc / 'data/CIFAR-N/CIFAR-10_human.pt')
    cifar_h5 = str(file_loc / 'data/CIFAR-N/cifar_feats.h5')


    noise_file = torch.load(cifar10n_pt)
    clean_label = noise_file['clean_label']
    worst_label = noise_file['worse_label']
    aggre_label = noise_file['aggre_label']
    random_label1 = noise_file['random_label1']
    random_label2 = noise_file['random_label2']
    random_label3 = noise_file['random_label3']

    with h5py.File(cifar_h5, "r") as f:
        X, X_test, y, y_test = f['train_feats'], f['test_feats'], f['train_labels'], f['test_labels']
        X, X_test, y, y_test = np.array(X), np.array(X_test), np.array(y), np.array(y_test)

    if NOISE_TYPE == 'clean':
        y = clean_label
    elif NOISE_TYPE == 'aggre':
        y = aggre_label
    elif NOISE_TYPE == 'random1':
        y = random_label1
    elif NOISE_TYPE == 'random2':
        y = random_label2
    elif NOISE_TYPE == 'random3':
        y = random_label3
    elif NOISE_TYPE == 'worst':
        y = worst_label
    else:
        print('Noise type not recognized.')


    X = pd.DataFrame(X)
    X_test = pd.DataFrame(X_test)
    y = pd.Series(y)
    y_test = pd.Series(y_test)

    noisy = [0 if clean_label[x] == y[x] else 1 for x in range(SUBSET_LENGTH)]
    gt = np.where(np.array(noisy)==1)[0]
    a, b = np.unique(y[:SUBSET_LENGTH], return_counts=True)
    print(f"Labels distribution: {b}")
    X = X[:SUBSET_LENGTH]
    y = y[:SUBSET_LENGTH]

    print(f"Number of noisy samples: {np.sum(noisy)}")

    kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    fold_splits = list(kf.split(X, y))

    memory = np.zeros_like(y)
    identified = []

    for run in range(N_RUNS):
        # train for one run
        aucs, test_ids, pred_probs, test_labels = train_one_run(fold_splits,filter_noise=True,noisy_idx=identified)
        print(f"Iteration {run}: {aucs}")
        # rank the ids
        memory = rank_weights(aucs, test_ids, pred_probs, test_labels, memory)
        # Generate new set of folds based on weights
        fold_splits, fold_ids = sample_folds(N_FOLDS, memory, TAU)
        # Get K worst samples for dropping from training
        # noise_set = np.argsort(memory)[:TOP_K]
        identified = np.where(memory<=MEMORY_NOISE_THRES)[0]
        print(f"Number of noise labels identified: {len(identified)}")
        # Evaluate
        F = set(identified)
        G = set(range(0, len(y))) - set(gt)
        F_t = set(range(0, len(y))) - set(identified)
        M = set(gt)
        ER1 = len(F.intersection(G)) / len(G)
        ER2 = len(F_t.intersection(M)) / len(M)
        NEP = len(F.intersection(M)) / len(F)
        F1 = len(F.intersection(M)) / (len(F.intersection(M)) + 0.5*(len(F.intersection(G))+len(F_t.intersection(M))))
        print("F1 score: {}\nTrue noisy labels identified:{}\nFalse Positive: {}\nFalse Negative: {}".format(F1,NEP, ER1, ER2))

        plot_weights(memory, gt)

    print(identified)

    identified = np.where(memory<=MEMORY_NOISE_THRES)[0]
    F = set(identified)
    G = set(range(0, len(y))) - set(gt)
    F_t = set(range(0, len(y))) - set(identified)
    M = set(gt)
    ER1 = len(F.intersection(G)) / len(G)
    ER2 = len(F_t.intersection(M)) / len(M)
    NEP = len(F.intersection(M)) / len(F)
    print("True noisy labels identified:{}\nFalse noisy: {}\nFalse good: {}".format(NEP, ER1, ER2))

    X1 = np.delete(X.to_numpy(), identified, axis=0)
    y1 = np.delete(y.to_numpy(), identified, axis=0)
    X1= pd.DataFrame(X1)
    y1 = pd.Series(y1)

    # rf = RandomForestClassifier(n_jobs=-1)
    rf = LogisticRegression()
    print(f'--------Training RF model on noisy data')
    rf.fit(X, y)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc*100:.3f}%')

    # rf = RandomForestClassifier(n_jobs=-1)
    rf = LogisticRegression()
    print(f'--------Training RF model on cleaned data')
    rf.fit(X1, y1)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc*100:.3f}%')

    with open(str(file_loc/ f"results/cifar/memory_cifarn_{NOISE_TYPE}_{TAU}_v2.npy"),"wb") as file:
        np.save(file,memory)