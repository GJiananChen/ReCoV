import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import warnings
warnings.filterwarnings("ignore")
import random
import h5py
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

from utils.sample import sample_folds
from cifar10n.cifar_utils import cifar_evaluate, load_cifardataset

file_loc = Path(__file__).resolve().parent.parent

def train_one_run(fold_splits, filter_noise=False, noisy_idx=[]):
    accs = []
    test_ids = []
    pred_probs = []
    test_labels = []
    for fold, (train, test) in enumerate(fold_splits):
        if filter_noise and len(noisy_idx)>0:
            #select x% indices to drop randomly
            drop_idx = np.random.permutation(noisy_idx)[:int(NOISY_DROP*len(noisy_idx))]
            train = list(set(train) - set(drop_idx))
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train].to_list(), y.iloc[test].to_list()
        rf = LogisticRegression()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        probs = rf.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        test_ids.append(test)
        pred_probs.append(probs)
        test_labels.append(y_test)
    return accs, test_ids, pred_probs, test_labels

def rank_weights(test_ids, pred_probs, test_labels, memory)->np.array:
    '''
    Gives weighting to all the samples in the dataset
    High weight implies more probability of being selected in the top fold
    '''
    test_ids_all = np.concatenate(test_ids)
    pred_probs_all = np.concatenate(pred_probs)
    test_labels_all = np.concatenate(test_labels)
    idx_temp = np.stack((np.arange(len(test_labels_all)),test_labels_all))
    temp =  pred_probs_all[idx_temp[0,:],idx_temp[1,:]]
    #likelihood
    weights = 1 - (temp.max() - temp)
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.3*weights + 0.7*memory

    true = test_labels_all[np.argsort(test_ids_all)]

    return memory, true

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
    plt.savefig(str(file_loc / f"results/cifar/cifar_n_fastrecov_{EXP_NAME}.png"))


if __name__ == '__main__':
    N_RUNS = 10
    N_FOLDS = 5
    TAU = 0.1
    NOISE_TYPE = 'aggre' # ['clean', 'random1', 'random2', 'random3', 'aggre', 'worst']
    FEAT = 'dinov2' # ['dinov2', 'imagenet']
    RANDOM_STATE = 1
    SUBSET_LENGTH = 50000
    MEMORY_NOISE_THRES = 0.3
    #Dropping bottom x% of the dataset
    NOISY_DROP = 0.8
    EXP_NAME = f"{NOISE_TYPE}_{TAU}_{FEAT}_{N_RUNS}_v5"

    Path.mkdir(file_loc / "results/cifar",exist_ok=True,parents=True)

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    cifar10n_pt = str(file_loc / f"data/CIFAR-10_human.pt")
    cifar_h5 = str(file_loc / f"data/cifar_feats_{FEAT}.h5")

    X, y, X_test, y_test, noisy, gt, clean_label = load_cifardataset(cifar10n_pt, cifar_h5, NOISE_TYPE, SUBSET_LENGTH)

    kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE, shuffle=True)
    fold_splits = list(kf.split(X, y))

    memory = np.zeros_like(y)
    identified = []

    for run in range(N_RUNS):
        # train for one run
        accs, test_ids, pred_probs, test_labels = train_one_run(fold_splits,filter_noise=True,noisy_idx=identified)
        print(f"Iteration {run}: {accs}")
        # rank the ids
        memory, true = rank_weights(test_ids, pred_probs, test_labels, memory)
        # Generate new set of folds based on weights
        fold_splits, fold_ids = sample_folds(N_FOLDS, memory, TAU)
        # Get K worst samples for dropping from training
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

    identified = np.where(memory<=MEMORY_NOISE_THRES)[0]

    cifar_evaluate(cifar10n_pt,cifar_h5,identified,NOISE_TYPE,SUBSET_LENGTH,RANDOM_STATE)