import os
import numpy as np
import pandas as pd
import h5py
import torch
import json
import random
import time
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import pickle
import math
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
# from sklearn.externals import joblib
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score, auc

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
    cifar10n_pt = './data/CIFAR-N/CIFAR-10_human.pt'
    cifar_h5 = r'./data/cifar_feats.h5'

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

    NOISE_TYPE = 'aggre' # ['clean', 'random1', 'random2', 'random3', 'aggre', 'worst']
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


    subset_length = 50000
    noisy = [0 if clean_label[x] == y[x] else 1 for x in range(subset_length)]
    a, b = np.unique(y[:subset_length], return_counts=True)
    print(b)
    X = X[:subset_length]
    y = y[:subset_length]


    with open('samples.json', 'rb') as fp:
        target_indice = json.load(fp)

    # target_indice = np.array([ 10,  21,  56,  71,  85, 110, 128, 142, 157, 182, 191, 249, 270,
    #    277, 287, 301, 303, 310, 321, 334, 338, 342, 346, 370, 424, 446,
    #    467, 479, 535, 537, 618, 633, 652, 693, 703, 705, 708, 729, 750,
    #    805, 865, 904, 922, 930, 941, 979, 995])

    X1 = np.delete(X.to_numpy(), target_indice, axis=0)
    y1 = np.delete(y.to_numpy(), target_indice, axis=0)
    X1= pd.DataFrame(X1)
    y1 = pd.Series(y1)

    # np.array(clean_label[target_indice])
    # np.array(aggre_label[target_indice])

    random_state = 1
    random.seed(random_state)
    np.random.seed(random_state)
    n_runs = 16000

    t1= time.time()
    rf = RandomForestClassifier(n_jobs=-1)
    print(f'--------Training RF model on noisy data')
    rf.fit(X, y)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc*100:.3f}%')

    t2 = time.time()
    print(t2-t1)
    rf = RandomForestClassifier(n_jobs=-1)
    print(f'--------Training RF model on cleaned data')
    rf.fit(X1, y1)

    t2 = time.time()
    print(t2-t1)
    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc*100:.3f}%')
