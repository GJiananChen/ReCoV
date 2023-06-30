import os
import warnings
warnings.filterwarnings("ignore")
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

from sample import sample_folds

def train_one_run(fold_splits):
    aucs = []
    test_ids = []
    pred_probs = []
    test_labels = []
    for fold, (train, test) in enumerate(fold_splits):
        X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train].to_list(), y.iloc[test].to_list()
        lr = LogisticRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        probs = lr.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        # print(f'Fold: {fold}, ACC={acc:.3f}, AUC={auc:.3f}')
        aucs.append(acc)
        test_ids.append(test)
        pred_probs.append(probs)
        test_labels.append(y_test)
    return aucs, test_ids, pred_probs, test_labels

def rank_weights(aucs, test_ids, pred_probs, test_labels)->np.array:
    '''
    Gives weighting to all the samples in the dataset
    '''
    n_folds = len(aucs)
    number_ids = [len(i) for i in test_ids]
    test_ids_all = np.concatenate(test_ids)
    pred_probs_all = np.concatenate(pred_probs)
    test_labels_all = np.concatenate(test_labels)
    # aucsweights = np.linspace(0,1,n_folds)
    # weights_auc = aucsweights[np.argsort(aucs)]
    # weights_auc = np.max(aucs)/aucs
    weights_auc = np.max(aucs) - aucs
    weights_auc = (weights_auc - weights_auc.min())/(weights_auc.max()-weights_auc.min())
    weights_auc = np.concatenate([[weights_auc[i]]*number_ids[i] for i in range(n_folds)])
    idx_temp = np.stack((np.arange(len(test_labels_all)),test_labels_all))
    temp =  pred_probs_all[idx_temp[0,:],idx_temp[1,:]]
    weights_like = temp.max() - temp
    # weights = (weights_auc**80)*weights_like
    weights = weights_auc*5 + weights_like
    weights = weights[test_ids_all]
    # print(weights[gt])
    return weights

N_RUNS = 500
NOISE_RATIO = 10
N_FOLDS = 5
TAU = 1
random.seed(1)
np.random.seed(1)

mushroom = pd.read_csv(r'./mushrooms.csv')

cols = mushroom.columns.to_list()
cols.remove('class')

mushroom = pd.get_dummies(mushroom, columns=cols)

X = mushroom.drop(['class'], axis=1).copy()
y = mushroom['class'].copy()
y[y == 'p'] = 1
y[y == 'e'] = 0
y = y.astype('int').to_list()

noisy = random.sample(list(enumerate(y)), int(0.01*NOISE_RATIO*len(y)))
for i in noisy:
    index, label = i[0], i[1]
    y[index] = 1 - label
    y = pd.Series(y)

TOP_K = len(noisy)
random_state = 1
kf = KFold(n_splits=N_FOLDS, random_state=random_state,shuffle=True)
fold_splits = list(kf.split(X,y))
gt = [x[0] for x in noisy]
for run in range(N_RUNS):
    #train for one run
    aucs, test_ids, pred_probs, test_labels = train_one_run(fold_splits)
    print(aucs)
    #rank the ids
    weights = rank_weights(aucs,test_ids, pred_probs, test_labels)
    #Generate new set of folds based on weights
    fold_splits, fold_ids = sample_folds(N_FOLDS,weights,TAU)
    #Get K worst samples
    identified = np.argsort(weights)[:TOP_K]
    #Evaluate
    gt = [x[0] for x in noisy]
    F = set(identified)
    G = set(range(0,len(y))) - set(gt)
    F_t = set(range(0,len(y))) - set(identified)
    M = set(gt)
    ER1 = len(F.intersection(G))/len(G)
    ER2 = len(F_t.intersection(M))/len(M)
    NEP = len(F.intersection(M))/ len(F)
    print("True noisy labels identified:{}\nFalse noisy: {}\nFalse good: {}".format(NEP,ER1,ER2))