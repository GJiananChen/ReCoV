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

file_loc = Path(__file__).resolve().parent.parent

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
    weights_auc = np.max(aucs) - aucs
    weights_auc = 1 - (weights_auc - weights_auc.min())/(weights_auc.max()-weights_auc.min())
    weights_auc = np.concatenate([[weights_auc[i]]*number_ids[i] for i in range(n_folds)])
    idx_temp = np.stack((np.arange(len(test_labels_all)),test_labels_all))
    temp =  pred_probs_all[idx_temp[0,:],idx_temp[1,:]]
    weights_like = 1 - (temp.max() - temp)
    # weights = (weights_auc**80)*weights_like
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
    plt.savefig(str(file_loc / "results/mushroom_sample.png"))


N_RUNS = 10
NOISE_RATIO = 10
N_FOLDS = 5
TAU = 0.1
random.seed(1)
np.random.seed(1)

mushroom = pd.read_csv(str(file_loc/'data/mushrooms.csv'))

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
    # y[index] = label
    y = pd.Series(y)

TOP_K = len(noisy)
random_state = 1
kf = KFold(n_splits=N_FOLDS, random_state=random_state,shuffle=True)
fold_splits = list(kf.split(X,y))
gt = [x[0] for x in noisy]
memory = np.zeros_like(y)
for run in range(N_RUNS):
    #train for one run
    aucs, test_ids, pred_probs, test_labels = train_one_run(fold_splits)
    print(f"Iteration {run}: {aucs}")
    #rank the ids
    memory = rank_weights(aucs,test_ids, pred_probs, test_labels, memory)
    # random.seed(run)
    # np.random.seed(run)
    # kf = KFold(n_splits=N_FOLDS, random_state=random_state,shuffle=True)
    # fold_splits = list(kf.split(X,y))
    #Generate new set of folds based on weights
    fold_splits, fold_ids = sample_folds(N_FOLDS,memory,TAU)
    #Get K worst samples
    identified = np.argsort(memory)[:TOP_K]
    #Evaluate
    gt = [x[0] for x in noisy]
    F = set(identified)
    G = set(range(0,len(y))) - set(gt)
    F_t = set(range(0,len(y))) - set(identified)
    M = set(gt)
    ER1 = len(F.intersection(G))/len(G)
    ER2 = len(F_t.intersection(M))/len(M)
    NEP = len(F.intersection(M))/ len(F)
    # print(np.sort(np.array(list(F.intersection(G))))[:30])
    print("True noisy labels identified:{}\nFalse noisy: {}\nFalse good: {}".format(NEP,ER1,ER2))
plot_weights(memory,gt)
#     if (run+1)%10==0:
#         print("ok")
# print("ok")