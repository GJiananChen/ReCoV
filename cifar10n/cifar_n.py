from pathlib import Path
import os
import numpy as np
import pandas as pd
import h5py
import torch
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
    file_loc = Path(__file__).resolve().parent.parent
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

    subset_length = 1000
    noisy = [0 if clean_label[x] == y[x] else 1 for x in range(subset_length)]
    gt = np.where(np.array(noisy)==1)[0].tolist()

    a, b = np.unique(y[:subset_length], return_counts=True)
    print(b)
    X = X[:subset_length]
    y = y[:subset_length]

    target_indice = [ 10, 270, 301, 338, 703, 930]
    np.array(clean_label[target_indice])
    np.array(aggre_label[target_indice])

    random_state = 1
    random.seed(random_state)
    np.random.seed(random_state)
    n_runs = 16000


    all_candidates = np.array([], dtype='int')

    for random_state in range(1, n_runs + 1):
        print(f'Current Seed: {random_state}')
        accs = []
        test_ids = []
        kf = KFold(n_splits=5, random_state=random_state, shuffle=True)
        for fold, (train, val) in enumerate(kf.split(X, y)):
            X_train, X_val, y_train, y_val = X.iloc[train], X.iloc[val], y.iloc[train].to_list(), y.iloc[val].to_list()
            rf = RandomForestClassifier()
            print(f'--------Training RF model Run: {random_state}, Fold: {fold} --------')
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_val)
            pred_proba = rf.predict_proba(X_val)

            acc = accuracy_score(y_val, y_pred)
            # auc = roc_auc_score(y_val, pred_proba, multi_class='ovr')
            # print(f'Fold: {fold}, ACC={acc:.5f}, AUC={auc:.5f}')
            print(f'Fold: {fold}, ACC={acc:.5f}')
            accs.append(acc)
            test_ids.append(val)

        candidates = test_ids[smallest_index(accs)]
        all_candidates = np.append(all_candidates, candidates)


        if random_state == 5 ** 3 or random_state == 5 ** 4 or random_state == 5 ** 5 or random_state == 5 ** 6 or random_state == 5 ** 7:
            print(all_candidates.shape)
            ids, counts = np.unique(all_candidates, return_counts=True)
            plt.clf()
            outlier_df = pd.DataFrame({'ids': ids, 'counts': counts})
            ax = sns.histplot(data=counts, stat='count')
            ax.xaxis.set_major_locator(ticker.MultipleLocator(400))
            plt.savefig(f'cifar10n_{subset_length}_hist_{random_state}.png')
            upper, lower = random_state * 1 / 5 + 2 * math.sqrt(
                random_state * (5 - 1) / 5 ** 2), random_state * 1 / 5 - 2 * math.sqrt(random_state * (5 - 1) / 5 ** 2)
            print(np.where(counts > upper))

            identified = np.where(counts > upper)[0].tolist()
            identified2 = np.where(counts < lower)[0].tolist()

            # gt = [x[0] for x in noisy]


            # print(len(gt))
            # print(len(identified))
            # print(len(identified2))
            # print(len(set(gt)-set(identified)))
            # print(len(set(gt)-set(identified2)))
            # X_train, X_test, y_train, y_test = train_test_split(
            #     X, y, test_size=0.2, random_state=0)
            # [n=900, 406, 308, 234, 406]
            # [n=1600, 406, 399, 230, 135, 406]
            # [n=10000, 406, xxx, xxx, xxx, xxx]
            # NEP P(ER1) P(ER2)
            # [n, M, F, X, ]
            F = set(identified)
            G = set(range(0, len(y))) - set(gt)
            F_t = set(range(0, len(y))) - set(identified)
            M = set(gt)
            ER1 = len(F.intersection(G)) / len(G)
            ER2 = len(F_t.intersection(M)) / len(M)
            NEP = len(F.intersection(M)) / len(F)

            with open(f'cifar10n_{subset_length}_{random_state}_with_candidates.pkl', 'wb') as f:
                pickle.dump(F, f)
                pickle.dump(G, f)
                pickle.dump(F_t, f)
                pickle.dump(M, f)
                pickle.dump(ER1, f)
                pickle.dump(ER2, f)
                pickle.dump(NEP, f)
                pickle.dump(all_candidates, f)

    print(1)
    print(1)
    # average # noisy label difference per run
