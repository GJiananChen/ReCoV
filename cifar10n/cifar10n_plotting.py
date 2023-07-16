import os.path
import os

import glob
import numpy as np
import pandas as pd
import torch
import h5py
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score, auc
from statsmodels.stats.multitest import multipletests

from scipy.stats import binom

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

import random
import math
import pickle


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
    # a, b = np.unique(y[:2000], return_counts=True)

    X = X[:subset_length]
    y = y[:subset_length]

    random_state = 1
    random.seed(random_state)
    np.random.seed(random_state)


    pkl_folder = r'F:\Projects2023\self-supervised-data-cleaning\cifar10n\pkl'

    all_candidates = []
    all_counts = np.zeros(subset_length)
    # Code for reading a folder
    for pkl in glob.glob(pkl_folder+'\*.pkl'):
        print(pkl)
        pkl_file = os.path.join(pkl_folder,pkl)
        with open(pkl_file, 'rb') as f:
            while True:
                try:
                    data = pickle.load(f)
                    ids, counts = np.unique(np.array(data).astype(np.int32),
                                            return_counts=True)
                    all_counts+=counts
                except EOFError:
                    break
    # Code for reading one file
    # pkl_file = os.path.join(pkl_folder, f'cifar10n_{subset_length}_15625_with_candidates.pkl')
    # pkl_file = os.path.join(pkl_folder, f'cifar10n_{subset_length}_0_1000.pkl')
    # data = []
    # with open(pkl_file, 'rb') as f:
    #     while True:
    #         try:
    #             data.append(pickle.load(f))
    #         except EOFError:
    #             break
    #             all_candidates = data[0]
    output_path = r'E:\PycharmProjects\AMINN_torch_dev\figures\cifar10n'
    figure_path = output_path
    if not os.path.exists(figure_path):
        os.mkdir(figure_path)


    random.seed(1)
    np.random.seed(1)



    gt = noisy
    # F = data[0]
    # G = data[1]
    # F_t = data[2]
    # M = data[3]
    # ER1 = data[4]
    # ER2 = data[5]
    # NEP = data[6]
    # all_candidates = data[7]
    c_length = len(y) / 5

    noisy_labels = gt
    for i in [120000]:
        # ids, counts = np.unique(np.array(all_candidates[:int(c_length * i)]).astype(np.int32), return_counts=True)
        ids, counts = ids, all_counts
        pvalues = [1 - pbinom(x, i, 0.2) for x in counts]
        reject, corrected_pvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')

        p = pvalues[np.where(corrected_pvalues == closest_value(corrected_pvalues, 0.05))[0][0]]
        corrected_threshold = qbinom(1 - p, i, 0.2)

        outlier_df = pd.DataFrame({'ids': ids, 'counts': counts, 'pvalues': pvalues,
                                   'corrected': corrected_pvalues, 'noisy_label': noisy_labels})

        plt.clf()
        ax = sns.histplot(data=outlier_df, x='pvalues', stat='count', multiple="stack", kde=False,
                          palette="pastel", hue="noisy_label",
                          element="bars", legend=True)

        plt.legend(title='Noisy GT', loc='best', labels=['Noisy', 'Clean'], borderaxespad=0.5)
        ax.axes.set_title(f"n_runs={i}", fontsize=20)
        ax.set_xlabel('Number of recurrance', fontsize=15)
        ax.set_ylabel('Number of samples', fontsize=15)

        # plt.xlabel('Number of recurrance')
        # plt.ylabel('Number of samples')
        plt.savefig(os.path.join(figure_path, f'cifar10n_{subset_length}_hist_{i}_pvalues.png'))

        plt.clf()
        ax = sns.histplot(data=outlier_df, x='corrected', stat='count', multiple="stack", kde=False,
                          palette="pastel", hue="noisy_label",
                          element="bars", legend=True)
        plt.legend(title='Noisy GT', loc='best', labels=['Noisy', 'Clean'], borderaxespad=0.5)
        ax.axes.set_title(f"n_runs={i}", fontsize=20)
        ax.set_xlabel('Number of recurrances', fontsize=15)
        ax.set_ylabel('Number of samples', fontsize=15)
        plt.savefig(os.path.join(figure_path, f'cifar10n_{subset_length}_hist_{i}_mtc_pvalues.png'))

        import matplotlib as mpl

        plt.clf()
        mpl.rcParams['axes.linewidth'] = 1.5
        mpl.rcParams.update({'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})
        ax = sns.histplot(data=outlier_df, x='counts', stat='count', multiple="stack", kde=False,
                          hue="noisy_label", binwidth= 5, binrange=(850, 1200),
                          element="bars", legend=True)
        # plt.legend(title='Noisy GT', loc='upper right', labels=['Noisy', 'Clean'], borderaxespad=0.5)
        plt.legend(title='Noisy GT', loc='upper left', labels=['Clean'], borderaxespad=0.5)
        plt.title(f'Cifar10N: n_runs={i}', fontsize=22)
        plt.xlabel('Number of recurrances', fontsize=18)
        plt.ylabel('Number of samples', fontsize=18)
        plt.xlim((850,1200))
        plt.xticks(np.arange(850, 1201, 50))
        plt.ylim((0,750))
        plt.yticks(np.arange(0, 751, 250))
        plt.axvline(int(corrected_threshold), 0, 1, color='r')
        plt.savefig(f'CIFAR10N_nruns{i}_counts.png', dpi=300)

        plt.clf()
        ax = sns.histplot(data=outlier_df, x='counts', stat='count', multiple="stack", kde=False,
                          palette="pastel", hue="noisy_label",
                      element="bars", legend=True)
        plt.legend(title='Noisy GT', loc='best', labels=['Noisy', 'Clean'], borderaxespad=0.5)
        ax.axes.set_title(f"CIFAR10N, n_runs={i}", fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(math.ceil(i / 100)))
        upper, lower = i * 1 / 5 + 2 * math.sqrt(i * (5 - 1) / 5 ** 2), i * 1 / 5 - 2 * math.sqrt(
            i * (5 - 1) / 5 ** 2)
        plt.axvline(int(corrected_threshold), 0, 1, color='r')
        # plt.axvline(int(upper), 0, 1, linestyle='--')
        ax.set_xlabel('Number of recurrances', fontsize=15)
        ax.set_ylabel('Number of samples', fontsize=15)
        # ax.axvline(int(upper))
        plt.savefig(os.path.join(figure_path, f'CIFAR10N_hist_{i}_counts.svg'), format='svg', dpi=600)
        plt.savefig(os.path.join(figure_path, f'CIFAR10N_hist_{i}_counts.png'), dpi=600)
    print(1)
    identified = np.where(counts > corrected_threshold)[0].tolist()

    import json
    with open("samples.json", "w") as fp:
        json.dump(identified, fp)

    highcounts = np.where(counts>3450)[0]
    loc = np.array(gt)[highcounts]
    false_pos = highcounts[loc==0]
    print(false_pos)

    lowcounts = np.where(counts<3000)[0]
    loc = np.array(gt)[lowcounts]
    false_neg = lowcounts[loc==1]
    print(false_neg)
    # if NR == 20:
    #     print(1)
    # F = set(identified)
    # G = set(range(0, len(y))) - set(gt)
    # F_t = set(range(0, len(y))) - set(identified)
    # M = set(gt)
    # mask_acc = (len(F.intersection(M)) + len(F_t.intersection(G))) / 8124
    #
    # print(mask_acc)
    # # ER1 = len(F.intersection(G)) / len(G)
    # # ER2 = len(F_t.intersection(M)) / len(M)
    # # NEP = len(F.intersection(M)) / len(F)
    # # F = {5281, 5508, 5126, 5128, 4364, 5107, 5237, 5717, 7739} for NR=0 mask_acc = 0.9988921713441654
    # # F = {720, 4364, 5126, 5128, 5237, 5281, 5508, 5717} for NR=10 mask_acc = 0.999015263417036
    # # F= {} for NR=20 mask_acc = 0.999015263417036
    # y = y.loc[list(F_t)]
    # # for index in sorted(list(F), reverse=True):
    # #     del y[index]
    # X = X.loc[F_t]
    #
    # aucs = []
    # accs = []
    # kf = KFold(n_splits=5, random_state=1, shuffle=True)
    # test_ids = []
    #
    # for fold, (train, test) in enumerate(kf.split(X, y)):
    #     X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train].to_list(), y.iloc[
    #         test].to_list()
    #     lr = LogisticRegression()
    #     lr.fit(X_train, y_train)
    #
    #     y_pred = lr.predict(X_test)
    #     acc = accuracy_score(y_test, y_pred)
    #     auc = roc_auc_score(y_test, y_pred)
    #     print(f'Fold: {fold}, ACC={acc:.3f}, AUC={auc:.3f}')
    #     aucs.append(auc)
    #     accs.append(acc)
    #     test_ids.append(test)
    # print(sum(aucs) / len(aucs))
    # print(sum(accs) / len(accs))




