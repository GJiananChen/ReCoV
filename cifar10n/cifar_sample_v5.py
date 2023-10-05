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

def calc_entropy(pred_probs):
    '''
    Calculates per negative sample entropy with a bias term and normalization
    Parameters:
        pred_probs: (n x n_classes)
    '''
    n_classes = np.shape(pred_probs)[1]
    entropy = -pred_probs*np.log2(pred_probs)
    # return 1 - np.sum(entropy,axis=1)/np.log2(n_classes)
    return np.sum(entropy,axis=1)/np.log2(n_classes)

def calc_margin(pred_probs, test_labels):
    # order = np.arange(len(test_labels))
    # temp = np.argsort(pred_probs,axis=1)
    # return 1 - (pred_probs[order,temp[:,-1]] - pred_probs[order,temp[:,-2]])
    test_index = np.stack((np.arange(len(test_labels)),test_labels))
    highest_index = np.argsort(pred_probs,axis=1)[:,-1]
    return pred_probs[test_index[0,:],test_index[1,:]] - pred_probs[test_index[0,:],highest_index]

def catch_label(pred_probs,test_labels, top_hits=3):
    '''
    Hypothesis: Faulty labels will be in one of the top hit labels for clean and wont be there for noisy
    '''
    # preds = np.argsort(pred_probs,axis=1)[:,-1]
    n_classes = pred_probs.shape[1]
    top_hit_preds = np.argsort(pred_probs,axis=1)[:,::-1]
    hit_loc = np.where((top_hit_preds-test_labels.reshape(-1,1))==0)[1]
    # score_array = np.array([(top_hits-i)*(1/top_hits) if i < top_hits else 0 for i in range(n_classes)])
    # score_array = np.array([1,0.8,0.7,0,0,0,0,0,0,0])
    score_array = np.array([1 if i < top_hits else 0 for i in range(n_classes)])
    return score_array[hit_loc]

def rank_weights(aucs, test_ids, pred_probs, test_labels, memory)->np.array:
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
    # weights_like = 1 - (temp.max() - temp)
    weights_like = temp
    # weights_like = (np.exp(-5*temp)-1)/(np.exp(-5)-1)
    #Entropy lies between 0 to 2
    weights_entropy = calc_entropy(pred_probs_all)
    # preds = 1*(np.argmax(pred_probs_all,axis=1)==test_labels_all)
    # weights_entropy = weights_entropy*(2*preds-1) + 1
    #margin lies between 0 to 2
    weights_margin = calc_margin(pred_probs_all,test_labels_all) + 1
    # weights_margin = calc_margin(pred_probs_all,test_labels_all)*(2*preds-1) + 1
    # weight_hits = catch_label(pred_probs_all,test_labels_all)
    # weights_hits = 4 - filter_tophits(pred_probs_all, test_labels_all, thresh=0.05, hits_thresh=3)
    # weights = 3*weights_like + weights_entropy + weights_margin
    # weights = 2*weights_like + 0.5*weights_entropy + weights_margin
    weights = weights_like + 0.1*weights_margin
    # weights = weights_like
    weights = weights[np.argsort(test_ids_all)]
    memory = 0.3*weights + 0.7*memory

    consistency_matrix.append(np.argmax(pred_probs_all,axis=1)[np.argsort(test_ids_all)])
    prob_matrix.append(pred_probs_all[np.argsort(test_ids_all)])
    true = test_labels_all[np.argsort(test_ids_all)]

    # print(weights[gt])
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
    plt.savefig(str(file_loc / f"results/cifar/cifar_n_sample_{EXP_NAME}.png"))

def consistency_metric():
    b = []
    for i in range(len(consistency_matrix.T)):
        temp = np.unique(consistency_matrix[:,i],return_counts=True)
        k = np.sort(temp[1])
        b.append(np.sum(k[:-1]))
    return np.array(b)

def filter_tophits_matrix(prob_matrix, test_labels, thresh=0.05, hits_thresh=3):
    nruns, nsamples, nclasses = np.shape(prob_matrix)
    prob_matrix_filt = np.where(prob_matrix<=thresh, 0, prob_matrix)
    h_run = []
    for run in range(nruns):
        #Replace items where prob < 0.03 to 0
        top_hit_preds = np.argsort(prob_matrix[run,:,:],axis=1)[:,::-1]
        hit_loc = np.where((top_hit_preds-test_labels.reshape(-1,1))==0)[1]
        hit_mask = np.sort(prob_matrix_filt[run,:,:],axis=1)[:,::-1]>0
        hit_loc_final = (hit_loc+1)*hit_mask[np.arange(len(hit_loc)),hit_loc]-1
        # score_array = np.array([i if i < top_hits else 0 for i in range(nclasses)])
        h_run.append(hit_loc_final)
    hits = np.array(h_run)
    hits = np.where(hits==-1,hits_thresh+1,hits)
    hits = np.where(hits>=hits_thresh+1,hits_thresh+1,hits)
    return hits

def filter_tophits(prob_matrix, test_labels, thresh=0.05, hits_thresh=3):
    # nruns, nsamples, nclasses = np.shape(prob_matrix)
    prob_matrix_filt = np.where(prob_matrix<=thresh, 0, prob_matrix)
    #Replace items where prob < 0.03 to 0
    top_hit_preds = np.argsort(prob_matrix,axis=1)[:,::-1]
    hit_loc = np.where((top_hit_preds-test_labels.reshape(-1,1))==0)[1]
    hit_mask = np.sort(prob_matrix_filt,axis=1)[:,::-1]>0
    hit_loc_final = (hit_loc+1)*hit_mask[np.arange(len(hit_loc)),hit_loc]-1
    # score_array = np.array([i if i < top_hits else 0 for i in range(nclasses)])
    hits = hit_loc_final
    hits = np.where(hits==-1,hits_thresh+1,hits)
    hits = np.where(hits>=hits_thresh+1,hits_thresh+1,hits)
    return hits

consistency_matrix = []
prob_matrix = []

if __name__ == '__main__':
    N_RUNS = 20
    N_FOLDS = 5
    TAU = 0.5
    NOISE_TYPE = 'aggre' # ['clean', 'random1', 'random2', 'random3', 'aggre', 'worst']
    FEAT = 'dinov2' # ['dinov2', 'imagenet', 'resnet']
    # FEAT = "dinov2"
    RANDOM_STATE = 1
    SUBSET_LENGTH = 50000
    MEMORY_NOISE_THRES = 0.4
    TOP_K = 4500
    #Dropping bottom 5% of the dataset
    NOISY_DROP = 0.5
    EXP_NAME = f"{NOISE_TYPE}_{TAU}_{FEAT}_{N_RUNS}_v5"

    if not (file_loc / "results/cifar").is_dir():
        os.mkdir(file_loc / "results/cifar")

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    cifar10n_pt = str(file_loc / 'data/CIFAR-N/CIFAR-10_human.pt')

    if FEAT == 'dinov2':
        cifar_h5 = str(file_loc / 'data/CIFAR-N/cifar_feats.h5')
    elif FEAT == 'imagenet':
        cifar_h5 = str(file_loc / 'data/CIFAR-N/cifar_feats_imagenet.h5')
    elif FEAT == 'resnet':
        cifar_h5 = str(file_loc / 'data/CIFAR-N/cifar_feats_resnet18.h5')
    else:
        print('Feature type not suppported.')


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
        # kf = KFold(n_splits=N_FOLDS, random_state=RANDOM_STATE+run, shuffle=True)
        # fold_splits = list(kf.split(X, y))
        aucs, test_ids, pred_probs, test_labels = train_one_run(fold_splits,filter_noise=True,noisy_idx=identified)
        print(f"Iteration {run}: {aucs}")
        # rank the ids
        memory, true = rank_weights(aucs, test_ids, pred_probs, test_labels, memory)
        # Generate new set of folds based on weights
        fold_splits, fold_ids = sample_folds(N_FOLDS, memory, TAU)
        # Get K worst samples for dropping from training
        # noise_set = np.argsort(memory)[:TOP_K]
        identified = np.argsort(memory)[:TOP_K]
        # identified = np.where(memory<=MEMORY_NOISE_THRES)[0]
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
    false_noisy = np.array(list(F.intersection(G)))
    false_clean = np.array(list(F_t.intersection(M)))
    true_noisy = np.array(list(F.intersection(M)))
    true_clean = np.array(list(F_t.intersection(G)))
    fn_ord = false_noisy[np.argsort(memory[false_noisy])]
    fc_ord = false_clean[np.argsort(memory[false_clean])]
    tn_ord = true_noisy[np.argsort(memory[true_noisy])]
    tc_ord = true_clean[np.argsort(memory[true_clean])]
    def get_vals(idx):
        print(consistency_matrix[:,idx])
        print(f"given label: {true[idx]}")
        print(f"clean label: {clean_label[idx]}")
        print(f"entropy : {entropy_mod[idx]}")
        print(f"margin : {margin_mod[idx]}")
        print(f"prob_matrix : {prob_matrix[-1,idx,:]}")
        print(f"memory : {memory[idx]}")
        print(f"top hits : {check[idx]}")
        print(f"hits: {hits[:,idx]}")
        # print(f"hits_mean : {np.mean(hits[:,idx])}")
    consistency_matrix = np.array(consistency_matrix)
    prob_matrix = np.stack(prob_matrix)
    margin_mod = calc_margin(prob_matrix[-1,:,:],y) + 1
    entropy = calc_entropy(prob_matrix[-1,:,:])
    preds = 1*(np.argmax(prob_matrix[-1,:,:],axis=1)==y).values
    entropy_mod = entropy*(2*preds-1) + 1
    check = catch_label(prob_matrix[-1,:,:],y.values)
    # hits = filter_tophits(prob_matrix, y.values)
    hits = filter_tophits_matrix(prob_matrix, y.values)

    identified = np.where(memory<=MEMORY_NOISE_THRES)[0]
    identified = identified[np.where(np.mean(hits[:,identified],axis=0)>=3)[0]]

    # identified_old = identified
    # identified_old = np.where(memory<=0.7)[0]
    # print(identified_old)
    # cons = consistency_metric()
    # identified = identified_old[np.where(cons[identified_old]<=1)[0]]

    # identified = np.argsort(memory)[:TOP_K]
    # identified = np.argsort(memory)[:4505]

    F = set(identified)
    G = set(range(0, len(y))) - set(gt)
    F_t = set(range(0, len(y))) - set(identified)
    M = set(gt)
    ER1 = len(F.intersection(G)) / len(G) #False noisy
    ER2 = len(F_t.intersection(M)) / len(M) #False good
    NEP = len(F.intersection(M)) / len(F)
    print("True noisy labels identified:{}\nFalse noisy: {}\nFalse good: {}".format(NEP, ER1, ER2))

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

    y_clean = clean_label
    y_clean = pd.Series(y_clean)
    # rf = RandomForestClassifier(n_jobs=-1)
    rf = LogisticRegression()
    print(f'--------Training RF model on clean data')
    rf.fit(X, y_clean)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc * 100:.3f}%')

    subset_length = 50000
    noisy = [0 if clean_label[x] == y[x] else 1 for x in range(subset_length)]
    a, b = np.unique(y[:subset_length], return_counts=True)
    print(b)
    X = X[:subset_length]
    y = y[:subset_length]

    target_indice = identified
    # target_indice_gt = gt # when testing upper bound

    X1 = np.delete(X.to_numpy(), target_indice, axis=0)
    y1 = np.delete(y.to_numpy(), target_indice, axis=0)
    X1 = pd.DataFrame(X1)
    y1 = pd.Series(y1)

    X2 = np.delete(X.to_numpy(), gt, axis=0)
    y2 = np.delete(y.to_numpy(), gt, axis=0)
    X2 = pd.DataFrame(X2)
    y2 = pd.Series(y2)

    # np.array(clean_label[target_indice])
    # np.array(aggre_label[target_indice])

    random_state = 1
    random.seed(random_state)
    np.random.seed(random_state)
    n_runs = 16000

    # rf = RandomForestClassifier(n_jobs=-1)
    rf = LogisticRegression()
    print(f'--------Training RF model on noisy data')
    rf.fit(X, y)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc * 100:.3f}%')

    # rf = RandomForestClassifier(n_jobs=-1)
    rf = LogisticRegression()
    print(f'--------Training RF model on recov cleaned data')
    rf.fit(X1, y1)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc * 100:.3f}%')

    # rf = RandomForestClassifier(n_jobs=-1)
    rf = LogisticRegression()
    print(f'--------Training RF model on gt cleaned data')
    rf.fit(X2, y2)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc * 100:.3f}%')
    import pickle

    # with open(str(file_loc / f"results/memory_cifar_{EXP_NAME}.npy"), "wb") as file:
    #     np.save(file, memory, allow_pickle=True)

