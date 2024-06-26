import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import warnings
warnings.filterwarnings("ignore")
import random
import h5py

import torch
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

file_loc = Path(__file__).resolve().parent.parent
random.seed(1)
np.random.seed(1)

def load_cifardataset(cifar10n_pt,cifar_h5,noise_type="aggre",subset_length=50000):
    """
    Load cifar10-N dataset with the extracted feature vectors for the images
    """
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

    if noise_type == 'clean':
        y = clean_label
    elif noise_type == 'aggre':
        y = aggre_label
    elif noise_type == 'random1':
        y = random_label1
    elif noise_type == 'random2':
        y = random_label2
    elif noise_type == 'random3':
        y = random_label3
    elif noise_type == 'worst':
        y = worst_label
    else:
        print('Noise type not recognized.')

    X = pd.DataFrame(X)
    X_test = pd.DataFrame(X_test)
    y = pd.Series(y)
    y_test = pd.Series(y_test)

    noisy = [0 if clean_label[x] == y[x] else 1 for x in range(subset_length)]
    gt = np.where(np.array(noisy)==1)[0]
    a, b = np.unique(y[:subset_length], return_counts=True)
    print(f"Labels distribution: {b}")
    X = X[:subset_length]
    y = y[:subset_length]

    print(f"Number of noisy samples: {np.sum(noisy)}")

    return X, y, X_test, y_test, noisy, gt, clean_label


def cifar_evaluate(cifar10n_dataset_path, cifar_features_path, identified, noise_type="aggre", subset_length=50000, seed=1):
    """
    Evaluates the identified index by removing them , training a new model and evaluating on the test set
    Parameters:
        cifar10n_dataset_path: CIFAR_10N dataset path
        cifar_features_path: path for extracted features
        identified: list of indices evaluated as noisy
    """
    
    random.seed(seed)
    np.random.seed(seed)

    print(f"Number of identified samples: {len(identified)}")
    X, y, X_test, y_test, noisy, gt, clean_label = load_cifardataset(cifar10n_dataset_path, cifar_features_path, noise_type, subset_length)

    F = set(identified)
    G = set(range(0, len(y))) - set(gt)
    F_t = set(range(0, len(y))) - set(identified)
    M = set(gt)
    ER1 = len(F.intersection(G)) / len(G)
    ER2 = len(F_t.intersection(M)) / len(M)
    NEP = len(F.intersection(M)) / len(F)
    F1 = len(F.intersection(M)) / (len(F.intersection(M)) + 0.5*(len(F.intersection(G))+len(F_t.intersection(M))))
    print("F1 score: {}\nTrue noisy labels identified:{}\nFalse Positive: {}\nFalse Negative: {}".format(F1,NEP, ER1, ER2))

    y_clean = clean_label
    y_clean = pd.Series(y_clean)
    rf = LogisticRegression()
    print(f'--------Training RF model on clean data')
    rf.fit(X, y_clean)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc * 100:.3f}%')

    noisy = [0 if clean_label[x] == y[x] else 1 for x in range(subset_length)]
    a, b = np.unique(y[:subset_length], return_counts=True)
    print(b)
    X = X[:subset_length]
    y = y[:subset_length]

    target_indice = identified

    X1 = np.delete(X.to_numpy(), target_indice, axis=0)
    y1 = np.delete(y.to_numpy(), target_indice, axis=0)
    X1 = pd.DataFrame(X1)
    y1 = pd.Series(y1)

    X2 = np.delete(X.to_numpy(), gt, axis=0)
    y2 = np.delete(y.to_numpy(), gt, axis=0)
    X2 = pd.DataFrame(X2)
    y2 = pd.Series(y2)

    rf = LogisticRegression()
    print(f'--------Training RF model on noisy data')
    rf.fit(X, y)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc * 100:.3f}%')

    rf = LogisticRegression()
    print(f'--------Training RF model on recov cleaned data')
    rf.fit(X1, y1)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc * 100:.3f}%')

    rf = LogisticRegression()
    print(f'--------Training RF model on gt cleaned data')
    rf.fit(X2, y2)

    y_pred = rf.predict(X_test)
    pred_proba = rf.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'ACC={acc * 100:.3f}%')