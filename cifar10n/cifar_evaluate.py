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
from sklearn.ensemble import RandomForestClassifier

file_loc = Path(__file__).resolve().parent.parent
random.seed(1)
np.random.seed(1)
cifar10n_pt = str(file_loc / 'data/CIFAR-N/CIFAR-10_human.pt')
cifar_h5 = str(file_loc / 'data/CIFAR-N/cifar_feats_resnet18.h5')
# cifar_h5 = str(file_loc / 'data/CIFAR-N/cifar_feats.h5')

NOISE_TYPE = "aggre"
SUBSET_LENGTH = 50000

# with open(str(file_loc / f"results/memory_cifarn_aggre_0.3.npy"),"rb") as file:
# with open(str(file_loc / f"results/memory_cifarn_worst_0.1_v2.npy"),"rb") as file:
# with open(str(file_loc / f"results/memory_cifarn_aggre_0.1_v2.npy"),"rb") as file:
with open(str(file_loc / f"results/cifar/memory_cifarn_resnet_aggre_0.6_v2.npy"),"rb") as file:
    memory = np.load(file)

identified = np.where(memory<=0.001)[0]
print(f"Number of identified samples: {len(identified)}")
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
# rf = RandomForestClassifier(n_jobs=-1)
rf = LogisticRegression()
print(f'--------Training RF model on clean data')
rf.fit(X, y_clean)

y_pred = rf.predict(X_test)
pred_proba = rf.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'ACC={acc * 100:.3f}%')

noisy = [0 if clean_label[x] == y[x] else 1 for x in range(SUBSET_LENGTH)]
a, b = np.unique(y[:SUBSET_LENGTH], return_counts=True)
print(b)
X = X[:SUBSET_LENGTH]
y = y[:SUBSET_LENGTH]

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